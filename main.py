import os
import sys
import shutil
import pickle
import argparse

import numpy as np
import os.path as osp
from psbody.mesh import Mesh
from collections import OrderedDict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.distributed as dist

from utils import utils, mesh_sampling
from utils.writer import Writer
from utils.scheduler_utils import StepLRSchedule, MultiplicativeLRSchedule, adjust_learning_rate
from utils import ddp_utils
from train_eval_unit import train_one_epoch, interp_from_lat_vecs
import datasets
from pyutils import *

from loguru import logger
from omegaconf import OmegaConf


def update_config_each_epoch(config, epoch):
    config.update({
        'use_sdf_grad': True,
        'use_sdf_asap': epoch >= config.loss.use_sdf_asap_epoch,
    })
    return config


def main(config):
    ##################################################
    # set up and deterministic
    ##################################################
    set_random_seed(config.seed)

    if config.launcher == 'none':
        config.dist_train = False
        total_gpus = 1
    else:
        total_gpus, config.local_rank = getattr(ddp_utils, 'init_dist_%s' % config.launcher)(
            config.tcp_port, config.local_rank, backend='nccl'
        )
        config.dist_train = True

    device = config.local_rank

    ##################################################
    # set up directory
    ##################################################
    config.config_path = os.path.normpath(config.config_path)
    config_path_sep = config.config_path.split(os.sep)
    assert(config_path_sep[0] == 'config' and config_path_sep[-1][-5:] == '.yaml')
    config_path_sep[-1] = config_path_sep[-1][:-5]
    exp_name = '/'.join(config_path_sep[1:])
    exp_dir = f"{config.work_dir}/{exp_name}"
    log_dir = get_directory( f"{exp_dir}/log" )
    ckpt_train_dir = get_directory( f"{log_dir}/ckpt_train/{config.rep}" )
    if config.mode == 'test_opt':
        ckpt_test_dir  = get_directory( f"{log_dir}/ckpt_test/{config.rep}_{config.epoch_continue}" )
    if config.local_rank == 0:
        if config.mode == 'test_opt':
            log_path = f"{log_dir}/{config.mode}_{config.rep}_log_{config.epoch_continue}.log"
        else:
            log_path = f"{log_dir}/{config.mode}_{config.rep}_log.log"
        logger.add(log_path, format="{message}", level="DEBUG")
        logger.info(config)
        logger.info(f"total_batch_size: {total_gpus * config.optimization[config.rep].batch_size}")
        shutil.copy2(config.config_path, log_dir)

    ##################################################
    # load dataset
    ##################################################
    MeshSdfDataset = load_module(f"datasets.{config.dataset.module_name}", config.dataset.class_name)
    train_mesh_sdf_dataset = MeshSdfDataset(mode='train', rep=config.rep, config=config.dataset)
    test_mesh_sdf_dataset  = MeshSdfDataset(mode='test',  rep=config.rep, config=config.dataset)

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_mesh_sdf_dataset) if config.dist_train else None
    test_opt_sampler = torch.utils.data.distributed.DistributedSampler(test_mesh_sdf_dataset) if config.dist_train else None # for test_opt

    train_loader = DataLoader(train_mesh_sdf_dataset, batch_size=config.optimization[config.rep].batch_size, 
                              shuffle=(train_sampler is None), pin_memory=True, num_workers=config.num_workers, sampler=train_sampler,
                              collate_fn=train_mesh_sdf_dataset.collate_batch if hasattr(train_mesh_sdf_dataset, 'collate_batch') else None)
    test_opt_loader = DataLoader(test_mesh_sdf_dataset, batch_size=config.optimization[config.rep].batch_size,
                                 shuffle=(test_opt_sampler is None), pin_memory=True, num_workers=config.num_workers, sampler=test_opt_sampler,
                                 collate_fn=test_mesh_sdf_dataset.collate_batch if hasattr(test_mesh_sdf_dataset, 'collate_batch') else None)

    ##################################################
    # setup model
    ##################################################
    Net = load_module("models." + config.model[config.rep]["module_name"], config.model[config.rep]["class_name"])
    model = Net(config=config, dataset=train_mesh_sdf_dataset)
    model = model.to(device)

    if config.local_rank == 0:
        logger.info(f"load dataset: datasets.{config.dataset.module_name}.{config.dataset.class_name}")
        logger.info(model)
    config.auto_decoder = config.model[config.rep].auto_decoder
    if config.mode == 'test_opt' or config.use_test_opt_model:
        config.auto_decoder = True

    ##################################################
    # initialized latent codes
    # NOTE: currently for the temporal model, we don't # need a learnable latent code,
    # we use a fixed time step as latent code, here we define latent for code compatibility
    ##################################################
    lat_vecs = None
    test_lat_vecs = None
    if config.rep == 'sdf':
        if config.auto_decoder:
            # TODO: design better initialization method
            lat_vecs      = torch.nn.Embedding(len(train_mesh_sdf_dataset), config.latent_dim)
            test_lat_vecs = torch.nn.Embedding(len(test_mesh_sdf_dataset),  config.latent_dim)
            torch.nn.init.normal_(lat_vecs.weight.data, 0.0, 1.0 / np.sqrt(config.latent_dim))
            torch.nn.init.normal_(test_lat_vecs.weight.data, 0.0, 1.0 / np.sqrt(config.latent_dim))
            lat_vecs      = lat_vecs.to(device)
            test_lat_vecs = test_lat_vecs.to(device)

    ##################################################
    #  setup learning rate scheduler and optimizer
    ##################################################
    if config.mode == 'train':
        lr_group_init_train = [ config.optimization[config.rep].lr]
        opt_params_group_init_train = [ { "params": model.parameters(), "lr": config.optimization[config.rep].lr, } ]

        scheduler_train = MultiplicativeLRSchedule(lr_group_init_train,
                                                   config.optimization.sdf.gammas,
                                                   config.optimization.sdf.milestones)
        optimizer_train = torch.optim.Adam(opt_params_group_init_train)

    ##################################################
    #  setup writer and state_info
    ##################################################
    writer = Writer(log_dir, config)

    state_info = {} # store epoch, basic training info, loss & so on.
    state_info['device'] = device
    state_info['len_train_loader'] = len(train_loader)
    state_info['len_test_opt_loader']  = len(test_opt_loader)


    ##################################################
    #  different mode: train, interp, ...
    ##################################################
    if config.mode == 'train':
        logger.info('-' * 30 + f" train {config.local_rank}" + '-' * 30 )

        start_epoch = 0
        end_epoch = config.optimization[config.rep].num_epochs

        if config.epoch_continue is not None:
            start_epoch = 1 + writer.load_checkpoint(f"{ckpt_train_dir}/checkpoint_{config.epoch_continue:05d}.pt",
                                                     model, lat_vecs, optimizer_train)
            logger.info(f"continue to train from previous epoch = {start_epoch}")

        if config.dist_train:
            model = nn.parallel.DistributedDataParallel(model, device_ids=[config.local_rank])
        if config.data_parallel:
            model = torch.nn.DataParallel(model)

        for epoch in range(start_epoch, end_epoch):
            lr_group = adjust_learning_rate(scheduler_train, optimizer_train, epoch)
            state_info.update( {'epoch': epoch, 'lr': lr_group} )

            config = update_config_each_epoch(config, epoch)

            if train_sampler is not None:
                train_sampler.set_epoch(epoch)

            train_one_epoch(state_info, config, train_loader, model, lat_vecs, optimizer_train, writer)

            # save checkpoint
            if config.local_rank == 0:
                model_module = model.module if config.dist_train or config.data_parallel else model 
                if (epoch + 1) % config.log.save_epoch_interval == 0: 
                    writer.save_checkpoint(f"{ckpt_train_dir}/checkpoint_{epoch:05d}.pt", epoch, model_module, lat_vecs, optimizer_train)
                if (epoch + 1) % config.log.save_latest_epoch_interval == 0: 
                    writer.save_checkpoint(f"{ckpt_train_dir}/checkpoint_latest.pt", epoch, model_module, lat_vecs, optimizer_train)


    elif config.mode in ['interp']:
        assert(not config.dist_train)
        logger.info('>' * 30 + f' {config.mode} ' + '>' * 30)
        if config.split == 'train':
            interp_lat_vecs = torch.nn.Embedding(len(train_mesh_sdf_dataset), config.latent_dim).to(device) if config.auto_decoder else None
            mesh_sdf_dataset = train_mesh_sdf_dataset
        elif config.split == 'test':
            interp_lat_vecs = torch.nn.Embedding(len(test_mesh_sdf_dataset), config.latent_dim).to(device) if config.auto_decoder else None
            mesh_sdf_dataset = test_mesh_sdf_dataset
        else:
            raise ValueError("split is train or test")
        interp_loader = DataLoader(mesh_sdf_dataset, batch_size=1, shuffle=False, pin_memory=True, num_workers=config.num_workers, 
                                   collate_fn=mesh_sdf_dataset.collate_batch if hasattr(mesh_sdf_dataset, 'collate_batch') else None)

        epoch = config.epoch_continue

        results_dir = f"{exp_dir}/results/{config.split}/{config.mode}_{config.rep}/{epoch}"
        if config.auto_decoder:
            if config.split == 'train':
                writer.load_checkpoint(f"{log_dir}/ckpt_{config.split}/{config.rep}/checkpoint_{config.epoch_continue:05d}.pt", model, interp_lat_vecs)
            else:
                # NOTE: epoch_continue is the train epoch not the test_opt epoch here
                writer.load_checkpoint(f"{log_dir}/ckpt_{config.split}/{config.rep}_{config.epoch_continue}/checkpoint_{config.test_opt_epoch:05d}.pt", model, interp_lat_vecs)
                results_dir = f"{exp_dir}/results/{config.split}/{config.mode}_{config.rep}_opt/{epoch}"
        else:
            writer.load_checkpoint(f"{log_dir}/ckpt_train/{config.rep}/checkpoint_{config.epoch_continue:05d}.pt", model)

        results_dir = get_directory(results_dir)

        interp_from_lat_vecs(state_info, config, interp_loader, model, interp_lat_vecs, results_dir)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", dest="config_path", required=True, help='config file path')
    parser.add_argument("--continue_from", dest="epoch_continue", type=int, help='epoch of loaded ckpt, so checkpoint_{epoch:05d}.pt is loaded')
    parser.add_argument("--mode", type=str, required=True, help='train, test_opt, recon, interp, eval')
    parser.add_argument("--split", type=str, default='test', help='{train, test}, use train or test dataset')
    parser.add_argument("--rep", type=str, required=True, help='{mesh, sdf, all}, representation to train/test/recon')
    # DDP
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm'], default='none')
    parser.add_argument('--tcp_port', type=int, default=18888, help='tcp port for distrbuted training')
    parser.add_argument('--local_rank', type=int, default=0, help='local rank for distributed training')
    parser.add_argument('--data_parallel', action='store_true', help='use DataParallel or not')
    # train
    parser.add_argument("--batch_size", type=int, default=None, help='if specified, it will override batch_size in the config')
    # evaluate
    parser.add_argument('--parallel_idx', type=int, default=-1, help='if parallel_idx larger than 0: [parallel_idx*parallel_interval, (parallel_idx+1)*parallel_interval)')
    parser.add_argument('--parallel_interval', type=int, default=100, help='if parallel_idx larger than 0: [parallel_idx*parallel_interval, (parallel_idx+1)*parallel_interval)')
    # interpolate lat_vecs
    parser.add_argument('--interp_src_fid', type=str, default=None, help='src_fid for interpolation')
    parser.add_argument('--interp_tgt_fid', type=str, default=None, help='tgt_fid for interpolation')
    # interpolate edges
    parser.add_argument('--edge_ids_path', type=str, default=None, help='path to edge_ids, .npy file')
    # use test_opt model
    parser.add_argument('--use_test_opt_model', action='store_true', help='all model and dir are primarily based on test_opt model')
    parser.add_argument("--test_opt_epoch", type=int, help='epoch of test_opt')
    args = parser.parse_args()

    config = OmegaConf.load(args.config_path)
    OmegaConf.resolve(config)
    update_config_from_args(config, args)

    assert(config.rep == 'sdf'), "Mesh Not Implemented"

    if args.batch_size is not None:
        config.optimization[config.rep].batch_size = args.batch_size
    if config.rep == 'mesh':
        config.latent_dim = config.latent_dim_mesh
    
    main(config)


