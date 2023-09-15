import os, sys
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import trimesh
from loguru import logger

from pyutils import get_directory, to_device
from utils import implicit_utils


def save_obj(fname, vertices, faces):
    mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
    mesh.export(fname)


def train_one_epoch(state_info, config, train_loader, model, lat_vecs, optimizer_train, writer, latents_all=None):
    model.train()

    epoch = state_info['epoch']
    device = state_info['device']

    # ASAP
    if config.local_rank == 0 and config.use_sdf_asap:
        logger.warning("use ARAP/ASAP loss")

    for b, batch_dict in enumerate(train_loader):
        state_info['b'] = b
        optimizer_train.zero_grad()
        batch_dict = to_device(batch_dict, device)

        batch_dict = model(None, batch_dict, config, state_info) # (B, N, 3)
        batch_dict.update({k : v.mean() for k, v in batch_dict.items() if 'loss' in k})
        state_info.update({k : v.item() for k, v in batch_dict.items() if 'loss' in k})

        loss = batch_dict["loss"]
        loss.backward()
        optimizer_train.step()

        if config.local_rank == 0 and b % config.log.log_batch_interval == 0:
            global_step = (state_info['epoch'] * state_info['len_train_loader'] + b ) * config.optimization[config.rep].batch_size
            writer.log_state_info(state_info)
            writer.log_summary(state_info, global_step, mode='train')

    return state_info


def interp_from_lat_vecs(state_info, config, interp_loader, model, interp_lat_vecs, results_dir):
    '''
    Args:
        recon_lat_vecs: Embedding(D, latent_dim)
    '''
    model.eval()
    device = state_info['device']

    # import ipdb; ipdb.set_trace()

    if config.split == 'train':
        src_fid = config.get('interp_src_fid', None)  # 'bear-pose-00010'
        tgt_fid = config.get('interp_tgt_fid', None)  # 'bear-pose-00011'
    elif config.split == 'test':
        src_fid = config.get('interp_src_fid', None)
        tgt_fid = config.get('interp_tgt_fid', None)
    else:
        raise NotImplementedError

    if src_fid is None and tgt_fid is None:  # interpolate the longest sequence
        src_fid = interp_loader.dataset.fid_list[0]
        tgt_fid = interp_loader.dataset.fid_list[-1]
        num_interp = interp_loader.dataset.num_data * 2 - 2
    else:
        num_interp = 10

    logger.info(" interpolate sdf predicted by sdfnet ")

    for i, batch_dict in enumerate(interp_loader):
        fid = interp_loader.dataset.fid_list[batch_dict['idx'][0]]
        if fid not in [src_fid, tgt_fid]:
            continue
        
        batch_dict = to_device(batch_dict, device)
        if config.auto_decoder:
            assert(interp_lat_vecs is not None)
            latent_vec = batch_dict['time']

        assert(latent_vec.shape[0] == 1) # batch_size == 1
        latent_vec = latent_vec[0]
        if fid == src_fid:
            latent_src = latent_vec
            src_idx = i
        if fid == tgt_fid:
            latent_tgt = latent_vec
            tgt_idx = i

    logger.info(f"interpolate {src_fid} ({src_idx}-th) and {tgt_fid} ({tgt_idx}-th)")
    for i_interp in range(0, num_interp + 1): 
        ri = i_interp / num_interp

        latent_interp = latent_src * (1 - ri) + latent_tgt * ri

        dump_dir = get_directory( f"{results_dir}/{src_idx}_{tgt_idx}" )
        x_range, y_range, z_range = config.loss.get('x_range', [-1, 1]), config.loss.get('y_range', [-0.7, 1.7]), config.loss.get('z_range', [-1.1, 0.9])
        verts, faces = implicit_utils.sdf_decode_mesh_from_single_lat(model, latent_interp, resolution=128, max_batch=int(2 ** 17), offset=None, scale=None, x_range=x_range, y_range=y_range, z_range=z_range)

        save_obj(f"{dump_dir}/{src_idx}_{tgt_idx}_{i_interp:02d}.obj", verts, faces)

    def _copy_raw_mesh(_fid, _idx):
        _fname = '/'.join(_fid.split('-'))
        _fpath = f"{interp_loader.dataset.raw_mesh_dir}/{_fname}.obj"
        os.system(f"cp {_fpath} ./{dump_dir}/{_idx}.obj")

    _copy_raw_mesh(src_fid, src_idx)
    _copy_raw_mesh(tgt_fid, tgt_idx)

