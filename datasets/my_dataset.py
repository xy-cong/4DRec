import os
import os.path as osp
import glob
import json
import pickle

import torch
import trimesh
import numpy as np
from tqdm import tqdm
from omegaconf import OmegaConf
from loguru import logger
from sklearn.decomposition import PCA


def remove_nans(tensor):
    tensor_nan = torch.isnan(tensor[:, 3])
    return tensor[~tensor_nan, :]


class TempSdfDataset(torch.utils.data.Dataset):

    def __init__(self, 
                 mode,
                 rep,
                 config,
                 **kwargs):
        '''
        Args:
            sdf_dir: raw sdf dir
            raw_mesh_dir: raw mesh dir, might not have consistent topology
            registration_dir: registered mesh dir, must have consistent topology
            num_samples: num of samples used to train sdfnet
        '''
        super().__init__()

        self.rep = rep
        self.config = config
        self.mode = mode
        if self.mode == 'train':
            split = 'train'
        elif self.mode == 'test':
            split = 'test'
        else:
            raise ValueError('invalid mode')

        self.data_dir = config.data_dir
        self.sdf_dir = config.sdf_dir
        # self.raw_mesh_dir = config.raw_mesh_dir
        self.registration_dir = config.registration_dir
        self.num_samples = config.num_samples
        self.template_path = config.template_path
        self.dataset_type = config.dataset_type
        self.dataset_name = config.dataset_name

        # load data split
        split_cfg_fname = config.split_cfg[split]
        current_dir = os.path.dirname(os.path.realpath(__file__))
        split_path = f"{current_dir}/splits/{self.dataset_type}/{split_cfg_fname}"
        with open(split_path, "r") as f:
            split_names = json.load(f)

        self.fid_list = self.get_fid_list(split_names)
        
        self.num_data = len(self.fid_list)

        logger.info(f"dataset mode = {mode}, split = {split}, len = {self.num_data}\n")

        # load temlate mesh for meshnet. Share topology. NOTE: used for meshnet, different from temp(late) in sdfnet
        # template_mesh = trimesh.load(self.template_path, process=False, maintain_order=True)
        # self.template_points = torch.from_numpy(template_mesh.vertices)
        # self.template_faces = np.asarray(template_mesh.faces)
        # self.num_nodes = self.template_points.shape[0]


    # def get_fid_list(self, split_names):
    #     fid_list = []
    #     assert(len(split_names) == 1)
    #     for dataset in split_names:
    #         for class_name in split_names[dataset]:
    #             for instance_name in split_names[dataset][class_name]:
    #                 for shape in split_names[dataset][class_name][instance_name]:
    #                     fid = f"{class_name}-{instance_name}-{shape}"
    #                     fid_list.append(fid)
    #     return fid_list
    def get_fid_list(self, split_names):
        fid_list = []
        assert(len(split_names) == 1)

        for instance_name in split_names[self.dataset_name]:
            for shape in split_names[self.dataset_name][instance_name]:
                fid = f"{self.dataset_name}-{instance_name}-{shape}"
                fid_list.append(fid)
        return fid_list


    def update_pca_sv(self, train_pca_axes, train_pca_sv_mean, train_pca_sv_std):
        pca_sv = np.matmul(self.verts_init_nml.reshape(self.num_data, -1), train_pca_axes.transpose())
        self.pca_sv = (pca_sv - train_pca_sv_mean) / train_pca_sv_std


    def __len__(self):
        return self.num_data


    def __getitem__(self, idx):
        data_dict = {}
        data_dict['idx'] = idx
        fid = self.fid_list[idx]
        fname = '/'.join(fid.split('-'))

        samples_nonmnfld = torch.from_numpy(np.load(f"{self.sdf_dir}/{fname}.npy")).float()

        # load sdf data
        if self.rep in ['sdf']:
            _, _, L = samples_nonmnfld.shape
            samples_nonmnfld = samples_nonmnfld[..., :L-1].reshape(-1, L-1)
    
            if samples_nonmnfld[:,-1].min() < 0:
                samples_nonmnfld[:,-1] = samples_nonmnfld[:,-1].abs()
            # import ipdb; ipdb.set_trace()

            off_surface_points = torch.from_numpy(np.random.uniform(-1, 1, size=(self.num_samples, 3)).astype(np.float32)).float()
            data_dict['off_surface_points'] = off_surface_points


            random_idx = (torch.rand(self.num_samples) * samples_nonmnfld.shape[0]).long()
            samples_nonmnfld = torch.index_select(samples_nonmnfld, 0, random_idx)

            data_dict['points_mnfld'] = 0
            data_dict['normals_mnfld'] = 0
            data_dict['samples_nonmnfld'] = samples_nonmnfld

            # # load mesh data
            # raw_mesh = trimesh.load(f"{self.raw_mesh_dir}/{fname}.obj", process=False, maintain_order=True)
            # data_dict['raw_mesh_verts'] = np.asarray(raw_mesh.vertices).astype(np.float32)
            # data_dict['raw_mesh_faces'] = np.asarray(raw_mesh.faces)

            # temporal time step
            # ------------------------------------------------ Time ------------------------------------------ #
            data_dict['time'] = torch.FloatTensor([idx / self.num_data]).reshape(-1)
            # data_dict['time'] = torch.rand(1).reshape(-1)
            # ------------------------------------------------ Time ------------------------------------------ #

        return data_dict




