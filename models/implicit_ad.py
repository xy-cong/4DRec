import torch
import torch.nn as nn
import torch.nn.functional as F

import trimesh
import numpy as np

from models.saldnet import ImplicitMap
from utils.diff_operators import gradient
from utils import implicit_utils
from utils.time_utils import *
from models.asap import compute_asap3d_sparse

import torch_sparse as ts


class ImplicitGenerator(nn.Module):
    def __init__(self,
                 config,
                 dataset,
                 **kwargs,
                ):
        super().__init__()
        self.config = config
        self.dataset = dataset
        self.model_cfg = config.model.sdf
        self.auto_decoder = self.model_cfg.auto_decoder
        self.latent_size = self.model_cfg.decoder.latent_size

        assert(self.auto_decoder)
        self.decoder = ImplicitMap(**self.model_cfg.decoder)


    def forward(self, latent, batch_dict, config, state_info=None, only_encoder_forward=False, only_decoder_forward=False):
        '''
        Args:
            latent: (B, latent_dim), in the 4DRecon model, just set it None
        Notes:
            latent is generated from encoder or nn.Parameter that can be initialized
        '''
        assert(latent is None)

        points_mnfld = batch_dict['points_mnfld'] # (B, S, 3)
        normals_mnfld = batch_dict['normals_mnfld'] # (B, S, 3)
        points_nonmnfld = batch_dict['samples_nonmnfld'][:, :, :3].clone().detach().requires_grad_(True) # (B, S, 3)
        latent = batch_dict['time']  # [B, 1], IMPORTANT: serve as the latent code

        # decode latent to sdf
        sdf_nonmnfld = self.decoder(points_nonmnfld, latent)

        batch_dict['points_nonmnfld'] = points_nonmnfld # (B, S, 3)
        batch_dict['sdf_nonmnfld'] = sdf_nonmnfld # (B, S, 1)

        if state_info is not None:
            self.get_loss(latent, batch_dict, config, state_info)

        return batch_dict


    def get_loss(self, latent, batch_dict, config, state_info):
        epoch = state_info['epoch']
        device = batch_dict['points_mnfld'].device
        loss = torch.zeros(1, device=device) 
        assert(config.rep in ['sdf'])

        # sdf loss
        sdf_loss_type = config.loss.get('sdf_loss_type', 'L1')
        if sdf_loss_type == 'L1':
            sdf_loss = F.l1_loss(batch_dict['sdf_nonmnfld'][:, :, 0].abs(), batch_dict['samples_nonmnfld'][:, :, -1])
            sdf_loss = sdf_loss * config.loss.sdf_weight
        else:
            raise NotImplementedError

        loss += sdf_loss
        batch_dict['sdf_loss'] = sdf_loss
        state_info['sdf_loss'] = sdf_loss.item()

        # grad loss
        if config.use_sdf_grad and config.loss.grad_loss_weight > 0:
            grad_nonmnfld = gradient(batch_dict['sdf_nonmnfld'], batch_dict['points_nonmnfld']) # (B, S, 3)
            normals_nonmnfld_gt = batch_dict['samples_nonmnfld'][:, :, 3:6] # (B, S, 3)

            grad_loss = torch.min(torch.abs(grad_nonmnfld - normals_nonmnfld_gt).sum(-1),
                                  torch.abs(grad_nonmnfld + normals_nonmnfld_gt).sum(-1)).mean()
            grad_loss = grad_loss * config.loss.grad_loss_weight
            loss += grad_loss
            batch_dict['grad_loss'] = grad_loss
            state_info['grad_loss'] = grad_loss.item()

        # sdf asap loss
        if config.use_sdf_asap:
            assert(len(latent.shape) == 2)
            B = latent.shape[0]

            # sample latents
            sample_latent_space = config.loss.get('sample_latent_space', None)
            assert(sample_latent_space is not None)
            if sample_latent_space:
                sample_latent_space_type = config.loss.get('sample_latent_space_type', 'line')
                if sample_latent_space_type == 'line':
                    rand_idx = np.random.choice(B, size=(B,))
                    rand_ratio = torch.rand((B, 1), device=device)
                    batch_vecs = latent * rand_ratio + latent[rand_idx] * (1 - rand_ratio) # (B, d)
                    batch_dict['rand_idx'] = rand_idx
                    batch_dict['rand_ratio'] = rand_ratio
                else:
                    raise NotImplementedError
            else:
                batch_vecs = latent # (B, d)

            sdf_asap_loss = self.get_sdf_asap_loss(batch_vecs, config.loss, batch_dict=batch_dict)
            sdf_asap_loss = sdf_asap_loss.mean() * config.loss.sdf_asap_weight
            loss += sdf_asap_loss
            batch_dict['sdf_asap_loss'] = sdf_asap_loss
            state_info['sdf_asap_loss'] = sdf_asap_loss.item()

        batch_dict["loss"] = loss


    def extract_iso_surface(self, batch_vecs, cfg, batch_dict=None):
        """
        Args:
            batch_vecs: (B, d)
            N: resolution of sampled sdf
        Returns:
            batch_verts_idx: (n1+...+nB, )
            batch_faces_idx: (m1+...+mB, )
            batch_verts: (n1+...+nB, 3)
            batch_faces: (m1+...+mB, 3)
        """
        device = batch_vecs.device
        B = batch_vecs.shape[0]

        batch_verts_idx = []
        batch_faces_idx = []
        batch_verts = []
        batch_faces = []
        for b in range(B):
            x_range, y_range, z_range = cfg.get('x_range', [-1, 1]), cfg.get('y_range', [-0.7, 1.7]), cfg.get('z_range', [-1.1, 0.9])
            verts, faces = implicit_utils.sdf_decode_mesh_from_single_lat(self, batch_vecs[b], resolution=cfg.sdf_grid_size, voxel_size=None,
                                                                          max_batch=int(2 ** 18), offset=None, scale=None, points_for_bound=None, verbose=False,
                                                                          x_range=x_range, y_range=y_range, z_range=z_range)
            # denoise mesh, remove small connected components
            split_mesh_list = trimesh.graph.split(trimesh.Trimesh(vertices=verts, faces=faces), only_watertight=False, engine='scipy')
            largest_mesh_idx = np.argmax([split_mesh.vertices.shape[0] for split_mesh in split_mesh_list])
            verts = np.asarray(split_mesh_list[largest_mesh_idx].vertices)
            faces = np.asarray(split_mesh_list[largest_mesh_idx].faces)
            if cfg.get('simplify_mesh', False):
                mesh_sim = trimesh.Trimesh(vertices=verts, faces=faces).simplify_quadratic_decimation(2000)
                verts = mesh_sim.vertices
                faces = mesh_sim.faces

            verts = torch.from_numpy(verts).float().to(device)
            faces = torch.from_numpy(faces).long().to(device)

            batch_verts_idx.append(torch.ones_like(verts[:, 0]) * b) # (n_b,)
            batch_faces_idx.append(torch.ones_like(faces[:, 0]) * b) # (m_b,)
            batch_verts.append(verts) # (n_b, 3)
            batch_faces.append(faces) # (m_b, 3)

        batch_verts_idx = torch.cat(batch_verts_idx) # (n1+...+nB)
        batch_faces_idx = torch.cat(batch_faces_idx) # (m1+...+mB)
        batch_verts = torch.cat(batch_verts) # (n1+...+nB, 3)
        batch_faces = torch.cat(batch_faces) # (m1+...+mB, 3)

        return batch_verts_idx, batch_faces_idx, batch_verts, batch_faces


    def get_sdf_asap_loss(self, batch_vecs, cfg, batch_dict=None):
        """
        Args:
            batch_vecs: (B, d)
            N: resolution of sampled sdf
        """
        device = batch_vecs.device
        B = batch_vecs.shape[0]

        with torch.no_grad():
            batch_verts_idx, batch_faces_idx, batch_verts, batch_faces = self.extract_iso_surface(batch_vecs, cfg, batch_dict=None)

        batch_vecs_expand = []
        for b in range(B):
            n_b = torch.where(batch_verts_idx == b)[0].shape[0]
            batch_vecs_expand.append(batch_vecs[b:(b+1)].repeat(n_b, 1))
        batch_vecs_expand = torch.cat(batch_vecs_expand) # (n1+...+nB, d)

        # XXXXXX compute gradient XXXXXX
        batch_verts = batch_verts.clone().detach().requires_grad_(True) # (n1+...+nB, 3)
        batch_vecs_expand = batch_vecs_expand.clone().detach().requires_grad_(True) # (n1+...+nB, d)
        iso_sdf_pred = self.decoder(batch_verts, batch_vecs_expand) # (n1+...+NB, 1)

        fx = gradient(iso_sdf_pred, batch_verts) # (n1+...+nB, 3)
        fz = gradient(iso_sdf_pred, batch_vecs_expand) # (n1+...+nB, d)

        # XXXXXX compute regularization loss XXXXXX
        trace_list = []

        for b in range(B):
            batch_verts_mask = (batch_verts_idx == b)
            batch_faces_mask = (batch_faces_idx == b)
            verts_b = batch_verts[batch_verts_mask]
            faces_b = batch_faces[batch_faces_mask]
            n_b = verts_b.shape[0]

            # compute C
            fx_b = fx[batch_verts_mask] # (n_b, 3)
            C0, C1, C_vals = [], [], []
            lin_b = torch.arange(n_b, device=device)
            C0 = torch.stack((lin_b, lin_b, lin_b), dim=0).T.reshape(-1)
            C1 = torch.stack((lin_b * 3, lin_b * 3 + 1, lin_b * 3 + 2), dim=0).T.reshape(-1)
            C_vals = fx_b.reshape(-1)
            C_indices, C_vals = ts.coalesce([C0, C1], C_vals, n_b, n_b * 3)
            C = torch.sparse_coo_tensor(C_indices, C_vals, (n_b, 3*n_b))

            # compute F
            F = fz[batch_verts_mask] # (n_b, d)

            hessian_b = compute_asap3d_sparse(verts_b, faces_b, weight_asap=cfg.weight_asap) # (3*n_b, 3*n_b), sparse
            hessian_b = hessian_b.float()

            implicit_reg_type = cfg.get('implicit_reg_type', None)
            if implicit_reg_type == 'dense_inverse':
                hessian_b = hessian_b.to_dense()
                hessian_b = hessian_b + cfg.mu_asap * torch.eye(n_b * 3, device=device) if cfg.get('add_mu_diag_to_hessian', True) else hessian_b

                hessian_b_pinv = torch.linalg.inv(hessian_b)
                hessian_b_pinv = (hessian_b_pinv + hessian_b_pinv.T) / 2.0 # hessian_b_pinv is symmetric

                CH = ts.spmm(C_indices, C_vals, n_b, n_b * 3, hessian_b_pinv) # (n_b, 3*n_b)
                CHCT = ts.spmm(C_indices, C_vals, n_b, n_b * 3, CH.T) # (n_b, n_b)
                CHCT = (CHCT + CHCT.T) / 2
                CHCT = CHCT + cfg.mu_asap * torch.eye(n_b, device=device) # some row of C might be 0

                CHCT_inv = torch.linalg.inv(CHCT)
                CHCT_inv = (CHCT_inv + CHCT_inv.T) / 2

                R = F.T @ CHCT_inv @ F
            else:
                raise NotImplementedError

            e = torch.linalg.eigvalsh(R).clamp(0)
            e = e ** 0.5
            trace = e.sum()
            trace_list.append(trace)

        traces = torch.stack(trace_list)
        return traces

