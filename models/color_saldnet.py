import numpy as np
from loguru import logger

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import distributions as dist
import tinycudann as tcnn
from models.encoder import get_encoder

import tinycudann as tcnn

def maxpool(x, dim=-1, keepdim=False):
    out, _ = x.max(dim=dim, keepdim=keepdim)
    return out


class SimplePointnet(nn.Module):
    ''' PointNet-based encoder network.
    Args:
        c_dim (int): dimension of latent code c
        dim (int): input points dimension
        hidden_dim (int): hidden dimension of the network
    '''

    def __init__(self, c_dim=128, dim=3, hidden_dim=128):
        super().__init__()
        self.c_dim = c_dim

        self.fc_pos = nn.Linear(dim, 2 * hidden_dim)
        self.fc_0 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc_1 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc_2 = nn.Linear(2 * hidden_dim, hidden_dim)
        self.fc_3 = nn.Linear(2 * hidden_dim, hidden_dim)

        self.fc_mean = nn.Linear(hidden_dim, c_dim)
        self.fc_std = nn.Linear(hidden_dim, c_dim)
        
        torch.nn.init.constant_(self.fc_mean.weight, 0)
        torch.nn.init.constant_(self.fc_mean.bias, 0)

        torch.nn.init.constant_(self.fc_std.weight, 0)
        torch.nn.init.constant_(self.fc_std.bias, -10)

        self.actvn = nn.ReLU()
        self.pool = maxpool

    def forward(self, p):
        net = self.fc_pos(p)
        net = self.fc_0(self.actvn(net))
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.fc_1(self.actvn(net))
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.fc_2(self.actvn(net))
        pooled = self.pool(net, dim=1, keepdim=True).expand(net.size())
        net = torch.cat([net, pooled], dim=2)

        net = self.fc_3(self.actvn(net))

        net = self.pool(net, dim=1)

        c_mean = self.fc_mean(self.actvn(net))
        c_std = self.fc_std(self.actvn(net))

        return c_mean, c_std


class Implicit_Map(nn.Module):
    def __init__(
        self,
        latent_size,
        dims,
        norm_layers=(),
        latent_in=(),
        weight_norm=False,
        activation=None,
        xyz_dim=3,
        out_dim=1,
        geometric_init=True,
        beta=100,
        use_encoder=False,
        encoder=None,
        **kwargs
    ):
        super().__init__()

        bias = 1.0
        self.latent_size = latent_size

        self.use_encoder = use_encoder
        if use_encoder:
            self.encoder, xyz_dim = get_encoder(encoder)


        last_out_dim = out_dim
        dims = [latent_size + xyz_dim] + list(dims) + [out_dim]
        self.d_in = latent_size + xyz_dim
        self.latent_in = latent_in
        self.num_layers = len(dims)

        for l in range(0, self.num_layers - 1):
            if l + 1 in latent_in:
                out_dim = dims[l + 1] - dims[0]
            else:
                out_dim = dims[l + 1]

            lin = nn.Linear(dims[l], out_dim)

            if geometric_init:
                if l == self.num_layers - 2:
                    torch.nn.init.normal_(lin.weight, mean=np.sqrt(np.pi) / np.sqrt(dims[l]), std=0.0001)
                    torch.nn.init.constant_(lin.bias, -bias)
                else:
                    torch.nn.init.constant_(lin.bias, 0.0)
                    torch.nn.init.normal_(lin.weight, 0.0, np.sqrt(2) / np.sqrt(out_dim))
            if weight_norm:
                lin = nn.utils.weight_norm(lin)

            setattr(self, "lin" + str(l), lin)

        self.softplus = nn.Softplus(beta=beta)
        self.relu = nn.ReLU()

    def forward(self, inputs, latent):
        '''
        Args:
            inputs: (B, N, 3) or (N1+...+NB, 3)
            latent: (B, din) or (N1+...+NB, din)
        return:
            x: (B, N, 1) or (N1+...+NB, 1)
            of
            x: (B, N, 4) or (N1+...+NB, 4)
        '''
        assert(self.latent_size > 0)
        assert(len(latent.shape) == 2)
        assert(latent.shape[0] == inputs.shape[0])
        if len(inputs.shape) == 3:
            # inputs: (B, N, 3), latent: (B, din)
            B, N = inputs.shape[0], inputs.shape[1]
            inputs_con = latent.unsqueeze(1).repeat(1, N, 1) # (B, N, din)
        elif len(inputs.shape) == 2:
            # inputs: (N1+...+NB, 3), latent: (N1+...+NB, din)
            inputs_con = latent
        else:
            raise AssertionError
        
        if self.use_encoder:
            inputs = self.encoder(inputs)
        
        x = torch.cat([inputs, inputs_con], dim=-1) # (B, N, din + 3) or (N1+...+NB, din+3)

        to_cat = x

        for l in range(0, self.num_layers - 1):
            lin = getattr(self, "lin" + str(l))

            if l in self.latent_in:
                x = torch.cat([x, to_cat], -1) / np.sqrt(2)

            x = lin(x)

            if l < self.num_layers - 2:
                x = self.softplus(x)

        return x


class Color_NGP(nn.Module):
    def __init__(self):
        super().__init__()
        self.color_encoder = tcnn.Encoding(
            n_input_dims=4,
            encoding_config={
                "otype": "HashGrid",
                "n_levels": 16, # 16
                "n_features_per_level": 2, # 2
                "log2_hashmap_size": 19,
                "base_resolution": 16,
                "per_level_scale": 1.3819,
            },
        )
        self.color_backbone = nn.Sequential(
                nn.Linear(32, 64),
                nn.GELU(),
                nn.Linear(64, 64),
                nn.GELU(),
                nn.Linear(64, 3)
            )
        
    def forward(self, inputs, latent):
        assert(len(latent.shape) == 2)
        assert(latent.shape[0] == inputs.shape[0])
        if len(inputs.shape) == 3:
            # inputs: (B, N, 3), latent: (B, din)
            B, N = inputs.shape[0], inputs.shape[1]
            inputs_con = latent.unsqueeze(1).repeat(1, N, 1) # (B, N, din)
        elif len(inputs.shape) == 2:
            # inputs: (N1+...+NB, 3), latent: (N1+...+NB, din)
            inputs_con = latent
        else:
            raise AssertionError
        
        x = torch.cat([inputs, inputs_con], dim=-1) # (B, N, din + 3) or (N1+...+NB, din+3)
        x = (x+1)/2
        x = x.reshape(-1, 4)
        color_enc = self.color_encoder(x).float()
        color = self.color_backbone(color_enc)
        return color.reshape(B, N, 3)

class Implicit_Color_Map(nn.Module):
    def __init__(
        self,
        latent_size,
        dims,
        norm_layers=(),
        latent_in=(),
        weight_norm=False,
        activation=None,
        xyz_dim=3,
        geometric_init=True,
        beta=100,
        use_encoder=False,
        encoder=None,
        colornet=None,
        **kwargs
    ):
        super().__init__()

        self.sdf_net = Implicit_Map(
            latent_size=latent_size,
            dims=dims,
            norm_layers=norm_layers,
            latent_in=latent_in,
            weight_norm=weight_norm,
            activation=activation,
            xyz_dim=xyz_dim,
            out_dim=1,
            geometric_init=geometric_init,
            beta=100,
            use_encoder=use_encoder,
            encoder=encoder
        )

        self.color_net = Color_NGP()

        # self.color_net = Implicit_Map(
        #     latent_size=latent_size,
        #     dims=dims,
        #     norm_layers=norm_layers,
        #     latent_in=latent_in,
        #     weight_norm=weight_norm,
        #     activation=activation,
        #     xyz_dim=xyz_dim,
        #     out_dim=3,
        #     geometric_init=geometric_init,
        #     beta=100,
        #     use_encoder=use_encoder,
        #     encoder=encoder
        # )

        self.softplus = nn.Softplus(beta=beta)

    def forward(self, inputs, latent):
        # import ipdb; ipdb.set_trace()
        
        color = self.color_net(inputs, latent)
        sdf = self.sdf_net(inputs, latent)
        # sdf = torch.ones_like(color)
        ret = torch.cat([sdf, color], axis=-1)
        return ret