import torch
import torch.nn as nn
import torch.nn.functional as F

class BRDF_Network(nn.Module):
    def __init__(self, brdf_input_dim=3, 
                 brdf_encoder_dims=[512, 512, 512, 512],
                 brdf_decoder_dims=[128, 128],
                 latent_dim=32, 
                 num_roughness_basis=1):
        super().__init__()

        self.n_basis = num_roughness_basis

        self.latent_dim = latent_dim
        self.actv_fn = nn.LeakyReLU(0.2)

        brdf_encoder_layer = []
        dim = brdf_input_dim
        for i in range(len(brdf_encoder_dims)):
            brdf_encoder_layer.append(nn.Linear(dim, brdf_encoder_dims[i]))
            brdf_encoder_layer.append(self.actv_fn)
            dim = brdf_encoder_dims[i]
        brdf_encoder_layer.append(nn.Linear(dim, self.latent_dim))
        self.brdf_encoder_layer = nn.Sequential(*brdf_encoder_layer)
        
        brdf_decoder_layer = []
        dim = self.latent_dim
        for i in range(len(brdf_decoder_dims)):
            brdf_decoder_layer.append(nn.Linear(dim, brdf_decoder_dims[i]))
            brdf_decoder_layer.append(self.actv_fn)
            dim = brdf_decoder_dims[i]
        brdf_decoder_layer.append(nn.Linear(dim, 3 + self.n_basis))
        self.brdf_decoder_layer = nn.Sequential(*brdf_decoder_layer)

        self.nn_num = 1
        self.nn_displacment = 0.0001
    def forward(self, points):
        
        points_nn = torch.cat([points + d for d in torch.randn((self.nn_num, 3)).cuda() * self.nn_displacment], dim=0)

        brdf_latent = self.brdf_encoder_layer(points)
        brdf_lc = torch.sigmoid(brdf_latent)
        brdf = self.brdf_decoder_layer(brdf_lc)
        roughness = torch.sigmoid(brdf[..., 3:])
        albedo = torch.sigmoid(brdf[..., :3])

        brdf_latent_nn = self.brdf_encoder_layer(points_nn)
        brdf_lc_nn = torch.sigmoid(brdf_latent_nn)
        brdf_nn = self.brdf_decoder_layer(brdf_lc_nn)
        roughness_nn = torch.sigmoid(brdf_nn[..., 3:])
        albedo_nn = torch.sigmoid(brdf_nn[..., :3])

        ret = dict([
            ('roughness', roughness),
            ('albedo', albedo),
            ('roughness_nn', roughness_nn),
            ('albedo_nn', albedo_nn),
            ('brdf_latent', brdf_latent),
        ])

        return ret
