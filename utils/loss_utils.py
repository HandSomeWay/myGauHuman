#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from utils.image_utils import erode
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp

import numpy as np

def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()

def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)

def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

def predicted_normal_loss(normal, normal_ref, alpha=None):
    """Computes the predicted normal supervision loss defined in ref-NeRF."""
    # normal: (3, H, W), normal_ref: (3, H, W), alpha: (3, H, W)
    if alpha is not None:
        device = alpha.device
        weight = alpha.detach().cpu().numpy()[0]
        weight = (weight*255).astype(np.uint8)

        weight = erode(weight, erode_size=4)

        weight = torch.from_numpy(weight.astype(np.float32)/255.)
        weight = weight[None,...].repeat(3,1,1)
        weight = weight.to(device) 
    else:
        weight = torch.ones_like(normal_ref)

    w = weight.permute(1,2,0).reshape(-1,3)[...,0].detach()
    # print("normal_ref" , normal_ref.shape)
    n = normal_ref.permute(1,2,0).reshape(-1,3).detach()    #(HxW, 3)
    n_pred = normal.permute(1,2,0).reshape(-1,3)    #(HxW, 3)
    loss = (w * (1.0 - torch.sum(n * n_pred, axis=-1))).mean()

    return loss

def kl_divergence(rho, rho_hat):
    rho_hat = torch.mean(torch.sigmoid(rho_hat), 0)
    rho = torch.tensor([rho] * len(rho_hat)).cuda()
    return torch.mean(rho * torch.log(rho/rho_hat) + (1 - rho) * torch.log((1 - rho)/(1 - rho_hat)))

def get_kl_loss(latent_values):
    latent_values = latent_values.view(-1, 32)
    loss = kl_divergence(0.05, latent_values)
    return loss

def get_albedo_smooth_loss(albedo_values, albedo_nn_values):
    pt_num = albedo_values.shape[-2]
    nn_num = albedo_nn_values.shape[-2] // pt_num
    albedo_values = albedo_values.view(-1, pt_num, 1, 3)
    albedo_nn_values = albedo_nn_values.view(-1, pt_num, nn_num, 3)
    albedo_diff = albedo_values - albedo_nn_values

    scale = torch.mean(albedo_nn_values, dim=-2, keepdim=True) + 1e-6
    loss = torch.mean(torch.abs(albedo_diff) / scale)

    return loss

def get_roughness_smooth_loss(roughness_values, roughness_nn_values):
    pt_num = roughness_values.shape[-2]
    nn_num = roughness_nn_values.shape[-2] // pt_num
    ch_num = roughness_values.shape[-1]
    roughness_values = roughness_values.view(-1, pt_num, 1, ch_num)
    roughness_nn_values = roughness_nn_values.view(-1, pt_num, nn_num, ch_num)
    diff = roughness_values - roughness_nn_values

    scale = torch.sum(roughness_nn_values, dim=-2, keepdim=True) + 1e-6
    loss = torch.mean(torch.abs(diff) / scale)

    return loss
