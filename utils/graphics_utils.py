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

import torch
import math
import numpy as np
from typing import NamedTuple

class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array

def geom_transform_points(points, transf_matrix):
    P, _ = points.shape
    ones = torch.ones(P, 1, dtype=points.dtype, device=points.device)
    points_hom = torch.cat([points, ones], dim=1)
    points_out = torch.matmul(points_hom, transf_matrix.unsqueeze(0))

    denom = points_out[..., 3:] + 0.0000001
    return (points_out[..., :3] / denom).squeeze(dim=0)

def getWorld2View(R, t):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return np.float32(Rt)

def getWorld2View2(R, t, translate=np.array([.0, .0, .0]), scale=1.0):
    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R.transpose()
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = C2W[:3, 3]
    cam_center = (cam_center + translate) * scale
    C2W[:3, 3] = cam_center
    Rt = np.linalg.inv(C2W)
    return np.float32(Rt)

def getProjectionMatrix(znear, zfar, fovX, fovY):
    tanHalfFovY = math.tan((fovY / 2))
    tanHalfFovX = math.tan((fovX / 2))

    top = tanHalfFovY * znear
    bottom = -top
    right = tanHalfFovX * znear
    left = -right

    P = torch.zeros(4, 4)

    z_sign = 1.0
    P[0, 0] = 2.0 * znear / (right - left)
    P[1, 1] = 2.0 * znear / (top - bottom)
    P[0, 2] = (right + left) / (right - left)
    P[1, 2] = (top + bottom) / (top - bottom)
    P[3, 2] = z_sign
    P[2, 2] = z_sign * zfar / (zfar - znear)
    P[2, 3] = -2 * (zfar * znear) / (zfar - znear) #-(zfar * znear) / (zfar - znear)
    return P

# tensor([[ 2.0463e+00,  0.0000e+00,  0.0000e+00,  0.0000e+00],
#         [ 0.0000e+00,  2.0546e+00,  0.0000e+00,  0.0000e+00],
#         [ 0.0000e+00,  0.0000e+00,  1.0000e+00, -1.0000e-03],
#         [ 0.0000e+00,  0.0000e+00,  1.0000e+00,  0.0000e+00]]) [12/10 09:44:14]

# tensor([[ 2.0463e+00,  0.0000e+00,  3.9123e-02,  0.0000e+00],
#         [ 0.0000e+00,  2.0546e+00, -5.0103e-03,  0.0000e+00],
#         [ 0.0000e+00,  0.0000e+00,  1.0000e+00, -2.0000e-03],
#         [ 0.0000e+00,  0.0000e+00,  1.0000e+00,  0.0000e+00]], device='cuda:0') [12/10 09:42:17]

def getProjectionMatrix_refine(K: torch.Tensor, H, W, znear=0.001, zfar=1000):
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    s = K[0, 1]
    P = torch.zeros(4, 4, dtype=K.dtype, device=K.device)
    z_sign = 1.0

    P[0, 0] = 2 * fx / W
    P[0, 1] = 2 * s / W
    P[0, 2] = -1 + 2 * (cx / W)

    P[1, 1] = 2 * fy / H
    P[1, 2] = -1 + 2 * (cy / H)

    P[2, 2] = z_sign * (zfar + znear) / (zfar - znear)
    P[2, 3] = -1 * z_sign * 2 * zfar * znear / (zfar - znear) # z_sign * 2 * zfar * znear / (zfar - znear)
    P[3, 2] = z_sign

    return P

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def focal2fov(focal, pixels):
    return 2*math.atan(pixels/(2*focal))


def ndc_2_cam(ndc_xyz, intrinsic, W, H):
    inv_scale = torch.tensor([[W - 1, H - 1]], device=ndc_xyz.device)
    cam_z = ndc_xyz[..., 2:3]
    cam_xy = ndc_xyz[..., :2] * inv_scale * cam_z
    cam_xyz = torch.cat([cam_xy, cam_z], dim=-1)
    cam_xyz = cam_xyz @ torch.inverse(intrinsic[0, ...].t())
    return cam_xyz

def depth2point_cam(sampled_depth, ref_intrinsic):
    B, N, C, H, W = sampled_depth.shape
    valid_z = sampled_depth
    valid_x = torch.arange(W, dtype=torch.float32, device=sampled_depth.device) / (W - 1)
    valid_y = torch.arange(H, dtype=torch.float32, device=sampled_depth.device) / (H - 1)
    valid_y, valid_x = torch.meshgrid(valid_y, valid_x)
    # B,N,H,W
    valid_x = valid_x[None, None, None, ...].expand(B, N, C, -1, -1)
    valid_y = valid_y[None, None, None, ...].expand(B, N, C, -1, -1)
    ndc_xyz = torch.stack([valid_x, valid_y, valid_z], dim=-1).view(B, N, C, H, W, 3)  # 1, 1, 5, 512, 640, 3
    cam_xyz = ndc_2_cam(ndc_xyz, ref_intrinsic, W, H) # 1, 1, 5, 512, 640, 3
    return ndc_xyz, cam_xyz

def depth2point_world(depth_image, intrinsic_matrix, extrinsic_matrix):
    # depth_image: (H, W), intrinsic_matrix: (3, 3), extrinsic_matrix: (4, 4)
    _, xyz_cam = depth2point_cam(depth_image[None,None,None,...], intrinsic_matrix[None,...])
    xyz_cam = xyz_cam.reshape(-1,3)
    xyz_world = torch.cat([xyz_cam, torch.ones_like(xyz_cam[...,0:1])], axis=-1) @ torch.inverse(extrinsic_matrix).transpose(0,1)
    xyz_world = xyz_world[...,:3]

    return xyz_world

def depth_pcd2normal(xyz):
    # print(xyz.shape)
    # exit()
    hd, wd, _ = xyz.shape 

    mask = xyz[..., 1:hd-1, 1:wd-1, 2] != 0
    # print(xyz.shape)
    # print(mask.shape)

    # print(xyz[..., 2:hd,   1:wd-1, :].shape)
    # print(xyz[..., 2:hd,   1:wd-1, :][mask.reshape(510,510,1)].shape)
    bottom_point = xyz[..., 2:hd,   1:wd-1, :]#[mask].reshape(510, 510, 3)
    top_point    = xyz[..., 0:hd-2, 1:wd-1, :]#[mask].reshape(510, 510, 3)
    right_point  = xyz[..., 1:hd-1, 2:wd,   :]#[mask].reshape(510, 510, 3)
    left_point   = xyz[..., 1:hd-1, 0:wd-2, :]#[mask].reshape(510, 510, 3)
    left_to_right = right_point - left_point
    bottom_to_top = top_point - bottom_point 
    # print(left_to_right.shape)
    xyz_normal = torch.cross(left_to_right, bottom_to_top, dim=-1)
    xyz_normal = torch.nn.functional.normalize(xyz_normal, p=2, dim=-1)
    xyz_normal = torch.nn.functional.pad(xyz_normal.permute(2,0,1), (1,1,1,1), mode='constant').permute(1,2,0)
    return xyz_normal

def normal_from_depth_image(depth, intrinsic_matrix, extrinsic_matrix):
    # depth: (H, W), intrinsic_matrix: (3, 3), extrinsic_matrix: (4, 4)
    # xyz_normal: (H, W, 3)
    depth = depth.squeeze(0)
    xyz_world = depth2point_world(depth, intrinsic_matrix, extrinsic_matrix) # (HxW, 3)
    xyz_world = xyz_world.reshape(*depth.shape, 3)
    xyz_normal = depth_pcd2normal(xyz_world)

    return xyz_normal
