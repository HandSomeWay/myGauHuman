import os
import itertools
import math
from argparse import ArgumentParser
from os import makedirs
from typing import Dict, List, Tuple

import imageio.v2 as imageio
import numpy as np
import nvdiffrast.torch as dr
import torch
import torch.nn.functional as F
from scene import Scene
from tqdm import trange
from diff_gaussian_rasterization import _C
from gs_ir import _C as gs_ir_ext
from tqdm import tqdm
import pickle
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel
from utils.graphics_utils import getProjectionMatrix
from utils.sh_utils import components_from_spherical_harmonics
from utils.general_utils import safe_state
import torchvision
from torchvision.transforms import Grayscale

def getWorld2ViewTorch(R: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    Rt = torch.zeros((4, 4), device=R.device)
    Rt[:3, :3] = R[:3, :3].T
    Rt[:3, 3] = t
    Rt[3, 3] = 1.0
    return Rt

# inverse the mapping from https://github.com/NVlabs/nvdiffrec/blob/dad3249af8ede96c7dd72c30328272117fabb710/render/light.py#L22
def get_envmap_dirs(res: List[int] = [16, 32]) -> Tuple[torch.Tensor, torch.Tensor]:
    # gy, gx = torch.meshgrid(
    #     torch.linspace(0.0, 1.0 - 1.0 / res[0], res[0], device="cuda"),
    #     torch.linspace(-1.0, 1.0 - 1.0 / res[1], res[1], device="cuda"),
    #     indexing="ij",
    # )
    gy, gx = torch.meshgrid(
        torch.linspace(0.0, 1.0, res[0], device="cuda"),
        torch.linspace(-1.0, 1.0, res[1], device="cuda"),
        indexing="ij",
    )
    d_theta, d_phi = np.pi / res[0], 2 * np.pi / res[1]

    sintheta, costheta = torch.sin(gy * np.pi), torch.cos(gy * np.pi)
    sinphi, cosphi = torch.sin(gx * np.pi), torch.cos(gx * np.pi)

    reflvec = torch.stack((sintheta * sinphi, costheta, -sintheta * cosphi), dim=-1)  # [H, W, 3]

    # get solid angles
    solid_angles = ((costheta - torch.cos(gy * np.pi + d_theta)) * d_phi)[..., None]  # [H, W, 1]
    
    return solid_angles, reflvec


def lookAt(eye: torch.Tensor, center: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    Z = F.normalize(eye - center, dim=0)
    Y = up
    X = F.normalize(torch.cross(Y, Z), dim=0)
    Y = F.normalize(torch.cross(Z, X), dim=0)

    matrix = torch.tensor(
        [
            [X[0], Y[0], Z[0]],
            [X[1], Y[1], Z[1]],
            [X[2], Y[2], Z[2]],
        ]
    )

    return matrix


def get_canonical_rays(H: int, W: int, tan_fovx: float, tan_fovy: float) -> torch.Tensor:
    cen_x = W / 2
    cen_y = H / 2
    focal_x = W / (2.0 * tan_fovx)
    focal_y = H / (2.0 * tan_fovy)

    x, y = torch.meshgrid(
        torch.arange(W),
        torch.arange(H),
        indexing="xy",
    )
    x = x.flatten()  # [H * W]
    y = y.flatten()  # [H * W]
    camera_dirs = F.pad(
        torch.stack(
            [
                (x - cen_x + 0.5) / focal_x,
                (y - cen_y + 0.5) / focal_y,
            ],
            dim=-1,
        ),
        (0, 1),
        value=1.0,
    )  # [H * W, 3]
    # NOTE: it is not normalized
    return camera_dirs.cuda()


def pc_to_grid(pc, res):
    # 确定边界
    min_coords = torch.min(pc, dim=0)[0]
    max_coords = torch.max(pc, dim=0)[0]
    
    # 计算网格大小
    x_size = (max_coords[0] - min_coords[0]) / res
    y_size = (max_coords[1] - min_coords[1]) / res
    z_size = (max_coords[2] - min_coords[2]) / res
    grid_sizes = torch.tensor([x_size, y_size, z_size], device=min_coords.device)
    
    # 计算网格索引
    pc_indices = torch.floor((pc - min_coords) / grid_sizes).long()
    
    # 确保索引不会超出网格范围
    pc_indices = torch.stack([
        torch.clamp(pc_indices[:, 0], min=0, max=res-1),
        torch.clamp(pc_indices[:, 1], min=0, max=res-1),
        torch.clamp(pc_indices[:, 2], min=0, max=res-1)
    ], dim=1)
    
    # 使用unique来找到非空的网格索引
    unique_indices, unique_inverse = torch.unique(pc_indices, return_inverse=True, dim=0)
    
    # 计算非空网格中心
    grid_centers = min_coords[None, :] + (unique_indices * grid_sizes[None, :]) + grid_sizes[None, :] / 2
    
    # 使用unique_inverse来获取每个点的网格索引
    pc_grid_indices = unique_inverse
    
    return grid_centers, grid_sizes, pc_grid_indices, unique_indices

def bake_set(view, gaussians, means3D, normal, H, W, light_map=None):
    
    # grayscale_transforms = Grayscale(num_output_channels=1)
    # light_map = grayscale_transforms(light_map)
    # Set up rasterization configuration
    res = 32
    bg_color = torch.zeros([3, res, res], device="cuda")

    # NOTE: capture 6 views with fov=90
    rotations: List[torch.Tensor] = [
        torch.tensor(
            [
                [0.0, 0.0, 1.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        ).cuda(),  # lookAt(torch.tensor([0, 0, 0]), torch.tensor([-1.0, 0.0, 0.0]), torch.tensor([0.0, -1.0, 0.0]))  [eye, center, up]
        torch.tensor(
            [
                [0.0, 0.0, -1.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        ).cuda(),  # lookAt(torch.tensor([0, 0, 0]), torch.tensor([1.0, 0.0, 0.0]), torch.tensor([0.0, -1.0, 0.0]))  [eye, center, up]
        torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        ).cuda(),  # lookAt(torch.tensor([0, 0, 0]), torch.tensor([0.0, -1.0, 0.0]), torch.tensor([0.0, 0.0, -1.0]))  [eye, center, up]
        torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        ).cuda(),  # lookAt(torch.tensor([0, 0, 0]), torch.tensor([0.0, 1.0, 0.0]), torch.tensor([0.0, 0.0, 1.0]))  [eye, center, up]
        torch.tensor(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        ).cuda(),  # lookAt(torch.tensor([0, 0, 0]), torch.tensor([0.0, 0.0, -1.0]), torch.tensor([0.0, 1.0, 0.0]))  [eye, center, up]
        torch.tensor(
            [
                [-1.0, 0.0, 0.0, 0.0],
                [0.0, -1.0, 0.0, 0.0],
                [0.0, 0.0, -1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
        ).cuda(),  # lookAt(torch.tensor([0, 0, 0]), torch.tensor([0.0, 0.0, 1.0]), torch.tensor([0.0, -1.0, 0.0]))  [eye, center, up]
    ]

    zfar = 5.0
    znear = 0.01
    projection_matrix = (
        getProjectionMatrix(znear=znear, zfar=zfar, fovX=math.pi * 0.5, fovY=math.pi * 0.5)
        .transpose(0, 1)
        .cuda()
    )

    points = means3D
    grid_centers, grid_sizes, pc_grid_indices, unique_indices = pc_to_grid(points, 10)
    num_grid = grid_centers.shape[0]
    # num_grid = points.shape[0]

    # prepare
    screenspace_points = (
        torch.zeros_like(
            gaussians.get_xyz, dtype=gaussians.get_xyz.dtype, requires_grad=False, device="cuda"
        )
        + 0
    )
    # means3D = gaussians.get_xyz
    means3D = points
    means2D = screenspace_points
    opacity = gaussians.get_opacity
    shs = gaussians.get_features
    scales = gaussians.get_scaling
    rots = gaussians.get_rotation

    (
        solid_angles,  # [H, W, 1]
        envmap_dirs,  # [H, W, 3]
    ) = get_envmap_dirs()


    with torch.no_grad():
        _occlusion = torch.zeros((opacity.shape[0], H, W, 1), device=opacity.device)
        dot_map = ((envmap_dirs * normal.unsqueeze(1).unsqueeze(1)).sum(dim=-1, keepdim=True))
        # _occlusion = torch.zeros((opacity.shape[0], 1), device=opacity.device)
        for grid_id in range(num_grid):
            grid_mask = (pc_grid_indices == grid_id).int()
            render_mask = pc_grid_indices != grid_id
            position = grid_centers[grid_id]
            # position = means3D[grid_id]
            opacity_cubemap = []
            # NOTE: crop by position
            valid_means3D = means3D[render_mask]
            valid_means2D = means2D[render_mask]
            valid_opacity = opacity[render_mask]
            valid_shs = shs[render_mask]
            valid_scales = scales[render_mask]
            valid_rots = rots[render_mask]
            for r_idx, rotation in enumerate(rotations):
                c2w = rotation
                c2w[:3, 3] = position
                w2c = torch.inverse(c2w)
                T = w2c[:3, 3]
                R = w2c[:3, :3].T
                world_view_transform = getWorld2ViewTorch(R, T).transpose(0, 1)
                full_proj_transform = (
                    world_view_transform.unsqueeze(0).bmm(projection_matrix.unsqueeze(0))
                ).squeeze(0)
                camera_center = world_view_transform.inverse()[3, :3]

                input_args = (
                    bg_color,               # background
                    valid_means3D,          # means3D
                    torch.Tensor([]),       # colors
                    valid_opacity,          # opacity
                    valid_scales,           # scales
                    valid_rots,             # rotations
                    1.0,                    # scale_modifier
                    torch.Tensor([]),       # cov3D_precomp
                    world_view_transform,   # viewmatrix,
                    full_proj_transform,    # projmatrix,
                    1.0,                    # tanfovx,
                    1.0,                    # tanfovy,
                    res,                    # image_height,
                    res,                    # image_width,
                    shs,                    # sh
                    gaussians.active_sh_degree,
                    camera_center,          # campos,
                    False,                  # prefiltered,
                    False,                  # debug,
                )
                (num_rendered, rendered_image, depth_map, opacity_map, radii, *_) = _C.rasterize_gaussians(
                    *input_args
                )
                # rgb_cubemap.append(rendered_image.permute(1, 2, 0))

                opacity_cubemap.append(opacity_map.permute(1, 2, 0))
                # depth_map = depth_map * (opacity_map > 0.5).float()  # NOTE: import to filter out the floater
                # depth_cubemap.append(depth_map.permute(1, 2, 0) * norm)

            # convert cubemap to HDRI
            opacity_envmap = dr.texture(
                torch.stack(opacity_cubemap)[None, ...],
                envmap_dirs[None, ...].contiguous(),
                # filter_mode="linear",
                filter_mode="nearest",
                boundary_mode="cube",
            )[
                 0
            ]  # [H, W, 1]
            # dot_map = torch.sum(envmap_dirs * (1 * normal[grid_id]), dim=2).unsqueeze(-1)
            # _occlusion[grid_id] = (dot_map * (1 - opacity_envmap) * light_map.permute(1, 2, 0)).sum()
            # _occlusion[grid_id] = dot_map * (1 - opacity_envmap)
            # _occlusion[grid_id] = F.interpolate((dot_map * (1 - opacity_envmap)).reshape(1, 1, 256, 512), size=(H, W), mode='bilinear', align_corners=False).reshape(H, W, 1).clamp(min=0.0, max=1.0)
            grid_mask_expanded = (grid_mask.unsqueeze(1).unsqueeze(1).unsqueeze(1)).expand(grid_mask.shape[0], H, W, 1)
            # opacity_bool = (1 - opacity_envmap) > 0
            _occlusion += grid_mask_expanded * (1 - opacity_envmap)
            # _occlusion += grid_mask_expanded * opacity_bool
        occlusion = dot_map/dot_map.max() * _occlusion
        view.set_occlusion(occlusion)
        return occlusion 


