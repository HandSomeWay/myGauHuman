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

import numpy as np
from transform import transformVector3x3
import torch
import math
from diff_gaussian_rasterization import GaussianRasterizationSettings, GaussianRasterizer
from scene.gaussian_model import GaussianModel
from utils.sh_utils import eval_sh
from utils.graphics_utils import normal_from_depth_image

def render_normal(viewpoint_cam, depth, bg_color, alpha):
    # depth: (H, W), bg_color: (3), alpha: (H, W)
    # normal_ref: (3, H, W)
    intrinsic_matrix, extrinsic_matrix = viewpoint_cam.get_calib_matrix_nerf()

    normal_ref = normal_from_depth_image(depth, intrinsic_matrix.to(depth.device), extrinsic_matrix.to(depth.device))
    background = bg_color[None,None,...]
    normal_ref = normal_ref*alpha[...,None] + background*(1. - alpha[...,None])
 
    normal_ref = normal_ref.squeeze(0).permute(2,0,1)
    return normal_ref

def render(viewpoint_camera, pc : GaussianModel, pipe, bg_color : torch.Tensor, scaling_modifier = 1.0, override_color = None, return_smpl_rot=False, transforms=None, translation=None):
    """
    Render the scene. 
    
    Background tensor (bg_color) must be on GPU!
    """
 
    # Create zero tensor. We will use it to make pytorch return gradients of the 2D (screen-space) means
    screenspace_points = torch.zeros_like(pc.get_xyz, dtype=pc.get_xyz.dtype, requires_grad=True, device="cuda") + 0
    try:
        screenspace_points.retain_grad()
    except:
        pass

    # Set up rasterization configuration
    tanfovx = math.tan(viewpoint_camera.FoVx * 0.5)
    tanfovy = math.tan(viewpoint_camera.FoVy * 0.5)

    H = int(viewpoint_camera.image_height)
    W = int(viewpoint_camera.image_width)

    raster_settings = GaussianRasterizationSettings(
        image_height=H,
        image_width=W,
        tanfovx=tanfovx,
        tanfovy=tanfovy,
        bg=bg_color,
        scale_modifier=scaling_modifier,
        viewmatrix=viewpoint_camera.world_view_transform,
        projmatrix=viewpoint_camera.full_proj_transform,
        sh_degree=pc.active_sh_degree,
        campos=viewpoint_camera.camera_center,
        prefiltered=False,
        debug=pipe.debug
    )

    rasterizer = GaussianRasterizer(raster_settings=raster_settings)
    means3D = pc.get_xyz
    normal, delta_normal = pc.get_normal()
    if not pc.motion_offset_flag:
        _, means3D, _, transforms, _, world_normal = pc.coarse_deform_c2source(means3D[None], viewpoint_camera.smpl_param,
            viewpoint_camera.big_pose_smpl_param,
            viewpoint_camera.big_pose_world_vertex[None], normals=normal[None])
    else:
        if transforms is None:
            # pose offset
            dst_posevec = viewpoint_camera.smpl_param['poses'][:, 3:] #[1, 69]--[1, 3 * 23]
            pose_out = pc.pose_decoder(dst_posevec)
            correct_Rs = pose_out['Rs']

            # SMPL lbs weights
            lbs_weights = pc.lweight_offset_decoder(means3D[None].detach())
            lbs_weights = lbs_weights.permute(0,2,1)

            # transform points
            _, means3D, _, transforms, translation, world_normal = pc.coarse_deform_c2source(means3D[None], viewpoint_camera.smpl_param,
                viewpoint_camera.big_pose_smpl_param,
                viewpoint_camera.big_pose_world_vertex[None], lbs_weights=lbs_weights, correct_Rs=correct_Rs, return_transl=return_smpl_rot, normals=normal[None])
        else:
            correct_Rs = None
            means3D = torch.matmul(transforms, means3D[..., None]).squeeze(-1) + translation
            world_normal = torch.matmul(transforms, normal[..., None]).squeeze(-1)

    means3D = means3D.squeeze()
    means2D = screenspace_points
    opacity = pc.get_opacity
    albedo = pc.get_albedo
    roughness = pc.get_roughness
    metallic = pc.get_metallic
    occlusion = pc.get_occlusion
    assert albedo.shape[0] == roughness.shape[0] and albedo.shape[0] == metallic.shape[0]

    viewmatrix = viewpoint_camera.world_view_transform

    world_normal = world_normal.squeeze()
    # normal = world_normal


    # normal = pc.get_normal(dir_pp_normalized, transforms)  
    normal = transformVector3x3(world_normal, viewmatrix)
    normal = (normal * 0.5) + 0.5
    # normal = normal[:, [2, 0, 1]]

    # If precomputed 3d covariance is provided, use it. If not, then it will be computed from
    # scaling / rotation by the rasterizer.
    scales = None
    rotations = None
    cov3D_precomp = None
    if pipe.compute_cov3D_python:
        cov3D_precomp = pc.get_covariance(scaling_modifier, transforms.squeeze())
        # cov3D_precomp = pc.get_covariance(scaling_modifier)
    else:
        scales = pc.get_scaling
        rotations = pc.get_rotation

    # If precomputed colors are provided, use them. Otherwise, if it is desired to precompute colors
    # from SHs in Python, do it. If not, then SH -> RGB conversion will be done by rasterizer.
    shs = None
    colors_precomp = None
    if override_color is None:
        if pipe.convert_SHs_python:
            shs_view = pc.get_features.transpose(1, 2).view(-1, 3, (pc.max_sh_degree+1)**2)
            dir_pp = (means3D - viewpoint_camera.camera_center.repeat(pc.get_features.shape[0], 1))
            dir_pp_normalized = dir_pp/dir_pp.norm(dim=1, keepdim=True)
            sh2rgb = eval_sh(pc.active_sh_degree, shs_view, dir_pp_normalized)
            colors_precomp = torch.clamp_min(sh2rgb + 0.5, 0.0)
            

        else:
            shs = pc.get_features
    else:
        colors_precomp = override_color

    # Rasterize visible Gaussians to image, obtain their radii (on screen). 
    rendered_image, radii, depth, alpha = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = colors_precomp,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)

    rendered_normal, *_ = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = normal,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    
    rendered_albedo, *_ = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = albedo,
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    rendered_roughness, *_ = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = roughness.repeat(1, 3),
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    rendered_metallic, *_ = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = metallic.repeat(1, 3),
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    rendered_occlusion, *_ = rasterizer(
        means3D = means3D,
        means2D = means2D,
        shs = shs,
        colors_precomp = occlusion.repeat(1, 3),
        opacities = opacity,
        scales = scales,
        rotations = rotations,
        cov3D_precomp = cov3D_precomp)
    
    viewmatrix = viewpoint_camera.world_view_transform
    normal_ref = render_normal(viewpoint_cam=viewpoint_camera, depth=depth, bg_color=bg_color, alpha=alpha)

    normal_ref = transformVector3x3(normal_ref.permute(1, 2, 0).reshape(H * W, 3), viewmatrix).permute(1, 0).reshape(3, H, W)
    normal_ref = (normal_ref * 0.5) + 0.5
    occ = pc.occ_decoder(torch.cat((rendered_normal, depth), dim=0).unsqueeze(0)).squeeze(0)
    # print(normal_ref.shape)
    # Those Gaussians that were frustum culled or had a radius of 0 were not visible.
    # They will be excluded from value updates used in the splitting criteria.
    return {"render": rendered_image,
            "render_depth": depth,
            "render_alpha": alpha,
            "viewspace_points": screenspace_points,
            "visibility_filter" : radii > 0,
            "radii": radii,
            "transforms": transforms,
            "translation": translation,
            "correct_Rs": correct_Rs,
            "normal": rendered_normal,
            "albedo": rendered_albedo,
            "roughness":rendered_roughness,
            "metallic":rendered_metallic,
            "occlusion":rendered_occlusion,
            "occ":occ,
            "normal_ref": normal_ref,}
