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


import os
import torch
from random import randint
from utils.loss_utils import l1_loss, l2_loss, ssim, predicted_normal_loss, delta_normal_loss
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state
import uuid
import imageio
import numpy as np
import cv2
import pickle
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

import lpips
loss_fn_vgg = lpips.LPIPS(net='vgg').to(torch.device('cuda', torch.cuda.current_device()))

import time
import torch.nn.functional as F
from pbr import CubemapLight, get_brdf_lut, pbr_shading
from gs_ir import recon_occlusion, IrradianceVolumes
from typing import Dict, List, Optional, Tuple, Union
import nvdiffrast.torch as dr
from torchvision.transforms import Grayscale

class GaussianHistogram(torch.nn.Module):
    def __init__(self, bins, min, max, sigma):
        super(GaussianHistogram, self).__init__()
        self.bins = bins
        self.min = min
        self.max = max
        self.sigma = sigma
        self.delta = float(max - min) / float(bins)
        self.centers = float(min) + self.delta * (torch.arange(bins, device=torch.device('cuda:0')).float() + 0.5)

    def forward(self, x):
        x = torch.unsqueeze(x, 0) - torch.unsqueeze(self.centers, 1)
        x = torch.exp(-0.5*(x/self.sigma)**2) / (self.sigma * np.sqrt(np.pi*2)) * self.delta
        x = x.sum(dim=1)
        return x
    
def get_entropy_loss(albedo):
    albedo_pred = albedo.reshape(albedo.shape[1] * albedo.shape[2], 3)
    # albedo_pred = albedo
    albedo_entropy = 0
    for i in range(3):
        channel = albedo_pred[..., i]
        hist = GaussianHistogram(8, 0., 1., sigma=torch.var(channel))
        h = hist(channel)
        if h.sum() > 1e-6:
            h = h.div(h.sum()) + 1e-6
        else:
            h = torch.ones_like(h).to(h)
        albedo_entropy += torch.sum(-h*torch.log(h))

    return albedo_entropy

def get_tv_loss(
    gt_image: torch.Tensor,  # [3, H, W]
    prediction: torch.Tensor,  # [C, H, W]
    pad: int = 1,
    step: int = 1,
) -> torch.Tensor:
    if pad > 1:
        gt_image = F.avg_pool2d(gt_image, pad, pad)
        prediction = F.avg_pool2d(prediction, pad, pad)
    rgb_grad_h = torch.exp(
        -(gt_image[:, 1:, :] - gt_image[:, :-1, :]).abs().mean(dim=0, keepdim=True)
    )  # [1, H-1, W]
    rgb_grad_w = torch.exp(
        -(gt_image[:, :, 1:] - gt_image[:, :, :-1]).abs().mean(dim=0, keepdim=True)
    )  # [1, H-1, W]
    tv_h = torch.pow(prediction[:, 1:, :] - prediction[:, :-1, :], 2)  # [C, H-1, W]
    tv_w = torch.pow(prediction[:, :, 1:] - prediction[:, :, :-1], 2)  # [C, H, W-1]
    tv_loss = (tv_h * rgb_grad_h).mean() + (tv_w * rgb_grad_w).mean()

    if step > 1:
        for s in range(2, step + 1):
            rgb_grad_h = torch.exp(
                -(gt_image[:, s:, :] - gt_image[:, :-s, :]).abs().mean(dim=0, keepdim=True)
            )  # [1, H-1, W]
            rgb_grad_w = torch.exp(
                -(gt_image[:, :, s:] - gt_image[:, :, :-s]).abs().mean(dim=0, keepdim=True)
            )  # [1, H-1, W]
            tv_h = torch.pow(prediction[:, s:, :] - prediction[:, :-s, :], 2)  # [C, H-1, W]
            tv_w = torch.pow(prediction[:, :, s:] - prediction[:, :, :-s], 2)  # [C, H, W-1]
            tv_loss += (tv_h * rgb_grad_h).mean() + (tv_w * rgb_grad_w).mean()

    return tv_loss


def get_masked_tv_loss(
    mask: torch.Tensor,  # [1, H, W]
    gt_image: torch.Tensor,  # [3, H, W]
    prediction: torch.Tensor,  # [C, H, W]
    erosion: bool = False,
) -> torch.Tensor:
    rgb_grad_h = torch.exp(
        -(gt_image[:, 1:, :] - gt_image[:, :-1, :]).abs().mean(dim=0, keepdim=True)
    )  # [1, H-1, W]
    rgb_grad_w = torch.exp(
        -(gt_image[:, :, 1:] - gt_image[:, :, :-1]).abs().mean(dim=0, keepdim=True)
    )  # [1, H-1, W]
    tv_h = torch.pow(prediction[:, 1:, :] - prediction[:, :-1, :], 2)  # [C, H-1, W]
    tv_w = torch.pow(prediction[:, :, 1:] - prediction[:, :, :-1], 2)  # [C, H, W-1]

    # erode mask
    mask = mask.float()
    if erosion:
        kernel = mask.new_ones([7, 7])
        mask = kornia.morphology.erosion(mask[None, ...], kernel)[0]
    mask_h = mask[:, 1:, :] * mask[:, :-1, :]  # [1, H-1, W]
    mask_w = mask[:, :, 1:] * mask[:, :, :-1]  # [1, H, W-1]

    tv_loss = (tv_h * rgb_grad_h * mask_h).mean() + (tv_w * rgb_grad_w * mask_w).mean()

    return tv_loss


def get_envmap_dirs(res: List[int] = [256, 512]) -> torch.Tensor:
    # gy, gx = torch.meshgrid(
    #     torch.linspace(0.0 + 1.0 / res[0], 1.0 - 1.0 / res[0], res[0], device="cuda"),
    #     torch.linspace(-1.0 + 1.0 / res[1], 1.0 - 1.0 / res[1], res[1], device="cuda"),
    #     indexing="ij",
    # )
    gy, gx = torch.meshgrid(
        torch.linspace(0.0, 1.0, res[0], device="cuda"),
        torch.linspace(-1.0 , 1.0, res[1], device="cuda"),
        indexing="ij",
    )

    sintheta, costheta = torch.sin(gy * np.pi), torch.cos(gy * np.pi)
    sinphi, cosphi = torch.sin(gx * np.pi), torch.cos(gx * np.pi)

    reflvec = torch.stack((sintheta * sinphi, costheta, -sintheta * cosphi), dim=-1)  # [H, W, 3]
    return reflvec


def resize_tensorboard_img(
    img: torch.Tensor,  # [C, H, W]
    max_res: int = 800,
) -> torch.Tensor:
    _, H, W = img.shape
    ratio = min(max_res / H, max_res / W)
    target_size = (int(H * ratio), int(W * ratio))
    transform = T.Resize(size=target_size)
    img = transform(img)  # [C, H', W']
    return img
def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    # torch.autograd.set_detect_anomaly(True)
    first_iter = 0
    pbr_iteration = 5000
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, dataset.smpl_type, dataset.motion_offset_flag, dataset.actor_gender)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    bound = 1.5
    # NOTE: prepare for PBR
    brdf_lut = get_brdf_lut().cuda()
    envmap_dirs = get_envmap_dirs()
    cubemap = CubemapLight(base_res=32, train=True).cuda()
    cubemap.train()

    aabb = torch.tensor([-bound, -bound, -bound, bound, bound, bound]).cuda()
    irradiance_volumes = IrradianceVolumes(aabb=aabb).cuda()
    irradiance_volumes.train()
    param_groups = [
        {
            "name": "irradiance_volumes",
            "params": irradiance_volumes.parameters(),
            "lr": opt.opacity_lr,
        },
        {"name": "cubemap", "params": cubemap.parameters(), "lr": opt.opacity_lr},
    ]
    light_optimizer = torch.optim.Adam(param_groups, lr=opt.opacity_lr)

    canonical_rays = scene.get_canonical_rays()
    viewpoint_stack = None
    ema_loss_for_log = 0.0
    Ll1_loss_for_log = 0.0
    mask_loss_for_log = 0.0
    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1

    # lpips_test_lst = []

    elapsed_time = 0
    for iteration in range(first_iter, opt.iterations + 1):        
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifer)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None

        with torch.no_grad():
            env_map = cubemap.export_envmap(return_img=True, res = [16, 32]).permute(2, 0, 1).clamp(min=0.0, max=1.0)
            # grayscale_transforms = Grayscale(num_output_channels=1)
            # env_map = grayscale_transforms(env_map)

        iter_start.record()

        gaussians.update_learning_rate(iteration, pbr_iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Start timer
        start_time = time.time()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = [i for i in range(len(scene.getTrainCameras()))]
        viewpoint_idx = viewpoint_stack.pop(randint(0, len(viewpoint_stack)-1))
        viewpoint_cam = scene.getTrainCameras()[viewpoint_idx]
        try:
            c2w = torch.inverse(viewpoint_cam.world_view_transform.T)  # [4, 4]
        except:
            continue

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True
        render_pkg = render(iteration, viewpoint_cam, gaussians, pipe, background, envmap=env_map)
        image, alpha, viewspace_point_tensor, visibility_filter, radii = render_pkg["render"], render_pkg["render_alpha"], render_pkg["viewspace_points"], render_pkg["visibility_filter"], render_pkg["radii"]
        normal = render_pkg["normal"]
        world_normal = render_pkg["world_normal"]
        albedo = render_pkg["albedo"]
        roughness = render_pkg["roughness"][0, ...].unsqueeze(0)
        metallic = render_pkg["metallic"][0, ...].unsqueeze(0)
        occlusion = render_pkg["occlusion"][0, ...].unsqueeze(0)
        # formulate roughness
        rmax, rmin = 1.0, 0.04
        roughness = roughness * (rmax - rmin) + rmin

        # NOTE: mask normal map by view direction to avoid skip value
        H, W = viewpoint_cam.image_height, viewpoint_cam.image_width
        view_dirs = -(
            (F.normalize(canonical_rays[:, None, :], p=2, dim=-1) * c2w[None, :3, :3])  # [HW, 3, 3]
            .sum(dim=-1)
            .reshape(H, W, 3)
        )  # [H, W, 3]


        # Loss
        gt_image = viewpoint_cam.original_image.cuda()
        gt_normal = viewpoint_cam.original_normal.cuda()
        bkgd_mask = viewpoint_cam.bkgd_mask.cuda()
        bound_mask = viewpoint_cam.bound_mask.cuda()
        loss: torch.Tensor
        
        if iteration <= pbr_iteration:
            
            scaling = gaussians.get_scaling
            scale_loss = torch.sum(torch.max(scaling - 0.015, torch.zeros_like(scaling)))

            point_posed = render_pkg['means3D']
            distance, _ = gaussians.knn_near_2(point_posed[None], point_posed[None])
            point_std = torch.std(distance[..., 1])

            Ll1 = l1_loss(image.permute(1,2,0)[bound_mask[0]==1], gt_image.permute(1,2,0)[bound_mask[0]==1])
            mask_loss = l2_loss(alpha[bound_mask==1], (bkgd_mask[0].unsqueeze(0))[bound_mask==1])
        
            normal_loss = l1_loss(normal.permute(1,2,0)[bound_mask[0]==1], gt_normal.permute(1,2,0)[bound_mask[0]==1])

            # crop the object region
            x, y, w, h = cv2.boundingRect(bound_mask[0].cpu().numpy().astype(np.uint8))
            img_pred = image[:, y:y + h, x:x + w].unsqueeze(0)
            img_gt = gt_image[:, y:y + h, x:x + w].unsqueeze(0)
            normal_pred = normal[:, y:y + h, x:x + w].unsqueeze(0)
            normal_gt = gt_normal[:, y:y + h, x:x + w].unsqueeze(0)
            # ssim loss
            ssim_loss = ssim(img_pred, img_gt)
            ssim_loss += ssim(normal_pred, normal_gt)
            # lipis loss
            lpips_loss = loss_fn_vgg(img_pred, img_gt).reshape(-1)
            lpips_loss += loss_fn_vgg(normal_pred, normal_gt).reshape(-1)


            loss = Ll1 + 0.1 * mask_loss + 0.01 * normal_loss + 0.01 * lpips_loss + 0.01 * (1.0 - ssim_loss) + 0.00 * scale_loss #+ point_std
            if tb_writer:
                tb_writer.add_scalar('train_loss_patches/normal_loss', normal_loss.item(), iteration)
                tb_writer.add_scalar('train_loss_patches/scale_loss', scale_loss.item(), iteration)
                tb_writer.add_scalar('train_loss_patches/point_std_loss', point_std.item(), iteration)
        else: # NOTE: PBR
            
            cubemap.build_mips() # build mip for environment light
            
            mask_loss = l2_loss(alpha[bound_mask==1], (bkgd_mask[0].unsqueeze(0))[bound_mask==1])
            pbr_result = pbr_shading(
                light=cubemap,
                normals=world_normal.permute(1, 2, 0).detach(),  # [H, W, 3]
                view_dirs=view_dirs,
                mask=alpha.permute(1, 2, 0),  # [H, W, 1]
                albedo=albedo.permute(1, 2, 0),  # [H, W, 3]
                roughness=roughness.permute(1, 2, 0),  # [H, W, 1]
                metallic=metallic.permute(1, 2, 0) if (metallic is not None) else None,  # [H, W, 1]
                tone=False,
                gamma=False,
                occlusion=occlusion.permute(1, 2, 0),
                brdf_lut=brdf_lut,
            )
            render_rgb = pbr_result["render_rgb"].permute(2, 0, 1)  # [3, H, W]
            # render_diffuse = render_pkg["diffuse"]
            # render_specular = pbr_result["specular_rgb"].permute(2, 0, 1)
            # render_rgb = render_diffuse + render_specular
            Ll1 = l1_loss(render_rgb.permute(1,2,0)[bound_mask[0]==1], gt_image.permute(1,2,0)[bound_mask[0]==1])
            
            x, y, w, h = cv2.boundingRect(bound_mask[0].cpu().numpy().astype(np.uint8))
            img_pred = render_rgb[:, y:y + h, x:x + w].unsqueeze(0)
            img_gt = gt_image[:, y:y + h, x:x + w].unsqueeze(0)
            ssim_loss = ssim(img_pred, img_gt)
            lpips_loss = loss_fn_vgg(img_pred, img_gt).reshape(-1)
            loss = Ll1 + 0.01 * (1.0 - ssim_loss) + 0.01 * lpips_loss

            ### BRDF loss
            if (alpha == 0).sum() > 0:
                brdf_tv_loss = get_masked_tv_loss(
                    alpha,
                    gt_image,  # [3, H, W]
                    torch.cat([albedo, roughness, metallic], dim=0),  # [5, H, W]
                )
            else:
                brdf_tv_loss = get_tv_loss(
                    gt_image,  # [3, H, W]
                    torch.cat([albedo, roughness, metallic], dim=0),  # [5, H, W]
                    pad=1,  # FIXME: 8 for scene
                    step=1,
                )
            entropy_loss = get_entropy_loss(albedo)
            # entropy_loss = get_entropy_loss(gaussians.get_albedo)
            loss += brdf_tv_loss * 1.0 + 0.0000 * entropy_loss
            lamb_loss = (1.0 - roughness[alpha > 0]).mean() + metallic[alpha > 0].mean() 
            loss += lamb_loss * 0.001

            if tb_writer:
                tb_writer.add_scalar('train_loss_patches/entropy_loss', entropy_loss.item(), iteration)
            #### envmap
            # TV smoothness
            envmap = dr.texture(
                cubemap.base[None, ...],
                envmap_dirs[None, ...].contiguous(),
                filter_mode="linear",
                boundary_mode="cube",
            )[
                0
            ]  # [H, W, 3]
            tv_h1 = torch.pow(envmap[1:, :, :] - envmap[:-1, :, :], 2).mean()
            tv_w1 = torch.pow(envmap[:, 1:, :] - envmap[:, :-1, :], 2).mean()
            env_tv_loss = tv_h1 + tv_w1
            env_tv_weight = 0.01
            loss += env_tv_loss * env_tv_weight

        loss.backward()

        # end time
        end_time = time.time()
        # Calculate elapsed time
        elapsed_time += (end_time - start_time)

        if (iteration in testing_iterations):
            print("[Elapsed time]: ", elapsed_time) 

        iter_end.record()

        with torch.no_grad():
            # Progress bar
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            Ll1_loss_for_log = 0.4 * Ll1.item() + 0.6 * Ll1_loss_for_log
            mask_loss_for_log = 0.4 * mask_loss.item() + 0.6 * mask_loss_for_log
            
            if iteration % 10 == 0:
                progress_bar.set_postfix({"#pts": gaussians._xyz.shape[0], "Ll1 Loss": f"{Ll1_loss_for_log:.{3}f}", "mask Loss": f"{mask_loss_for_log:.{2}f}"})
                                        #   "ssim": f"{ssim_loss_for_log:.{2}f}", "lpips": f"{lpips_loss_for_log:.{2}f}", "normal":f"{normal_loss_for_log:.{2}f}"})
                progress_bar.update(10)
            if iteration == opt.iterations:
                progress_bar.close()

            # Log and save
            training_report(tb_writer, iteration, Ll1, loss, l1_loss, iter_start.elapsed_time(iter_end), testing_iterations, scene, render, (pipe, background),
                            cubemap, env_map, brdf_lut, view_dirs)
            
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # Start timer
            start_time = time.time()
            # Densification
            if iteration < opt.densify_until_iter:
                # Keep track of max radii in image-space for pruning
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, kl_threshold=0.4, t_vertices=viewpoint_cam.big_pose_world_vertex, iter=iteration)
                    # gaussians.densify_and_prune(opt.densify_grad_threshold, 0.01, scene.cameras_extent, 1)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()
            
            gaussians.update_learning_rate(iteration, pbr_iteration)
            # Optimizer step
            if iteration < opt.iterations:
                gaussians.optimizer.step()
                gaussians.optimizer.zero_grad(set_to_none = True)
                if iteration >= pbr_iteration:
                    light_optimizer.step()
                    light_optimizer.zero_grad(set_to_none=True)
                    cubemap.clamp_(min=0.0)

            # end time
            end_time = time.time()
            # Calculate elapsed time
            elapsed_time += (end_time - start_time)

            # if (iteration in checkpoint_iterations):
            if (iteration in testing_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")
                torch.save({"cubemap":cubemap.state_dict(),}, scene.model_path + "/env_map" + str(iteration) + ".pth")
        
def prepare_output_and_logger(args):    
    if not args.model_path:
        args.model_path = os.path.join("./output/", args.exp_name)

        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        from datetime import datetime
        TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S/}".format(datetime.now())

        tb_writer = SummaryWriter(os.path.join(args.model_path, "Logs",TIMESTAMP))
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

def training_report(tb_writer, iteration, Ll1, loss, l1_loss, elapsed, testing_iterations, scene : Scene, renderFunc, renderArgs,
                    cubemap, light_map, brdf_lut, view_dirs):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/l1_loss', Ll1.item(), iteration)
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # Report test and samples of training set
    if iteration in testing_iterations:
        

        validation_configs = ({'name': 'test', 'cameras' : scene.getTestCameras()}, 
                              {'name': 'train', 'cameras' : scene.getTrainCameras()})

        smpl_rot = {}
        smpl_rot['train'], smpl_rot['test'] = {}, {}
        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0: 
                l1_test = 0.0
                psnr_test = 0.0
                ssim_test = 0.0
                lpips_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    
                    torch.cuda.empty_cache()
                    smpl_rot[config['name']][viewpoint.pose_id] = {}
                    render_output = renderFunc(iteration, viewpoint, scene.gaussians, *renderArgs, return_smpl_rot=True, envmap=light_map)
                    image = torch.clamp(render_output["render"], 0.0, 1.0)
                    normal = torch.clamp(render_output["normal"], 0.0, 1.0)
                    if iteration> 5000:
                        pbr_result = pbr_shading(
                            light=cubemap,
                            normals=render_output["world_normal"].permute(1, 2, 0).detach(),  # [H, W, 3]
                            view_dirs=view_dirs,
                            mask=render_output["render_alpha"].permute(1, 2, 0),  # [H, W, 1]
                            albedo=render_output["albedo"].permute(1, 2, 0),  # [H, W, 3]
                            roughness=render_output["roughness"].permute(1, 2, 0)[..., 0].unsqueeze(-1),  # [H, W, 1]
                            metallic=render_output["metallic"].permute(1, 2, 0)[..., 0].unsqueeze(-1),  # [H, W, 1]
                            tone=False,
                            gamma=False,
                            occlusion=render_output["occlusion"].permute(1, 2, 0)[..., 0].unsqueeze(-1),
                            brdf_lut=brdf_lut,
                        )
                        # render_diffuse = render_output["diffuse"]
                        render_diffuse = pbr_result["diffuse_rgb"].permute(2, 0, 1)
                        render_specular = pbr_result["specular_rgb"].permute(2, 0, 1)
                        # image = render_diffuse + render_specular
                        image = torch.clamp(pbr_result["render_rgb"], 0.0, 1.0).permute(2, 0, 1)
                    envmap = cubemap.export_envmap(return_img=True, res = [16, 32]).permute(2, 0, 1).clamp(min=0.0, max=1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    bound_mask = viewpoint.bound_mask
                    image.permute(1,2,0)[bound_mask[0]==0] = 0 if renderArgs[1].sum().item() == 0 else 1 
                    normal.permute(1,2,0)[bound_mask[0]==0] = 0 if renderArgs[1].sum().item() == 0 else 1 
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                        tb_writer.add_images("normal/" + config['name'] + "_view_{}".format(viewpoint.image_name), normal[None], global_step=iteration)
                        tb_writer.add_images("albedo/" + config['name'] + "_view_{}".format(viewpoint.image_name), render_output["albedo"][None], global_step=iteration)
                        tb_writer.add_images("occlusion/" + config['name'] + "_view_{}".format(viewpoint.image_name), render_output["occlusion"][None], global_step=iteration)
                        tb_writer.add_images("roughness/" + config['name'] + "_view_{}".format(viewpoint.image_name), render_output["roughness"][None], global_step=iteration)
                        tb_writer.add_images("metallic/" + config['name'] + "_view_{}".format(viewpoint.image_name), render_output["metallic"][None], global_step=iteration)
                        if iteration > 5000:
                            tb_writer.add_images("diffuse/" + config['name'] + "_view_{}".format(viewpoint.image_name), render_diffuse[None], global_step=iteration)
                            tb_writer.add_images("specular/" + config['name'] + "_view_{}".format(viewpoint.image_name), render_specular[None], global_step=iteration)
                        if idx < 1:
                            tb_writer.add_images("envmap_cube/" + config['name'] + "_view_{}".format(viewpoint.image_name), envmap[None], global_step=iteration)
                            tb_writer.add_images("envmap/" + config['name'] + "_view_{}".format(viewpoint.image_name), light_map[None], global_step=iteration)
                        if iteration == testing_iterations[0]:
                            tb_writer.add_images(config['name'] + "_view_{}/ground_truth".format(viewpoint.image_name), gt_image[None], global_step=iteration)
                    l1_test += l1_loss(image, gt_image).mean().double()
                    psnr_test += psnr(image, gt_image).mean().double()
                    ssim_test += ssim(image, gt_image).mean().double()
                    lpips_test += loss_fn_vgg(image, gt_image).mean().double()
                
                    smpl_rot[config['name']][viewpoint.pose_id]['transforms'] = render_output['transforms']
                    smpl_rot[config['name']][viewpoint.pose_id]['translation'] = render_output['translation']

                l1_test /= len(config['cameras']) 
                psnr_test /= len(config['cameras'])   
                ssim_test /= len(config['cameras'])
                lpips_test /= len(config['cameras'])      
                print("\n[ITER {}] Evaluating {} #{}: L1 {} PSNR {} SSIM {} LPIPS {}".format(iteration, config['name'], len(config['cameras']), l1_test, psnr_test, ssim_test, lpips_test))
                if tb_writer:
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - l1_loss', l1_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - psnr', psnr_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - ssim', ssim_test, iteration)
                    tb_writer.add_scalar(config['name'] + '/loss_viewpoint - lpips', lpips_test, iteration)

        # Store data (serialize)
        save_path = os.path.join(scene.model_path, 'smpl_rot', f'iteration_{iteration}')
        os.makedirs(save_path, exist_ok=True)
        with open(save_path+"/smpl_rot.pickle", 'wb') as handle:
            pickle.dump(smpl_rot, handle, protocol=pickle.HIGHEST_PROTOCOL)

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[3_000, 5_000, 7_000, 10_000, 15_000, 20_000, 25_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default= [3_000, 5_000, 7_000, 10_000, 15_000, 20_000, 25_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)
    # Initialize system state (RNG)
    safe_state(args.quiet)

    # Start GUI server, configure and run training
    # network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    # All done
    print("\nTraining complete.")
