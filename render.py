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

from typing import Dict, List, Tuple
import torch
import torch.nn.functional as F
from scene import Scene
import os
import time
import pickle
from tqdm import tqdm
from os import makedirs
from gaussian_renderer import render
import torchvision
from utils.general_utils import safe_state
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, get_combined_args
from gaussian_renderer import GaussianModel

from utils.image_utils import psnr
from utils.loss_utils import ssim
import lpips
from pbr import CubemapLight, get_brdf_lut, pbr_shading
import cv2
import numpy as np
import nvdiffrast.torch as dr
from torchvision.transforms import Grayscale

from testcopy import SSAO

loss_fn_vgg = lpips.LPIPS(net='vgg').to(torch.device('cuda', torch.cuda.current_device()))


def read_hdr(path: str) -> np.ndarray:
    """Reads an HDR map from disk.

    Args:
        path (str): Path to the .hdr file.

    Returns:
        numpy.ndarray: Loaded (float) HDR map with RGB channels in order.
    """
    with open(path, "rb") as h:
        buffer_ = np.frombuffer(h.read(), np.uint8)
    bgr = cv2.imdecode(buffer_, cv2.IMREAD_UNCHANGED)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    return rgb


def cube_to_dir(s: int, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if s == 0:
        rx, ry, rz = torch.ones_like(x), -y, -x
    elif s == 1:
        rx, ry, rz = -torch.ones_like(x), -y, x
    elif s == 2:
        rx, ry, rz = x, torch.ones_like(x), y
    elif s == 3:
        rx, ry, rz = x, -torch.ones_like(x), -y
    elif s == 4:
        rx, ry, rz = x, -y, torch.ones_like(x)
    elif s == 5:
        rx, ry, rz = -x, -y, -torch.ones_like(x)
    return torch.stack((rx, ry, rz), dim=-1)


def latlong_to_cubemap(latlong_map: torch.Tensor, res: List[int]) -> torch.Tensor:
    cubemap = torch.zeros(
        6, res[0], res[1], latlong_map.shape[-1], dtype=torch.float32, device="cuda"
    )
    for s in range(6):
        gy, gx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, res[0], device="cuda"),
            torch.linspace(-1.0, 1.0, res[1], device="cuda"),
            indexing="ij",
        )
        v = F.normalize(cube_to_dir(s, gx, gy), p=2, dim=-1)

        tu = torch.atan2(v[..., 0:1], -v[..., 2:3]) / (2 * np.pi) + 0.5
        tv = torch.acos(torch.clamp(v[..., 1:2], min=-1, max=1)) / np.pi
        texcoord = torch.cat((tu, tv), dim=-1)

        # cubemap[s, ...] = v
        cubemap[s, ...] = dr.texture(
            latlong_map[None, ...], texcoord[None, ...], filter_mode="linear"
        )[0]
    return cubemap


def render_set(model_path, name, iteration, views, gaussians, pipeline, background, scene=None):
   

    render_path = os.path.join(model_path, name, "ours_{}".format(iteration), "renders")
    gts_path = os.path.join(model_path, name, "ours_{}".format(iteration), "gt")
    normal_path = os.path.join(model_path, name, "ours_{}".format(iteration), "normal")
    render_normal_path = os.path.join(model_path, name, "ours_{}".format(iteration), "render_normal")
    world_normal_path = os.path.join(model_path, name, "ours_{}".format(iteration), "world_normal")
    render_albedo_path = os.path.join(model_path, name, "ours_{}".format(iteration), "render_albedo")
    render_roughness_path = os.path.join(model_path, name, "ours_{}".format(iteration), "render_roughness")
    render_pbr_path = os.path.join(model_path, name, "ours_{}".format(iteration), "render_pbr")
    render_diffuse_path = os.path.join(model_path, name, "ours_{}".format(iteration), "render_diffuse")
    render_specular_path = os.path.join(model_path, name, "ours_{}".format(iteration), "render_specular")
    render_depth_path = os.path.join(model_path, name, "ours_{}".format(iteration), "render_depth")
    render_alpha_path = os.path.join(model_path, name, "ours_{}".format(iteration), "render_alpha")
    render_ao_path = os.path.join(model_path, name, "ours_{}".format(iteration), "render_ao")

    makedirs(render_path, exist_ok=True)
    makedirs(gts_path, exist_ok=True)
    makedirs(normal_path, exist_ok=True)
    makedirs(render_normal_path, exist_ok=True)
    makedirs(world_normal_path, exist_ok=True)
    makedirs(render_albedo_path, exist_ok=True)
    makedirs(render_roughness_path, exist_ok=True)
    makedirs(render_pbr_path, exist_ok=True)
    makedirs(render_diffuse_path, exist_ok=True)
    makedirs(render_specular_path, exist_ok=True)
    makedirs(render_depth_path, exist_ok=True)
    makedirs(render_alpha_path, exist_ok=True)
    makedirs(render_ao_path, exist_ok=True)

    cubemap = CubemapLight(base_res=32).cuda()

    # Read HDR file as novel light.
    # hdri = read_hdr("/home/shangwei/data/my_HDR_map/blaubeuren_night_2k.hdr")
    # hdri = torch.from_numpy(hdri).cuda()

    # Read EXR file as novel light.
    # os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
    # image = cv2.imread('/home/shangwei/data/mixamo/envmaps/sunset.exr', cv2.IMREAD_UNCHANGED)
    # # image = cv2.imread('/home/shangwei/codes/RelightableAvatar/relighting/0000.exr', cv2.IMREAD_UNCHANGED)
    # # image = np.power(image, 1./2.2)
    # hdri = torch.from_numpy(image).cuda()[:,:,[2,1,0]]
    # # cubemap.base.data = latlong_to_cubemap(ldri[..., :3].contiguous(), [32, 32])
    # cubemap.base.data = latlong_to_cubemap(hdri[..., :3].contiguous(), [32, 32])

    # # Regularize the novel light.
    # new_cubemap = torch.zeros_like(cubemap.base)
    # new_cubemap[0] = cubemap.base[5].transpose(0, 1).flip(0)
    # new_cubemap[1] = cubemap.base[4].transpose(0, 1).flip(1)
    # new_cubemap[2] = cubemap.base[1].flip(0).flip(1)
    # new_cubemap[3] = cubemap.base[0]
    # new_cubemap[4] = cubemap.base[2].transpose(0, 1).flip(1)
    # new_cubemap[5] = cubemap.base[3].transpose(0, 1).flip(1)
    # cubemap.base.data = new_cubemap
    # cubemap.eval()

    # Reconstructed light
    cubemap_path = model_path + f'/env_map{iteration}.pth'
    checkpoint = torch.load(cubemap_path)
    cubemap_params = checkpoint["cubemap"]
    cubemap.load_state_dict(cubemap_params)
    
    cubemap.build_mips()
    
    # envmap = cubemap.export_envmap(return_img=True).permute(2, 0, 1).clamp(min=0.0, max=1.0)
    envmap_path = os.path.join(model_path, name, "ours_{}".format(iteration), "envmap.png")
    envmap = cubemap.export_envmap(return_img=True, res = [16, 32]).permute(2, 0, 1).clamp(min=0.0, max=1.0)
    torchvision.utils.save_image(envmap, envmap_path)
    grayscale_transforms = Grayscale(num_output_channels=1)
    envmap = grayscale_transforms(envmap)
    # Load data (deserialize)
    with open(model_path + '/smpl_rot/' + f'iteration_{iteration}/' + 'smpl_rot.pickle', 'rb') as handle:
        smpl_rot = pickle.load(handle)

    rgbs = []
    rgbs_gt = []
    rgbs_normal = []
    rgbs_normal_rd = []
    rgbs_normal_wd = []
    albedo_rd = []
    roughness_rd = []
    pbr_rd = []
    pbr_diffuse = []
    pbr_specular = []
    depth_rd = []
    alpha_rd = []
    ao_rd = []
    elapsed_time = 0
    for _, view in enumerate(tqdm(views, desc="Rendering progress")):
        gt = view.original_image[0:3, :, :].cuda()
        gt_normal = view.original_normal[0:3, :, :].cuda()
        
        if 'zju' in model_path:
            gt_normal = (gt_normal * 2) - 1.
            gt_normal[2, ...] = -gt_normal[2, ...]
            gt_normal = (gt_normal + 1) / 2.
        bound_mask = view.bound_mask
        transforms, translation = smpl_rot[name][view.pose_id]['transforms'], smpl_rot[name][view.pose_id]['translation']

        # Start timer
        start_time = time.time() 

        render_output = render(iteration, view, gaussians, pipeline, background, transforms=transforms, translation=translation, envmap=envmap)
        rendering = render_output["render"]
        render_alpha = render_output["render_alpha"]
        render_normal = render_output["normal"]
        world_normal = render_output["world_normal"]
        render_albedo = render_output["albedo"]
        render_roughness = render_output["roughness"]
        render_occlusion = render_output["occlusion"]
        render_depth = render_output["render_depth"]

        # ssao = SSAO(render_depth, render_normal)
        if iteration > 3000 :
            alpha = render_output["render_alpha"]
            H, W = view.image_height, view.image_width
            ref_view = views[0]
            c2w = torch.inverse(ref_view.world_view_transform.T)  # [4, 4]
            if scene:
                canonical_rays = scene.get_canonical_rays()
            view_dirs = -(
                (F.normalize(canonical_rays[:, None, :], p=2, dim=-1) * c2w[None, :3, :3])  # [HW, 3, 3]
                .sum(dim=-1)
                .reshape(H, W, 3)
            )  # [H, W, 3]

            brdf_lut = get_brdf_lut().cuda()
            pbr_result = pbr_shading(
                light=cubemap,
                normals=world_normal.permute(1, 2, 0).detach(),  # [H, W, 3]
                # normals=normal.permute(1, 2, 0).detach(),  # [H, W, 3]
                view_dirs=view_dirs,
                mask=alpha.permute(1, 2, 0),  # [H, W, 1]
                albedo=render_albedo.permute(1, 2, 0),  # [H, W, 3]
                # albedo=albedo.permute(1, 2, 0),  # [H, W, 3]
                roughness=render_roughness[0, ...].unsqueeze(0).permute(1, 2, 0),  # [H, W, 1]
                metallic=None,  # [H, W, 1]
                tone=False,
                gamma=False,
                occlusion=render_occlusion.permute(1, 2, 0)[..., 0][..., None],    # [H, W, 1]
                brdf_lut=brdf_lut,
            )
            render_pbr = pbr_result["render_rgb"].clamp(min=0.0, max=1.0).permute(2, 0, 1) # [3, H, W]
            render_diffuse = pbr_result["diffuse_rgb"].clamp(min=0.0, max=1.0).permute(2, 0, 1) # [3, H, W]
            # render_diffuse = render_output['diffuse']
            render_specular = pbr_result["specular_rgb"].clamp(min=0.0, max=1.0).permute(2, 0, 1) # [3, H, W]
            


            # render_pbr = render_diffuse + render_specular

            render_ao = render_occlusion.clamp(min=0.0, max=1.0) # [1, H, W]
            render_pbr.permute(1,2,0)[bound_mask[0]==0] = 0 if background.sum().item() == 0 else 1
            render_diffuse.permute(1,2,0)[bound_mask[0]==0] = 0 if background.sum().item() == 0 else 1
            render_specular.permute(1,2,0)[bound_mask[0]==0] = 0 if background.sum().item() == 0 else 1
            render_ao.permute(1,2,0)[bound_mask[0]==0] = 0 if background.sum().item() == 0 else 1
            pbr_rd.append(render_pbr)
            pbr_diffuse.append(render_diffuse)
            pbr_specular.append(render_specular)
            ao_rd.append(render_ao)

        # end time
        end_time = time.time()
        # Calculate elapsed time
        elapsed_time += end_time - start_time

        rendering.permute(1,2,0)[bound_mask[0]==0] = 0 if background.sum().item() == 0 else 1
        render_normal.permute(1,2,0)[bound_mask[0]==0] = 0 if background.sum().item() == 0 else 1
        world_normal.permute(1,2,0)[bound_mask[0]==0] = 0 if background.sum().item() == 0 else 1
        render_albedo.permute(1,2,0)[bound_mask[0]==0] = 0 if background.sum().item() == 0 else 1
        render_roughness.permute(1,2,0)[bound_mask[0]==0] = 0 if background.sum().item() == 0 else 1
        render_depth.permute(1,2,0)[bound_mask[0]==0] = 0 if background.sum().item() == 0 else 1
        render_alpha.permute(1,2,0)[bound_mask[0]==0] = 0 if background.sum().item() == 0 else 1

        rgbs.append(rendering)
        rgbs_gt.append(gt)
        # rgbs_normal.append(normal)
        rgbs_normal.append(gt_normal)
        rgbs_normal_rd.append(render_normal)
        rgbs_normal_wd.append(world_normal)
        albedo_rd.append(render_albedo)
        roughness_rd.append(render_roughness)
        depth_rd.append(render_depth)
        alpha_rd.append(render_alpha)


    # Calculate elapsed time
    print("Elapsed time: ", elapsed_time, " FPS: ", len(views)/elapsed_time) 

    psnrs = 0.0
    ssims = 0.0
    lpipss = 0.0

    for id in range(len(views)):
        rendering = rgbs[id]
        gt = rgbs_gt[id]
        normal = rgbs_normal[id]
        render_normal = rgbs_normal_rd[id]
        world_normal = rgbs_normal_wd[id]
        render_albedo = albedo_rd[id]
        render_roughness = roughness_rd[id]
        render_depth = depth_rd[id]
        render_alpha = alpha_rd[id]


        rendering = torch.clamp(rendering, 0.0, 1.0)
        gt = torch.clamp(gt, 0.0, 1.0)
        normal = torch.clamp(normal, 0.0, 1.0)
        render_normal = torch.clamp(render_normal, 0.0, 1.0)
        world_normal = torch.clamp(world_normal, 0.0, 1.0)
        render_albedo = torch.clamp(render_albedo, 0.0, 1.0)
        render_roughness = torch.clamp(render_roughness, 0.0, 1.0)
        render_depth = torch.clamp(render_depth, 0.0, 1.0)
        render_alpha = torch.clamp(render_alpha, 0.0, 1.0)

        torchvision.utils.save_image(rendering, os.path.join(render_path, '{0:05d}'.format(id) + ".png"))
        torchvision.utils.save_image(gt, os.path.join(gts_path, '{0:05d}'.format(id) + ".png"))
        torchvision.utils.save_image(normal, os.path.join(normal_path, '{0:05d}'.format(id) + ".png"))
        torchvision.utils.save_image(render_normal, os.path.join(render_normal_path, '{0:05d}'.format(id) + ".png"))
        torchvision.utils.save_image(world_normal, os.path.join(world_normal_path, '{0:05d}'.format(id) + ".png"))
        torchvision.utils.save_image(render_albedo, os.path.join(render_albedo_path, '{0:05d}'.format(id) + ".png"))
        torchvision.utils.save_image(render_roughness, os.path.join(render_roughness_path, '{0:05d}'.format(id) + ".png"))
        torchvision.utils.save_image(render_depth, os.path.join(render_depth_path, '{0:05d}'.format(id) + ".png"))
        torchvision.utils.save_image(render_alpha, os.path.join(render_alpha_path, '{0:05d}'.format(id) + ".png"))
        if iteration > 3000 :
            render_pbr = pbr_rd[id]
            render_diffuse = pbr_diffuse[id]
            render_specular = pbr_specular[id]
            render_ao = ao_rd[id]
            render_pbr = torch.clamp(render_pbr, 0.0, 1.0)
            render_diffuse = torch.clamp(render_diffuse, 0.0, 1.0)
            render_specular = torch.clamp(render_specular, 0.0, 1.0)
            render_ao = torch.clamp(render_ao, 0.0, 1.0)
            torchvision.utils.save_image(render_pbr, os.path.join(render_pbr_path, '{0:05d}'.format(id) + ".png"))
            torchvision.utils.save_image(render_diffuse, os.path.join(render_diffuse_path, '{0:05d}'.format(id) + ".png"))
            torchvision.utils.save_image(render_specular, os.path.join(render_specular_path, '{0:05d}'.format(id) + ".png"))
            torchvision.utils.save_image(render_ao, os.path.join(render_ao_path, '{0:05d}'.format(id) + ".png"))
        # metrics
        if iteration > 3000 :
            psnrs += psnr(render_pbr, gt).mean().double()
            ssims += ssim(render_pbr, gt).mean().double()
            lpipss += loss_fn_vgg(render_pbr, gt).mean().double()
        else:
            psnrs += psnr(rendering, gt).mean().double()
            ssims += ssim(rendering, gt).mean().double()
            lpipss += loss_fn_vgg(rendering, gt).mean().double()

    psnrs /= len(views)   
    ssims /= len(views)
    lpipss /= len(views)  

    # evalution metrics
    print("\n[ITER {}] Evaluating {} #{}: PSNR {} SSIM {} LPIPS {}".format(iteration, name, len(views), psnrs, ssims, lpipss))

def render_sets(dataset : ModelParams, iteration : int, pipeline : PipelineParams, skip_train : bool, skip_test : bool):
    print(dataset)
    with torch.no_grad():
        gaussians = GaussianModel(dataset.sh_degree, dataset.smpl_type, dataset.motion_offset_flag, dataset.actor_gender)
        scene = Scene(dataset, gaussians, load_iteration=iteration, shuffle=False)

        bg_color = [1,1,1] if dataset.white_background else [0, 0, 0]
        background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

        if not skip_train:
             render_set(dataset.model_path, "train", scene.loaded_iter, scene.getTrainCameras(), gaussians, pipeline, background, scene)

        if not skip_test:
             render_set(dataset.model_path, "test", scene.loaded_iter, scene.getTestCameras(), gaussians, pipeline, background, scene)

if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Testing script parameters")
    model = ModelParams(parser, sentinel=True)
    pipeline = PipelineParams(parser)
    parser.add_argument("--iteration", default=-1, type=int)
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument("--skip_test", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = get_combined_args(parser)
    print("Rendering " + args.model_path)

    # Initialize system state (RNG)
    safe_state(args.quiet)

    render_sets(model.extract(args), args.iteration, pipeline.extract(args), args.skip_train, args.skip_test)