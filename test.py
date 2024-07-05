
import imageio
from pbr import CubemapLight
import torch
import torchvision
import os
import nvdiffrast.torch as dr
import numpy as np
from torchvision import transforms
from PIL import Image
cubemap = CubemapLight(base_res=256).cuda()
filepath = "/home/shangwei/codes/myGauHuman/output/zju_mocap_refine/my_377_env/env_map5000.pth"
checkpoint = torch.load(filepath)
cubemap_params = checkpoint["cubemap"]
cubemap.load_state_dict(cubemap_params)
cubemap.build_mips()
envmap = cubemap.export_envmap(return_img=True).permute(2, 0, 1).clamp(min=0.0, max=1.0)
envmap_path = os.path.join( "envmap.png")
torchvision.utils.save_image(envmap, envmap_path)

normalpath = "/home/shangwei/codes/myGauHuman/output/zju_mocap_refine/my_377_env/test/ours_5000/render_normal/00001.png"
normal = Image.open(normalpath)
transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),  # 将图片转换为Tensor
])
normals = transform(normal).cuda() # [3, H, W]
torchvision.utils.save_image(normals, os.path.join("check2.png"))

check = normals.permute(1, 2, 0).reshape(1, 512, 512, 3).contiguous().reshape(512, 512, 3).permute(2, 0, 1)
torchvision.utils.save_image(check, os.path.join("check3.png"))
diffuse_light = dr.texture(
    cubemap.diffuse[None, ...],  # [1, 6, 16, 16, 3]
    normals.permute(1, 2, 0).reshape(1, 512, 512, 3).contiguous(),  # [1, H, W, 3]
    filter_mode="linear",
    boundary_mode="cube",
    ).squeeze()  # [H, W, 3]
# print(diffuse_light.shape)
result = diffuse_light.permute(2, 0, 1).clamp(min=0.0, max=1.0)
torchvision.utils.save_image(result, os.path.join("check.png"))

albedopath = "/home/shangwei/codes/myGauHuman/output/zju_mocap_refine/my_377_env/test/ours_5000/render_albedo/00001.png"
albedo = Image.open(albedopath)
albedos = transform(albedo).cuda() # [3, H, W]

diffuse_rgb = result * albedos
torchvision.utils.save_image(diffuse_rgb, os.path.join("check1.png"))
