
from pbr import CubemapLight, get_brdf_lut, pbr_shading
import torchvision
import torch
import os
cubemap = CubemapLight(base_res=32).cuda()
cubemap_path = '/home/shangwei/codes/myGauHuman/output/mixamo/ch21/env_map7000.pth'
checkpoint = torch.load(cubemap_path)
cubemap_params = checkpoint["cubemap"]
cubemap.load_state_dict(cubemap_params)
    
cubemap.build_mips()
    
for i in range(6):
    torchvision.utils.save_image(cubemap.base[i].permute(2, 0, 1), 'saved_{0:01d}.png'.format(i))
envmap_path = "envmap.png"
# envmap = cubemap.export_envmap(return_img=True).permute(2, 0, 1).clamp(min=0.0, max=1.0)
envmap = cubemap.export_envmap(return_img=True, res = [16, 32]).permute(2, 0, 1).clamp(min=0.0, max=1.0)
torchvision.utils.save_image(envmap, envmap_path)
