
import os
from typing import Dict, List, Tuple
import numpy as np
import trimesh
import pickle
from smpl.smpl_numpy import SMPL
import matplotlib.pyplot as plt
import plyfile
from pbr import CubemapLight, get_brdf_lut, pbr_shading
import torch
import torch.nn.functional as F
import cv2
from gs_ir import recon_occlusion, IrradianceVolumes
import OpenEXR
import imageio
import torchvision
import nvdiffrast.torch as dr
import open3d as o3d
from PIL import Image
from utils.image_utils import psnr
from utils.loss_utils import ssim
import lpips

loss_fn_vgg = lpips.LPIPS(net='vgg').to(torch.device('cuda', torch.cuda.current_device()))

psnrs = 0.0
ssims = 0.0
lpipss = 0.0

for i in range(4):
    for j in range(4): #0,2,5,8:
        j_idx = [0, 2, 5, 8]
        render_path = '/home/shangwei/codes/myGauHuman/output/mixamo/ch21_kl/test/ours_25000/render_albedo/'+'{0:05d}'.format(i*6*4+j)+'.png'
        # render_path = '/home/shangwei/codes/RelightableAvatar/data/result/material_rec/material_syn_ch21/albedo/frame'+'{0:04d}'.format(i*30)+'_view{0:04d}.png'.format(j_idx[j])
        
        gt_path = '/home/shangwei/data/mixamo/ch21/albedo/'+'{0:02d}'.format(j_idx[j])+'/'+'{0:04d}'.format(i*30)+'.png'
        print(render_path)
        print(gt_path)
        print('ok')
        render = Image.open(render_path)
        gt = Image.open(gt_path)
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize((512, 512)),  # 调整图片大小
            torchvision.transforms.ToTensor()           # 将图片转换为Tensor
        ])
        render = transform(render).cuda()
        gt = transform(gt).cuda()

        psnrs += psnr(render, gt[:3,:,:]).mean().double()
        ssims += ssim(render, gt[:3,:,:]).mean().double()
        lpipss += loss_fn_vgg(render, gt[:3,:,:]).mean().double()
print(psnrs/16)
print(ssims/16)
print(lpipss/16)
# for i in range(4):
#     for j in range(4): #2, 5, 8:
#         # render_path = '/home/shangwei/codes/myGauHuman/output/mixamo/ch21/test/ours_30000/render_pbr/'+'{0:05d}'.format(i*6*4+j)+'.png'
#         j_idx = [0, 2, 5, 8]
#         # gt_path = '/home/shangwei/data/mixamo/ch21/images/'+'{0:02d}'.format(j_idx[j])+'/'+'{0:03d}'.format(i*3)+'0.jpg'
#         gt_path = '/home/shangwei/data/mixamo/ch21/relighting/0003/images/'+'{0:02d}'.format(j_idx[j])+'/'+'{0:04d}'.format(i*3)+'.png'
#         # render_path = '/home/shangwei/codes/myGauHuman/output/mixamo/ch38/train/ours_5000/render_albedo/'+'{0:05d}'.format(i)+'.png'
#         # gt_path = '/home/shangwei/data/mixamo/ch38/albedo/00/'+'{0:04d}'.format(i)+'.png'
#         render_path = '/home/shangwei/codes/RelightableAvatar/relighting/ch21/0003/frame'+'{0:04d}'.format(i*30)+'_view{0:04d}.png'.format(j_idx[j])
#         print(render_path)
#         print(gt_path)
#         print('ok')
#         render = Image.open(render_path)
#         gt = Image.open(gt_path)
#         transform = torchvision.transforms.Compose([
#             torchvision.transforms.Resize((512, 512)),  # 调整图片大小
#             torchvision.transforms.ToTensor()           # 将图片转换为Tensor
#         ])
#         render = transform(render).cuda()
#         gt = transform(gt).cuda()

#         psnrs += psnr(render, gt[:3,:,:]).mean().double()
#         ssims += ssim(render, gt[:3,:,:]).mean().double()
#         lpipss += loss_fn_vgg(render, gt[:3,:,:]).mean().double()

# print(psnrs/16)
# print(ssims/16)
# print(lpipss/16)
# ply_path = "/home/shangwei/codes/GauHuman/output/zju_mocap_refine/my_377/point_cloud/iteration_3000/point_cloud.ply"
# ply_data = plyfile.PlyData.read(ply_path)

# vertex_data = ply_data['vertex'].data
# vertices_1 = np.array([list(item) for item in vertex_data])

# smpl_model = SMPL(sex='neutral', model_dir='assets/SMPL_NEUTRAL_renderpeople.pkl')
# # 提取顶点、面和法线信息
# vertices = smpl_model.v_template
# faces = smpl_model.faces

# # 创建trimesh对象
# mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)
# upsampled_mesh = mesh.subdivide()


# mesh_1 = trimesh.Trimesh(vertices=vertices_1[...,:3], faces=upsampled_mesh.faces, process=False)
# trimesh.smoothing.filter_humphrey(mesh_1, alpha=0.1, beta=0.5, iterations=50, laplacian_operator=None)
# # 输出mesh
# trimesh会自动计算mesh的法线
# mesh.export('smpl_mesh.ply')
# upsampled_mesh.export('smpl_mesh_upsampled.ply')
# mesh_1.export('smpl_mesh_trained_smooth.ply')
