
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

import torchvision
import nvdiffrast.torch as dr
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
            torch.linspace(-1.0 + 1.0 / res[0], 1.0 - 1.0 / res[0], res[0], device="cuda"),
            torch.linspace(-1.0 + 1.0 / res[1], 1.0 - 1.0 / res[1], res[1], device="cuda"),
            indexing="ij",
        )
        v = F.normalize(cube_to_dir(s, gx, gy), p=2, dim=-1)

        tu = torch.atan2(v[..., 0:1], -v[..., 2:3]) / (2 * np.pi) + 0.5
        tv = torch.acos(torch.clamp(v[..., 1:2], min=-1, max=1)) / np.pi
        texcoord = torch.cat((tu, tv), dim=-1)

        cubemap[s, ...] = dr.texture(
            latlong_map[None, ...], texcoord[None, ...], filter_mode="linear"
        )[0]
    return cubemap

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

cubemap = CubemapLight(base_res=256).cuda()

hdri = read_hdr("/home/shangwei/data/my_HDR_map/blaubeuren_night_2k.hdr")   #[1024, 2048, 3]
hdri = torch.from_numpy(hdri).cuda()   #[1024, 2048, 3]
cubemap.base.data = latlong_to_cubemap(hdri, [256, 256])    #[6, 256, 256, 3]
cubemap.eval()
envmap = cubemap.export_envmap(return_img=True).permute(2, 0, 1).clamp(min=0.0, max=1.0)    #[3, 512, 1024]

envmap_path = os.path.join("envmap.png")
torchvision.utils.save_image(envmap, envmap_path)

cubemap.build_mips()
brdf_lut = get_brdf_lut().cuda()    #[1, 256, 256, 2]
pbr_result = pbr_shading(
    light=cubemap,
    normals=torch.randn(512, 512, 3).cuda(),  # [H, W, 3]
    view_dirs=torch.randn(512, 512, 3).cuda(),
    mask=torch.randn(512, 512, 1).cuda(),  # [H, W, 1]
    albedo=torch.randn(512, 512, 3).cuda(),  # [H, W, 3]
    roughness=torch.randn(512, 512, 1).cuda(),  # [H, W, 1]
    metallic=torch.randn(512, 512, 1).cuda(),  # [H, W, 1]
    tone=False,
    gamma=False,
    occlusion=torch.randn(512, 512, 1).cuda(),    # [H, W, 1]
    irradiance=torch.randn(512, 512, 1).cuda(),
    brdf_lut=brdf_lut,
)
smpl_model = SMPL(sex='neutral', model_dir='assets/SMPL_NEUTRAL_renderpeople.pkl')
smpl_pose = smpl_model(pose=np.random.rand(smpl_model.pose.size) * .2, beta=np.random.rand(smpl_model.beta.size) * .03)
# 提取顶点、面和法线信息
vertices = smpl_pose[0]
faces = smpl_model.faces

# 创建trimesh对象
mesh = trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


# 如果你需要将法线保存为图像，你可以使用以下代码
# 首先计算每个顶点的法线
vertex_normals = mesh.vertex_normals
normal_colors = (vertex_normals + 1.0) / 2.0
normal_image = (normal_colors * 255).astype(np.uint8)


# 输出mesh
# trimesh会自动计算mesh的法线
mesh.export('smpl_mesh.ply')
