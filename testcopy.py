import numpy as np
from scipy.special import sph_harm
from PIL import Image
import torch
import torchvision
res = [256, 512]
gy, gx = torch.meshgrid(
torch.linspace(0.0 + 1.0 / res[0], 1.0 - 1.0 / res[0], res[0], device="cuda"),
torch.linspace(-1.0 + 1.0 / res[1], 1.0 - 1.0 / res[1], res[1], device="cuda"),
indexing="ij",
    )

sintheta, costheta = torch.sin(gy * np.pi), torch.cos(gy * np.pi)
sinphi, cosphi = torch.sin(gx * np.pi), torch.cos(gx * np.pi)

reflvec = torch.stack(
    (sintheta * sinphi, costheta, -sintheta * cosphi), dim=-1
)  # [H, W, 3]
torchvision.utils.save_image((reflvec.permute(2, 0, 1) + 1.0) / 2, 'dir_map.png')