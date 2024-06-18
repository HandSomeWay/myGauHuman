import torch

a = torch.randn(3, 5, 5)
b = a[0, :, :].unsqueeze(0)
print(a.shape)
print(b.shape)