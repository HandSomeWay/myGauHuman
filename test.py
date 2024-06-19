import torch

a = torch.randn(5, 1)
b = a.repeat(1, 3)
print(a)
print(b)