import torch

def transformsVector3x3(v, matrix):
    transformed = torch.empty(v.shape[0], 3, device="cuda")
    v = v.unsqueeze(2).expand(v.shape[0], 3, 3)
    transformed[:, :3] =  v[:, 0, :3] * matrix[:, 0, :3].squeeze(1) +   v[:, 1, :3] * matrix[:, 1, :3].squeeze(1) +  v[:, 2, :3] * matrix[:, 2, :3].squeeze(1)
    return transformed

def transformVector3x3(v, matrix):
    transformed = torch.empty(v.shape[0], 3, device="cuda")
    # transformed[:, 0] = matrix[0, 0] * v[:, 0] +  matrix[1, 0] * v[:, 1] + matrix[2, 0] * v[:, 2]
    # transformed[:, 1] = matrix[0, 1] * v[:, 0] +  matrix[1, 1] * v[:, 1] + matrix[2, 1] * v[:, 2]
    # transformed[:, 2] = matrix[0, 2] * v[:, 0] +  matrix[1, 2] * v[:, 1] + matrix[2, 2] * v[:, 2]
    # matrix 4 4, n 3 
    # print(matrix.size(), v.suze())
    transformed[:, :] = v[:, 0].unsqueeze(dim=-1) * matrix[0, :3] + v[:, 1].unsqueeze(dim=-1) * matrix[1, :3] + v[:, 2].unsqueeze(dim=-1) * matrix[2, :3] 
    return transformed


def transformPoint4x4(p, matrix):
    transformed = torch.empty(p.shape[0], 4, device="cuda")
    # transformed[:, 0] = matrix[0, 0] * p[:, 0] +  matrix[1, 0] * p[:, 1] + matrix[2, 0] * p[:, 2] + matrix[3, 0]
    # transformed[:, 1] = matrix[0, 1] * p[:, 0] +  matrix[1, 1] * p[:, 1] + matrix[2, 1] * p[:, 2] + matrix[3, 1]
    # transformed[:, 2] = matrix[0, 2] * p[:, 0] +  matrix[1, 2] * p[:, 1] + matrix[2, 2] * p[:, 2] + matrix[3, 2]
    # transformed[:, 3] = matrix[0, 3] * p[:, 0] +  matrix[1, 3] * p[:, 1] + matrix[2, 3] * p[:, 2] + matrix[3, 3]
    transformed[:, :] = p[:, 0].unsqueeze(-1) * matrix[0, :]+ p[:, 1].unsqueeze(-1) * matrix[1, :]+ p[:, 2].unsqueeze(-1) * matrix[2, :] + matrix[3, :]
    return transformed

def transformPoint4x3(p, matrix):
    transformed = torch.empty(p.shape[0], 3, device="cuda") 
    # transformed[:, 0] = matrix[0, 0] * p[:, 0] +  matrix[1, 0] * p[:, 1] + matrix[2, 0] * p[:, 2] + matrix[3, 0]
    # transformed[:, 1] = matrix[0, 1] * p[:, 0] +  matrix[1, 1] * p[:, 1] + matrix[2, 1] * p[:, 2] + matrix[3, 1]
    # transformed[:, 2] = matrix[0, 2] * p[:, 0] +  matrix[1, 2] * p[:, 1] + matrix[2, 2] * p[:, 2] + matrix[3, 2]
    transformed[:, :] = p[:, 0].unsqueeze(-1) * matrix[0, :3] + p[:, 1].unsqueeze(-1) * matrix[1, :3] + p[:, 2].unsqueeze(-1) * matrix[2, :3] + matrix[3, :3]
    return transformed

def ndc2Pix(p, H, W):
    #image_X
    p[:, 0] = ((p[:, 0]  + 1.0) * W - 1.0) * 0.5
    #image_Y
    p[:, 1] = ((p[:, 1]  + 1.0) * H - 1.0) * 0.5

    return p

