import torch
import torch.nn.functional as F
from torchvision import transforms, datasets

def SSAO(depth_map, normal_map, kernel_size=32, radius=0.5):
    transform = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 标准化
    ])
    depth_map = transform(depth_map)[0, :, :]
    normal_map = transform(normal_map).permute(1, 2 ,0)
    H, W = depth_map.shape[0], depth_map.shape[1]

    # 创建一个随机的采样核
    kernel = torch.rand((kernel_size, 2), device=depth_map.device) * 2 - 1
    kernel = kernel / torch.norm(kernel, dim=1, keepdim=True)
    kernel = kernel * radius

    # 初始化输出遮蔽图
    occlusion = torch.zeros((H, W), dtype=torch.float32, device=depth_map.device)

    # 扩展法线图和深度图以便进行采样
    normal_map = normal_map.permute(2, 0, 1).unsqueeze(0)  # 变换为 (1, 3, H, W)
    depth_map = depth_map.unsqueeze(0).unsqueeze(0)  # 变换为 (1, 1, H, W)

    # 对每个采样点进行SSAO操作
    for i in range(kernel_size):
        offset = kernel[i]

        # 计算采样坐标
        y, x = torch.meshgrid(torch.linspace(-1, 1, H, device=depth_map.device), 
                              torch.linspace(-1, 1, W, device=depth_map.device), 
                              indexing='ij')
        grid = torch.stack([x, y], dim=-1)  # 生成网格坐标
        grid = grid + offset  # 添加偏移量
        grid = grid.unsqueeze(0)  # 变换为 (1, H, W, 2)

        # 对深度图和法线图进行采样
        sampled_depth_map = F.grid_sample(depth_map, grid, align_corners=True, mode='bilinear', padding_mode='border')
        sampled_normal_map = F.grid_sample(normal_map, grid, align_corners=True, mode='bilinear', padding_mode='border')

        # 计算向量
        vec = torch.cat([
            (grid[..., 0] * (W - 1) / 2).unsqueeze(1),
            (grid[..., 1] * (H - 1) / 2).unsqueeze(1),
            sampled_depth_map - depth_map
        ], dim=1)
        vec = vec / torch.norm(vec, dim=1, keepdim=True)

        # 计算遮蔽因子
        center_normal = normal_map
        ao = torch.clamp(torch.sum(center_normal * vec, dim=1), min=0.0)

        # 累加遮蔽因子
        occlusion += ao.squeeze(0)

    # 平均遮蔽因子
    occlusion /= kernel_size

    # 归一化遮蔽图
    occlusion = (occlusion - occlusion.min()) / (occlusion.max() - occlusion.min())

    return occlusion.unsqueeze(0).unsqueeze(0)  # 增加通道和批量大小维度


def check():
    normal_path = "/home/shangwei/codes/myGauHuman/output/zju_mocap_refine/my_377_env/test/ours_5000/render_normal/00321.png"

    transform = transforms.Compose([
        transforms.Resize((512, 512)),  # 调整图片大小
        transforms.ToTensor(),           # 将图片转换为张量
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 标准化
    ])

    from PIL import Image
    # 读取图片
    normal = Image.open(normal_path)

    # 应用转换
    N = transform(normal).cuda()    #[3, H, W]

    depth_path = "/home/shangwei/codes/myGauHuman/output/zju_mocap_refine/my_377_env/test/ours_5000/render_depth/00321.png"
    # 读取图片
    depth = Image.open(depth_path)

    # 应用转换
    D = transform(depth).cuda()    #[H, W]
    # 假设D和N已经被加载为PyTorch张量，并且已经移动到相应的设备上（如GPU）
    # D[H, W, 1] - 深度图
    # N[H, W, 3] - 法线图
    ssao_map = SSAO(D, N)
    ssao_map.squeeze(0)
    from torchvision.utils import save_image
    save_image(D, 'imD.png')
    save_image(N, 'imN.png')
    print(D.max())

    save_image(1 - ssao_map, 'image1.png')
    # 打印结果
    # print(ssao_map.shape)  # 应该是 [1, 1, H, W]

# check()