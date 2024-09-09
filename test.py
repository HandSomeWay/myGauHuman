import torch
import torch.nn.functional as F


def SSAO(depth_map, normal_map, kernel_size=9, radius=0.5):
    # 获取深度图和法线图的大小
    H, W = depth_map.shape[0], depth_map.shape[1]

    # 创建一个随机的采样核
    kernel = torch.rand((kernel_size, 2)) * 2 - 1
    kernel = kernel / torch.norm(kernel, dim=1, keepdim=True)
    kernel = kernel * radius

    # 初始化输出遮蔽图
    occlusion = torch.zeros((H, W), dtype=torch.float32, device=depth_map.device)

    # 对每个像素点进行SSAO操作
    for y in range(H):
        for x in range(W):
            # 获取当前像素点的深度和法线
            center_depth = depth_map[y, x]
            center_normal = normal_map[y, x]

            # 计算遮蔽因子
            ao = 0.0
            for i in range(kernel_size):
                # 采样坐标
                sample_x = int(x + kernel[i, 0])
                sample_y = int(y + kernel[i, 1])

                # 确保采样坐标在图像范围内
                sample_x = max(0, min(W - 1, sample_x))
                sample_y = max(0, min(H - 1, sample_y))

                # 获取采样点的深度和法线
                sample_depth = depth_map[sample_y, sample_x]
                sample_normal = normal_map[sample_y, sample_x]

                # 计算向量
                vec = torch.tensor([sample_x - x, sample_y - y, sample_depth - center_depth], dtype=torch.float32).cuda()

                # 归一化向量
                vec = vec / torch.norm(vec)
                # print(center_normal.shape)
                # print(vec.shape)
                # 计算遮蔽因子
                ao += max(0.0, torch.dot(center_normal, vec))

            # 平均遮蔽因子并存储结果
            occlusion[y, x] = ao / kernel_size

    # 归一化遮蔽图
    occlusion = (occlusion - occlusion.min()) / (occlusion.max() - occlusion.min())

    return occlusion.unsqueeze(0).unsqueeze(0)  # 增加通道和批量大小维度



normal_path = "/home/shangwei/codes/myGauHuman/output/zju_mocap_refine/my_377_env/test/ours_5000/render_normal/00000.png"

from torchvision import transforms, datasets
transform = transforms.Compose([
    transforms.Resize((512, 512)),  # 调整图片大小
    transforms.ToTensor(),           # 将图片转换为张量
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # 标准化
])

from PIL import Image
# 读取图片
normal = Image.open(normal_path)

# 应用转换
N = transform(normal).cuda()

depth_path = "/home/shangwei/codes/myGauHuman/output/zju_mocap_refine/my_377_env/test/ours_5000/render_depth/00000.png"
# 读取图片
depth = Image.open(depth_path)

# 应用转换
D = transform(depth)[0, :, :].cuda()
D *= 3.50

# 假设D和N已经被加载为PyTorch张量，并且已经移动到相应的设备上（如GPU）
# D[H, W, 1] - 深度图
# N[H, W, 3] - 法线图
ssao_map = SSAO(D, N.permute(1, 2, 0))
ssao_map.squeeze(0)
from torchvision.utils import save_image
save_image(1 - ssao_map, 'image.png')
# 打印结果
print(ssao_map.shape)  # 应该是 [1, 1, H, W]


