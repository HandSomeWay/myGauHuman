import torch
import torchvision.transforms as transforms
from sklearn.cluster import KMeans
from torch_kmeans import KMeans as tk
from PIL import Image
import numpy as np
import torchvision
# 读取图像
image = Image.open('/home/shangwei/codes/myGauHuman/output/mixamo/ch21/test/ours_30000/render_albedo/00000.png')

# 将图像转换为张量
transform = transforms.Compose([
    transforms.ToTensor(),
])
image_tensor = transform(image).permute(1, 2, 0)  # 转换为 HxWxC 格式
# mask = (image  [0,0,0])
# image[mask]
# model = tk(n_clusters=4)
# result = model(image_tensor)


# torchvision.utils.save_image(result, "label_torch.png")
# 扁平化图像数据
h, w, c = image_tensor.shape
pixels = image_tensor.reshape(-1, c)


# 将张量转换为numpy数组以使用sklearn的KMeans
pixels_np = pixels.numpy()

# 使用K-means算法
kmeans = KMeans(n_clusters=4)  # 假设我们想要将图像分为3块
kmeans.fit(pixels_np)
labels = kmeans.labels_

# 将聚类标签转换回张量并重塑为原始图像形状
labels_tensor = torch.from_numpy(labels).long().view(h, w)

# 根据聚类结果重构图像
clustered_image = torch.zeros_like(image_tensor)
for cluster_idx in range(kmeans.n_clusters):
    clustered_image[labels_tensor == cluster_idx] = torch.tensor(kmeans.cluster_centers_[cluster_idx])

# 将张量转换回图像以进行显示
clustered_image = clustered_image.permute(2, 0, 1)  # 转换回 CxHxW 格式

torchvision.utils.save_image(clustered_image, "label.png")

