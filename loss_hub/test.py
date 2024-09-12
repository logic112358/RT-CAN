import torch
import torch.nn.functional as F

def calculate_gradient(image):
    # 定义Sobel算子
    sobel_x = torch.Tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    sobel_y = torch.Tensor([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    # 分别计算每个通道的梯度
    gradient_x = F.conv2d(image, sobel_x.view(1, 1, 3, 3), padding=1)
    gradient_y = F.conv2d(image, sobel_y.view(1, 1, 3, 3), padding=1)

    # 计算梯度幅值
    gradient = torch.sqrt(gradient_x**2 + gradient_y**2)

    return gradient

# 创建一个具有 (1, 256, 128, 160) 大小的输入图像
image = torch.randn(2, 1, 128, 160)

# 计算梯度
gradient = calculate_gradient(image)

# 打印梯度的形状
print(gradient.shape)
