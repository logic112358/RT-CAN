import torch
import torch.nn.functional as F

def min_max_normalization(data):
    min_val = data.min()
    max_val = data.max()
    normalized_data = (data - min_val) / (max_val - min_val)
    return normalized_data

def texture_loss(fused_image, infrared_image, visible_image, device):
    B, C, H, W = fused_image.shape
    sobel_kernel = torch.Tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).view(1, 1, 3, 3).to(device)
    fused_image = min_max_normalization(fused_image)
    infrared_image = min_max_normalization(infrared_image)
    visible_image = min_max_normalization(visible_image)

    fused_image = fused_image.view(B * C, 1, H, W)
    infrared_image = infrared_image.view(B * C, 1, H, W)
    visible_image = visible_image.view(B * C, 1, H, W)

    gradient_fused = torch.abs(F.conv2d(fused_image, sobel_kernel, padding=1))
    gradient_infrared = torch.abs(F.conv2d(infrared_image, sobel_kernel, padding=1))
    gradient_visible = torch.abs(F.conv2d(visible_image, sobel_kernel, padding=1))

    gradient_fused = gradient_fused.view(B, C, H, W)
    gradient_infrared = gradient_infrared.view(B, C, H, W)
    gradient_visible = gradient_visible.view(B, C, H, W)

    min_gradient = torch.min(gradient_infrared, gradient_visible)
    value = torch.norm(gradient_fused - min_gradient, p=2) / (H * W * C)
    return value

def intensity_loss(fused_image, infrared_image, visible_image):
    B,C,H,W = fused_image.shape
    fused_image = min_max_normalization(fused_image)
    infrared_image = min_max_normalization(infrared_image)
    visible_image = min_max_normalization(visible_image)
    min_intensity = torch.min(infrared_image, visible_image)
    intensity_diff = fused_image - min_intensity
    intensity_loss = torch.norm(intensity_diff, p=2, dim=None, keepdim=False, out=None)/ (H*W*C)

    return intensity_loss


def fusion_loss(fused_image, infrared_image, visible_image, alpha, device):
    loss1 = texture_loss(fused_image, infrared_image, visible_image,device)
    loss2 = intensity_loss(fused_image, infrared_image, visible_image)
    # print(loss1)
    # print(loss2)
    fusion_loss = alpha*loss1 + loss2

    return fusion_loss

def FusionLoss(fused_image, infrared_image, visible_image,device):
    loss = 0
    alpha = 0.5
    for i in range(len(fused_image)):
        loss += 0.1*i*fusion_loss(fused_image[i], infrared_image[i], visible_image[i], alpha, device)
    # print(loss)
    loss = loss*1000
    
    return loss

if __name__ == '__main__':
    import sys
    sys.path.append("..")  # 将上级文件夹添加到搜索路径中
    from model.fuseDiff import FusionNetwork

    device = torch.device('cuda')

    input_T = torch.rand(2, 256, 128, 160).to(device)
    input_RGB = torch.rand(2, 256, 128, 160).to(device)

    # input_T = torch.rand(2, 2048, 16, 20).to(device)
    # input_RGB = torch.rand(2, 2048, 16, 20).to(device)

    alpha = 0.5

    model = FusionNetwork(256).to(device)
    out,weight = model(input_T,input_RGB)
    loss = fusion_loss(out,input_T,input_RGB, alpha, device)

    print(loss)

