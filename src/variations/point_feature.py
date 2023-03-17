import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.nn as nn

class ResNetEncoder(nn.Module):
    def __init__(self):
        super(ResNetEncoder, self).__init__()
        self.resnet = models.resnet18()
        self.resnet.eval()
    def encoder(self, image):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image_tensor = transform(image)

        # 提取特征
        features = self.resnet.layer4(image_tensor.unsqueeze(0))

        # 将特征张量插值为与原始图像相同的大小
        features = F.interpolate(features, size=image_tensor.shape[1:], mode='bilinear', align_corners=False)
        return features
