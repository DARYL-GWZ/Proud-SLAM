import torch
import torch.nn as nn
import torchvision.models as models

class PointsResNet(nn.Module):
    def __init__(self, feature_n):
        super(PointsResNet, self).__init__()
        resnet = models.resnet18()
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.fc = nn.Linear(resnet.fc.in_features, feature_n)

    def forward(self, x):
        # print("\033[0;33;40m",'x1',x.shape, x.dtype, "\033[0m")
        x = x.permute(1, 0).reshape(-1, 3, 1, 1)
        # print("\033[0;33;40m",'x2',x.shape,x.dtype, "\033[0m")
        x = self.resnet(x)
        # print("\033[0;33;40m",'x3',x.shape, "\033[0m")
        x = x.view(-1, 512)
        # print("\033[0;33;40m",'x4',x.shape, "\033[0m")
        x = self.fc(x)
        return x
    
# model = PointsResNet(128)
# image = torch.rand(120000, 3)
# features = model(image)
# print(features.shape) # 输出: torch.Size([120000, 128])



# class PixelResNet(nn.Module):
#     def __init__(self, feature_n):
#         super(PixelResNet, self).__init__()
#         self.resnet = nn.Sequential(
#             nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
#             nn.BatchNorm2d(64),
#             nn.ReLU(inplace=True),
#             nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
#             nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(128),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(256),
#             nn.ReLU(inplace=True),
#             nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(512),
#             nn.ReLU(inplace=True),
#             nn.AdaptiveAvgPool2d((1,1))
#         )
#         self.fc = nn.Linear(512, feature_n)

    # def forward(self, x):
    #     x = x.permute(1, 0).reshape(-1, 3, 1, 1)
    #     x = self.resnet(x)
    #     x = x.view(x.size(0), -1)
    #     x = self.fc(x)
    #     return x



