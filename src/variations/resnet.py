import torch
import torch.nn as nn
import torchvision.models as models
import  torch
from    torch import  nn
from    torch.nn import functional as F
# ---------------预训练resnet-----------------------
# class PointsResNet(nn.Module):
#     def __init__(self, feature_n):
#         super(PointsResNet, self).__init__()
#         resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
#         for param in resnet.parameters():
#             param.requires_grad = True
#         modules = list(resnet.children())[:-1]
#         self.resnet = nn.Sequential(*modules)
#         self.fc = nn.Linear(resnet.fc.in_features, feature_n)
#         # for param in self.fc.parameters():
#         #     param.requires_grad = False
#     def forward(self, x1):
#         # with torch.no_grad():
#         # x1 = x1.clone()
#         # print("\033[0;33;40m",'x1',x1.shape, "\033[0m")
#         x = x1.reshape(-1, 3, 1, 1)
#         # print("\033[0;33;40m",'x2',x.shape, "\033[0m")
#         x = self.resnet(x)
#         # print("\033[0;33;40m",'x3',x.shape, "\033[0m")
#         x = x.view(-1, 512)
#         # print("\033[0;33;40m",'x4',x.shape, "\033[0m")
#         x = self.fc(x)
#         # print("\033[0;33;40m",'x5',x.shape, "\033[0m")
#         x= x.reshape(-1, x1.shape[1], x.shape[1])
#         # print("\033[0;33;40m",'x6',x.shape, "\033[0m")
#         return x
    
# model = PointsResNet(128)
# image = torch.rand(120000, 3)
# features = model(image)
# print(features.shape) # 输出: torch.Size([120000, 128])



# class PointsResNet(nn.Module):
#     def __init__(self, feature_n):
#         super(PointsResNet, self).__init__()
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
#         # for param in self.fc.parameters():
#         #     param.requires_grad = False
#     def forward(self, x):
#         x = x.reshape(-1, 3, 1, 1)
#         x = self.resnet(x)
#         x = x.view(x.size(0), -1)
        
#         x = self.fc(x)
#         return x
# -------------------------------------------------------------------
# class ResBlk(nn.Module):
#     """
#     resnet block
#     """
#     def __init__(self, ch_in, ch_out):
#         """
#         :param ch_in:
#         :param ch_out:
#         """
#         super(ResBlk, self).__init__()
#         self.conv1 = nn.Conv2d(ch_in, ch_out, kernel_size=3, stride=1, padding=1)
#         self.bn1 = nn.BatchNorm2d(ch_out)
#         self.conv2 = nn.Conv2d(ch_out, ch_out, kernel_size=3, stride=1, padding=1)
#         self.bn2 = nn.BatchNorm2d(ch_out)
#         self.extra = nn.Sequential()
#         if ch_out != ch_in:
#             # [b, ch_in, h, w] => [b, ch_out, h, w]
#             self.extra = nn.Sequential(
#                 nn.Conv2d(ch_in, ch_out, kernel_size=1, stride=1),
#                 nn.BatchNorm2d(ch_out)
#             )

#     def forward(self, x):
#         """
#         :param x: [b, ch, h, w]
#         :return:
#         """
#         out = F.relu(self.bn1(self.conv1(x)))
#         # print("\033[0;33;40m",'out',out.shape, "\033[0m")
#         out = self.bn2(self.conv2(out))
#         out = self.extra(x) + out
#         return out

# class PointsResNet(nn.Module):
#     def __init__(self, feature_n):
#         super(PointsResNet, self).__init__()
#         self.conv1 = nn.Sequential(
#             nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
#             nn.BatchNorm2d(16)
#         )
#         # followed 4 blocks
#         # [b, 64, h, w] => [b, 128, h ,w]
#         self.blk1 = ResBlk(16, 16)
#         # [b, 128, h, w] => [b, 256, h, w]
#         self.blk2 = ResBlk(16, 32)
#         # # [b, 256, h, w] => [b, 512, h, w]
#         # self.blk3 = ResBlk(128, 256)
#         # # [b, 512, h, w] => [b, 1024, h, w]
#         # self.blk4 = ResBlk(256, 512)
#         self.outlayer = nn.Linear(32, feature_n)

#     def forward(self, x):
#         """
#         :param x:
#         :return:
#         """
#         x = x.view(-1, 3, 1,1)
#         x = F.relu(self.conv1(x))
#         # print("\033[0;33;40m",'x1',x.shape, "\033[0m")
#         # [b, 64, h, w] => [b, 1024, h, w]
#         x = self.blk1(x)
#         x = self.blk2(x)
#         # print("\033[0;33;40m",'xx',x.shape, "\033[0m")
#         x = x.view(x.size(0), -1)
#         # print("\033[0;33;40m",'xxx',x.shape, "\033[0m")
#         x = self.outlayer(x)
#         x = x.view(-1, 8, x.size(1))
#         return x


class PointsResNet(nn.Module):
    def __init__(self, feature_n):
        super(PointsResNet, self).__init__()
        self.resnet = nn.Sequential(
            # nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.Linear(6, 64),
            nn.ReLU(inplace=False),
            nn.Linear(64, 128),
            nn.ReLU(inplace=False),
            nn.Linear(128, 256),
            nn.ReLU(inplace=False),
            nn.Linear(256, 512),
            nn.ReLU(inplace=False),           
        )
        self.fc = nn.Linear(512, feature_n)
        # for param in self.fc.parameters():
        #     param.requires_grad = False
    def forward(self, x1 ,y):
        x = torch.cat((x1,y),2)
        x = x.reshape(-1, x.shape[2])
        x = self.resnet(x)
        x = x.view(x1.size(0),x1.size(1), -1)
        
        x = self.fc(x)
        return x