# import open3d as o3d
# import torch
# device = o3d.core.Device("CPU:0")
# dtype = o3d.core.float32
# import numpy as np

# device = o3d.core.Device("CPU:0")
# dtype = o3d.core.float32
# Create an empty point cloud
# Use pcd.point to access the points' attributes
# pcd = o3d.t.geometry.PointCloud()

# Default attribute: "positions".
# This attribute is created by default and is required by all point clouds.
# The shape must be (N, 3). The device of "positions" determines the device
# of the point cloud.
# pcd.point.positions = o3d.core.Tensor([[0, 0, 0],
#                                           [1, 1, 1],
#    
# [2, 2, 2]], dtype, device)
# pcd = o3d.t.geometry.PointCloud()
# image = np.random.random((10, 3))
# aa = np.random.random((10, 4))

# pcd.point.positions = o3d.core.Tensor(image, o3d.core.float32)
# pcd.point.colors = o3d.core.Tensor(image, o3d.core.float32)

# # print("\033[0;33;40m",'pcd',pcd.point.colors.numpy(), "\033[0m")
# # arr = pcd.point.colors.numpy()
# # print("\033[0;33;40m",'np',arr, "\033[0m")

# # print("\033[0;33;40m",'arr',a.shape, "\033[0m")
# per_colors = torch.tensor(pcd.point.colors.numpy())
# print("\033[0;33;40m",'per_colors',per_colors, "\033[0m")

# # Common attributes: "normals", "colors".
# # Common attributes are used in built-in point cloud operations. The
# # spellings must be correct. For example, if "normal" is used instead of
# # "normals", some internal operations that expects "normals" will not work.
# # "normals" and "colors" must have shape (N, 3) and must be on the same
# # device as the point cloud.
# pcd.point.normals = o3d.core.Tensor([[0, 0, 1],
#                                         [0, 1, 0],
#                                         [1, 0, 0]], dtype, device)
# pcd.point.colors = o3d.core.Tensor([[0.0, 0.0, 0.0],
#                                         [0.1, 0.1, 0.1],
#                                         [0.2, 0.2, 0.2]], dtype, device)

# # User-defined attributes.
# # You can also attach custom attributes. The value tensor must be on the
# # same device as the point cloud. The are no restrictions on the shape and
# # dtype, e.g.,
# pcd.point.intensities = o3d.core.Tensor([0.3, 0.1, 0.4], dtype, device)
# pcd.point.labels = o3d.core.Tensor([3, 1, 4], o3d.core.int32, device)

# -------------------最近邻搜索----------------------------
# import torch
# import time
# from annoy import AnnoyIndex
# def find_nearest_points(tensor, point, m):
#     # """
#     # 在给定的tensor中找到离point最近的m个点，并返回它们的索引。
    
#     # 参数：
#     # tensor: PyTorch张量，形状为[n, 3]，其中n是点的数量，每个点都有三个坐标。
#     # point: 一个包含三个坐标的PyTorch张量，形状为[3,]。
#     # m: 要查找的最近点的数量。
    
#     # 返回值：
#     # 一个包含m个最近点的索引的PyTorch张量，形状为[m,]。
#     # """
#     # 创建一个AnnoyIndex对象，用于查找最近邻点
#     index = AnnoyIndex(3, 'euclidean')
    
#     # 将每个点添加到索引中
#     for i, x in enumerate(tensor):
#         index.add_item(i, x.numpy())
    
    # 构建索引树，以便进行快速查找
    # index.build(10)  # 10个近邻
    
    # # 查找最近的m个点的索引
    # indices = index.get_nns_by_vector(point.numpy(), m)
    
    # # 将结果转换为PyTorch张量
    # indices = torch.tensor(indices)
    
    # return indices

# 测试数据
# n = 20000  # 点云中的点数
# m = 5  # 要查找的最近点的数量
# point = torch.tensor([1, 2, 3])

# # 生成一个随机的点云张量
# tensor = torch.randn(n, 3)

# # 测试函数
# start = time.time()
# indices = find_nearest_points(tensor, point, m)
# end = time.time()

# # 输出结果
# print(f"最近的{m}个点的索引：{indices}")
# print(f"函数执行时间：{end - start:.6f}秒")
# ------------------------------------------------------------------------

# ------------------GPU最近邻搜索--------------------------------
# import faiss
# import numpy as np
# import torch
# import time

# def find_nearest_points(points, point, m):
#     """
#     使用faiss库中的K近邻算法在点云中查找最近的m个点，并返回它们的索引。
#     """
#     # 将点云转换为NumPy数组并转置
#     points_np = points.cpu().numpy().T.astype('float32')
    
#     # 创建faiss索引
#     d = points_np.shape[0]  # 点的维度
#     index = faiss.IndexFlatL2(d)
#     index.add(points_np)
    
#     # 查找最近的m个点
#     query = point.cpu().numpy().astype('float32').reshape(1, d)
#     distances, indices = index.search(query, m)
    
#     # 将结果转换为PyTorch张量并返回
#     return torch.tensor(indices[0])

# # 测试数据
# n = 1000000  # 点云中的点数
# m = 5  # 要查找的最近点的数量
# point = torch.tensor([1, 2, 3], dtype=torch.float32)

# # 生成一个随机的点云张量
# points = torch.randn(n, 3)

# start = time.time()
# indices = find_nearest_points(points, point, m)
# end = time.time()
# -------------------------resnet text----------------
# import torch
# import torch.nn as nn
# import torchvision.models as models
# import time

# class PointsResNet(nn.Module):
#     def __init__(self, feature_n):
#         super(PointsResNet, self).__init__()
#         resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
#         modules = list(resnet.children())[:-1]
#         self.resnet = nn.Sequential(*modules)
#         self.fc = nn.Linear(resnet.fc.in_features, feature_n)

    # def forward(self, x1):
    #     # print("\033[0;33;40m",'x1',x1.shape, "\033[0m")
    #     x = x1.reshape(-1, 3, 1, 1)
    #     # print("\033[0;33;40m",'x2',x.shape, "\033[0m")
    #     x = self.resnet(x)
        # # print("\033[0;33;40m",'x3',x.shape, "\033[0m")
        # x = x.view(-1, 512)
        # # print("\033[0;33;40m",'x4',x.shape, "\033[0m")
        # x = self.fc(x)
        # # print("\033[0;33;40m",'x5',x.shape, "\033[0m")
        # x= x.reshape(-1, x1.shape[1], x.shape[1])
        # # print("\033[0;33;40m",'x6',x.shape, "\033[0m")
        # return x


# model = PointsResNet(64)
# start = time.time()
# image = torch.rand(3000, 8, 3)
# features = model(image) # 输出: torch.Size([120000, 128])
# end = time.time()

# # # # 输出结果
# print(f"函数执行时间：{end - start:.6f}秒")
# class PointsResNet(nn.Module):
#     def __init__(self, num_output_features):
#         super(PointsResNet, self).__init__()
#         resnet = models.resnet18(pretrained=True)
#         self.features = nn.Sequential(*list(resnet.children())[:-1])
#         self.output_layer = nn.Linear(resnet.fc.in_features, num_output_features)

#     def forward(self, x):
#         print("\033[0;33;40m",'x1',x.shape, "\033[0m")
#         x = self.features(x)
#         print("\033[0;33;40m",'x2',x.shape, "\033[0m")
#         x = x.view(x.size(0), -1)
#         print("\033[0;33;40m",'x3',x.shape, "\033[0m")
#         x = self.output_layer(x)
#         print("\033[0;33;40m",'x4',x.shape, "\033[0m")
#         return x
    
# model = PointsResNet(128)
# start = time.time()
# image = torch.rand(3000,8, 3)
# features = model(image)
# print(features.shape) # 输出: torch.Size([120000, 128])
# end = time.time()

# # 输出结果
# print(f"函数执行时间：{end - start:.6f}秒")
# ---------------------octree text----------------
# import torch
# torch.classes.load_library(
#     "third_party/sparse_octree/build/lib.linux-x86_64-cpython-310/svo.cpython-310-x86_64-linux-gnu.so")

# svo = torch.classes.svo.Octree()
# svo.init(256, 16, 0.2, 8)

# voxels = torch.rand(10, 3)
# colors = torch.rand(10, 3)

# svo.insert(voxels.cpu().int(),colors.cpu().int())
# voxels, children, features, pcd_xyz, pcd_color = svo.get_centres_and_children()
# print("\033[0;33;40m",'voxels',voxels.shape, "\033[0m")
# print("\033[0;33;40m",'children',children.shape, "\033[0m")
# print("\033[0;33;40m",'features',features.shape, "\033[0m")
# print("\033[0;33;40m",'pointclous',pcd_xyz.shape, "\033[0m")
# print("\033[0;33;40m",'pointclous',pcd_color.shape, "\033[0m")


# ---------------------get_embeddings_pcd text----------------
# def get_sampled_feature(query_xyz, query_fea, samples_xyz):
#     d = torch.Tensor([])
#     fea = torch.zeros((1, query_fea.shape[2]))
#     for i in range(query_xyz.shape[0]):
#         dis = torch.norm(samples_xyz - query_xyz[i], dim=1)
#         d = torch.cat((d, 1 / dis))
#     all_dis = torch.sum(d)
#     for j in range(query_xyz.shape[0]):
#         feature = torch.mul((d[j] / all_dis), query_fea[j]).reshape(1, query_fea.shape[1])
#         fea = torch.add(fea, feature)
#     return fea

# sample 表示采样点的位置数据
# positions 表示点云位置数据
# features 表示点云特征数据
# 函数首先计算每个采样点和每个点云的距离，然后基于距离计算每个点云对采样点的权重
# 最后将权重与特征值相乘并求和，得到采样点的特征
# import  torch
# import time
# import numpy as np
# def compute_sample_features(sample, positions, features):
#     """
#     Computes features for the given sample points based on positions and features of the point cloud.
#     :param sample: Tensor of shape (M, 3) representing the coordinates of M sample points.
#     :param positions: Tensor of shape (M, N, 3) representing the coordinates of N points in M voxels.
#     :param features: Tensor of shape (M, N, 64) representing the features of N points in M voxels.
#     :return: Tensor of shape (M, 64) representing the features of M sample points.
#     """
#     M, N = positions.shape[:2]

#     # Compute distances between sample points and voxel points
    
#     # print("\033[0;33;40m",'sample.unsqueeze(1) - positions',(torch.sum((sample.unsqueeze(1) - positions) ** 2, dim=-1)).shape, "\033[0m")
#     distances = torch.sqrt(torch.sum((sample.unsqueeze(1) - positions) ** 2, dim=-1))  # Shape: (M, N)
    
#     # print("\033[0;33;40m",'distances',distances.shape, "\033[0m")
#     # Compute weights based on distances.
#     # np.savetxt('test_distances.txt', distances.detach().cpu().numpy())
#     print("\033[0;33;40m",'distances',distances, "\033[0m")
#     weights = torch.softmax(-(distances), dim=-1)  # Shape: (M, N)
#     print("\033[0;33;40m",'weights',weights, "\033[0m")
#     print("\033[0;33;40m",'weights.shape',weights.shape, "\033[0m")
#     print("\033[0;33;40m",'weights.unsqueeze(-1)',weights.unsqueeze(-1), "\033[0m")
    
#     # print("\033[0;33;40m",'weights',weights.shape, "\033[0m")
#     # print("\033[0;33;40m",'weights.unsqueeze(-1) ',(weights.unsqueeze(-1) ).shape, "\033[0m")
#     print("\033[0;33;40m",'features',features, "\033[0m")
#     # Compute weighted average of features.
#     sample_features = torch.sum(weights.unsqueeze(-1) * features, dim=1)  # Shape: (M, 64)

#     return sample_features

# start = time.time()
# feature = torch.tensor([[[1,8,3,2,4,111],[1,8,3,2,4,5],
#                          [1,8,3,2,4,5],[1,8,3,2,4,5],
#                          [1,8,3,2,4,5],[1,8,3,2,4,5],
#                          [1,8,3,2,4,5],[1,8,3,2,4,5]]]).float()
# positions = torch.tensor([[[1,8,5],[1,8,3],[1,8,3],
#                            [1,8,3],[1,8,3],[1,8,3],
#                            [1,8,3],[1,8,3]]]).float()
# sample = torch.tensor([[1,8,3]]).float()
# print("\033[0;33;40m",'sample',sample.shape, "\033[0m")
# print("\033[0;33;40m",'feature',feature.shape, "\033[0m")
# print("\033[0;33;40m",'positions',positions.shape, "\033[0m")
# # print("\033[0;33;40m",'sample',sample.shape, "\033[0m")
# features = compute_sample_features(sample,positions, feature)
# print("\033[0;33;40m",'sample features',features.shape, "\033[0m")
# print("\033[0;33;40m",'sample features',features, "\033[0m")



# samples_xyz = torch.rand(10000, 3)
# query_xyz = torch.rand(10000,8, 3)
# query_fea = torch.rand(10000,8, 128)
# features = compute_sample_features(samples_xyz,query_xyz, query_fea)
# end = time.time()
# print(f"函数执行时间：{end - start:.6f}秒")
# print(features.shape) # 输出: torch.Size([120000, 128])



# dis = torch.tensor([[1,8,3,2,4,5,6,7],[1,8,3,2,4,5,6,7],[1,8,3,2,4,5,6,7]]).float()
# print("\033[0;33;40m",'dis',dis, "\033[0m")
# weights = torch.softmax(-dis, dim=-1)
# weights1 = dis/ 
# print("\033[0;33;40m",'weights',weights, "\033[0m")

# ============== pcd feature cal test=============
# import torch
# import torch.nn.functional as F
# pointclouds_xyz = torch.rand(2272, 8, 3)
# sampled_idx = torch.rand(10000).long()
# point_feats = torch.rand(10000, 3)
# pointclouds_feature = torch.rand(2272, 8, 64)
# #  values torch.Size([20000, 16]) 
# #  原始pointclouds_xyz torch.Size([2272, 8, 3]) 
# #  point_feats torch.Size([2272, 8]) 
# #  pointclouds_feature torch.Size([2272, 8, 64]) 


# xyz_list = []
# feats_list = []
# for i in range(pointclouds_xyz.shape[1]):
#     per_pcd_xyz = F.embedding(sampled_idx, pointclouds_xyz[:,i,:])
#     print("\033[0;33;40m",'per_pcd_xyz',per_pcd_xyz.shape, "\033[0m")
#     per_pcd_feats = F.embedding(sampled_idx, pointclouds_feature[:,i,:])
#                                 # .long(), pointclouds_feature[:,i,:]).view(pointclouds_xyz[:,i,:].size(0), -1)
#     print("\033[0;33;40m",'per_pcd_feats',per_pcd_feats.shape, "\033[0m")
#     xyz_list.append(per_pcd_xyz)
#     feats_list.append(per_pcd_feats)
    
# pointclouds_xyz = torch.stack(xyz_list, dim=1)
# pointclouds_feats = torch.stack(feats_list, dim=1)
# print("\033[0;33;40m",'pointclouds_xyz',pointclouds_xyz.shape, "\033[0m")
# print("\033[0;33;40m",'pointclouds_feats',pointclouds_feats.shape, "\033[0m")


# tensor = torch.randn(3, 4)
# with open('tensor.txt', 'w') as file:
#     file.write(str(tensor))

# # ============== tensorboardx test=============
# from tensorboardX import SummaryWriter
# writer = SummaryWriter()
# for i in range(10):
#     writer.add_scalar('quadratic', i**2, global_step=i)
#     writer.add_scalar('exponential', 2**i, global_step=i)

# # ============== resnet test=============
# import  torch
# from    torch import  nn
# from    torch.nn import functional as F
# from    torch.utils.data import DataLoader
# from    torchvision import datasets
# from    torchvision import transforms
# from    torch import nn, optim

# # from    torchvision.models import resnet18

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
#         print("\033[0;33;40m",'out',out.shape, "\033[0m")
#         out = self.bn2(self.conv2(out))
#         out = self.extra(x) + out
#         return out

# class ResNet18(nn.Module):
#     def __init__(self, feature_n):
#         super(ResNet18, self).__init__()
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



# def main():

#     device = torch.device('cuda')
#     model = ResNet18(16).to(device)
#     x = torch.rand(3000,8,3).to(device)

#     print("\033[0;33;40m",'x',x.shape, "\033[0m")

#     logits = model(x)
#     print("\033[0;33;40m",'logits',logits.shape, "\033[0m")





# if __name__ == '__main__':
#     main()
    
# import torch
# import torch.nn as nn
# import time

# class BasicBlock(nn.Module):
#     def __init__(self, in_planes, out_planes, stride=1):
#         super(BasicBlock, self).__init__()
#         self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(out_planes)
#         self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn2 = nn.BatchNorm2d(out_planes)

#         self.shortcut = nn.Sequential()
#         if stride != 1 or in_planes != out_planes:
#             self.shortcut = nn.Sequential(
#                 nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False),
#                 nn.BatchNorm2d(out_planes)
#             )

#     def forward(self, x):
#         out = nn.ReLU()(self.bn1(self.conv1(x)))
#         out = self.bn2(self.conv2(out))
#         out += self.shortcut(x)
#         out = nn.ReLU()(out)
#         return out


# class ResNetFeatureExtractor(nn.Module):
#     def __init__(self, num):
#         super(ResNetFeatureExtractor, self).__init__()
#         self.num = num
#         self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
#         self.bn1 = nn.BatchNorm2d(64)
#         self.layer1 = nn.Sequential(BasicBlock(64, 64), BasicBlock(64, 64))
#         self.layer2 = nn.Sequential(BasicBlock(64, 128, stride=2), BasicBlock(128, 128))
#         self.layer3 = nn.Sequential(BasicBlock(128, 256, stride=2), BasicBlock(256, 256))
#         self.layer4 = nn.Sequential(BasicBlock(256, 512, stride=2), BasicBlock(512, 512))
#         self.avgpool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Linear(512, num)

#     def forward(self, x):
#         print("\033[0;33;40m",'x',x.shape, "\033[0m")
#         out = nn.ReLU()(self.bn1(self.conv1(x)))
#         out = self.layer1(out)
#         out = self.layer2(out)
#         out = self.layer3(out)
#         out = self.layer4(out)
#         out = self.avgpool(out)
#         out = out.view(out.size(0), -1)
#         out = self.fc(out)
#         return out


# model = ResNetFeatureExtractor(16)
# start = time.time()
# image = torch.rand(3000, 8, 3)
# features = model(image) # 输出: torch.Size([120000, 128])
# end = time.time()
# print("\033[0;33;40m",'image',image.shape, "\033[0m")
# print("\033[0;33;40m",'features',features.shape, "\033[0m")

# import numpy as np

# def positional_encoding_3d(x, y, z, d):
#     pos = np.array([x, y, z])
#     encoding = np.zeros((d))
#     print("\033[0;33;40m",'encoding',encoding.shape, "\033[0m")
    
#     for i in range(d//2):
#         encoding[2*i] = np.sum(np.sin(np.dot(pos, 1/10000**(2*i/d))))
#         encoding[2*i+1] = np.sum(np.cos(np.dot(pos, 1/10000**(2*i/d))))
#     return encoding

# # 示例代码
# d = 8  # 编码向量的维度
# x, y, z = 10, 20, 30  # 三维位置
# encoding = positional_encoding_3d(x, y, z, d)
# print(encoding)

# import torch

# def encoding_3d(pos, d):
#     encoding = torch.zeros(pos.shape[0],pos.shape[1],d)
#     # print("\033[0;33;40m",'encoding',encoding.shape, "\033[0m")
#     for i in range(d//2):
#         x = 1/10000**(2*i/d)
#         encoding[:,:,2*i] = torch.sum(torch.sin(pos * x),dim = -1)
#         encoding[:,:,2*i+1] = torch.sum(torch.cos(pos * x),dim = -1)
#     return encoding

# d = 16 # 编码向量的维度
# # x, y, z = 10, 20, 30  # 三维位置
# pos = torch.tensor([[[1,2,3],
#                    [1,2,4],
#                    [1,2,5],
#                    [1,2,6],
#                    [1,2,7],
#                    [1,2,6],
#                    [1,2,6],
#                    [1,2,6]]])
# encoding = encoding_3d(pos, d)
# print(encoding)