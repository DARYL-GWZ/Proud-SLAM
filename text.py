import open3d as o3d
import torch
device = o3d.core.Device("CPU:0")
dtype = o3d.core.float32
import numpy as np

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
import torch
import time
from annoy import AnnoyIndex
def find_nearest_points(tensor, point, m):
    # """
    # 在给定的tensor中找到离point最近的m个点，并返回它们的索引。
    
    # 参数：
    # tensor: PyTorch张量，形状为[n, 3]，其中n是点的数量，每个点都有三个坐标。
    # point: 一个包含三个坐标的PyTorch张量，形状为[3,]。
    # m: 要查找的最近点的数量。
    
    # 返回值：
    # 一个包含m个最近点的索引的PyTorch张量，形状为[m,]。
    # """
    # 创建一个AnnoyIndex对象，用于查找最近邻点
    index = AnnoyIndex(3, 'euclidean')
    
    # 将每个点添加到索引中
    for i, x in enumerate(tensor):
        index.add_item(i, x.numpy())
    
    # 构建索引树，以便进行快速查找
    index.build(10)  # 10个近邻
    
    # 查找最近的m个点的索引
    indices = index.get_nns_by_vector(point.numpy(), m)
    
    # 将结果转换为PyTorch张量
    indices = torch.tensor(indices)
    
    return indices

# 测试数据
n = 20000  # 点云中的点数
m = 5  # 要查找的最近点的数量
point = torch.tensor([1, 2, 3])

# 生成一个随机的点云张量
tensor = torch.randn(n, 3)

# 测试函数
start = time.time()
indices = find_nearest_points(tensor, point, m)
end = time.time()

# 输出结果
print(f"最近的{m}个点的索引：{indices}")
print(f"函数执行时间：{end - start:.6f}秒")
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
# ------------------------------------------------------------
