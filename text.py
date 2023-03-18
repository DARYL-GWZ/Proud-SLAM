import open3d as o3d
import torch
device = o3d.core.Device("CPU:0")
dtype = o3d.core.float32
import numpy as np

# device = o3d.core.Device("CPU:0")
# dtype = o3d.core.float32
# Create an empty point cloud
# Use pcd.point to access the points' attributes
pcd = o3d.t.geometry.PointCloud()

# Default attribute: "positions".
# This attribute is created by default and is required by all point clouds.
# The shape must be (N, 3). The device of "positions" determines the device
# of the point cloud.
# pcd.point.positions = o3d.core.Tensor([[0, 0, 0],
#                                           [1, 1, 1],
#    
# [2, 2, 2]], dtype, device)
pcd = o3d.t.geometry.PointCloud()
image = np.random.random((10, 3))
aa = np.random.random((10, 4))

pcd.point.positions = o3d.core.Tensor(image, o3d.core.float32)
pcd.point.colors = o3d.core.Tensor(image, o3d.core.float32)

# print("\033[0;33;40m",'pcd',pcd.point.colors.numpy(), "\033[0m")
# arr = pcd.point.colors.numpy()
# print("\033[0;33;40m",'np',arr, "\033[0m")

# print("\033[0;33;40m",'arr',a.shape, "\033[0m")
per_colors = torch.tensor(pcd.point.colors.numpy())
print("\033[0;33;40m",'per_colors',per_colors, "\033[0m")

# Common attributes: "normals", "colors".
# Common attributes are used in built-in point cloud operations. The
# spellings must be correct. For example, if "normal" is used instead of
# "normals", some internal operations that expects "normals" will not work.
# "normals" and "colors" must have shape (N, 3) and must be on the same
# device as the point cloud.
pcd.point.normals = o3d.core.Tensor([[0, 0, 1],
                                        [0, 1, 0],
                                        [1, 0, 0]], dtype, device)
pcd.point.colors = o3d.core.Tensor([[0.0, 0.0, 0.0],
                                        [0.1, 0.1, 0.1],
                                        [0.2, 0.2, 0.2]], dtype, device)

# User-defined attributes.
# You can also attach custom attributes. The value tensor must be on the
# same device as the point cloud. The are no restrictions on the shape and
# dtype, e.g.,
pcd.point.intensities = o3d.core.Tensor([0.3, 0.1, 0.4], dtype, device)
pcd.point.labels = o3d.core.Tensor([3, 1, 4], o3d.core.int32, device)