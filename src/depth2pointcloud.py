# import numpy as np

# def rgbmap(rgb):
#     rgb = (rgb-rgb.min())/(rgb.max()-rgb.min())
#     return rgb

# def pixel2cam(p, c_m): #像素坐标转相机归一化坐标
#     return np.array([[(p[0] - c_m[0, 2]) / c_m[0, 0]], [(p[1] - c_m[1, 2]) / c_m[1, 1]]])

# def depth_to_3dpoints(pose1,rgb,depth,K):
#     pts_3d = []
#     color_3d = []
#     rgb = rgbmap(rgb)
#     # print('rgb',rgb.shape)
#     for m in range(0,depth.shape[0],3):
#         for n in range(0,depth.shape[1],3):
#             d = depth[m, n] 
#             key_points = [n, m]
#             # if d < 0.01 or d>10:
#             if d < 0.01 :
#                 continue
#             dd = d   #深度越大移动越大
#             p1 = pixel2cam(key_points, K)
#             p3d= [p1[0, 0]*dd, p1[1, 0]*dd, dd]
#             c3d =[rgb[m,n,2],rgb[m,n,1],rgb[m,n,0]]
#             #     p3d= np.append(p3d,[1])
#             #     a = copy.deepcopy(p3d[0]) 
#             #     b = copy.deepcopy(p3d[1])
#             #     c = copy.deepcopy(p3d[2])
#             #     p3d[0] = a  #正向向左
#             #     p3d[1] = -b  #正向向上
#             #     p3d[2] = -c #正向向后
#             p3d= pose1[:3,:4] @ p3d
#             color_3d.append(c3d)
#             pts_3d.append(p3d)
#     pts_3d = np.array(pts_3d)
#     color_3d = np.array(color_3d)
#     return pts_3d,color_3d


# import open3d as o3d
# import numpy as np

# # 生成点云数据
# points = np.random.rand(1000, 3)
# pcd = o3d.geometry.PointCloud()
# pcd.points = o3d.utility.Vector3dVector(points)

# # 创建 KD 树
# tree = o3d.geometry.KDTreeFlann(pcd)

# # 给定点和半径
# query_point = np.array([0.5, 0.5, 0.5])
# radius = 0.2

# # 查询 KD 树
# [k, idx, _] = tree.search_radius_vector_3d(query_point, radius)

# # 获取查询结果
# selected_points = np.asarray(pcd.points)[idx, :]

# import open3d as o3d
# import numpy as np

# # 创建一个空的点云对象
# pcd = o3d.geometry.PointCloud()

# # 生成随机点
# n_points = 100
# points = np.random.rand(n_points, 3)

# # 为每个点添加自定义属性info，存储随机数值
# info = np.random.rand(n_points)
# pcd.point["info"] = o3d.utility.Vector3dVector(info)

# # 将点设置为点云对象的坐标
# pcd.points = o3d.utility.Vector3dVector(points)

# # 可以将自定义属性存储为numpy数组并在之后使用
# info_retrieved = np.asarray(pcd.point["info"])

# # 可以使用自定义属性来进行可视化
# o3d.visualization.draw_geometries([pcd], point_show_normal=False)
import open3d as o3d
import numpy as np

# 创建一个点云对象
pcd = o3d.geometry.PointCloud()

# 创建点坐标数组
points = np.array([[0, 0, 0], [0, 1, 0], [1, 0, 0]])

# 将点坐标数组设置为点云的坐标
pcd.points = o3d.utility.Vector3dVector(points)

# 创建自定义属性数组
custom_property = np.array([1, 2, 3])

# 将自定义属性数组转换为C++的int类型
custom_property_int = custom_property.astype(np.int32)

# 将自定义属性数组添加为点云对象的新属性
pcd.custom_property = o3d.utility.IntVector(custom_property_int)

# 将点云对象保存为PLY文件
o3d.io.write_point_cloud("custom_property.ply", pcd)
