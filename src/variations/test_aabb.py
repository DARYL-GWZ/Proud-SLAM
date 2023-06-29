from voxel_helpers import ray_intersect_vox_AABB, ray_intersect_vox
import torch
import pandas as pd
import numpy as np


torch.classes.load_library(
    "/home/guowzh/code/Proud-slam/third_party/sparse_octree/build/lib.linux-x86_64-cpython-310/svo.cpython-310-x86_64-linux-gnu.so")

points_path='/home/guowzh/code/Proud-slam/vox_0_points.txt'
colors_path='/home/guowzh/code/Proud-slam/vox_0_colors.txt'
points_data = pd.read_table(points_path,sep=' ',header=None)
colors_data = pd.read_table(colors_path,sep=' ',header=None)
points = torch.from_numpy(points_data.values)
colors = torch.from_numpy(colors_data.values)
print("\033[0;33;40m",'points',points.shape, "\033[0m")
print("\033[0;33;40m",'points',colors.shape, "\033[0m")

svo = torch.classes.svo.Octree()
svo.init(256, 16, 0.2, 8)


svo.insert_hash(points.cpu().float(),colors.cpu().int())
voxels_center = svo.get_centres()
voxels_center = voxels_center.cuda()

print("\033[0;33;40m",'hash voxels',voxels_center.shape, "\033[0m")
# np.savetxt(f'hash_voxels_center.txt', voxels_center.detach().cpu().numpy())

rays_d_path='/home/guowzh/code/Proud-slam/vox_1_rays_d.txt'
rays_o_path='/home/guowzh/code/Proud-slam/vox_1_rays_o.txt'
rays_d_data = pd.read_table(rays_d_path,sep=' ',header=None)
rays_o_data = pd.read_table(rays_o_path,sep=' ',header=None)
rays_d = torch.from_numpy(rays_d_data.values).unsqueeze(0).contiguous().cuda()
rays_o = torch.from_numpy(rays_o_data.values).unsqueeze(0).contiguous().cuda()
print("\033[0;33;40m",'rays_d',rays_d.shape, "\033[0m")
print("\033[0;33;40m",'rays_o',rays_o.shape, "\033[0m")


# voxels = torch.div(points, 0.2, rounding_mode='floor')
# svo.insert(voxels.cpu().int(),colors.cpu().int(),points.cpu().float())
# voxels, children, features, pcd_xyz, pcd_color = svo.get_centres_and_children()

# # print("\033[0;33;40m",'octree voxels',voxels.shape, "\033[0m")
# centres = (voxels[:, :3] + voxels[:, -1:] / 2) * 0.2
# centres = centres.cuda()
# # np.savetxt(f'octree_voxels_center.txt', centres.detach().cpu().numpy())
# # print("\033[0;33;40m",'octree centres',centres.shape, "\033[0m")
# childrens = torch.cat([children, voxels[:, -1:]], -1).cuda()
# intersections, hits = ray_intersect_vox(
#     rays_o, rays_d, centres,
#     childrens, 0.2, 10, 10)
# print("\033[0;33;40m",'oc_hits',hits.shape, "\033[0m")
# print("\033[0;33;40m",'oc_intersections[min_depth]',intersections["min_depth"].shape, "\033[0m")
# print("\033[0;33;40m",'oc_intersections[max_depth]',intersections["max_depth"].shape, "\033[0m")
# print("\033[0;33;40m",'oc_intersections[intersected_voxel_idx]',intersections["intersected_voxel_idx"].shape, "\033[0m")
# np.savetxt(f'oc_hits.txt', hits.detach().cpu().numpy())
# np.savetxt(f'oc_intersections["min_depth"].txt', intersections["min_depth"][0].detach().cpu().numpy())
# np.savetxt(f'oc_intersections["max_depth"].txt', intersections["max_depth"][0].detach().cpu().numpy())
# np.savetxt(f'oc_intersections["intersected_voxel_idx"].txt', intersections["intersected_voxel_idx"][0].detach().cpu().numpy())



intersections, hits = ray_intersect_vox_AABB(
    rays_o, rays_d, voxels_center, 0.2, 10, 10)
print("\033[0;33;40m",'ha_hits',hits.shape, "\033[0m")
print("\033[0;33;40m",'ha_intersections[min_depth]',intersections["min_depth"].shape, "\033[0m")
print("\033[0;33;40m",'ha_intersections[max_depth]',intersections["max_depth"].shape, "\033[0m")
print("\033[0;33;40m",'ha_intersections[intersected_voxel_idx]',intersections["intersected_voxel_idx"].shape, "\033[0m")
# np.savetxt(f'ha_hits.txt', hits.detach().cpu().numpy())
# np.savetxt(f'ha_intersections["min_depth"].txt', intersections["min_depth"][0].detach().cpu().numpy())
# np.savetxt(f'ha_intersections["max_depth"].txt', intersections["max_depth"][0].detach().cpu().numpy())
# np.savetxt(f'ha_intersections["intersected_voxel_idx"].txt', intersections["intersected_voxel_idx"][0].detach().cpu().numpy())

