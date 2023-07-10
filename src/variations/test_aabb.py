from voxel_helpers import ray_intersect_vox_AABB, ray_sample
import torch
import pandas as pd
import numpy as np
# from .voxel_helpers import ray_intersect_vox, ray_sample
import torch.nn.functional as F
torch.classes.load_library(
    "/home/guowzh/code/Proud-slam/third_party/sparse_octree/build/lib.linux-x86_64-cpython-310/svo.cpython-310-x86_64-linux-gnu.so")
def ray(ray_start, ray_dir, depths):
    return ray_start + ray_dir * depths



# @torch.enable_grad()
# def encoding_3d(pos, d):
#     encoding = torch.zeros(pos.shape[0],d)
#     # print("\033[0;33;40m",'encoding',encoding.shape, "\033[0m")
#     for i in range(d//2):
#         x = 1/10000**(2*i/d)
#         encoding[:,2*i] = torch.sum(torch.sin(pos * x),dim = -1)
#         encoding[:,2*i+1] = torch.sum(torch.cos(pos * x),dim = -1)
#     return encoding

# @torch.enable_grad()
# def get_features_pcd(samples, map_states):
#     # global flag
#     # pointclouds_xyz = map_states["pointclouds_xyz"].cuda()
#     # pointclouds_color = map_states["pointclouds_color"].cuda()
#     sampled_idx = samples["sampled_point_voxel_idx"].long()
#     sampled_xyz = samples["sampled_point_xyz"].requires_grad_(True)
#     sampled_d = samples["sampled_point_ray_direction"].cuda()
#     # sampled_dep = samples["sampled_point_depth"].cuda()
#     pcd_xyz = F.embedding(sampled_idx, pointclouds_xyz.reshape(pointclouds_xyz.shape[0],-1))
#     # pcd_color = F.embedding(sampled_idx, pointclouds_color.reshape(pointclouds_color.shape[0],-1))

    
#     # pcd_feats = resnet(pcd_xyz.reshape(-1,10,3), pcd_color.reshape(-1,10,3))
    
#     feats_color = get_embeddings_pcd(sampled_xyz, pcd_xyz.reshape(pcd_xyz.shape[0],-1,3), pcd_feats.reshape(pcd_feats.shape[0],10,-1))
#     feats_xyz = encoding_3d(sampled_xyz,16).cuda()
#     feats_d = encoding_3d(sampled_d,16).cuda()
#     feats = torch.cat([feats_xyz,feats_d], 1)

#     inputs = { "emb": feats}
#     return inputs

# @torch.enable_grad()
# def get_embeddings_pcd(sample, positions, features):
    
#     distances = torch.sqrt(torch.sum((sample.unsqueeze(1) - positions) ** 2, dim=-1))  # Shape: (M, N)
#     weights = torch.softmax(-(distances*10), dim=-1)  # Shape: (M, N)
#     sample_features = torch.sum(weights.unsqueeze(-1) * features, dim=1)  # Shape: (M, 64)
#     return sample_features


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
# voxels_center = svo.get_centres()
voxels_center,v_points,v_colors,v_childrne = svo.getVoxelPoints()
voxels_center = voxels_center.cuda()

print("\033[0;33;40m",'hash voxels',voxels_center.shape, "\033[0m")
# np.savetxt(f'hash_voxels_center.txt', voxels_center.detach().cpu().numpy())

rays_d_path='/home/guowzh/code/Proud-slam/vox_0_rays_d.txt'
rays_o_path='/home/guowzh/code/Proud-slam/vox_0_rays_o.txt'
rays_d_data = pd.read_table(rays_d_path,sep=' ',header=None)
rays_o_data = pd.read_table(rays_o_path,sep=' ',header=None)
rays_d = torch.from_numpy(rays_d_data.values).unsqueeze(0).contiguous().cuda()
rays_o = torch.from_numpy(rays_o_data.values).unsqueeze(0).contiguous().cuda()
print("\033[0;33;40m",'rays_d',rays_d.shape, "\033[0m")
print("\033[0;33;40m",'rays_o',rays_o.shape, "\033[0m")


intersections, hits = ray_intersect_vox_AABB(
    rays_o, rays_d, voxels_center, 0.2, 10, 10)
print("\033[0;33;40m",'ha_hits',hits.shape, "\033[0m")
print("\033[0;33;40m",'ha_intersections[min_depth]',intersections["min_depth"].shape, "\033[0m")
print("\033[0;33;40m",'ha_intersections[max_depth]',intersections["max_depth"].shape, "\033[0m")
print("\033[0;33;40m",'ha_intersections[intersected_voxel_idx]',intersections["intersected_voxel_idx"].shape, "\033[0m")
# np.savetxt(f'ha_hits.txt', hits.detach().cpu().numpy())
np.savetxt(f'ha_intersections["min_depth"].txt', intersections["min_depth"][0].detach().cpu().numpy())
np.savetxt(f'ha_intersections["max_depth"].txt', intersections["max_depth"][0].detach().cpu().numpy())
np.savetxt(f'ha_intersections["intersected_voxel_idx"].txt', intersections["intersected_voxel_idx"][0].detach().cpu().numpy())

ray_mask = hits.view(1, -1)
intersections = {
        name: outs[ray_mask].reshape(-1, outs.size(-1))
        for name, outs in intersections.items()
    }
    
    # the ray after hit test, remove the hole or too far item-------------
rays_o = rays_o[ray_mask].reshape(-1, 3)
rays_d = rays_d[ray_mask].reshape(-1, 3)
# ---------rays_o torch.Size([1024, 3]) 
# ---------rays_d torch.Size([1024, 3])
samples = ray_sample(intersections, 0.02)
# sample configure caculation 计算光线上各个采样点的深度和索引有效值
sampled_depth = samples['sampled_point_depth']
sampled_idx = samples['sampled_point_voxel_idx'].long()
# ---------sampled_depth torch.Size([1024, 57]) 
# ---------sampled_idx torch.Size([1024, 57]) 
# only compute when the rays hits  [1017, 60]
sample_mask = sampled_idx.ne(-1)

# ---------rays_o torch.Size([1024, 3]) 
# ---------rays_d torch.Size([1024, 3])


# if sample_mask.sum() == 0:  # miss everything skip
#     return None, 0

# sample points xyz through the ray-----通过点深度和有效值算出采样点xyz和方向
sampled_xyz = ray(rays_o.unsqueeze(
    1), rays_d.unsqueeze(1), sampled_depth.unsqueeze(2))
sampled_dir = rays_d.unsqueeze(1).expand(
    *sampled_depth.size(), rays_d.size()[-1])
sampled_dir = sampled_dir / \
    (torch.norm(sampled_dir, 2, -1, keepdim=True) + 1e-8)
# caculate the final sampled point's position and direction

# ---------sampled_xyz torch.Size([1024, 57, 3]) 
# ---------sampled_dir torch.Size([1024, 57, 3]) 
print("\033[0;33;40m",'sampled_xyz',sampled_xyz.shape, "\033[0m")
# print("\033[0;33;40m",'sampled_dir',sampled_dir.shape, "\033[0m")

samples['sampled_point_xyz'] = sampled_xyz
samples['sampled_point_ray_direction'] = sampled_dir

#     samples_valid = {
#     "sampled_point_depth": sampled_depth,
#     "sampled_point_distance": sampled_dists,
#     "sampled_point_voxel_idx": sampled_idx,
#      sampled_point_xyz
#      sampled_point_ray_direction
# }

samples_valid = {name: s[sample_mask] for name, s in samples.items()}
num_points = samples_valid['sampled_point_depth'].shape[0]
field_outputs = []

chunk_size = 10000

# print("\033[0;33;40m",'samples_valid[sampled_point_xyz]',samples_valid['sampled_point_xyz'].shape, "\033[0m")


for i in range(0, num_points, chunk_size):
    chunk_samples = {name: s[i:i+chunk_size]
                        for name, s in samples_valid.items()}
    # print("\033[0;33;40m",'chunk_samples[sampled_point_xyz]',chunk_samples['sampled_point_xyz'].shape, "\033[0m")
    # sampled_xyz = chunk_samples["sampled_point_xyz"]
    
    
    # clos_points,clos_colors = svo.getClosePoints(sampled_xyz.cpu().float())
    # print("\033[0;33;40m",'clos_points',clos_points.shape, "\033[0m")
    # print("\033[0;33;40m",'clos_colors',clos_colors.shape, "\033[0m")
    # np.savetxt(f'clos_points.txt', clos_points[:,14,:].detach().cpu().numpy())
    # np.savetxt(f'clos_colors.txt', clos_colors[:,14,:].detach().cpu().numpy())
    # np.savetxt(f'sampled_xyz.txt', sampled_xyz.detach().cpu().numpy())
        

    
