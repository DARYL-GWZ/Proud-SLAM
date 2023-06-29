from voxel_helpers import ray_intersect_vox_AABB, ray_sample
import torch
import pandas as pd
import numpy as np
# from .voxel_helpers import ray_intersect_vox, ray_sample

torch.classes.load_library(
    "/home/guowzh/code/Proud-slam/third_party/sparse_octree/build/lib.linux-x86_64-cpython-310/svo.cpython-310-x86_64-linux-gnu.so")
def ray(ray_start, ray_dir, depths):
    return ray_start + ray_dir * depths

# def get_features_pcd(samples, map_states,voxel_size):
#     global flag
#     pointclouds_xyz = map_states["pointclouds_xyz"].cuda()
#     pointclouds_color = map_states["pointclouds_color"].cuda()
#     sampled_idx = samples["sampled_point_voxel_idx"].long()
#     sampled_xyz = samples["sampled_point_xyz"].requires_grad_(True)

#     # pcd_feats = resnet(pcd_xyz.reshape(-1,8,3), pcd_color.reshape(-1,8,3))
    
#     pcd_xyz = F.embedding(sampled_idx, pointclouds_xyz.reshape(pointclouds_xyz.shape[0],-1))
#     pcd_color = F.embedding(sampled_idx, pointclouds_color.reshape(pointclouds_color.shape[0],-1))
#     # print("\033[0;33;40m",'pcd_xyz',pcd_xyz.shape, "\033[0m")
#     pcd_feats = resnet(pcd_xyz.reshape(-1,8,3), pcd_color.reshape(-1,8,3))
    
#     # pcd_feats = F.embedding(sampled_idx, pointclouds_feature.reshape(pointclouds_feature.shape[0],-1))
#     print("\033[0;33;40m",'====print pcd======', "\033[0m")
#     # print("\033[0;33;40m",'pcd_xyz',pcd_xyz.shape, "\033[0m")
#     print("\033[0;33;40m",'pcd_point_features',pcd_feats.shape, "\033[0m")
#     # print("\033[0;33;40m",'sampled_xyz',sampled_xyz.shape, "\033[0m")
#     # np.savetxt('pcd_xyz.txt', pcd_xyz.detach().cpu().numpy())
#     # np.savetxt('pcd_feats0.txt', pcd_feats[:,0,:].detach().cpu().numpy())
#     # np.savetxt('pcd_feats1.txt', pcd_feats[:,1,:].detach().cpu().numpy())
#     # np.savetxt('pcd_feats2.txt', pcd_feats[:,2,:].detach().cpu().numpy())
#     pcd_point_features = pcd_feats.reshape(pcd_feats.shape[0],-1)
#     # np.savetxt(f'pcd_{flag}_point_xyz.txt', pcd_xyz.detach().cpu().numpy())
#     # np.savetxt(f'pcd_{flag}_point_features.txt', pcd_point_features.detach().cpu().numpy())
    
#     feats = get_embeddings_pcd(sampled_xyz, pcd_xyz.reshape(pcd_xyz.shape[0],-1,3), pcd_feats.reshape(pcd_feats.shape[0],-1,16),voxel_size)
#     # feats = torch.ones(sampled_xyz.shape[0],16).float().cuda()
#     # print("\033[0;33;40m",'===============', "\033[0m")
#     # np.savetxt(f'pcd_{flag}_feats.txt', feats.detach().cpu().numpy())
#     # flag = flag + 1 
#     # np.savetxt('feats_pcd.txt', feats.detach().cpu().numpy())
#     # print("\033[0;33;40m",'feats_pcd',feats.shape, "\033[0m")

#     # print("\033[0;33;40m",'feats_pcd',feats.shape, "\033[0m")
#     print("\033[0;33;40m",'-----print pcd over-----', "\033[0m")
#     inputs = { "emb": feats}
#     return inputs

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

# sample configure caculation------计算各个ray上采样点的深度和有效采样点
# 根据光线和体素的碰撞，在体素中进行采样
# intersections为经过碰撞检测后的每条光线经过体素的最大和最小深度，和光线穿过的index
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
# print("\033[0;33;40m",'sampled_idx',sampled_idx.shape, "\033[0m")
# print("\033[0;33;40m",'sampled_idx',sampled_idx[0,:], "\033[0m")
# print("\033[0;33;40m",'sampled_depth',sampled_depth.shape, "\033[0m")
# print("\033[0;33;40m",'sampled_depth',sampled_depth[0,:], "\033[0m")

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

print("\033[0;33;40m",'samples_valid[sampled_point_xyz]',samples_valid['sampled_point_xyz'].shape, "\033[0m")

for i in range(0, num_points, chunk_size):
    chunk_samples = {name: s[i:i+chunk_size]
                        for name, s in samples_valid.items()}
    print("\033[0;33;40m",'chunk_samples[sampled_point_xyz]',chunk_samples['sampled_point_xyz'].shape, "\033[0m")
    sampled_xyz = chunk_samples["sampled_point_xyz"]
    

    clos_points,clos_colors = svo.getClosePoints(sampled_xyz.cpu().float())
    print("\033[0;33;40m",'clos_points',clos_points.shape, "\033[0m")
    print("\033[0;33;40m",'clos_colors',clos_colors.shape, "\033[0m")
    np.savetxt(f'clos_points.txt', clos_points[:,0,:].detach().cpu().numpy())
    np.savetxt(f'clos_colors.txt', clos_colors[:,0,:].detach().cpu().numpy())
    np.savetxt(f'sampled_xyz.txt', sampled_xyz.detach().cpu().numpy())
        

    
