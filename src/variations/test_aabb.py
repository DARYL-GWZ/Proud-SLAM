from .voxel_helpers import ray_intersect_vox_AABB
import torch


rays_o = torch.tensor([0,0,0])
intersections, hits = ray_intersect_vox_AABB(
    rays_o, rays_d, centres,
     voxel_size, max_voxel_hit, max_distance)