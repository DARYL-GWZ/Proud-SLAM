from copy import deepcopy
import torch
import torch.nn.functional as F
import open3d as o3d
from .voxel_helpers import ray_intersect_vox, ray_sample
import numpy as np
from math import sqrt
from annoy import AnnoyIndex
from tensorboardX import SummaryWriter
from time import sleep


flag = 0
def ray(ray_start, ray_dir, depths):
    return ray_start + ray_dir * depths


def fill_in(shape, mask, input, initial=1.0):
    if isinstance(initial, torch.Tensor):
        output = initial.expand(*shape)
    else:
        output = input.new_ones(*shape) * initial
    return output.masked_scatter(mask.unsqueeze(-1).expand(*shape), input)


def masked_scatter(mask, x):
    B, K = mask.size()
    if x.dim() == 1:
        return x.new_zeros(B, K).masked_scatter(mask, x)
    return x.new_zeros(B, K, x.size(-1)).masked_scatter(
        mask.unsqueeze(-1).expand(B, K, x.size(-1)), x
    )


def masked_scatter_ones(mask, x):
    B, K = mask.size()
    if x.dim() == 1:
        return x.new_ones(B, K).masked_scatter(mask, x)
    return x.new_ones(B, K, x.size(-1)).masked_scatter(
        mask.unsqueeze(-1).expand(B, K, x.size(-1)), x
    )

# 这段代码是一个函数，用于进行三线性插值，其中p是一个3D坐标，表示当前点的位置，
# q是一个3D坐标，表示当前点周围的8个点的位置，point_feats是一个特征向量，
# 表示当前点周围的8个点的特征。这段代码的主要作用是计算当前点的特征，
@torch.enable_grad()
def trilinear_interp(p, q, point_feats):
    global flag
    # print("\033[0;33;40m",'p',p.shape, "\033[0m")
    # print("\033[0;33;40m",'q',q.shape, "\033[0m")
    weights = (p * q + (1 - p) * (1 - q)).prod(dim=-1, keepdim=True)
    if point_feats.dim() == 2:
        point_feats = point_feats.view(point_feats.size(0), 8, -1)
    # print("\033[0;33;40m",'weights_vox',weights.shape, "\033[0m")
    # print("\033[0;33;40m",'point_feats',point_feats.shape, "\033[0m")
    # np.savetxt(f'vox_{flag}_weights.txt', weights[:,:,0].detach().cpu().numpy())
    
    point_feats = (weights * point_feats).sum(1)
    return point_feats

# 这段代码是一个函数，用于计算当前点周围的8个点的位置，其中point_xyz是一个3D坐标，
# 表示当前点的位置，
# quarter_voxel是一个标量，表示体素的大小，
# offset_only是一个布尔值，表示是否只计算偏移量，
# bits是一个标量，表示偏移量的精度。
# 这段代码的主要作用是将当前点周围的8个点的位置转换为体素坐标，以便进行三线性插值。
def offset_points(point_xyz, quarter_voxel=1, offset_only=False, bits=2):
    c = torch.arange(1, 2 * bits, 2, device=point_xyz.device)
    # print("\033[0;33;40m",'c',c, "\033[0m")
    ox, oy, oz = torch.meshgrid([c, c, c], indexing='ij')
    # print("\033[0;33;40m",'ox',ox, "\033[0m")
    # print("\033[0;33;40m",'ox.shape',ox.shape, "\033[0m")
    
    offset = (torch.cat([
        ox.reshape(-1, 1),
        oy.reshape(-1, 1),
        oz.reshape(-1, 1)], 1).type_as(point_xyz) - bits) / float(bits - 1)
    # print("\033[0;33;40m",'offset',offset.shape, "\033[0m")
    
    if not offset_only:
        return (
            point_xyz.unsqueeze(1) + offset.unsqueeze(0).type_as(point_xyz) * quarter_voxel)
    return offset.type_as(point_xyz) * quarter_voxel


@torch.enable_grad()
def get_embeddings_vox(sampled_xyz, point_xyz, point_feats, voxel_size):
    global flag
    # tri-linear interpolation
    # 转换到体素归一化坐标
    p = ((sampled_xyz - point_xyz) / voxel_size + 0.5).unsqueeze(1)
    q = offset_points(p, 0.5, offset_only=True).unsqueeze(0) + 0.5
    # np.savetxt(f'vox_{flag}_p.txt', p[:,0,:].detach().cpu().numpy())
    # np.savetxt(f'vox_{flag}_q.txt', q[0,:,:].detach().cpu().numpy())

    feats = trilinear_interp(p, q, point_feats).float()
    # if self.args.local_coord:
    # feats = torch.cat([(p-.5).squeeze(1).float(), feats], dim=-1)
    return feats

# // voxels是一个N*4的张量，其中N是八叉树中非FEATURE类型的节点的数量。每一行代表一个节点对应的体素，包含了x,y,z坐标和边长。
# // children是一个N*8的张量，其中N和上面相同。每一行代表一个节点和其八个子节点之间的索引关系。如果某个子节点不存在或者是FEATURE类型，则对应位置为-1。
# // point_feats是N*8的张量，其中N和上面相同。每一行代表一个节点对应体素的八个顶点是否有特征。如果某个顶点有特征，则对应位置为该特征节点在八叉树中的索引；否则为-1。
@torch.enable_grad()
def get_features_vox(samples, map_states, voxel_size):
    # encoder states
    global flag
    point_feats = map_states["voxel_vertex_idx"].cuda()
    point_xyz = map_states["voxel_center_xyz"].cuda()
    values = map_states["voxel_vertex_emb"].cuda()
    # print("\033[0;33;40m",'----------------', "\033[0m")
    # # print("\033[0;33;40m",'values',values.shape, "\033[0m")
    # print("\033[0;33;40m",'原始point_xyz',point_xyz.shape, "\033[0m")
    # print("\033[0;33;40m",'原始point_feats',point_feats.shape, "\033[0m")
    # print("\033[0;33;40m",'point_xyz222',point_xyz.shape, "\033[0m")
    
    # ray point samples
    sampled_idx = samples["sampled_point_voxel_idx"].long()
    # sampled_xyz = samples["sampled_point_xyz"]
    sampled_xyz = samples["sampled_point_xyz"].requires_grad_(True)
    sampled_dis = samples["sampled_point_distance"]
    # print("\033[0;33;40m",'sampled_idx',sampled_idx.shape, "\033[0m")
    # print("\033[0;33;40m",'sampled_idx',sampled_idx[0], "\033[0m")
    point_xyz = F.embedding(sampled_idx, point_xyz)
    # print("\033[0;33;40m",'point_xyz',point_xyz.shape, "\033[0m")
    # print("\033[0;33;40m",'point_xyz',point_xyz[0,:], "\033[0m")
    
    point_feats = F.embedding(F.embedding(
        sampled_idx, point_feats), values).view(point_xyz.size(0), -1)
    
    # print("\033[0;33;40m",'----vox----', "\033[0m")
    # print("\033[0;33;40m",'point_xyz',point_xyz.shape, "\033[0m")
    # print("\033[0;33;40m",'point_feats',point_feats.shape, "\033[0m")
    # np.savetxt(f'vox_{flag}_point_xyz.txt', point_xyz.detach().cpu().numpy())
    # np.savetxt(f'vox_{flag}_point_feats.txt', point_feats.detach().cpu().numpy())
    
    # np.savetxt('point_xyz.txt', point_xyz.detach().cpu().numpy())
    # np.savetxt('vox_feats.txt', point_feats.detach().cpu().numpy())
    
    # print("\033[0;33;40m",'F.embedding(sampled_idx, point_feats)',F.embedding(sampled_idx, point_feats).shape, "\033[0m")
    # print("\033[0;33;40m",'point_feats',point_feats.shape, "\033[0m")
    # print("\033[0;33;40m",'sampled_xyz',sampled_xyz.shape, "\033[0m")
    
    feats = get_embeddings_vox(sampled_xyz, point_xyz, point_feats, voxel_size)
    # print("\033[0;33;40m",'feats',feats.shape, "\033[0m")
    # print("\033[0;33;40m",'------------', "\033[0m")
    # np.savetxt(f'vox_{flag}_feats.txt', feats.detach().cpu().numpy())
    # flag = flag +1
    # np.savetxt('feats_vox.txt', feats.detach().cpu().numpy())
    # print("\033[0;33;40m",'feats_vox',feats.shape, "\033[0m")
    # print("\033[0;33;40m",'sampled_dis',sampled_dis.shape, "\033[0m")
    
    # print("\033[0;33;40m",'=====vox======', "\033[0m")
    
    inputs = {"dists": sampled_dis, "emb": feats}
    return inputs

@torch.enable_grad()
def encoding_3d(pos, d):
    encoding = torch.zeros(pos.shape[0],pos.shape[1],d)
    # print("\033[0;33;40m",'encoding',encoding.shape, "\033[0m")
    for i in range(d//2):
        x = 1/10000**(2*i/d)
        encoding[:,:,2*i] = torch.sum(torch.sin(pos * x),dim = -1)
        encoding[:,:,2*i+1] = torch.sum(torch.cos(pos * x),dim = -1)
    return encoding

@torch.enable_grad()
def get_features_pcd(samples, map_states, resnet,voxel_size):
    global flag
    pointclouds_xyz = map_states["pointclouds_xyz"].cuda()
    pointclouds_color = map_states["pointclouds_color"].cuda()
    sampled_idx = samples["sampled_point_voxel_idx"].long()
    sampled_xyz = samples["sampled_point_xyz"].requires_grad_(True)

    # pcd_feats = resnet(pcd_xyz.reshape(-1,8,3), pcd_color.reshape(-1,8,3))
    
    pcd_xyz = F.embedding(sampled_idx, pointclouds_xyz.reshape(pointclouds_xyz.shape[0],-1))
    pcd_color = F.embedding(sampled_idx, pointclouds_color.reshape(pointclouds_color.shape[0],-1))
    # print("\033[0;33;40m",'pcd_xyz',pcd_xyz.shape, "\033[0m")
    pcd_feats = resnet(pcd_xyz.reshape(-1,8,3), pcd_color.reshape(-1,8,3))
    
    # pcd_feats = F.embedding(sampled_idx, pointclouds_feature.reshape(pointclouds_feature.shape[0],-1))
    print("\033[0;33;40m",'====print pcd======', "\033[0m")
    # print("\033[0;33;40m",'pcd_xyz',pcd_xyz.shape, "\033[0m")
    print("\033[0;33;40m",'pcd_point_features',pcd_feats.shape, "\033[0m")
    # print("\033[0;33;40m",'sampled_xyz',sampled_xyz.shape, "\033[0m")
    # np.savetxt('pcd_xyz.txt', pcd_xyz.detach().cpu().numpy())
    # np.savetxt('pcd_feats0.txt', pcd_feats[:,0,:].detach().cpu().numpy())
    # np.savetxt('pcd_feats1.txt', pcd_feats[:,1,:].detach().cpu().numpy())
    # np.savetxt('pcd_feats2.txt', pcd_feats[:,2,:].detach().cpu().numpy())
    pcd_point_features = pcd_feats.reshape(pcd_feats.shape[0],-1)
    # np.savetxt(f'pcd_{flag}_point_xyz.txt', pcd_xyz.detach().cpu().numpy())
    # np.savetxt(f'pcd_{flag}_point_features.txt', pcd_point_features.detach().cpu().numpy())
    
    feats = get_embeddings_pcd(sampled_xyz, pcd_xyz.reshape(pcd_xyz.shape[0],-1,3), pcd_feats.reshape(pcd_feats.shape[0],-1,16),voxel_size)
    # feats = torch.ones(sampled_xyz.shape[0],16).float().cuda()
    # print("\033[0;33;40m",'===============', "\033[0m")
    # np.savetxt(f'pcd_{flag}_feats.txt', feats.detach().cpu().numpy())
    # flag = flag + 1 
    # np.savetxt('feats_pcd.txt', feats.detach().cpu().numpy())
    # print("\033[0;33;40m",'feats_pcd',feats.shape, "\033[0m")

    # print("\033[0;33;40m",'feats_pcd',feats.shape, "\033[0m")
    print("\033[0;33;40m",'-----print pcd over-----', "\033[0m")
    inputs = { "emb": feats}
    return inputs

@torch.enable_grad()
def get_embeddings_pcd(sample, positions, features,voxel_size):
    global flag
    """
    Computes features for the given sample points based on positions and features of the point cloud.
    :param sample: Tensor of shape (M, 3) representing the coordinates of M sample points.
    :param positions: Tensor of shape (M, N, 3) representing the coordinates of N points in M voxels.
    :param features: Tensor of shape (M, N, 16) representing the features of N points in M voxels.
    :return: Tensor of shape (M, 16) representing the features of M sample points.
    """
    # sample = ((sample - point_xyz) / voxel_size + 0.5).unsqueeze(1)
    # M, N = positions.shape[:2]
    # Compute distances between sample points and voxel points
    # print("\033[0;33;40m",'sample.unsqueeze(1) - positions',(torch.sum((sample.unsqueeze(1) - positions) ** 2, dim=-1)).shape, "\033[0m")
    distances = torch.sqrt(torch.sum(((sample+ (voxel_size * 0.5)).unsqueeze(1) - positions) ** 2, dim=-1))  # Shape: (M, N)
    # print("\033[0;33;40m",'distances',distances.shape, "\033[0m")
    # Compute weights based on distances.
    # np.savetxt(f'pcd_{flag}_distances.txt', distances.detach().cpu().numpy())
    # print("\033[0;33;40m",'distances',distances.shape, "\033[0m")
    weights = torch.softmax(-(distances*10), dim=-1)  # Shape: (M, N)
    # print("\033[0;33;40m",'weights',weights.shape, "\033[0m")
    # np.savetxt(f'pcd_{flag}weights.txt', weights.detach().cpu().numpy())
    
    # print("\033[0;33;40m",'weights.unsqueeze(-1) ',(weights.unsqueeze(-1) ).shape, "\033[0m")
    # print("\033[0;33;40m",'features',features.shape, "\033[0m")
    # Compute weighted average of features.
    # np.savetxt('weights_pcd.txt', weights.detach().cpu().numpy())
    # print("\033[0;33;40m",'weights_pcd',weights.shape, "\033[0m")
    sample_features = torch.sum(weights.unsqueeze(-1) * features, dim=1)  # Shape: (M, 64)
    return sample_features




@torch.no_grad()
def get_scores(sdf_network, map_states, voxel_size, bits=8):
    feats = map_states["voxel_vertex_idx"]
    points = map_states["voxel_center_xyz"]
    values = map_states["voxel_vertex_emb"]
    chunk_size = 32
    res = bits  # -1

    @torch.no_grad()
    def get_scores_once(feats, points, values):
        # sample points inside voxels
        start = -.5
        end = .5  # - 1./bits

        x = y = z = torch.linspace(start, end, res)
        xx, yy, zz = torch.meshgrid(x, y, z)
        sampled_xyz = torch.stack([xx, yy, zz], dim=-1).float().cuda()

        sampled_xyz *= voxel_size
        sampled_xyz = sampled_xyz.reshape(1, -1, 3) + points.unsqueeze(1)

        sampled_idx = torch.arange(points.size(0), device=points.device)
        sampled_idx = sampled_idx[:, None].expand(*sampled_xyz.size()[:2])
        sampled_idx = sampled_idx.reshape(-1)
        sampled_xyz = sampled_xyz.reshape(-1, 3)

        if sampled_xyz.shape[0] == 0:
            return

        field_inputs = get_features_vox(
            {
                "sampled_point_xyz": sampled_xyz,
                "sampled_point_voxel_idx": sampled_idx,
                "sampled_point_ray_direction": None,
                "sampled_point_distance": None,
            },
            {
                "voxel_vertex_idx": feats,
                "voxel_center_xyz": points,
                "voxel_vertex_emb": values,
            },
            voxel_size
        )

        # evaluation with density
        sdf_values = sdf_network.get_values(field_inputs['emb'].float().cuda())
        return sdf_values.reshape(-1, res ** 3, 4).detach().cpu()

    return torch.cat([
        get_scores_once(feats[i: i + chunk_size],
                        points[i: i + chunk_size], values)
        for i in range(0, points.size(0), chunk_size)], 0).view(-1, res, res, res, 4)


@torch.no_grad()
def eval_points(sdf_network, map_states, sampled_xyz, sampled_idx, voxel_size):
    feats = map_states["voxel_vertex_idx"]
    points = map_states["voxel_center_xyz"]
    values = map_states["voxel_vertex_emb"]

    # sampled_xyz = sampled_xyz.reshape(1, 3) + points.unsqueeze(1)
    # sampled_idx = sampled_idx[None, :].expand(*sampled_xyz.size()[:2])
    sampled_idx = sampled_idx.reshape(-1)
    sampled_xyz = sampled_xyz.reshape(-1, 3)

    if sampled_xyz.shape[0] == 0:
        return

    field_inputs = get_features_vox(
        {
            "sampled_point_xyz": sampled_xyz,
            "sampled_point_voxel_idx": sampled_idx,
            "sampled_point_ray_direction": None,
            "sampled_point_distance": None,
        },
        {
            "voxel_vertex_idx": feats,
            "voxel_center_xyz": points,
            "voxel_vertex_emb": values,
        },
        voxel_size
    )

    # evaluation with density
    sdf_values = sdf_network.get_values(field_inputs['emb'].float().cuda())
    return sdf_values.reshape(-1, 4)[:, :3].detach().cpu()

    # return torch.cat([
    #     get_scores_once(feats[i: i + chunk_size],
    #                     points[i: i + chunk_size], values)
    #     for i in range(0, points.size(0), chunk_size)], 0).view(-1, res, res, res, 4)

# def sample_feature(query_xyz,query_fea,samples_xyz):
#     d = np.array([])
    # fea = np.zeros(shape=(1,query_fea.shape[1]))
    # for i in range(0,query_xyz.shape[0]):
    #     dis = sqrt((samples_xyz[0] - query_xyz[i][0])**2 + (samples_xyz[1] - query_xyz[i][1])**2+ (samples_xyz[2] - query_xyz[i][2])**2)
    #     d = np.append(d, (1/dis))
    # all_dis = np.sum(d)
    # # query_fea [5 x 128]
    # for j in range(0,query_xyz.shape[0]):
    #     feature = np.multiply((d[j] / all_dis), query_fea[j][:]).reshape(1,query_fea.shape[1])
    #     fea = np.add(fea, feature)
    # # print("\033[0;33;40m",'fea',fea.shape, "\033[0m")
    # return fea



def render_rays(
        rays_o,
        rays_d,
        map_states,
        sdf_network,
        resnet,
        step_size,
        voxel_size,
        truncation,
        max_voxel_hit,
        max_distance,
        chunk_size=10000,
        profiler=None,
        return_raw=False
):
    global flag
    centres = map_states["voxel_center_xyz"]
    childrens = map_states["voxel_structure"]
    # print("\033[0;33;40m",'centres',centres.shape, "\033[0m")
    
    # hit test--------------------
    if profiler is not None:
        profiler.tick("ray_intersect")
        
    intersections, hits = ray_intersect_vox(
        rays_o, rays_d, centres,
        childrens, voxel_size, max_voxel_hit, max_distance)
    # 一共1024根光线
    # ---------rays_o torch.Size([1, 1024, 3]) 
    # ---------rays_d torch.Size([1, 1024, 3])
    # intersections["min_depth"]对应穿过体素的起点位置 [1, 1024, 9]
    # intersections["max_depth"]对应穿过体素的终点位置 [1, 1024, 9]
    # intersections["intersected_voxel_idx"]对应穿过体素的index [1, 1024, 9]
    # hits对应每根光线是否有效，为true或false [1, 1024]
   
    if profiler is not None:
        profiler.tok("ray_intersect")
    assert(hits.sum() > 0)

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

    
    np.savetxt(f'vox_{flag}_rays_o.txt', rays_o.detach().cpu().numpy())
    np.savetxt(f'vox_{flag}_rays_d.txt', rays_d.detach().cpu().numpy())
    flag = flag + 1
    # print("\033[0;33;40m",'rays_o',rays_o.shape, "\033[0m")
    # print("\033[0;33;40m",'rays_d',rays_d.shape, "\033[0m")
    if profiler is not None:
        profiler.tick("ray_sample")
    # sample configure caculation------计算各个ray上采样点的深度和有效采样点
    # 根据光线和体素的碰撞，在体素中进行采样
    # intersections为经过碰撞检测后的每条光线经过体素的最大和最小深度，和光线穿过的index
    samples = ray_sample(intersections, step_size=step_size)
   
    if profiler is not None:
        profiler.tok("ray_sample")
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
    
    if sample_mask.sum() == 0:  # miss everything skip
        return None, 0
    
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
    # print("\033[0;33;40m",'sampled_xyz',sampled_xyz.shape, "\033[0m")
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
    if chunk_size < 0:
        chunk_size = num_points
    
    # samples_valid[sampled_point_xyz] torch.Size([20125, 3]) 
    # print("\033[0;33;40m",'samples_valid[sampled_point_xyz]',samples_valid['sampled_point_xyz'].shape, "\033[0m")
    # print("\033[0;33;40m",'map_states["voxel_center_xyz"]',map_states["voxel_center_xyz"].shape, "\033[0m")
    for i in range(0, num_points, chunk_size):
        chunk_samples = {name: s[i:i+chunk_size]
                         for name, s in samples_valid.items()}
        # print("\033[0;33;40m",'chunk_samples',chunk_samples['sampled_point_depth'].shape, "\033[0m")


        # get encoder features as inputs
        # if profiler is not None:
        #     profiler.tick("get_features_vox")
            # caculate the  embeddings, 三线性插值
        # chunk_inputs {"dists": sampled_dis, "emb": feats}
        # print("\033[0;33;40m",'=====123===', "\033[0m")
        # chunk_inputs = get_features_pcd(chunk_samples, map_states, resnet, voxel_size)
        # print("\033[0;31;40m",'chunk_inputs11',chunk_inputs['emb'].shape, "\033[0m")

        chunk_inputs = get_features_vox(chunk_samples, map_states, voxel_size)

        # sleep(500)
        # chunk_inputs = get_features_vox(chunk_samples, map_states, voxel_size)
        
        # print("\033[0;31;40m",'chunk_inputs',chunk_inputs['emb'].shape, "\033[0m")
        
        if profiler is not None:
            profiler.tok("get_features_vox")

        # forward implicit fields
        if profiler is not None:
            profiler.tick("render_core")

        chunk_outputs = sdf_network(chunk_inputs)
        if profiler is not None:
            profiler.tok("render_core")

        field_outputs.append(chunk_outputs)
    # the sdf and rgb values from the net
    field_outputs = {name: torch.cat(
        [r[name] for r in field_outputs], dim=0) for name in field_outputs[0]}

    outputs = {'sample_mask': sample_mask}

    # 对sdf和color进行积分
    sdf = masked_scatter_ones(sample_mask, field_outputs['sdf']).squeeze(-1)
    colour = masked_scatter(sample_mask, field_outputs['color'])
    # colour = torch.sigmoid(colour)
    sample_mask = outputs['sample_mask']

    valid_mask = torch.where(
        sample_mask, torch.ones_like(
            sample_mask), torch.zeros_like(sample_mask)
    )

    # convert sdf to weight
    def sdf2weights(sdf_in, trunc):
        weights = torch.sigmoid(sdf_in / trunc) * \
            torch.sigmoid(-sdf_in / trunc)

        signs = sdf_in[:, 1:] * sdf_in[:, :-1]
        mask = torch.where(
            signs < 0.0, torch.ones_like(signs), torch.zeros_like(signs)
        )
        inds = torch.argmax(mask, axis=1)
        inds = inds[..., None]
        z_min = torch.gather(z_vals, 1, inds)
        mask = torch.where(
            z_vals < z_min + trunc,
            torch.ones_like(z_vals),
            torch.zeros_like(z_vals),
        )
        weights = weights * mask * valid_mask
        # weights = weights * valid_mask
        return weights / (torch.sum(weights, dim=-1, keepdims=True) + 1e-8), z_min

    z_vals = samples["sampled_point_depth"]

    weights, z_min = sdf2weights(sdf, truncation)
    rgb = torch.sum(weights[..., None] * colour, dim=-2)
    depth = torch.sum(weights * z_vals, dim=-1)

    return {
        "weights": weights,
        "color": rgb,
        "depth": depth,
        "z_vals": z_vals,
        "sdf": sdf,
        "weights": weights,
        "ray_mask": ray_mask,
        "raw": z_min if return_raw else None
    }


def bundle_adjust_frames(
    keyframe_graph,
    map_states,
    sdf_network,
    resnet,
    loss_criteria,
    voxel_size,
    step_size,
    N_rays=512,
    num_iterations=10,
    truncation=0.1,
    max_voxel_hit=10,
    max_distance=10,
    learning_rate=[1e-2, 5e-3],
    embed_optim=None,
    model_optim=None,
    resnet_optim=None,
    update_pose=True,
):
    # save_path = './proud_log/'
    # summary_writer = SummaryWriter(save_path)
    # optimize_params = [{'params': embeddings, 'lr': learning_rate[0]}]
    optimizers = [embed_optim]
    # optimizers = [model_optim]
    if model_optim is not None:
        # optimize_params += [{'params': sdf_network.parameters(),
        #                      'lr': learning_rate[0]}]
        optimizers += [model_optim]
        
    if resnet_optim is not None:
        optimizers += [resnet_optim]
        # print("\033[0;33;40m",'resnet_optim intro success', "\033[0m")
        
        
    # optimize_params=[]
    for keyframe in keyframe_graph:
        if keyframe.stamp != 0 and update_pose:
            optimizers += [keyframe.optim]
            # keyframe.pose.requires_grad_(True)
            # optimize_params += [{
            #     'params': keyframe.pose.parameters(), 'lr': learning_rate[1]
            # }]
    
    # if len(optimize_params) != 0:
    #     pose_optim = torch.optim.Adam(optimize_params)
    #     optimizers += [pose_optim]
    
    
    # writer = SummaryWriter()
    # sampling number after getted the new frame
    for i in range(num_iterations):

        rays_o = []
        rays_d = []
        rgb_samples = []
        depth_samples = []
        # print("\033[0;33;40m",'i= ',i , "\033[0m")
        # print("\033[0;33;40m",'voxels',voxels.shape, "\033[0m")
        
        
        # random sampling in the whole keyframe_graph
        for frame in keyframe_graph:
            pose = frame.get_pose().cuda()
            frame.sample_rays(N_rays)
            # print("\033[0;33;40m",'frame',frame, "\033[0m")
            sample_mask = frame.sample_mask.cuda()
            sampled_rays_d = frame.rays_d[sample_mask].cuda()
            # print("\033[0;33;40m",'sample_mask',sample_mask, "\033[0m")
            # print("\033[0;33;40m",'sample_mask_shape',sample_mask.shape, "\033[0m")
            
            R = pose[: 3, : 3].transpose(-1, -2)
            # 每条光线的在nerf世界坐标系下的方向
            sampled_rays_d = sampled_rays_d@R
            # print("\033[0;33;40m",'sampled_rays_d',sampled_rays_d, "\033[0m")
            # 每条光线的在nerf坐标系下的原点
            sampled_rays_o = pose[: 3, 3].reshape(
                1, -1).expand_as(sampled_rays_d)
            # print("\033[0;33;40m",'sampled_rays_o',sampled_rays_o, "\033[0m")
            rays_d += [sampled_rays_d]
            rays_o += [sampled_rays_o]
            rgb_samples += [frame.rgb.cuda()[sample_mask]]
            depth_samples += [frame.depth.cuda()[sample_mask]]

        rays_d = torch.cat(rays_d, dim=0).unsqueeze(0)
        rays_o = torch.cat(rays_o, dim=0).unsqueeze(0)
        
        rgb_samples = torch.cat(rgb_samples, dim=0).unsqueeze(0)
        depth_samples = torch.cat(depth_samples, dim=0).unsqueeze(0)
        # random sampling in the whole keyframe_graph
        final_outputs = render_rays(
            rays_o,
            rays_d,
            map_states,
            sdf_network,
            resnet,
            step_size,
            voxel_size,
            truncation,
            max_voxel_hit,
            max_distance,
            # chunk_size=-1
        )

        loss, _ = loss_criteria(
            final_outputs, (rgb_samples, depth_samples))
        # print("\033[0;33;40m",'loss',loss, "\033[0m")
        # with torch.autograd.set_detect_anomaly(True):
        # writer.add_scalar('loss', i**2, global_step=i)
        for optim in optimizers:
            optim.zero_grad()
        # print("\033[0;33;40m",'optimizers',optimizers, "\033[0m")
        # print("\033[0;33;40m",'backward', "\033[0m")
        loss.backward()
        # print("\033[0;33;40m",'backward后', "\033[0m")

        # print("\033[0;33;40m",'backward后', "\033[0m")
        for optim in optimizers:
            optim.step()


def track_frame(
    frame_pose,
    curr_frame,
    map_states,
    sdf_network,
    resnet,
    loss_criteria,
    voxel_size,
    N_rays=512,
    step_size=0.05,
    num_iterations=10,
    truncation=0.1,
    learning_rate=1e-3,
    max_voxel_hit=10,
    max_distance=10,
    profiler=None,
    depth_variance=False
):

    init_pose = deepcopy(frame_pose).cuda()
    init_pose.requires_grad_(True)
    optim = torch.optim.Adam(init_pose.parameters(), lr=learning_rate)

    for iter in range(num_iterations):
        if iter == 0 and profiler is not None:
            profiler.tick("sample_rays")
        curr_frame.sample_rays(N_rays)
        if iter == 0 and profiler is not None:
            profiler.tok("sample_rays")

        sample_mask = curr_frame.sample_mask
        ray_dirs = curr_frame.rays_d[sample_mask].unsqueeze(0).cuda()
        rgb = curr_frame.rgb[sample_mask].cuda()
        depth = curr_frame.depth[sample_mask].cuda()

        ray_dirs_iter = ray_dirs.squeeze(
            0) @ init_pose.rotation().transpose(-1, -2)
        ray_dirs_iter = ray_dirs_iter.unsqueeze(0)
        ray_start_iter = init_pose.translation().reshape(
            1, 1, -1).expand_as(ray_dirs_iter).cuda().contiguous()

        if iter == 0 and profiler is not None:
            profiler.tick("render_rays")
        final_outputs = render_rays(
            ray_start_iter,
            ray_dirs_iter,
            map_states,
            sdf_network,
            resnet,
            step_size,
            voxel_size,
            truncation,
            max_voxel_hit,
            max_distance,
            # chunk_size=-1,
            profiler=profiler if iter == 0 else None
        )
        if iter == 0 and profiler is not None:
            profiler.tok("render_rays")

        hit_mask = final_outputs["ray_mask"].view(N_rays)
        final_outputs["ray_mask"] = hit_mask

        if iter == 0 and profiler is not None:
            profiler.tick("loss_criteria")
        loss, _ = loss_criteria(
            final_outputs, (rgb, depth), weight_depth_loss=depth_variance)
        # print("\033[0;33;40m",'loss',loss, "\033[0m")
        if iter == 0 and profiler is not None:
            profiler.tok("loss_criteria")

        if iter == 0 and profiler is not None:
            profiler.tick("backward step")
        optim.zero_grad()
        # print("\033[0;33;40m",'tracking backward', "\033[0m")
        loss.backward()
        # print("\033[0;33;40m",'tracking backward后', "\033[0m")
        
        optim.step()
        if iter == 0 and profiler is not None:
            profiler.tok("backward step")

    return init_pose, optim, hit_mask
