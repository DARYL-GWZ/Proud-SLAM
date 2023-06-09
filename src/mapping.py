from copy import deepcopy,copy
import random
from time import sleep
import numpy as np

import torch
import trimesh

from criterion import Criterion
from loggers import BasicLogger
from utils.import_util import get_decoder, get_property, get_resnet
from variations.render_helpers import bundle_adjust_frames
from utils.mesh_util import MeshExtractor
import open3d as o3d
import cv2
# from variations.point_feature import PointsResNet

torch.classes.load_library(
    "third_party/sparse_octree/build/lib.linux-x86_64-cpython-310/svo.cpython-310-x86_64-linux-gnu.so")

def get_network_size(net):
    size = 0
    for param in net.parameters():
        size += param.element_size() * param.numel()
    return size / 1024 / 1024


class Mapping:
    def __init__(self, args, logger: BasicLogger, vis=None, **kwargs):
        super().__init__()
        print("\033[0;33;40m",'mapping初始化', "\033[0m")
        self.args = args
        self.logger = logger
        self.visualizer = vis
        self.decoder = get_decoder(args).cuda()
        self.points_encoder = get_resnet(args).cuda()

        self.loss_criteria = Criterion(args)
        self.keyframe_graph = []
        self.initialized = False

        mapper_specs = args.mapper_specs

        # optional args
        self.ckpt_freq = get_property(args, "ckpt_freq", -1)
        self.final_iter = get_property(mapper_specs, "final_iter", 0)
        self.mesh_res = get_property(mapper_specs, "mesh_res", 8)
        self.save_data_freq = get_property(
            args.debug_args, "save_data_freq", 0)

        # required args
        # self.overlap_th = mapper_specs["overlap_th"]
        self.voxel_size = mapper_specs["voxel_size"]
        self.window_size = mapper_specs["window_size"]
        self.num_iterations = mapper_specs["num_iterations"]
        self.n_rays = mapper_specs["N_rays_each"]
        self.sdf_truncation = args.criteria["sdf_truncation"]
        self.max_voxel_hit = mapper_specs["max_voxel_hit"]
        self.step_size = mapper_specs["step_size"]
        self.step_size = self.step_size * self.voxel_size
        self.max_distance = args.data_specs["max_depth"]
        # print("\033[0;33;40m",'self.window_size',self.window_size, "\033[0m")

        embed_dim = args.decoder_specs["in_dim"]
        use_local_coord = mapper_specs["use_local_coord"]
        self.embed_dim = embed_dim - 3 if use_local_coord else embed_dim
        num_embeddings = mapper_specs["num_embeddings"]
        self.mesh_freq = args.debug_args["mesh_freq"]
        self.mesher = MeshExtractor(args)

        self.embeddings = torch.zeros(
            (num_embeddings, self.embed_dim),
            requires_grad=True, dtype=torch.float32,
            device=torch.device("cuda"))
        # self.embeddings = torch.zeros(
        #     (num_embeddings, 2),
        #     requires_grad=True, dtype=torch.float32,
        #     device=torch.device("cuda"))
        # print("\033[0;33;40m",'self.embeddings',self.embeddings.shape, "\033[0m")
        torch.nn.init.normal_(self.embeddings, std=0.01)
        self.embed_optim = torch.optim.Adam([self.embeddings], lr=5e-3)
        self.model_optim = torch.optim.Adam(self.decoder.parameters(), lr=5e-3)
        # print("\033[0;33;40m",'self.voxel_size',self.voxel_size, "\033[0m")
        # print("\033[0;33;40m",'embed_dim',embed_dim, "\033[0m")

        self.svo = torch.classes.svo.Octree()
        self.svo.init(256, embed_dim, self.voxel_size, 8)
        
        self.frame_poses = []
        self.depth_maps = []
        self.last_tracked_frame_id = 0
        # self.points_encoder = PointsResNet(16)
        self.resnet_optim = torch.optim.Adam(self.points_encoder.parameters(), lr=5e-3) 
        self.flag = 0
        
    def spin(self, share_data, kf_buffer):
        print("\033[0;33;40m",'*****mapping process started!*****', "\033[0m")
        
        while True:
            # torch.cuda.empty_cache()
            # print("\033[0;33;40m",'kf_buffer.empty()1',kf_buffer.empty(), "\033[0m")      
            if not kf_buffer.empty():
                tracked_frame = kf_buffer.get()

                if not self.initialized:
                    if self.mesher is not None:
                        self.mesher.rays_d = tracked_frame.get_rays()
                    # print("\033[0;33;40m",'initialization', "\033[0m")      
                    self.create_voxels_pointcloud(tracked_frame)
                    # self.create_pointcloud(tracked_frame)
                    self.insert_keyframe(tracked_frame)
                    # print("\033[0;33;40m",'kf_buffer.empty()',kf_buffer.empty(), "\033[0m")      
                    while kf_buffer.empty():
                        # pass
                        # print("\033[0;33;40m",'initialization2', "\033[0m")      
                        self.do_mapping(share_data)
                        # print('==================')
                        # print("\033[0;33;40m",'initialization3', "\033[0m")      
                        # self.update_share_data(share_data, tracked_frame.stamp)
                    self.initialized = True
                    print("\033[0;33;40m",'第一帧初始化成功', "\033[0m")      
                    
                else:
                    # print("\033[0;33;40m",'tracked_frame',tracked_frame ,"\033[0m")
                    
                    # print("\033[0;33;40m",'initialization后', "\033[0m")      
                    # print("\033[0;33;40m",'-------3333--------', "\033[0m")      
                    self.do_mapping(share_data, tracked_frame)
                    # print("\033[0;33;40m",'-------1111--------', "\033[0m")      
                    
                    self.create_voxels_pointcloud(tracked_frame)
                    # print("\033[0;33;40m",'-------2222--------', "\033[0m")      
                    
                    # self.create_pointcloud(tracked_frame)
                    # if (tracked_frame.stamp - self.current_keyframe.stamp) > 50:
                    if (tracked_frame.stamp - self.current_keyframe.stamp) > 50:
                        self.insert_keyframe(tracked_frame)
                        print(
                            f"********** current num kfs: { len(self.keyframe_graph) } **********")

                # self.create_voxels(tracked_frame)
                tracked_pose = tracked_frame.get_pose().detach()
                ref_pose = self.current_keyframe.get_pose().detach()
                rel_pose = torch.linalg.inv(ref_pose) @ tracked_pose
                self.frame_poses += [(len(self.keyframe_graph) -
                                      1, rel_pose.cpu())]
                self.depth_maps += [tracked_frame.depth.clone().cpu()]

                if self.mesh_freq > 0 and (tracked_frame.stamp + 1) % self.mesh_freq == 0:
                    self.logger.log_mesh(self.extract_mesh(
                        res=self.mesh_res, clean_mesh=True), name=f"mesh_{tracked_frame.stamp:05d}.ply")
                if self.save_data_freq > 0 and (tracked_frame.stamp + 1) % self.save_data_freq == 0:
                    print("\033[0;33;40m",'mapping print debug img', "\033[0m")
                    self.save_debug_data(tracked_frame)
            elif share_data.stop_mapping:
                break

        print(f"********** post-processing {self.final_iter} steps **********")
        self.num_iterations = 1
        for iter in range(self.final_iter):
            self.do_mapping(share_data, tracked_frame=None,
                            update_pose=False, update_decoder=False,update_resnet=False)

        print("******* extracting final mesh *******")
        pose = self.get_updated_poses()
        mesh = self.extract_mesh(res=self.mesh_res, clean_mesh=False)
        self.logger.log_ckpt(self)
        self.logger.log_numpy_data(np.asarray(pose), "frame_poses")
        self.logger.log_mesh(mesh)
        self.logger.log_numpy_data(self.extract_voxels(), "final_voxels")
        print("******* mapping process died *******")

    def do_mapping(
            self,
            share_data,
            tracked_frame=None,
            update_pose=True,
            update_decoder=True,
            update_resnet=True
    ):
        # self.map.create_voxels(self.keyframe_graph[0])
        self.decoder.train()
        self.points_encoder.train()
        
        # 选择要ba优化的关键帧序列
        # print("\033[0;33;40m",'do_mapping', "\033[0m")
        # print("\033[0;33;40m",'mapping voxel1',self.map_states["voxel_center_xyz"].shape, "\033[0m")
        
        
        optimize_targets = self.select_optimize_targets(tracked_frame)
        # print("\033[0;33;40m",'tracked_frame',tracked_frame, "\033[0m")
        
        # optimize_targets = [f.cuda() for f in optimize_targets]
        # print("\033[0;33;40m",'=====123===', "\033[0m")
        bundle_adjust_frames(
            optimize_targets,
            self.map_states,
            self.decoder,
            self.points_encoder,
            self.loss_criteria,
            self.voxel_size,
            self.step_size,
            self.n_rays,
            self.num_iterations,
            self.sdf_truncation,
            self.max_voxel_hit,
            self.max_distance,
            learning_rate=[1e-2, 1e-3],
            embed_optim=self.embed_optim,
            model_optim=self.model_optim if update_decoder else None,
            resnet_optim = self.resnet_optim if update_resnet else None,
            update_pose=update_pose,
        )
        # print("\033[0;33;40m",'=====456===', "\033[0m")

        # optimize_targets = [f.cpu() for f in optimize_targets]
        self.update_share_data(share_data)
        # sleep(0.01)

    def select_optimize_targets(self, tracked_frame=None):
        # TODO: better ways
        targets = []
        selection_method = 'random'
        if len(self.keyframe_graph) <= self.window_size:
            targets = self.keyframe_graph[:]
        elif selection_method == 'random':
            targets = random.sample(self.keyframe_graph, self.window_size)
        elif selection_method == 'overlap':
            raise NotImplementedError(
                f"seletion method {selection_method} unknown")

        if tracked_frame is not None and tracked_frame != self.current_keyframe:
            targets += [tracked_frame]
        return targets

    def update_share_data(self, share_data, frameid=None):
        # print("\033[0;33;40m",'update_share_data', "\033[0m")
        share_data.decoder = deepcopy(self.decoder).cpu()
        share_data.points_encoder = deepcopy(self.points_encoder).cpu()
        # share_data.hash_voxel = deepcopy(self.svo)
        # print("\033[0;33;40m",'centres',centres.shape, "\033[0m")
        
        tmp_states = {}
        # print("\033[0;33;40m",'mapping update voxel',self.map_states["voxel_center_xyz"].shape, "\033[0m")
        for k, v in self.map_states.items():
            tmp_states[k] = v.detach().cpu()
        share_data.states = tmp_states
        # self.last_tracked_frame_id = frameid

    def insert_keyframe(self, frame):
        # kf check
        print("insert keyframe")
        self.current_keyframe = frame
        self.keyframe_graph += [frame]
        # self.update_grid_features()
    
    
    def create_voxels_pointcloud(self, frame):
        
        points = frame.get_points().cuda()
        colors = frame.get_color().cuda()
        pose = frame.get_pose().cuda()
        points = points@pose[:3, :3].transpose(-1, -2) + pose[:3, 3]
        voxels = torch.div(points, self.voxel_size, rounding_mode='floor')
        colors = colors * 255
        
        # test the frame
        # color_gt = frame.get_color_gt().cpu().numpy()
        # color_gt = color_gt * 255
        # cv2.imwrite(f'vox_{self.flag}_color.jpg',color_gt)
        
        # np.savetxt('colors.txt', colors.cpu().numpy())
        # print("\033[0;33;40m",'color_gt 输出完成',color_gt.shape, "\033[0m")
        
        # print("\033[0;33;40m",'self.voxel_size',self.voxel_size, "\033[0m")
        # print("\033[0;33;40m",'voxels',voxels.shape, "\033[0m")
        
        # test the input information
        # np.savetxt(f'pcd_{self.flag}_pose.txt', pose.detach().cpu().numpy())
        # np.savetxt(f'vox_{self.flag}_colors.txt', colors.detach().cpu().numpy())
        # print("\033[0;33;40m",'points',points.shape, "\033[0m")
        np.savetxt(f'vox_{self.flag}_points.txt', points.detach().cpu().numpy())
        np.savetxt(f'vox_{self.flag}_colors.txt', colors.detach().cpu().numpy())
        self.flag = self.flag + 1
        # np.savetxt(f'pcd_{self.flag}_voxels.txt', voxels.detach().cpu().numpy())
        # print("\033[0;33;40m",'======print over=====', "\033[0m")
        
        # voxels = voxels[:1, :3]
        # print("\033[0;33;40m",'voxels',voxels.shape, "\033[0m")
        # print("\033[0;33;40m",'----------------', "\033[0m")
        # self.svo.insert(voxels.cpu().int(),voxels.cpu().int(),voxels.cpu().float())
        self.svo.insert(voxels.cpu().int(),colors.cpu().int(),points.cpu().float())
        # print("\033[0;33;40m",'======222======', "\033[0m")

        self.update_grid_pcd_features()

# // voxels是一个N*4的张量，其中N是八叉树中非FEATURE类型的节点的数量。每一行代表一个节点对应的体素，包含了x,y,z坐标和边长。
# // children是一个N*8的张量，其中N和上面相同。每一行代表一个节点和其八个子节点之间的索引关系。如果某个子节点不存在或者是FEATURE类型，则对应位置为-1。
# // features是一个N*8的张量，其中N和上面相同。每一行代表一个节点对应体素的八个顶点是否有特征。如果某个顶点有特征，则对应位置为该特征节点在八叉树中的索引；否则为-1。
    @torch.enable_grad()
    def update_grid_pcd_features(self):
        voxels, children, features, pcd_xyz, pcd_color = self.svo.get_centres_and_children()
        # a = self.svo
        # a = deepcopy(self.svo)
        # print("\033[0;33;40m",'=====mapping print=======', "\033[0m")
        # self.flag = self.flag - 1
        # np.savetxt(f'voxels_{self.flag}.txt', (voxels* self.voxel_size).cpu().numpy())
        # np.savetxt(f'pcd_{self.flag}_xyz0.txt', pcd_xyz[:,0,:].cpu().numpy())
        # np.savetxt(f'pcd_{self.flag}_xyz1.txt', pcd_xyz[:,1,:].cpu().numpy())
        # np.savetxt(f'pcd_{self.flag}_xyz2.txt', pcd_xyz[:,2,:].cpu().numpy())
        # np.savetxt(f'pcd_{self.flag}_xyz3.txt', pcd_xyz[:,3,:].cpu().numpy())
        # np.savetxt(f'pcd_{self.flag}_xyz4.txt', pcd_xyz[:,4,:].cpu().numpy())
        # np.savetxt(f'pcd_{self.flag}_xyz5.txt', pcd_xyz[:,5,:].cpu().numpy())
        # np.savetxt(f'pcd_{self.flag}_xyz6.txt', pcd_xyz[:,6,:].cpu().numpy())
        # np.savetxt(f'pcd_{self.flag}_xyz7.txt', pcd_xyz[:,7,:].cpu().numpy())
        # np.savetxt(f'pcd_color0.txt', pcd_color[:,0,:].cpu().numpy())
        # np.savetxt(f'pcd_color1.txt', pcd_color[:,1,:].cpu().numpy())
        # print("\033[0;33;40m",'---------------', "\033[0m")
        
        # print("\033[0;33;40m",'voxels',voxels.shape, "\033[0m")
        # print("\033[0;33;40m",'children',children.shape, "\033[0m")
        # print("\033[0;33;40m",'features',features.shape, "\033[0m")
        # print("\033[0;33;40m",'pointclous',pcd_xyz.shape, "\033[0m")
        # print("\033[0;33;40m",'pcd_color',pcd_color.shape, "\033[0m")
        # pcd_xyz = pcd_xyz[:,:, :3]  * self.voxel_size
        # pcd_xyz = (pcd_xyz[:,:, :3] + pcd_xyz[:,:, -1:] / 2) * self.voxel_size
        # 将节点坐标从体素顶点移到体素中心
        centres = (voxels[:, :3] + voxels[:, -1:] / 2) * self.voxel_size
        # aa = (pcd_xyz[:,:, -1:] / 2) * self.voxel_size
        # bb = (voxels[:, :3] + voxels[:, -1:] / 2)
        
        # print("\033[0;33;40m",'aa',aa.shape, "\033[0m")
        # print("\033[0;33;40m",'bb',bb.shape, "\033[0m")
        # print("\033[0;33;40m",'pcd_xyz[:,:, :3]',pcd_xyz[:,:, :3].shape, "\033[0m")
        # print("\033[0;33;40m",'voxels[:, :3]',voxels[:, :3].shape, "\033[0m")
        
        
        # pcd_centres =  aa + pcd_xyz[:,:, :3] 

        children = torch.cat([children, voxels[:, -1:]], -1)
        # pcd_color = pcd_color.cuda()
        # print("\033[0;33;40m",'resnet',self.points_encoder.fc.weight.grad, "\033[0m")
        
        # pcd_features = self.points_encoder(pcd_color).requires_grad_(True).cuda()
        # print("\033[0;33;40m",'features',pcd_features.shape, "\033[0m")
        centres = centres.cuda().float()
        children = children.cuda().int()
        # pcd_xyz = pcd_xyz.cuda().float()
        # pcd_color = pcd_color.cuda().float()
        
        # print("\033[0;33;40m",'centres',centres.shape, "\033[0m")
        # print("\033[0;33;40m",'children',children.shape, "\033[0m")
        # print("\033[0;33;40m",'features',features.shape, "\033[0m")
        
        # print("\033[0;33;40m",'===mapping====', "\033[0m")
        # # np.savetxt('voxels.txt', voxels.cpu().numpy())
        # np.savetxt(f'centres_{self.flag}.txt', centres.cpu().numpy())
        # np.savetxt(f'pcd_centres_{self.flag}_0.txt', pcd_centres[:,0,:].cpu().numpy())
        # np.savetxt(f'pcd_centres_{self.flag}_1.txt', pcd_centres[:,1,:].cpu().numpy())
        # np.savetxt(f'pcd_centres_{self.flag}_6.txt', pcd_centres[:,6,:].cpu().numpy())
        # np.savetxt(f'pcd_centres_{self.flag}_7.txt', pcd_centres[:,7,:].cpu().numpy())
        
        # self.flag = self.flag + 1
        # np.savetxt('centres.txt', centres.cpu().numpy())
        # np.savetxt('children.txt', children.cpu().numpy())
        # np.savetxt('features.txt', features.cpu().numpy())
        # np.savetxt('pcd_color0.txt', pcd_color[:,0,:].cpu().numpy())
        # np.savetxt('pcd_color1.txt', pcd_color[:,1,:].cpu().numpy())
        # print("\033[0;33;40m",'----mapping print over------', "\033[0m")
        
        map_states = {}
        map_states["voxel_vertex_idx"] = features.cuda()
        map_states["voxel_center_xyz"] = centres
        map_states["voxel_structure"] = children
        map_states["voxel_vertex_emb"] = self.embeddings
        map_states["pointclouds_xyz"] = pcd_xyz
        map_states["pointclouds_color"] = pcd_color
        
        # print("\033[0;33;40m",'pcd_xyz',pcd_xyz.shape, "\033[0m")
        
        # map_states["pointclouds_feature"] = pcd_features
        # print("\033[0;33;40m",'===============', "\033[0m")
        # np.savetxt('pcd_xyz0.txt', pcd_xyz[:,0,:].cpu().numpy())
        # np.savetxt('pcd_xyz1.txt', pcd_xyz[:,1,:].cpu().numpy())
        # np.savetxt('pcd_xyz2.txt', pcd_xyz[:,2,:].cpu().numpy())
        # np.savetxt('pcd_xyz3.txt', pcd_xyz[:,3,:].cpu().numpy())
        # np.savetxt('pcd_xyz4.txt', pcd_xyz[:,4,:].cpu().numpy())
        # np.savetxt('pcd_xyz5.txt', pcd_xyz[:,5,:].cpu().numpy())
        # np.savetxt('pcd_xyz6.txt', pcd_xyz[:,6,:].cpu().numpy())
        # np.savetxt('pcd_xyz7.txt', pcd_xyz[:,7,:].cpu().numpy())
        
        # np.savetxt('pcd_xyz3.txt', pcd_xyz[:,3,:].cpu().numpy())
        # np.savetxt('pcd_xyz4.txt', pcd_xyz[:,4,:].cpu().numpy())
        # np.savetxt('pcd_xyz5.txt', pcd_xyz[:,5,:].cpu().numpy())
        # np.savetxt('pcd_xyz6.txt', pcd_xyz[:,6,:].cpu().numpy())
        # np.savetxt('pcd_xyz7.txt', pcd_xyz[:,7,:].cpu().numpy())
        # np.savetxt('centres.txt', centres.cpu().numpy())
        
        # np.savetxt('pcd_features.txt', pcd_features[:,2,:].detach().cpu().numpy())
        # np.savetxt('centres.txt', centres.cpu().numpy())
        # print("\033[0;33;40m",'----------------', "\033[0m")
        # print("\033[0;33;40m",'pcd_xyz',pcd_xyz.shape, "\033[0m")
        # print("\033[0;33;40m",'centres',centres.shape, "\033[0m")
        # print("\033[0;33;40m",'pcd_color',pcd_color.shape, "\033[0m")

        self.map_states = map_states



    @torch.no_grad()
    def get_updated_poses(self):
        frame_poses = []
        for i in range(len(self.frame_poses)):
            ref_frame_ind, rel_pose = self.frame_poses[i]
            ref_frame = self.keyframe_graph[ref_frame_ind]
            ref_pose = ref_frame.get_pose().detach().cpu()
            pose = ref_pose @ rel_pose
            frame_poses += [pose.detach().cpu().numpy()]
        return frame_poses

    @torch.no_grad()
    def extract_mesh(self, res=8, clean_mesh=False):
        sdf_network = self.decoder
        sdf_network.eval()

        voxels, _, features,_,_ = self.svo.get_centres_and_children()
        index = features.eq(-1).any(-1)
        voxels = voxels[~index, :]
        features = features[~index, :]
        centres = (voxels[:, :3] + voxels[:, -1:] / 2) * self.voxel_size

        encoder_states = {}
        encoder_states["voxel_vertex_idx"] = features.cuda()
        encoder_states["voxel_center_xyz"] = centres.cuda()
        encoder_states["voxel_vertex_emb"] = self.embeddings

        frame_poses = self.get_updated_poses()
        mesh = self.mesher.create_mesh(
            self.decoder, encoder_states, self.voxel_size, voxels,
            frame_poses=frame_poses[-1], depth_maps=self.depth_maps[-1],
            clean_mseh=clean_mesh, require_color=True, offset=-10, res=res)
        return mesh

    @torch.no_grad()
    def extract_voxels(self, offset=-10):
        voxels, _, features,_,_ = self.svo.get_centres_and_children()
        index = features.eq(-1).any(-1)
        voxels = voxels[~index, :]
        features = features[~index, :]
        voxels = (voxels[:, :3] + voxels[:, -1:] / 2) * \
            self.voxel_size + offset
        print(torch.max(features)-torch.count_nonzero(index))
        return voxels

    @torch.no_grad()
    def save_debug_data(self, tracked_frame, offset=-10):
        """
        save per-frame voxel, mesh and pose 
        """
        pose = tracked_frame.get_pose().detach().cpu().numpy()
        pose[:3, 3] += offset
        frame_poses = self.get_updated_poses()
        mesh = self.extract_mesh(res=8, clean_mesh=True)
        voxels = self.extract_voxels().detach().cpu().numpy()
        keyframe_poses = [p.get_pose().detach().cpu().numpy()
                          for p in self.keyframe_graph]

        for f in frame_poses:
            f[:3, 3] += offset
        for kf in keyframe_poses:
            kf[:3, 3] += offset

        verts = np.asarray(mesh.vertices)
        faces = np.asarray(mesh.triangles)
        color = np.asarray(mesh.vertex_colors)

        self.logger.log_debug_data({
            "pose": pose,
            "updated_poses": frame_poses,
            "mesh": {"verts": verts, "faces": faces, "color": color},
            "voxels": voxels,
            "voxel_size": self.voxel_size,
            "keyframes": keyframe_poses,
            "is_keyframe": (tracked_frame == self.current_keyframe)
        }, tracked_frame.stamp)
