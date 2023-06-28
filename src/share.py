from multiprocessing.managers import BaseManager, NamespaceProxy
from copy import deepcopy,copy
import torch.multiprocessing as mp
from time import sleep
import sys
import open3d as o3d

class ShareDataProxy(NamespaceProxy):
    _exposed_ = ('__getattribute__', '__setattr__')


class SharedPointCloud:
    def __init__(self):
        self.lock = mp.Lock()
        # self.pointcloud = o3d.io.read_point_cloud(filepath)

    def get_pointcloud(self):
        with self.lock:
            return self.pointcloud

    def set_pointcloud(self, pointcloud):
        with self.lock:
            self.pointcloud = pointcloud



class ShareData:
    global lock
    lock = mp.RLock()

    def __init__(self):
        self.__stop_mapping = False
        self.__stop_tracking = False

        self.__decoder = None
        self.__points_encoder = None
        self.__voxels = None
        self.__octree = None
        self.__states = None
        self.__hash_voxel = None
        self.__tracking_trajectory = []

    @property
    def decoder(self):
        with lock:
            return deepcopy(self.__decoder)
            print("========== decoder get ==========")
            sys.stdout.flush()
    
    @decoder.setter
    def decoder(self, decoder):
        with lock:
            self.__decoder = deepcopy(decoder)
            # print("========== decoder set ==========")
            sys.stdout.flush()
    
    @property
    def points_encoder(self):
        with lock:
            return deepcopy(self.__points_encoder)
            print("========== decoder get ==========")
            sys.stdout.flush()
    
    @points_encoder.setter
    def points_encoder(self, points_encoder):
        with lock:
            self.__points_encoder = deepcopy(points_encoder)
            # print("========== decoder set ==========")
            sys.stdout.flush()


    @property
    def voxels(self):
        with lock:
            return deepcopy(self.__voxels)
            print("========== voxels get ==========")
            sys.stdout.flush()
    
    @voxels.setter
    def voxels(self, voxels):
        with lock:
            self.__voxels = deepcopy(voxels)
            print("========== voxels set ==========")
            sys.stdout.flush()

    @property
    def octree(self):
        with lock:
            return deepcopy(self.__octree)
            print("========== octree get ==========")
            sys.stdout.flush()
    
    @octree.setter
    def octree(self, octree):
        with lock:
            self.__octree = deepcopy(octree)
            print("========== octree set ==========")
            sys.stdout.flush()
    @property
    def hash_voxel(self):
        with lock:
            return deepcopy(self.__hash_voxel)
            print("========== hash_voxel get ==========")
            sys.stdout.flush()
    
    @hash_voxel.setter
    def hash_voxel(self, hash_voxel):
        with lock:
            self.__hash_voxel = deepcopy(hash_voxel)
            print("========== hash_voxel set ==========")
            sys.stdout.flush()
    @property
    def states(self):
        with lock:
            return deepcopy(self.__states)
            print("========== states get ==========")
            sys.stdout.flush()
    
    @states.setter
    def states(self, states):
        with lock:
            self.__states = deepcopy(states)
            # print("========== states set ==========")
            sys.stdout.flush()

    @property
    def stop_mapping(self):
        with lock:
            return self.__stop_mapping
            print("========== stop_mapping get ==========")
            sys.stdout.flush()
    
    @stop_mapping.setter
    def stop_mapping(self, stop_mapping):
        with lock:
           self.__stop_mapping = stop_mapping
           print("========== stop_mapping set ==========")
           sys.stdout.flush()

    @property
    def stop_tracking(self):
        with lock:
            return self.__stop_tracking
            print("========== stop_tracking get ==========")
            sys.stdout.flush()
    
    @stop_tracking.setter
    def stop_tracking(self, stop_tracking):
        with lock:
           self.__stop_tracking = stop_tracking
           print("========== stop_tracking set ==========")
           sys.stdout.flush()

    @property
    def tracking_trajectory(self):
        with lock:
            return deepcopy(self.__tracking_trajectory)
            print("========== tracking_trajectory get ==========")
            sys.stdout.flush()
    
    def push_pose(self, pose):
        with lock:
            self.__tracking_trajectory.append(deepcopy(pose))
            # print("========== push_pose ==========")
            sys.stdout.flush()
            
    # def get_pointcloud(self):
    #     with self.lock:
    #         return self.pointcloud

    # def set_pointcloud(self, pointcloud):
    #     with self.lock:
    #         self.pointcloud = pointcloud

    