from multiprocessing.managers import BaseManager
from time import sleep

import torch
import torch.multiprocessing as mp

from loggers import BasicLogger
from mapping import Mapping
from share import ShareData, ShareDataProxy
from tracking import Tracking
from utils.import_util import get_dataset
from visualization import Visualizer


class VoxSLAM:
    def __init__(self, args):
        self.args = args
        # print("\033[0;33;40m",'args',self.args, "\033[0m")
        # logger (optional)
        self.logger = BasicLogger(args)
        # visualizer (optional)
        self.visualizer = Visualizer(args, self)

        # shared data 
        # 设置进程启动方式，
        mp.set_start_method('spawn', force=True)
        # 将需要共享的类对象注册在类中
        BaseManager.register('ShareData', ShareData, ShareDataProxy)
        # 实例化一个共享对象
        manager = BaseManager()
        # 运行这个类
        manager.start()
        self.share_data = manager.ShareData()
        # keyframe buffer  共享的buffer序列
        self.kf_buffer = mp.Queue(maxsize=1)
        print("\033[0;33;40m",'buffer数',self.kf_buffer, "\033[0m")
        # data stream
        self.data_stream = get_dataset(args)
        # tracker 
        print("\033[0;33;40m",'数据流',self.data_stream, "\033[0m")
        # 加载参数，数据集
        self.tracker = Tracking(args, self.data_stream, self.logger, self.visualizer)
        # mapper
        self.mapper = Mapping(args, self.logger, self.visualizer)
        # initialize map with first frame
        self.tracker.process_first_frame(self.kf_buffer)
        self.processes = []

    def start(self):
        #建图进程
        mapping_process = mp.Process(
            target=self.mapper.spin, args=(self.share_data, self.kf_buffer))
        mapping_process.start()
        print("initializing the first frame ...")
        sleep(5)
        #追踪进程
        tracking_process = mp.Process(
            target=self.tracker.spin, args=(self.share_data, self.kf_buffer))
        tracking_process.start()
        #可视化进程
        vis_process = mp.Process(
            target=self.visualizer.spin, args=(self.share_data,))
        #存储建图和追踪进程
        self.processes = [tracking_process, mapping_process]

        if self.args.enable_vis:
            vis_process.start()
            self.processes += [vis_process]

    def wait_child_processes(self):
        for p in self.processes:
            p.join()

    @torch.no_grad()
    def get_raw_trajectory(self):
        return self.share_data.tracking_trajectory

    @torch.no_grad()
    def get_keyframe_poses(self):
        keyframe_graph = self.mapper.keyframe_graph
        poses = []
        for keyframe in keyframe_graph:
            poses.append(keyframe.get_pose().detach().cpu().numpy())
        return poses
