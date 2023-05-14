import os  # noqa
import sys  # noqa
sys.path.insert(0, os.path.abspath('src')) # noqa
import random
from parser import get_parser
import numpy as np
import torch
from voxslam import VoxSLAM

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

if __name__ == '__main__':
    args = get_parser().parse_args()
    # print("\033[0;33;40m",'args',args, "\033[0m")
    if hasattr(args, 'seeding'):
        setup_seed(args.seeding)

    slam = VoxSLAM(args)
    slam.start()
    print("\033[0;33;40m",'slam.start()运行结束', "\033[0m")
    slam.wait_child_processes()