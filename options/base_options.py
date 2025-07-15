import argparse
import torch

class BaseOptions:
    def __init__(self):
        self.initialized = False

    # ---------------------------------------------------------- #
    # 1. 定义所有可选参数
    # ---------------------------------------------------------- #
    def initialize(self, parser: argparse.ArgumentParser):
        # ---- 路径与实验 ----
        parser.add_argument('--dataroot', type=str, required=True,
                            help='数据集根目录')
        parser.add_argument('--name', type=str, required=True,
                            help='实验名，将生成 checkpoints/<name>/time/ 目录')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints',
                            help='模型&日志保存主目录')
        parser.add_argument('--model', type=str, default='cycle_gan', help='模型类型')
        parser.add_argument('--verbose', action='store_true', help='打印更多信息')

        # ---- 数据 & 训练超参（允许命令行覆盖，若不传则用默认）----
        parser.add_argument('--load_size', type=int, default=256,
                            help='Resize 的长边')
        parser.add_argument('--crop_size', type=int, default=256,
                            help='随机裁剪尺寸')
        parser.add_argument('--batch_size', type=int, default=10, help='批量大小')
        parser.add_argument('--n_epochs', type=int, default=50,
                            help='学习率恒定阶段 epoch 数')
        parser.add_argument('--n_epochs_decay', type=int, default=50,
                            help='学习率线性衰减阶段 epoch 数')

        self.initialized = True
        return parser

    # ---------------------------------------------------------- #
    def gather_options(self):
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
        return parser.parse_args()

    # ---------------------------------------------------------- #
    def parse(self):
        self.opt = self.gather_options()

        # ---- 自动检测设备 ----
        if torch.backends.mps.is_available():      # macOS Metal
            self.opt.device = torch.device('mps')
        elif torch.cuda.is_available():            # NVIDIA CUDA
            self.opt.device = torch.device('cuda:0')
        else:
            self.opt.device = torch.device('cpu')

        print(f"=> 使用设备: {self.opt.device}")    # 一定打印
        return self.opt
