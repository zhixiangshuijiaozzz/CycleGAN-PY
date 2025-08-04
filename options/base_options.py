import argparse
import torch

class BaseOptions:
    def __init__(self):
        self.initialized = False

    def initialize(self, parser: argparse.ArgumentParser):
        parser.add_argument('--dataroot', type=str, required=True,
                            help='数据集根目录')
        parser.add_argument('--name', type=str, required=True,
                            help='实验名，将生成 checkpoints/<name>/time/ 目录')
        parser.add_argument('--checkpoints_dir', type=str, default='./checkpoints',
                            help='模型&日志保存主目录')
        parser.add_argument('--model', type=str, default='cycle_gan', help='模型类型')
        parser.add_argument('--verbose', action='store_true', help='打印更多信息')
        parser.add_argument('--load_size', type=int, default=256,
                            help='Resize 的长边')
        parser.add_argument('--crop_size', type=int, default=256,
                            help='随机裁剪尺寸')
        parser.add_argument('--batch_size', type=int, default=5, help='批量大小')
        parser.add_argument('--n_epochs', type=int, default=50,
                            help='学习率恒定阶段 epoch 数')
        parser.add_argument('--n_epochs_decay', type=int, default=50,
                            help='学习率线性衰减阶段 epoch 数')

        self.initialized = True
        return parser

    def gather_options(self):
        if not self.initialized:
            parser = argparse.ArgumentParser(
                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
            parser = self.initialize(parser)
        return parser.parse_args()

    def parse(self):
        self.opt = self.gather_options()

        if torch.backends.mps.is_available():
            self.opt.device = torch.device('mps')
        elif torch.cuda.is_available():
            self.opt.device = torch.device('cuda:0')
        else:
            self.opt.device = torch.device('cpu')

        print(f"=> 使用设备: {self.opt.device}")
        return self.opt
