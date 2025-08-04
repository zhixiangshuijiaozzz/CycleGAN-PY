# 文件：options/test_options.py

from .base_options import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = super().initialize(parser)
        parser.add_argument('--phase', type=str, default='test',
                            help="数据集阶段（train、test、val），默认 'test'")
        parser.add_argument('--epoch', type=int, default=None,
                            help="要加载的模型 epoch 编号，默认 None（使用 latest_net_G.pth）")
        parser.add_argument('--direction', type=str, default='AtoB',
                            help="转换方向，格式 'AtoB' 或 'BtoA'，默认 'AtoB'")
        # 原有参数
        parser.add_argument('--results_dir', type=str, default='./results',
                            help='测试结果保存主目录')
        parser.add_argument('--how_many', type=int, default=float("inf"),
                            help='测试图像数量')
        return parser
