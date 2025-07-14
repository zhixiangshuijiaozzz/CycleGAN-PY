from .base_options import BaseOptions

class TestOptions(BaseOptions):
    def initialize(self, parser):
        parser = super().initialize(parser)
        parser.add_argument('--load_size', type=int, default=256, help='加载大小')
        parser.add_argument('--crop_size', type=int, default=256, help='裁剪大小')
        parser.add_argument('--results_dir', type=str, default='./results', help='测试结果保存目录')
        parser.add_argument('--how_many', type=int, default=float("inf"), help='测试图像数量')
        return parser
