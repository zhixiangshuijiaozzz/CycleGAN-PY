from .base_options import BaseOptions

class TrainOptions(BaseOptions):
    def initialize(self, parser):
        parser = super().initialize(parser)
        # 数据增强
        parser.add_argument('--load_size', type=int, default=286, help='加载后 resize 大小')
        parser.add_argument('--crop_size', type=int, default=256, help='随机裁剪大小')
        parser.add_argument('--serial_batches', action='store_true', help='是否按序读取 B 域')
        # 训练超参
        parser.add_argument('--batch_size', type=int, default=1, help='batch size')
        parser.add_argument('--ngf', type=int, default=64, help='生成器通道基数')
        parser.add_argument('--ndf', type=int, default=64, help='判别器通道基数')
        parser.add_argument('--n_blocks', type=int, default=9, help='ResNet 块数')
        parser.add_argument('--lr', type=float, default=0.0002, help='学习率')
        parser.add_argument('--beta1', type=float, default=0.5, help='Adam 优化 beta1')
        parser.add_argument('--lambda_A', type=float, default=10.0, help='循环一致性损失权重 A->B->A')
        parser.add_argument('--lambda_B', type=float, default=10.0, help='循环一致性损失权重 B->A->B')
        parser.add_argument('--lambda_identity', type=float, default=0.5, help='身份映射损失权重')
        parser.add_argument('--n_epochs', type=int, default=100, help='仅 warm-up 阶段 epochs')
        parser.add_argument('--n_epochs_decay', type=int, default=100, help='线性衰减阶段 epochs')
        return parser