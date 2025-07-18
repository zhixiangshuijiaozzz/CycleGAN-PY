import torch
import torch.nn as nn
from .weights_init import init_weights


class ResnetBlock(nn.Module):
    """ResNet 块"""

    def __init__(self, dim):
        super().__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0),
            nn.InstanceNorm2d(dim),
            nn.ReLU(True),

            nn.ReflectionPad2d(1),
            nn.Conv2d(dim, dim, kernel_size=3, padding=0),
            nn.InstanceNorm2d(dim)
        )

    def forward(self, x):
        return x + self.block(x)


class ResnetGenerator(nn.Module):
    """基于 ResNet 的生成器：c7s1-64，d128，d256，9 x ResnetBlock，u128，u64，c7s1-3"""

    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=9):
        super().__init__()
        model = []
        # c7s1-64
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0),
            nn.InstanceNorm2d(ngf),
            nn.ReLU(True)
        ]
        # 下采样
        curr_dim = ngf
        for _ in range(2):
            model += [
                nn.Conv2d(curr_dim, curr_dim * 2, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(curr_dim * 2),
                nn.ReLU(True)
            ]
            curr_dim *= 2
        # ResNet 块
        for _ in range(n_blocks):
            model += [ResnetBlock(curr_dim)]
        # 上采样
        for _ in range(2):
            model += [
                nn.ConvTranspose2d(curr_dim, curr_dim // 2, kernel_size=3, stride=2,
                                   padding=1, output_padding=1),
                nn.InstanceNorm2d(curr_dim // 2),
                nn.ReLU(True)
            ]
            curr_dim //= 2
        # c7s1-输出
        model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(curr_dim, output_nc, kernel_size=7, padding=0),
            nn.Tanh()
        ]
        self.model = nn.Sequential(*model)
        init_weights(self)

    def forward(self, x):
        return self.model(x)
