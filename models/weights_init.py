import torch.nn as nn

def init_weights(net, init_type='normal', init_gain=0.02):
    """
    初始化网络权重：
      - conv 层使用正态分布 N(0, init_gain)
      - BatchNorm 层权重初始化为 N(1, init_gain)，偏置为 0
    """
    def _init(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1):
            nn.init.normal_(m.weight.data, 0.0, init_gain)
        elif classname.find('BatchNorm2d') != -1:
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)
    net.apply(_init)
