import os
import torch
from torch.utils.data import DataLoader
from data.unpaired_dataset import UnpairedImageDataset
from models.generators import ResnetGenerator
from options.test_options import TestOptions
from PIL import Image

if __name__ == '__main__':
    opt = TestOptions().parse()
    # 加载数据
    dataset = UnpairedImageDataset(opt.dataroot, phase='test',
                                   load_size=opt.load_size, crop_size=opt.crop_size,
                                   serial_batches=True)
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    # 加载网络
    netG = ResnetGenerator(3, 3, opt.ngf if hasattr(opt,'ngf') else 64, opt.n_blocks if hasattr(opt,'n_blocks') else 9).cuda()
    checkpoint = torch.load(f"{opt.checkpoints_dir}/{opt.name}/latest_net_G.pth")
    netG.load_state_dict(checkpoint)
    netG.eval()

    # 创建结果目录
    save_dir = os.path.join(opt.results_dir, opt.name)
    os.makedirs(save_dir, exist_ok=True)

    # 推理
    for i, data in enumerate(loader):
        if i >= opt.how_many:
            break
        real_A = data['A'].cuda()
        fake_B = netG(real_A)
        # 反归一化并保存
        img = (fake_B[0].cpu().detach().numpy() + 1) / 2 * 255
        img = img.transpose((1,2,0)).astype('uint8')
        save_path = os.path.join(save_dir, f'{i:04d}.png')
        Image.fromarray(img).save(save_path)
        print(f'Saved {save_path}')
