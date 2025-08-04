# 文件：test.py

import os
import torch
from torch.utils.data import DataLoader
from data.unpaired_dataset import UnpairedImageDataset
from models.generators import ResnetGenerator
from options.test_options import TestOptions
from PIL import Image

if __name__ == '__main__':
    opt = TestOptions().parse()

    dataset = UnpairedImageDataset(
        opt.dataroot,
        phase=opt.phase,
        load_size=opt.load_size,
        crop_size=opt.crop_size,
        serial_batches=True
    )
    loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=4)

    ngf = opt.ngf if hasattr(opt, 'ngf') else 64
    n_blocks = opt.n_blocks if hasattr(opt, 'n_blocks') else 9

    if opt.direction == 'AtoB':
        input_nc, output_nc = 3, 3
    elif opt.direction == 'BtoA':
        input_nc, output_nc = 3, 3
    else:
        raise ValueError(f"未知的 direction: {opt.direction}，只能是 'AtoB' 或 'BtoA'")

    netG = ResnetGenerator(input_nc, output_nc, ngf, n_blocks).to(opt.device)

    if opt.epoch is None:
        ckpt_name = 'latest_net_G.pth'
    else:
        ckpt_name = f'G_{opt.direction}_ep{opt.epoch}.pth'
    ckpt_path = os.path.join(opt.checkpoints_dir, opt.name, ckpt_name)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"找不到模型文件：{ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=opt.device)
    netG.load_state_dict(checkpoint)
    netG.eval()

    save_dir = os.path.join(opt.results_dir, opt.name, opt.phase, opt.direction)
    os.makedirs(save_dir, exist_ok=True)

    for i, data in enumerate(loader):
        if i >= opt.how_many:
            break
        real = data['A'] if opt.direction == 'AtoB' else data['B']
        real = real.to(opt.device)
        fake = netG(real)
        img = (fake[0].cpu().detach().numpy() + 1) / 2 * 255
        img = img.transpose((1,2,0)).astype('uint8')
        save_path = os.path.join(save_dir, f'{i:04d}.png')
        Image.fromarray(img).save(save_path)
        print(f'Saved {save_path}')
