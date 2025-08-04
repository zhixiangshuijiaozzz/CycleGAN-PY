# 文件：train.py

import os
import time
import warnings
from datetime import datetime
import itertools

import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# -------- 感知损失相关依赖 --------
from torchvision.models import vgg16
import torchvision.transforms as T

from data.unpaired_dataset import UnpairedImageDataset
from models.generators import ResnetGenerator
from models.discriminators import NLayerDiscriminator
from utils.image_pool import ImagePool
from utils.logger import Logger
from options.base_options import BaseOptions

# ---------------- 感知损失设置 ----------------
# 1. 加载预训练 VGG16（只取前 16 层特征）
vgg = vgg16(pretrained=True).features[:16].eval()
for p in vgg.parameters():
    p.requires_grad = False

# 2. 定义输入到 VGG 的归一化（ImageNet 均值/方差）
vgg_normalize = T.Normalize(
    mean=[0.485, 0.456, 0.406],
    std =[0.229, 0.224, 0.225]
)

def perceptual_loss(fake, real):
    """
    计算 VGG 特征层的 L1 感知损失
    fake, real: Tensor, shape [B,3,H,W], 值域 [-1,1]
    """
    # 先把像素映射到 [0,1]
    fake_img = (fake + 1) * 0.5
    real_img = (real + 1) * 0.5
    # 归一化到 ImageNet 分布
    f_feat = vgg(vgg_normalize(fake_img))
    r_feat = vgg(vgg_normalize(real_img))
    return torch.nn.L1Loss()(f_feat, r_feat)

if __name__ == '__main__':
    # 1) 解析参数（自动检测 MPS / CUDA / CPU）
    opt = BaseOptions().parse()

    # 2) 统一 session 目录，不再使用时间戳
    session_dir = os.path.join(opt.checkpoints_dir, opt.name)
    os.makedirs(session_dir, exist_ok=True)

    # 3) TensorBoard（直接写入 tb_logs 子目录）
    writer = SummaryWriter(log_dir=os.path.join(session_dir, 'tb_logs'))

    # 4) 普通日志（写入 session_dir/loss_log.txt）
    logger = Logger(opt, session_dir=session_dir)

    # 5) 数据加载
    dataset = UnpairedImageDataset(
        root_dir       = opt.dataroot,
        phase          = 'train',
        load_size      = opt.load_size,
        crop_size      = opt.crop_size,
        serial_batches = False
    )
    loader = DataLoader(dataset,
                        batch_size   = opt.batch_size,
                        shuffle      = True,
                        num_workers  = 4)

    # ----------- 空数据显式报错 -----------
    if len(loader) == 0:
        raise RuntimeError(
            f"[数据集为空] 请确认以下两个文件夹存在且包含图像：\n"
            f"  {os.path.join(opt.dataroot, 'trainA')}\n"
            f"  {os.path.join(opt.dataroot, 'trainB')}"
        )

    # 6) 构建网络（搬到设备）
    netG_A2B = ResnetGenerator(3, 3).to(opt.device)
    netG_B2A = ResnetGenerator(3, 3).to(opt.device)
    netD_A   = NLayerDiscriminator(3).to(opt.device)
    netD_B   = NLayerDiscriminator(3).to(opt.device)

    # 7) 损失与优化器
    criterion_GAN   = torch.nn.MSELoss().to(opt.device)
    criterion_cycle = torch.nn.L1Loss().to(opt.device)
    criterion_id    = torch.nn.L1Loss().to(opt.device)

    optimizer_G   = torch.optim.Adam(
        itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
        lr=opt.lr, betas=(opt.beta1, 0.999)
    )
    optimizer_D_A = torch.optim.Adam(netD_A.parameters(),
                                     lr=opt.lr, betas=(opt.beta1, 0.999))
    optimizer_D_B = torch.optim.Adam(netD_B.parameters(),
                                     lr=opt.lr, betas=(opt.beta1, 0.999))

    # 8) 其它工具
    pool_A = ImagePool(50)
    pool_B = ImagePool(50)

    # 9) 训练参数
    n_epochs       = opt.n_epochs
    n_epochs_decay = opt.n_epochs_decay
    total_epochs   = n_epochs + n_epochs_decay
    total_iters    = 0

    # -------------------------------------------------- #
    # 10) 训练循环（带 tqdm 进度条）
    # -------------------------------------------------- #
    for epoch in range(1, total_epochs + 1):
        epoch_start = time.time()

        progress = tqdm(loader,
                        desc=f'Epoch {epoch}/{total_epochs}',
                        ncols=85)
        for i, data in enumerate(progress):
            total_iters += 1
            real_A = data['A'].to(opt.device)
            real_B = data['B'].to(opt.device)

            # ---------- 更新生成器 ----------
            fake_B = netG_A2B(real_A)
            rec_A  = netG_B2A(fake_B)
            fake_A = netG_B2A(real_B)
            rec_B  = netG_A2B(fake_A)

            # 身份映射损失
            idt_A = netG_B2A(real_A)
            idt_B = netG_A2B(real_B)
            loss_id_A = criterion_id(idt_A, real_A) * opt.lambda_identity * 0.5
            loss_id_B = criterion_id(idt_B, real_B) * opt.lambda_identity * 0.5

            # 对抗损失
            loss_GAN_A2B = criterion_GAN(
                netD_B(fake_B), torch.ones_like(netD_B(fake_B)))
            loss_GAN_B2A = criterion_GAN(
                netD_A(fake_A), torch.ones_like(netD_A(fake_A)))

            # 循环一致性损失
            loss_cycle_A = criterion_cycle(rec_A, real_A) * opt.lambda_A
            loss_cycle_B = criterion_cycle(rec_B, real_B) * opt.lambda_B

            # 感知损失
            loss_perc_A = perceptual_loss(fake_A, real_A) * 0.1
            loss_perc_B = perceptual_loss(fake_B, real_B) * 0.1

            # 总生成器损失
            loss_G = (
                loss_id_A + loss_id_B +
                loss_GAN_A2B + loss_GAN_B2A +
                loss_cycle_A + loss_cycle_B +
                loss_perc_A + loss_perc_B
            )

            optimizer_G.zero_grad()
            loss_G.backward()
            optimizer_G.step()

            # ---------- 更新判别器 A ----------
            loss_D_real_A = criterion_GAN(
                netD_A(real_A), torch.ones_like(netD_A(real_A)))
            fake_A_ = pool_A.query(fake_A.detach())
            loss_D_fake_A = criterion_GAN(
                netD_A(fake_A_), torch.zeros_like(netD_A(fake_A_)))
            loss_D_A = (loss_D_real_A + loss_D_fake_A) * 0.5
            optimizer_D_A.zero_grad()
            loss_D_A.backward()
            optimizer_D_A.step()

            # ---------- 更新判别器 B ----------
            loss_D_real_B = criterion_GAN(
                netD_B(real_B), torch.ones_like(netD_B(real_B)))
            fake_B_ = pool_B.query(fake_B.detach())
            loss_D_fake_B = criterion_GAN(
                netD_B(fake_B_), torch.zeros_like(netD_B(fake_B_)))
            loss_D_B = (loss_D_real_B + loss_D_fake_B) * 0.5
            optimizer_D_B.zero_grad()
            loss_D_B.backward()
            optimizer_D_B.step()

            # ---------- tqdm 动态后缀显示 ----------
            progress.set_postfix({
                'G':   f'{loss_G.item():.3f}',
                'D_A': f'{loss_D_A.item():.3f}',
                'D_B': f'{loss_D_B.item():.3f}'
            })

            # TensorBoard 写入
            writer.add_scalar('Loss/Generator',       loss_G.item(),   total_iters)
            writer.add_scalar('Loss/Discriminator_A', loss_D_A.item(), total_iters)
            writer.add_scalar('Loss/Discriminator_B', loss_D_B.item(), total_iters)

        # 一个 epoch 完成
        epoch_time = time.time() - epoch_start
        logger.log(f'>>> End epoch {epoch} | Time: {epoch_time:.1f}s')

        # 每 5 个 epoch 保存一次带 epoch 编号的模型
        if epoch % 5 == 0:
            torch.save(netG_A2B.state_dict(),
                       os.path.join(session_dir, f'G_A2B_ep{epoch}.pth'))
            torch.save(netG_B2A.state_dict(),
                       os.path.join(session_dir, f'G_B2A_ep{epoch}.pth'))
            torch.save(netD_A.state_dict(),
                       os.path.join(session_dir, f'D_A_ep{epoch}.pth'))
            torch.save(netD_B.state_dict(),
                       os.path.join(session_dir, f'D_B_ep{epoch}.pth'))

        # 每个 epoch 都保存最新模型（覆盖）
        torch.save(netG_A2B.state_dict(),
                   os.path.join(session_dir, 'latest_net_G.pth'))
        torch.save(netD_A.state_dict(),
                   os.path.join(session_dir, 'latest_net_D_A.pth'))
        torch.save(netD_B.state_dict(),
                   os.path.join(session_dir, 'latest_net_D_B.pth'))

    writer.close()
