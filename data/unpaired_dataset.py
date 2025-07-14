import os
from PIL import Image
from torch.utils.data import Dataset
from .transforms import UnpairedTransforms
import random

class UnpairedImageDataset(Dataset):
    """
    无配对图像数据集：
    假设根目录下有 trainA/ trainB/ testA/ testB/ 四个子目录
    """
    def __init__(self, root_dir, phase='train', load_size=286, crop_size=256, serial_batches=False):
        super().__init__()
        self.dir_A = os.path.join(root_dir, f'{phase}A')
        self.dir_B = os.path.join(root_dir, f'{phase}B')
        self.paths_A = sorted([os.path.join(self.dir_A, fn) for fn in os.listdir(self.dir_A)])
        self.paths_B = sorted([os.path.join(self.dir_B, fn) for fn in os.listdir(self.dir_B)])
        self.len_A = len(self.paths_A)
        self.len_B = len(self.paths_B)
        self.serial = serial_batches
        self.transform = UnpairedTransforms(load_size, crop_size)

    def __len__(self):
        return max(self.len_A, self.len_B)

    def __getitem__(self, idx):
        # A 图像
        index_A = idx % self.len_A
        path_A = self.paths_A[index_A]
        img_A = Image.open(path_A).convert('RGB')

        # B 图像，顺序 or 随机
        if self.serial:
            index_B = idx % self.len_B
        else:
            index_B = random.randint(0, self.len_B - 1)
        path_B = self.paths_B[index_B]
        img_B = Image.open(path_B).convert('RGB')

        # 预处理
        A = self.transform(img_A)
        B = self.transform(img_B)
        return {'A': A, 'B': B,
                'A_path': path_A, 'B_path': path_B}
