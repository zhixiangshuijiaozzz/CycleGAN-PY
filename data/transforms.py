import random
from PIL import Image
import torchvision.transforms as T

class UnpairedTransforms:
    def __init__(self, load_size, crop_size, flip=True):
        self.load_size = load_size
        self.crop_size = crop_size
        self.flip = flip
        self.resize = T.Resize(load_size, interpolation=Image.BICUBIC)
        self.random_crop = T.RandomCrop(crop_size)
        self.jitter = T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1)
        self.gaussian_blur = T.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize((0.5, 0.5, 0.5),
                                     (0.5, 0.5, 0.5))

    def __call__(self, img):
        # 1. Resize + 随机裁剪
        img = self.resize(img)
        img = self.random_crop(img)
        # 2. 随机水平翻转
        if self.flip and random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        # 3. 随机色彩抖动
        if random.random() > 0.5:
            img = self.jitter(img)
        # 4. 随机高斯模糊
        if random.random() > 0.5:
            img = self.gaussian_blur(img)
        # 5. 转张量 + 归一化
        img = self.to_tensor(img)
        img = self.normalize(img)
        return img
