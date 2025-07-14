import random
from PIL import Image
import torchvision.transforms as T

class UnpairedTransforms:
    def __init__(self, load_size, crop_size, flip=True):
        self.load_size = load_size
        self.crop_size = crop_size
        self.flip = flip
        # 定义统一的预处理流水线
        self.resize = T.Resize(load_size, interpolation=Image.BICUBIC)
        self.random_crop = T.RandomCrop(crop_size)
        self.to_tensor = T.ToTensor()
        self.normalize = T.Normalize((0.5, 0.5, 0.5),
                                     (0.5, 0.5, 0.5))

    def __call__(self, img):
        img = self.resize(img)
        img = self.random_crop(img)
        if self.flip and random.random() > 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        img = self.to_tensor(img)
        img = self.normalize(img)
        return img
