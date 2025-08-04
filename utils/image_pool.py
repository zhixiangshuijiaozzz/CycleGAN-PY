import random
import torch

class ImagePool:
    def __init__(self, pool_size):
        self.pool_size = pool_size
        if pool_size > 0:
            self.images = []

    def query(self, images):
        if self.pool_size == 0:
            return images
        return_images = []
        for img in images:
            img = torch.unsqueeze(img.data, 0)
            if len(self.images) < self.pool_size:
                self.images.append(img)
                return_images.append(img)
            else:
                if random.random() > 0.5:
                    idx = random.randint(0, self.pool_size - 1)
                    tmp = self.images[idx].clone()
                    self.images[idx] = img
                    return_images.append(tmp)
                else:
                    return_images.append(img)
        return torch.cat(return_images, 0)
