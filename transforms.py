import random
from torchvision.transforms import functional as F


class Compose(object):
    """组合多个transform函数"""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target,ground):
        for t in self.transforms:
            image, target ,ground= t(image, target,ground)
        return image, target ,ground


class ToTensor(object):
    """将PIL图像转为Tensor"""
    def __call__(self, image, target,ground):
        image = F.to_tensor(image)
        ground = F.to_tensor(ground)
        return image, target , ground


class RandomHorizontalFlip(object):
    """随机水平翻转图像以及bboxes"""
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, image, target ,ground):
        if random.random() < self.prob:
            height, width = image.shape[-2:]
            image = image.flip(-1)  # 水平翻转图片
            ground = ground.flip(-1)
            bbox = target["boxes"]
            # bbox: xmin, ymin, xmax, ymax
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]  # 翻转对应bbox坐标信息
            target["boxes"] = bbox
        return image, target ,ground
