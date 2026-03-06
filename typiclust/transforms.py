"""
SimCLR 与标准数据增强 / 变换。
"""
import torchvision.transforms as transforms

from .config import CIFAR10_MEAN, CIFAR10_STD


class SimCLRTransform:
    """为对比学习生成同一图像的两个增强视图。"""

    def __init__(self, size: int = 32):
        color_jitter = transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size=size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([color_jitter], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])

    def __call__(self, x):
        return self.transform(x), self.transform(x) #对同一张图生成两个不同的增强版本


class StandardTransform:
    """用于抽取嵌入的标准变换（无随机增强）。"""

    def __init__(self, size: int = 32):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])

    def __call__(self, x):
        return self.transform(x)
