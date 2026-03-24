"""
Data transforms for SimCLR learning
"""
import torchvision.transforms as transforms

from .config import CIFAR10_MEAN, CIFAR10_STD


class SimCLRTransform:
    #Data augmentation
    def __init__(self, size = 32):
        self.transform = transforms.Compose([
            transforms.RandomResizedCrop(size=size, scale=(0.2, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])

    def __call__(self, x):
        return self.transform(x), self.transform(x) #different kind but same picture


class StandardTransform:
    #Deterministic transform
    def __init__(self, size = 32):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
        ])

    def __call__(self, x):
        return self.transform(x)
