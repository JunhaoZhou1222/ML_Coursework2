"""
下游分类器：ResNet-18 全监督。
"""
import torch.nn as nn
import torchvision


class ResNet18Classifier(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        resnet = torchvision.models.resnet18(weights=None)
        resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        resnet.maxpool = nn.Identity()
        resnet.fc = nn.Linear(512, num_classes)
        self.net = resnet

    def forward(self, x):
        return self.net(x)
