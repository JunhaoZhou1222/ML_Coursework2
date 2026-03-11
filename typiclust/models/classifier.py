import torch.nn as nn
import torchvision


class ResNet18Classifier(nn.Module):
    #estimate acc in test
    def __init__(self, num_classes = 10):
        super().__init__()
        resnet = torchvision.models.resnet18(weights=None)
        resnet.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        resnet.maxpool = nn.Identity()
        resnet.fc = nn.Linear(512, num_classes)
        self.net = resnet

    def forward(self, x):
        return self.net(x)
