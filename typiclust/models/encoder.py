"""
SimCLR 编码器：ResNet-18 骨干 + 投影头。
"""
import torch
import torch.nn as nn
import torchvision


class SimCLREncoder(nn.Module):
    """
    ResNet-18 骨干 + 2 层 MLP 投影头。
    倒数第二层（512 维）作为嵌入空间。
    """

    def __init__(self, projection_dim: int = 128):
        super().__init__()
        resnet = torchvision.models.resnet18(weights=None)
        resnet.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        resnet.maxpool = nn.Identity()
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])

        self.projection_head = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, projection_dim),
        )

    def forward(self, x: torch.Tensor):
        h = self.backbone(x).squeeze(-1).squeeze(-1)  # (B, 512)
        z = self.projection_head(h)  # (B, projection_dim)
        return h, z
