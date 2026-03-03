"""
NT-Xent (Normalised Temperature-Scaled Cross-Entropy) 对比损失。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    """SimCLR 使用的对比损失。"""

    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        z1, z2: (B, D) 投影向量，将在此处做 L2 归一化。
        """
        B = z1.size(0)
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        z = torch.cat([z1, z2], dim=0)  # (2B, D)
        sim = torch.mm(z, z.T) / self.temperature  # (2B, 2B)

        mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
        sim.masked_fill_(mask, float("-inf"))

        labels = torch.cat([
            torch.arange(B, 2 * B),
            torch.arange(0, B),
        ]).to(z.device)

        return F.cross_entropy(sim, labels)
