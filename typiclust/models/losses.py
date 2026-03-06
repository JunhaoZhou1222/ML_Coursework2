import torch
import torch.nn as nn
import torch.nn.functional as F


class NTXentLoss(nn.Module):
    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.temperature = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        # l2 Normalization
        B = z1.size(0)
        # Set all vector lengths to 1
        z1 = F.normalize(z1, dim=1) 
        z2 = F.normalize(z2, dim=1)

        z = torch.cat([z1, z2], dim=0)  # (2B, D)
        sim = torch.mm(z, z.T) / self.temperature  # (2B, 2B) dot product

        mask = torch.eye(2 * B, dtype=torch.bool, device=z.device)
        sim.masked_fill_(mask, float("-inf"))

        labels = torch.cat([
            torch.arange(B, 2 * B), # eg. [4, 5, 6, 7]
            torch.arange(0, B), # eg. [0, 1, 2, 3]
        ]).to(z.device) # labels = [4, 5, 6, 7, 0, 1, 2, 3]

        return F.cross_entropy(sim, labels)
