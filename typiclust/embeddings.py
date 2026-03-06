from typing import Tuple
import numpy as np
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data import DataLoader
from .config import DEVICE
from .models import SimCLREncoder
from .transforms import StandardTransform


@torch.no_grad()
def extract_embeddings(
    model: SimCLREncoder,
    dataset_root: str = "./data",
    batch_size: int = 512,
) -> Tuple[np.ndarray, np.ndarray]:

    dataset = torchvision.datasets.CIFAR10(
        root=dataset_root,
        train=True,
        download=False,
        transform=StandardTransform(size=32),
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )

    model.eval()
    all_h, all_y = [], []
    for x, y in loader:
        x = x.to(DEVICE)
        h, _ = model(x)
        h = F.normalize(h, dim=1) #L2 norm
        all_h.append(h.cpu().numpy())
        all_y.append(y.numpy()) #Store label

    embeddings = np.concatenate(all_h, axis=0) #eg. (50000, 512)
    labels = np.concatenate(all_y, axis=0) #eg. (50000, )
    return embeddings, labels
