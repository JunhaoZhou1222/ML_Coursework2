"""
从 SimCLR 编码器提取 L2 归一化嵌入。
"""
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
    """
    对整个 CIFAR-10 训练集提取 L2 归一化的倒数第二层嵌入。

    Returns
    -------
    embeddings : np.ndarray  (N, 512)
    labels     : np.ndarray  (N,)
    """
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
        h = F.normalize(h, dim=1)
        all_h.append(h.cpu().numpy())
        all_y.append(y.numpy())

    embeddings = np.concatenate(all_h, axis=0)
    labels = np.concatenate(all_y, axis=0)
    return embeddings, labels
