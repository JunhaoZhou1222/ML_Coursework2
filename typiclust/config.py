"""
TPC_RP 全局配置与可复现性设置。
"""
import random
import numpy as np
import torch

SEED = 42


def set_seed(seed: int = SEED) -> None:
    """设置所有随机种子以保证可复现性。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# 在模块加载时设置种子
set_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CIFAR-10 标准化参数
CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2023, 0.1994, 0.2010)
