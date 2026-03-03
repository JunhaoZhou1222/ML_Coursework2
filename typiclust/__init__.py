"""
TypiClust with Representation + K-means (TPC_RP).

Based on: "Active Learning on a Budget: Opposite Strategies Suit High and Low Budgets"
Hacohen et al., ICML 2022.

Algorithm: SimCLR representation → K-means clustering → select most typical per cluster.
Dataset: CIFAR-10.
"""

from .config import SEED, DEVICE, set_seed
from .pipeline import run_typiclust_rp
from .models import SimCLREncoder, ResNet18Classifier, NTXentLoss
from .selection import typiclust_rp_select
from .typicality import compute_typicality
from .embeddings import extract_embeddings
from .train import train_simclr, train_classifier

__all__ = [
    "SEED",
    "DEVICE",
    "set_seed",
    "run_typiclust_rp",
    "SimCLREncoder",
    "ResNet18Classifier",
    "NTXentLoss",
    "typiclust_rp_select",
    "compute_typicality",
    "extract_embeddings",
    "train_simclr",
    "train_classifier",
]
