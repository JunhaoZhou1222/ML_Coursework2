"""
训练脚本：SimCLR 预训练与下游分类器。
"""
from .simclr import train_simclr
from .classifier import train_classifier

__all__ = ["train_simclr", "train_classifier"]
