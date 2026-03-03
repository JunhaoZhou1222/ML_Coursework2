"""
下游 ResNet-18 分类器训练与评估。
"""
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset

from ..config import CIFAR10_MEAN, CIFAR10_STD, DEVICE
from ..models import ResNet18Classifier


def train_classifier(
    labeled_indices: list[int],
    dataset_root: str = "./data",
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 0.025,
) -> float:
    """在已标注子集上训练 ResNet-18 分类器，返回测试集准确率（%）。"""
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR10_MEAN, CIFAR10_STD),
    ])

    full_train = torchvision.datasets.CIFAR10(
        root=dataset_root,
        train=True,
        download=False,
        transform=train_transform,
    )
    test_set = torchvision.datasets.CIFAR10(
        root=dataset_root,
        train=False,
        download=False,
        transform=test_transform,
    )

    labeled_set = Subset(full_train, labeled_indices)
    train_loader = DataLoader(
        labeled_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=256,
        shuffle=False,
        num_workers=2,
    )

    model = ResNet18Classifier(num_classes=10).to(DEVICE)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=5e-4,
        nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=0,
    )
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            loss = criterion(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            pred = model(x).argmax(dim=1)
            correct += (pred == y).sum().item()
            total += y.size(0)
    acc = 100.0 * correct / total
    return acc
