"""
下游 ResNet-18 分类器训练与评估。
训练时用验证集选最优 checkpoint，最终在测试集上报告最优模型的准确率。
"""
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import train_test_split

from ..config import CIFAR10_MEAN, CIFAR10_STD, DEVICE
from ..models import ResNet18Classifier


def train_classifier(
    labeled_indices: list[int],
    dataset_root: str = "./data",
    epochs: int = 100,
    batch_size: int = 64,
    lr: float = 0.025,
    val_ratio: float = 0.2,
) -> float:
    """
    在已标注子集上训练 ResNet-18：按 val_ratio 划分验证集，保存验证集最优模型，
    最后用该 checkpoint 在测试集上评估并返回准确率（%）。
    """
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

    # 标注样本过少时不划分验证集，全部用于训练
    n_labeled = len(labeled_indices)
    if n_labeled >= 5 and val_ratio > 0:
        labels = [full_train[i][1] for i in labeled_indices]
        try:
            train_idx, val_idx = train_test_split(
                range(n_labeled),
                test_size=val_ratio,
                stratify=labels,
                random_state=42,
            )
        except ValueError:
            train_idx, val_idx = train_test_split(
                range(n_labeled), test_size=val_ratio, random_state=42
            )
        train_indices = [labeled_indices[i] for i in train_idx]
        val_indices = [labeled_indices[i] for i in val_idx]
        train_set = Subset(full_train, train_indices)
        val_set = Subset(full_train, val_indices)
        train_loader = DataLoader(
            train_set,
            batch_size=min(batch_size, len(train_indices)),
            shuffle=True,
            num_workers=2,
        )
        val_loader = DataLoader(
            val_set,
            batch_size=min(256, len(val_indices)),
            shuffle=False,
            num_workers=2,
        )
        use_val = True
    else:
        labeled_set = Subset(full_train, labeled_indices)
        train_loader = DataLoader(
            labeled_set,
            batch_size=min(batch_size, n_labeled),
            shuffle=True,
            num_workers=2,
        )
        val_loader = None
        use_val = False

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

    best_val_acc = -1.0
    best_state = None

    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            loss = criterion(model(x), y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        scheduler.step()

        if use_val and val_loader is not None:
            model.eval()
            val_correct = val_total = 0
            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(DEVICE), y.to(DEVICE)
                    pred = model(x).argmax(dim=1)
                    val_correct += (pred == y).sum().item()
                    val_total += y.size(0)
            val_acc = 100.0 * val_correct / val_total if val_total > 0 else 0.0
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

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
