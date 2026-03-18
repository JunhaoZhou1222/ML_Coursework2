"""
- ResNet-18 trained on the labeled set
- SGD with 0.9 momentum and Nesterov momentum
- Initial learning rate 0.025, cosine scheduler
- Augmentations: random crops and horizontal flips
"""
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import numpy as np
from ..config import CIFAR10_MEAN, CIFAR10_STD, DEVICE
from ..models import ResNet18Classifier
from ..config import NUM_WORKERS

def mixup_data(x, y, alpha=0.2):
    """Mix two samples together to create virtual training examples."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    index = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    return mixed_x, y, y[index], lam

def train_classifier(
    labeled_indices,
    dataset_root = "./data",
    epochs = 300, #200
    batch_size = 64,
    lr = 0.025,
):
    """
    Train a fresh ResNet-18 on `labeled_indices`

    Optimizations over baseline paper reproduction:
    - [Opt] Mixup
    - [Opt] AutoAugment for stronger data augmentation
    - [Opt] Label smoothing (0.1) to reduce overconfidence
      
    """

    # When n_labeled < 20, validation set (2-4 samples) is pure noise.
    # Skip val split and train longer instead.
    #n_labeled = len(labeled_indices)
    #if n_labeled < 20:
        #val_ratio = 0.0
        #epochs = 200

    # Low budget benefits greatly from stronger augmentation to reduce overfitting
    train_transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.AutoAugment(transforms.AutoAugmentPolicy.CIFAR10),
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
    n_labeled = len(labeled_indices)
    labeled_set = Subset(full_train, labeled_indices)
    train_loader = DataLoader(
        labeled_set,
        batch_size=min(batch_size, n_labeled),
        shuffle=True,
        num_workers=NUM_WORKERS,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=256,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )
    model = ResNet18Classifier(num_classes=10).to(DEVICE)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=1e-4,
        nesterov=True,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=0,
    )
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1) #label_smoothing=0.1

    for epoch in range(1, epochs + 1):
        model.train()
        for x, y in train_loader:
            x, y = x.to(DEVICE), y.to(DEVICE)
            # [Opt 3] Mixup: 50% chance
            if np.random.rand() < 0.5:
                x, ya, yb, lam = mixup_data(x, y)
                outputs = model(x)
                loss = lam * criterion(outputs, ya) + (1 - lam) * criterion(outputs, yb)
            else:
                outputs = model(x)
                loss = criterion(outputs, y)
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
