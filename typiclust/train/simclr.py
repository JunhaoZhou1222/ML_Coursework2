import torch
import torchvision
from torch.utils.data import DataLoader

from ..config import DEVICE
from ..models import SimCLREncoder, NTXentLoss
from ..transforms import SimCLRTransform


def train_simclr(
    dataset_root: str = "./data",
    epochs: int = 50, 
    batch_size: int = 256,
    lr: float = 0.5,
    temperature: float = 0.5,
    projection_dim: int = 128,
) -> SimCLREncoder:
    
    print("\n=== Step 1: Training SimCLR Encoder ===")
    #Load data
    train_dataset = torchvision.datasets.CIFAR10(
        root=dataset_root,
        train=True,
        download=True,
        transform=SimCLRTransform(size=32),
    )
    loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
    )

    model = SimCLREncoder(projection_dim=projection_dim).to(DEVICE)
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=0.9,
        weight_decay=1e-4, #L2 norm
        nesterov=True, #faster than normal
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs,
        eta_min=0, #min lr
    )
    criterion = NTXentLoss(temperature=temperature)

    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for (x1, x2), _ in loader:
            x1, x2 = x1.to(DEVICE), x2.to(DEVICE)
            _, z1 = model(x1)
            _, z2 = model(x2)
            loss = criterion(z1, z2)
            optimizer.zero_grad()
            loss.backward() # Calculate grad
            optimizer.step()
            total_loss += loss.item()
        scheduler.step() #update rate
        avg = total_loss / len(loader)
        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch [{epoch:3d}/{epochs}]  Loss: {avg:.4f}")

    return model
