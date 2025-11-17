import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm

from vit_rgts.main import VitRGTS


# ----------------------------- #
# 1. Setup
# ----------------------------- #
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("Using device:", device)


# ----------------------------- #
# 2. Data Preparation
# ----------------------------- #
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

cifar10 = datasets.CIFAR10(root="./data", download=True, transform=transform)
train_size = int(0.9 * len(cifar10))
val_size = len(cifar10) - train_size
train_dataset, val_dataset = random_split(cifar10, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)


# ----------------------------- #
# 3. Model Init
# ----------------------------- #
model = VitRGTS(
    image_size=224,
    patch_size=14,
    num_classes=10,
    dim=1024,
    depth=12,
    heads=16,
    mlp_dim=2048,
    dropout=0.1,
    emb_dropout=0.1
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.0002)


# LR schedule: Warm-up + Cosine
def lr_schedule(step):
    if step < 2500:
        return step / 2500
    return 0.5 * (1 + torch.cos((step - 2500) / (300000 - 2500) * 3.14159))

scheduler = LambdaLR(optimizer, lr_schedule)


# ----------------------------- #
# 4. Training & Validation
# ----------------------------- #

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    total_correct = 0

    loop = tqdm(loader, desc="Training", leave=False)

    for imgs, labels in loop:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(imgs)

        # Fix: Model returns tuple → extract logits
        if isinstance(outputs, tuple):
            outputs = outputs[0]

        loss = criterion(outputs, labels)
        loss.backward()

        clip_grad_norm_(model.parameters(), 0.05)
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total_correct += predicted.eq(labels).sum().item()

        loop.set_postfix(loss=loss.item())

    acc = total_correct / len(train_dataset)
    return total_loss / len(loader), acc


def validate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    total_correct = 0

    loop = tqdm(loader, desc="Validating", leave=False)

    with torch.no_grad():
        for imgs, labels in loop:
            imgs, labels = imgs.to(device), labels.to(device)

            outputs = model(imgs)

            # Fix: extract logits from tuple
            if isinstance(outputs, tuple):
                outputs = outputs[0]

            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total_correct += predicted.eq(labels).sum().item()

            loop.set_postfix(loss=loss.item())

    acc = total_correct / len(val_dataset)
    return total_loss / len(loader), acc


# ----------------------------- #
# 5. Epoch Calculation (300k steps)
# ----------------------------- #
num_epochs = 2
print(f"Total epochs to reach 300k steps: {num_epochs}")


# ----------------------------- #
# 6. Training Loop
# ----------------------------- #
for epoch in range(num_epochs):
    print(f"\n----- Epoch {epoch+1}/{num_epochs} -----")

    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)

    print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.4f}")


# ----------------------------- #
# 7. Save Model
# ----------------------------- #
torch.save(model.state_dict(), "mega_vit_model.pth")
print("\nTraining finished. Model saved as mega_vit_model.pth")
