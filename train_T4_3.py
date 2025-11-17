import warnings
# Silence specific FutureWarning coming from PyTorch's sdp_kernel deprecation
# This prevents the noisy deprecation message during training. If you prefer
# to see future warnings (recommended for debugging), remove this filter and
# either upgrade PyTorch or update code that calls the deprecated API.
warnings.filterwarnings(
    "ignore",
    message=r".*torch\.backends\.cuda\.sdp_kernel.*",
    category=FutureWarning,
)

import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_
from torch.optim import Adam
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from tqdm import tqdm
import math

from vit_rgts.main import VitRGTS

# 1. Setup and Imports
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
# 2. Data Preparation
# 2. Data Preparation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Using STL-10 for demonstration purposes
# We load the 'train' split, which contains 5,000 labeled images
stl10_data = datasets.STL10(root="./data1", split='train', download=True, transform=transform)

# Now we split that 5,000-image training set into 90% train / 10% validation
train_size = int(0.9 * len(stl10_data))
val_size = len(stl10_data) - train_size
train_dataset, val_dataset = random_split(stl10_data, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# 3. Model Initialization
model = VitRGTS(
    image_size=224,
    patch_size=16,
    num_classes=10,  # CIFAR-10 has 10 classes
    dim=1024,
    depth=6,
    heads=16,
    mlp_dim=2048,
    dropout=0.1,
    emb_dropout=0.1
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.0002)

# Warm-up + Cosine schedule for the learning rate
def lr_schedule(epoch):
    if epoch < 2500:
        return epoch / 2500
    # use math.cos since epoch is a float/int here (not a Tensor)
    return 0.5 * (1 + math.cos((epoch - 2500) / (300000 - 2500) * 3.14159))

scheduler = LambdaLR(optimizer, lr_schedule)

# 4. Training Loop
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0

    # Use tqdm to show batch-level progress and metrics
    with tqdm(total=len(loader), desc="Epoch Training", unit="batch") as pbar:
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()

            outputs = model(imgs)
            
            # --- Start of fix ---
            if isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs
            # --- End of fix ---

            # Use 'logits' for everything from here on
            loss = criterion(logits, labels)
            loss.backward()  # Do backward pass on the correct loss

            clip_grad_norm_(model.parameters(), 0.05)
            optimizer.step()
            scheduler.step()

            batch_loss = loss.item() # Get loss value
            _, predicted = logits.max(1) # Get predictions from logits
            
            batch_correct = predicted.eq(labels).sum().item()
            batch_acc = batch_correct / imgs.size(0)

            total_loss += batch_loss
            correct += batch_correct

            # Update progress bar with current batch stats
            pbar.set_postfix({
                "Batch_Loss": f"{batch_loss:.4f}",
                "Batch_Acc": f"{batch_acc:.4f}"
            })
            pbar.update(1)

    avg_loss = total_loss / len(loader)
    # Make sure 'train_dataset' is defined, or pass len(loader.dataset)
    avg_acc = correct / len(train_dataset) 
    return avg_loss, avg_acc

def validate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    with torch.no_grad():
        with tqdm(total=len(loader), desc="Epoch Validation", unit="batch") as pbar:
            for imgs, labels in loader:
                imgs, labels = imgs.to(device), labels.to(device)

                outputs = model(imgs)
                if isinstance(outputs, tuple):
                    logits = outputs[0]
                else:
                    logits = outputs

                loss = criterion(logits, labels)

                batch_loss = loss.item()
                _, predicted = logits.max(1)
                batch_correct = predicted.eq(labels).sum().item()
                batch_acc = batch_correct / imgs.size(0)

                total_loss += batch_loss
                correct += batch_correct

                pbar.set_postfix({
                    "Batch_Loss": f"{batch_loss:.4f}",
                    "Batch_Acc": f"{batch_acc:.4f}"
                })
                pbar.update(1)

    avg_loss = total_loss / len(loader)
    avg_acc = correct / len(val_dataset)
    return avg_loss, avg_acc

# Assuming we will train for a certain number of epochs (in this case, calculated to reach 300k steps)
# num_epochs = (300000 * 64) // len(train_dataset)
num_epochs = 20


# for epoch in range(num_epochs):
#     train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
#     print(train_loss, train_acc)

#     val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
#     print(val_loss, val_acc)
    
#     print(f"Epoch: {epoch+1}/{num_epochs}, Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

# # 5. Final Steps
# torch.save(model.state_dict(), "mega_vit_model.pth")
# print("Training finished.")

print("="*40)
print("🚀 STARTING MODEL TRAINING 🚀")
print("="*40)

for epoch in range(num_epochs):
    # Print a header for the new epoch
    print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")

    # 1. Run Training for one epoch
    # We still need to calculate train_loss, just not print it here
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)

    # 2. Run Validation for one epoch
    val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
    
    # 3. Print a clean, combined summary for the epoch
    # This replaces all the separate print lines
    print(f"Epoch {epoch+1} Summary:")
    print(f"\tTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"\tVal Loss  : {val_loss:.4f} | Val Acc  : {val_acc:.4f}")

# 5. Final Steps
print("\n" + "="*40)
print("✅ TRAINING FINISHED SUCCESSFULLY ✅")
print("="*40)

torch.save(model.state_dict(), "mega_vit_model.pth")
print("Model state_dict saved to 'mega_vit_model.pth'")