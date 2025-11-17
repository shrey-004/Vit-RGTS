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
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Using CIFAR-10 for demonstration purposes
cifar10 = datasets.CIFAR10(root="./data", download=True, transform=transform)
train_size = int(0.9 * len(cifar10))
val_size = len(cifar10) - train_size
train_dataset, val_dataset = random_split(cifar10, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

# 3. Model Initialization
model = VitRGTS(
    image_size=224,
    patch_size=14,
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
# def train_epoch(model, loader, optimizer, criterion, device):
#     model.train()
#     total_loss = 0.0
#     correct = 0

#     # Use tqdm to show batch-level progress and metrics
#     with tqdm(total=len(loader), desc="Epoch Training", unit="batch") as pbar:
#         for imgs, labels in loader:
#             imgs, labels = imgs.to(device), labels.to(device)
#             optimizer.zero_grad()

#             outputs = model(imgs)
#             loss = criterion(outputs, labels)
#             loss.backward()

#             clip_grad_norm_(model.parameters(), 0.05)
#             optimizer.step()
#             scheduler.step()

#             batch_loss = loss.item()
#             _, predicted = outputs.max(1)
#             batch_correct = predicted.eq(labels).sum().item()
#             batch_acc = batch_correct / imgs.size(0)

#             total_loss += batch_loss
#             correct += batch_correct

#             # Update progress bar with current batch stats
#             pbar.set_postfix({
#                 "Batch_Loss": f"{batch_loss:.4f}",
#                 "Batch_Acc": f"{batch_acc:.4f}"
#             })
#             pbar.update(1)

#     avg_loss = total_loss / len(loader)
#     avg_acc = correct / len(train_dataset)
#     return avg_loss, avg_acc







def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0.0
    correct = 0

    # Use tqdm to show batch-level progress and metrics
    with tqdm(total=len(loader), desc="Epoch Training", unit="batch") as pbar:
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad()

            # <--- FIX IS HERE ---
            # Unpack the tuple: (predictions, stats)
            # We use _ for stats because we don't need them during training
            predictions, _ = model(imgs) 
            
            # Now use 'predictions' (which is a Tensor) for loss
            loss = criterion(predictions, labels)
            # <--- END FIX ---

            loss.backward()

            clip_grad_norm_(model.parameters(), 0.05)
            optimizer.step()
            scheduler.step()

            batch_loss = loss.item()

            # <--- FIX IS ALSO HERE ---
            # Use 'predictions' to calculate accuracy
            _, predicted = predictions.max(1)
            # <--- END FIX ---
            
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
    avg_acc = correct / len(train_dataset)
    return avg_loss, avg_acc
















# def validate_epoch(model, loader, criterion, device):
#     model.eval()
#     total_loss = 0.0
#     correct = 0
#     with torch.no_grad():
#         with tqdm(total=len(loader), desc="Epoch Validation", unit="batch") as pbar:
#             for imgs, labels in loader:
#                 imgs, labels = imgs.to(device), labels.to(device)

#                 outputs = model(imgs)
#                 loss = criterion(outputs, labels)

#                 batch_loss = loss.item()
#                 _, predicted = outputs.max(1)
#                 batch_correct = predicted.eq(labels).sum().item()
#                 batch_acc = batch_correct / imgs.size(0)

#                 total_loss += batch_loss
#                 correct += batch_correct

#                 pbar.set_postfix({
#                     "Batch_Loss": f"{batch_loss:.4f}",
#                     "Batch_Acc": f"{batch_acc:.4f}"
#                 })
#                 pbar.update(1)

#     avg_loss = total_loss / len(loader)
#     avg_acc = correct / len(val_dataset)
#     return avg_loss, avg_acc

# # Assuming we will train for a certain number of epochs (in this case, calculated to reach 300k steps)
# # num_epochs = (300000 * 64) // len(train_dataset)
num_epochs = 10




def validate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    
    # <--- NEW CODE: Initialize stats aggregator ---
    # We know the depth is 6 from our model definition
    num_layers = 6 
    epoch_stats = [
        {
            'num_high_norm': 0, 
            'total_tokens': 0, 
            'mean_norm_sum': 0.0, 
            'threshold_sum': 0.0
        } for _ in range(num_layers)
    ]
    num_batches = len(loader)
    # <--- END NEW CODE ---
    
    with torch.no_grad():
        with tqdm(total=len(loader), desc="Epoch Validation", unit="batch") as pbar:
            for imgs, labels in loader:
                imgs, labels = imgs.to(device), labels.to(device)

                # <--- MODIFIED CODE: Capture layer_stats ---
                outputs, layer_stats = model(imgs)
                # <--- END MODIFIED CODE ---
                
                loss = criterion(outputs, labels)

                batch_loss = loss.item()
                _, predicted = outputs.max(1)
                batch_correct = predicted.eq(labels).sum().item()
                batch_acc = batch_correct / imgs.size(0)

                total_loss += batch_loss
                correct += batch_correct

                # <--- NEW CODE: Aggregate stats from this batch ---
                for i, stats in enumerate(layer_stats):
                    epoch_stats[i]['num_high_norm'] += stats['num_high_norm']
                    epoch_stats[i]['total_tokens'] += stats['total_tokens']
                    epoch_stats[i]['mean_norm_sum'] += stats['mean_norm']
                    epoch_stats[i]['threshold_sum'] += stats['threshold']
                # <--- END NEW CODE ---

                pbar.set_postfix({
                    "Batch_Loss": f"{batch_loss:.4f}",
                    "Batch_Acc": f"{batch_acc:.4f}"
                })
                pbar.update(1)

    avg_loss = total_loss / num_batches
    avg_acc = correct / len(val_dataset)
    
    # <--- NEW CODE: Process and format the final stats ---
    final_report_stats = []
    for i in range(num_layers):
        total_high_norm = epoch_stats[i]['num_high_norm']
        total_tokens = epoch_stats[i]['total_tokens']
        
        # Avoid division by zero if dataset is empty
        percentage = (total_high_norm / total_tokens) * 100 if total_tokens > 0 else 0
        avg_mean_norm = epoch_stats[i]['mean_norm_sum'] / num_batches if num_batches > 0 else 0
        avg_threshold = epoch_stats[i]['threshold_sum'] / num_batches if num_batches > 0 else 0
        
        final_report_stats.append({
            'layer': i + 1,
            'percentage': percentage,
            'avg_mean_norm': avg_mean_norm,
            'avg_threshold': avg_threshold
        })
    # <--- END NEW CODE ---

    # <--- MODIFIED RETURN: Return the final stats report ---
    return avg_loss, avg_acc, final_report_stats



# print("="*40)
# print("🚀 STARTING MODEL TRAINING 🚀")
# print("="*40)

# for epoch in range(num_epochs):
#     # Print a header for the new epoch
#     print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")

#     # 1. Run Training for one epoch
#     # We still need to calculate train_loss, just not print it here
#     train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)

#     # 2. Run Validation for one epoch
#     val_loss, val_acc = validate_epoch(model, val_loader, criterion, device)
    
#     # 3. Print a clean, combined summary for the epoch
#     # This replaces all the separate print lines
#     print(f"Epoch {epoch+1} Summary:")
#     print(f"\tTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
#     print(f"\tVal Loss  : {val_loss:.4f} | Val Acc  : {val_acc:.4f}")

# # 5. Final Steps
# print("\n" + "="*40)
# print("✅ TRAINING FINISHED SUCCESSFULLY ✅")
# print("="*40)

# torch.save(model.state_dict(), "mega_vit_model.pth")
# print("Model state_dict saved to 'mega_vit_model.pth'")




print("="*40)
print("🚀 STARTING MODEL TRAINING 🚀")
print("="*40)

for epoch in range(num_epochs):
    # Print a header for the new epoch
    print(f"\n--- Epoch {epoch+1}/{num_epochs} ---")

    # 1. Run Training for one epoch
    # We will NOT be collecting stats from train_epoch to keep things clean
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)

    # 2. Run Validation for one epoch
    # <--- MODIFIED CODE: Capture the final_report_stats ---
    val_loss, val_acc, val_stats_report = validate_epoch(model, val_loader, criterion, device)
    # <--- END MODIFIED CODE ---
    
    # 3. Print a clean, combined summary for the epoch
    print(f"Epoch {epoch+1} Summary:")
    print(f"\tTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
    print(f"\tVal Loss  : {val_loss:.4f} | Val Acc  : {val_acc:.4f}")
    
    # <--- NEW CODE: Print the norm-thresholding report ---
    print("  Validation Norm-Thresholding Report:")
    print("  " + "-"*35)
    print("  Layer | % High-Norm | Avg Mean Norm | Avg Threshold")
    print("  " + "-"*35)
    for stats in val_stats_report:
        print(f"   {stats['layer']:<5} |   {stats['percentage']:<7.3f} % |     {stats['avg_mean_norm']:<7.2f} |   {stats['avg_threshold']:<7.2f}")
    # <--- END NEW CODE ---

# 5. Final Steps
print("\n" + "="*40)
print("✅ TRAINING FINISHED SUCCESSFULLY ✅")
print("="*40)

torch.save(model.state_dict(), "mega_vit_model.pth")
print("Model state_dict saved to 'mega_vit_model.pth'")