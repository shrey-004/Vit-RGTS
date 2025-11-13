import vit_rgts.main_T2
print("✅ Loaded VitRGTS from:", vit_rgts.main_T2.__file__)


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

from vit_rgts.main_T2 import VitRGTS

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
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            loss = criterion(outputs, labels)
            loss.backward()

            clip_grad_norm_(model.parameters(), 0.05)
            optimizer.step()
            scheduler.step()

            batch_loss = loss.item()
            _, predicted = outputs.max(1)
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
                    outputs = outputs[0]
                loss = criterion(outputs, labels)


                batch_loss = loss.item()
                _, predicted = outputs.max(1)
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
num_epochs = (len(train_dataset)) // len(train_dataset)+1


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



# ----------------- Begin coordinate probing helper functions -----------------

class CoordRegressor(nn.Module):
    """Simple MLP regressor predicting normalized (x,y) coordinates from token features."""
    def __init__(self, in_dim, hidden=512):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden//2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden//2, 2)  # predict (x_norm, y_norm) in [0,1]
        )

    def forward(self, x):
        return self.net(x)


def extract_patch_tokens_and_coords(model, loader, device):
    """
    Forward all images through model (in eval), collect patch tokens and their true patch coordinates.
    Returns:
      tokens: (N_total_tokens, dim) Tensor
      coords: (N_total_tokens, 2) Tensor with normalized coords in [0,1] -- (row_norm, col_norm)
    NOTE: excludes register tokens and CLS.
    """
    model.eval()
    all_tokens = []
    all_coords = []
    with torch.no_grad():
        for imgs, _ in tqdm(loader, desc="Extracting tokens for probe"):
            imgs = imgs.to(device)
            # model must support returning tokens:
            out = model(imgs, return_tokens=True)
            if isinstance(out, tuple) and len(out) == 2:
                logits, patch_tokens = out
            else:
                raise RuntimeError("model(img, return_tokens=True) must return (logits, patch_tokens)")

            # patch_tokens shape: (b, num_patches, dim)
            b, n_patches, dim = patch_tokens.shape

            # compute grid coords (row, col) for patches (0..H-1, 0..W-1)
            # We need the patch grid dims. Compute sqrt if square grid:
            p = int(n_patches ** 0.5)
            if p * p != n_patches:
                # fallback: try to infer from pos_embedding shape if available
                try:
                    pe = model.pos_embedding
                    p = int(pe.shape[0] ** 0.5)
                except Exception:
                    raise RuntimeError("Cannot infer patch grid size. Ensure num_patches is square.")
            # make grid coords normalized to [0,1]
            rows = torch.arange(p, dtype=torch.float32, device=device) / (p - 1)
            cols = torch.arange(p, dtype=torch.float32, device=device) / (p - 1)
            rr, cc = torch.meshgrid(rows, cols, indexing='ij')  # shape (p, p)
            coords_grid = torch.stack([rr.flatten(), cc.flatten()], dim=1)  # (n_patches, 2)

            # repeat for batch
            coords_b = coords_grid.unsqueeze(0).expand(b, -1, -1)  # (b, n_patches, 2)

            all_tokens.append(patch_tokens.cpu())
            all_coords.append(coords_b.cpu())

    tokens = torch.cat(all_tokens, dim=0)  # (total_images, n_patches, dim)
    coords = torch.cat(all_coords, dim=0)  # (total_images, n_patches, 2)

    # flatten across images and patches
    B, N, D = tokens.shape
    tokens = tokens.view(B * N, D)
    coords = coords.view(B * N, 2)

    return tokens, coords


def split_high_normal(tokens, threshold_rule='layerwise'):
    """
    tokens: (M, D) Tensor in CPU
    threshold_rule: currently uses mean + 3*std across the given token set.
    Returns two index masks (high_mask, normal_mask) boolean tensors of length M.
    """
    norms = torch.norm(tokens, dim=1)
    mu = norms.mean()
    std = norms.std()
    threshold = mu + 3.0 * std
    high_mask = norms > threshold
    normal_mask = ~high_mask
    return high_mask, normal_mask, threshold.item(), mu.item()


def train_probe_regressor(X_train, y_train, X_val, y_val, in_dim, device,
                          lr=1e-3, epochs=10, batch_size=1024):
    """
    Train CoordRegressor on X_train->y_train. Returns model and validation metrics.
    X_*: torch tensors (num_samples, dim)
    y_*: torch tensors (num_samples, 2) normalized coords
    """
    model = CoordRegressor(in_dim).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.MSELoss()

    # create simple dataloaders
    train_ds = torch.utils.data.TensorDataset(X_train, y_train)
    val_ds = torch.utils.data.TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    for ep in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(device); yb = yb.to(device)
            pred = model(xb)
            loss = crit(pred, yb)
            opt.zero_grad(); loss.backward(); opt.step()

    # compute errors on validation
    model.eval()
    with torch.no_grad():
        preds = []
        trues = []
        for xb, yb in val_loader:
            xb = xb.to(device)
            pred = model(xb).cpu()
            preds.append(pred)
            trues.append(yb)
        preds = torch.cat(preds, dim=0)
        trues = torch.cat(trues, dim=0)

        # mean L2 error (in normalized coordinate space)
        l2_error = torch.norm(preds - trues, dim=1).mean().item()

        # discrete accuracy: round predicted coords to patch grid indices and compare
        # We'll compute p by assuming coords normalized on [0,1] and number of patches p = round(sqrt(N_patches));
        # But we can compute accuracy within tolerance: count predictions within tolerance radius eps (in normalized units).
        eps = 1.0 /  (int((X_train.shape[0] / 1)**0.5) * 1.0 + 0.0001)  # not used exactly — we'll use a fixed eps
        # Simpler: report percentage of predictions within 1/(p-1) distance of true (i.e., within 1 patch)
        # We'll compute p from unique grid distances:
        # compute distance threshold as 1/(p-1) where p = round(sqrt(#patches))
        # But we don't pass #patches here; caller can compute accurate discrete accuracy if desired.
        # For robust metric, we'll compute fraction within absolute normalized distance 1/(p-1)
        # Let's estimate p:
        # estimate_p = int(round((X_train.shape[0]) ** 0.5))  # not reliable here
        # instead, compute a tolerant threshold = 0.07 (approx for 14x14 grid -> 1/(14-1)=~0.077)
        tol = 0.09
        within_tol = (torch.norm(preds - trues, dim=1) <= tol).float().mean().item()

    return model, l2_error, within_tol


def run_coordinate_probe(model, val_loader, device, probe_epochs=10, sample_frac=1.0):
    """
    Full pipeline: extract tokens+coords from val_loader, split into high/normal tokens,
    train small regressors for each set, print results.
    """
    print("→ Extracting tokens and coordinates from validation set...")
    tokens, coords = extract_patch_tokens_and_coords(model, val_loader, device)

    print(f"Collected tokens: {tokens.shape}, coords: {coords.shape}")
    # optionally subsample to speed up
    if sample_frac < 1.0:
        idx = torch.randperm(tokens.shape[0])[: int(tokens.shape[0] * sample_frac)]
        tokens = tokens[idx]
        coords = coords[idx]

    # split into high-norm and normal using mean+3std
    high_mask, normal_mask, thr, mu = split_high_normal(tokens)
    print(f"Threshold (mean + 3*std): {thr:.4f} (mean={mu:.4f}). High token count: {high_mask.sum().item()}, normal count: {normal_mask.sum().item()}")

    device_cpu = torch.device("cpu")
    # Convert to float32
    tokens = tokens.float()
    coords = coords.float()

    # Create train/val splits for each group (80/20)
    def make_splits(mask):
        idxs = torch.nonzero(mask).squeeze(1)
        if idxs.numel() == 0:
            return None
        perm = torch.randperm(idxs.numel())
        tr = idxs[perm[: int(0.8 * idxs.numel())]]
        va = idxs[perm[int(0.8 * idxs.numel()):]]
        return tr, va

    high_splits = make_splits(high_mask)
    normal_splits = make_splits(normal_mask)

    in_dim = tokens.shape[1]

    results = {}

    if high_splits is not None:
        tr_idx, va_idx = high_splits
        X_tr, y_tr = tokens[tr_idx], coords[tr_idx]
        X_va, y_va = tokens[va_idx], coords[va_idx]
        print("Training regressor on HIGH-NORM tokens (count:", tr_idx.numel(), ")")
        _, l2_err, within = train_probe_regressor(X_tr, y_tr, X_va, y_va, in_dim, device, epochs=probe_epochs)
        results['high'] = {'l2_error': l2_err, 'within_tol': within}
    else:
        print("No high-norm tokens found to train regressor.")

    if normal_splits is not None:
        tr_idx, va_idx = normal_splits
        X_tr, y_tr = tokens[tr_idx], coords[tr_idx]
        X_va, y_va = tokens[va_idx], coords[va_idx]
        print("Training regressor on NORMAL tokens (count:", tr_idx.numel(), ")")
        _, l2_err2, within2 = train_probe_regressor(X_tr, y_tr, X_va, y_va, in_dim, device, epochs=probe_epochs)
        results['normal'] = {'l2_error': l2_err2, 'within_tol': within2}
    else:
        print("No normal tokens found to train regressor.")

    # print summary
    print("\n=== Coordinate probing results ===")
    for k, v in results.items():
        print(f" {k.upper():6s} | Mean L2 error (normalized): {v['l2_error']:.4f} | within_tol (tol~0.09): {v['within_tol']*100:.2f}%")
    print("==================================\n")

    return results

# ----------------- End coordinate probing helper functions -----------------




if __name__ == "__main__":
    torch.save(model.state_dict(), "mega_vit_model.pth")
    print("Model state_dict saved to 'mega_vit_model.pth'")


    print("Extracting and saving validation patch tokens...")
    tokens, coords = extract_patch_tokens_and_coords(model, val_loader, device)
    torch.save({'tokens': tokens, 'coords': coords}, "val_tokens.pt")
    print("✅ Saved validation tokens to 'val_tokens.pt'")

    print("Running coordinate probing on validation set...")
    probe_results = run_coordinate_probe(model, val_loader, device, probe_epochs=10, sample_frac=1.0)

