"""
Baseline Transfer Learning — EfficientNet-B0
Document page classification with two-stage transfer learning.
Run: python3 run_baseline.py
"""
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from torchvision.models import EfficientNet_B0_Weights
from PIL import Image
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import copy, time, random, sys

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else
                       "mps" if hasattr(torch.backends, "mps") and torch.backends.mps.is_available() else
                       "cpu")
print(f"Device: {device}")
sys.stdout.flush()

# ==============================================================
# STEP 1 — Load Dataset
# ==============================================================
IMAGE_DIR = Path("pages_raw")

train_df = pd.read_csv("train_doc_split.csv")
val_df   = pd.read_csv("val_doc_split.csv")
test_df  = pd.read_csv("test_doc_split.csv")

classes = sorted(train_df["label"].unique())
label2idx = {c: i for i, c in enumerate(classes)}
idx2label = {i: c for c, i in label2idx.items()}

print(f"Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
print(f"Classes ({len(classes)}): {classes}")
sys.stdout.flush()


def pad_to_square(img):
    w, h = img.size
    max_dim = max(w, h)
    new_img = Image.new("RGB", (max_dim, max_dim), (255, 255, 255))
    new_img.paste(img, ((max_dim - w) // 2, (max_dim - h) // 2))
    return new_img


class DocPageDataset(Dataset):
    def __init__(self, df, image_dir, transform):
        self.records = df.reset_index(drop=True)
        self.image_dir = Path(image_dir)
        self.transform = transform

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        row = self.records.iloc[idx]
        img = Image.open(self.image_dir / row["image_name"]).convert("RGB")
        img = pad_to_square(img)
        img = self.transform(img)
        label = label2idx[row["label"]]
        return img, label


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomRotation(2),
    transforms.ColorJitter(brightness=0.1, contrast=0.1),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

eval_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
])

BATCH_SIZE = 32

train_ds = DocPageDataset(train_df, IMAGE_DIR, train_transform)
val_ds   = DocPageDataset(val_df,   IMAGE_DIR, eval_transform)
test_ds  = DocPageDataset(test_df,  IMAGE_DIR, eval_transform)

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  num_workers=0, drop_last=False)
val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

print(f"Batches — train: {len(train_loader)}, val: {len(val_loader)}, test: {len(test_loader)}")
sys.stdout.flush()

# ==============================================================
# STEP 2 — Handle Class Imbalance
# ==============================================================
class_counts = train_df["label"].value_counts().reindex(classes).values.astype(float)
inv_freq = 1.0 / class_counts
class_weights = inv_freq / inv_freq.sum() * len(classes)
class_weights_t = torch.FloatTensor(class_weights).to(device)

print("\nClass weights (inverse-frequency, normalized):")
for c, w in zip(classes, class_weights):
    print(f"  {c:<35} {w:.4f}")
sys.stdout.flush()

# ==============================================================
# STEP 3 — Model Setup & Training
# ==============================================================
model = models.efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1)
in_features = model.classifier[1].in_features
model.classifier = nn.Sequential(
    nn.Dropout(0.3),
    nn.Linear(in_features, len(classes)),
)
model = model.to(device)

total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"\nTotal params: {total_params:,}")
print(f"Trainable:    {trainable_params:,}")
sys.stdout.flush()

criterion = nn.CrossEntropyLoss(weight=class_weights_t)


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    for imgs, labels in loader:
        imgs, labels = imgs.to(device), labels.to(device)
        optimizer.zero_grad()
        loss = criterion(model(imgs), labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * imgs.size(0)
    return running_loss / len(loader.dataset)


def evaluate_epoch(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            running_loss += criterion(out, labels).item() * imgs.size(0)
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    n = len(loader.dataset)
    avg_loss = running_loss / n
    acc = accuracy_score(all_labels, all_preds)
    f1m = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return avg_loss, acc, f1m


# ---- Stage 1: frozen backbone ----
for p in model.features.parameters():
    p.requires_grad = False

optimizer1 = optim.AdamW(model.classifier.parameters(), lr=1e-3, weight_decay=1e-4)
history = {"epoch": [], "stage": [], "train_loss": [], "val_loss": [], "val_acc": [], "val_f1": []}

print("\n" + "=" * 70)
print("Stage 1 — Frozen backbone, training classifier head (5 epochs)")
print("=" * 70)
print(f"{'Epoch':>5} {'Train Loss':>11} {'Val Loss':>11} {'Val Acc':>9} {'Val F1':>9}")
sys.stdout.flush()

for epoch in range(1, 6):
    t0 = time.time()
    tr_loss = train_one_epoch(model, train_loader, criterion, optimizer1)
    vl_loss, vl_acc, vl_f1 = evaluate_epoch(model, val_loader, criterion)
    elapsed = time.time() - t0
    history["epoch"].append(epoch)
    history["stage"].append(1)
    history["train_loss"].append(tr_loss)
    history["val_loss"].append(vl_loss)
    history["val_acc"].append(vl_acc)
    history["val_f1"].append(vl_f1)
    print(f"{epoch:>5} {tr_loss:>11.4f} {vl_loss:>11.4f} {vl_acc:>8.1%} {vl_f1:>8.4f}  ({elapsed:.0f}s)")
    sys.stdout.flush()

print("Stage 1 complete.")
sys.stdout.flush()

# ---- Stage 2: full fine-tuning with discriminative LR ----
for p in model.features.parameters():
    p.requires_grad = True

optimizer2 = optim.AdamW([
    {"params": model.features.parameters(), "lr": 1e-4},
    {"params": model.classifier.parameters(), "lr": 1e-3},
], weight_decay=1e-4)

scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer2, T_max=30)

PATIENCE = 7
MAX_EPOCHS = 30
best_val_f1 = max(history["val_f1"])
best_model_state = copy.deepcopy(model.state_dict())
best_epoch = history["epoch"][int(np.argmax(history["val_f1"]))]
best_val_loss = min(history["val_loss"])
patience_counter = 0

print("\n" + "=" * 70)
print(f"Stage 2 — Full fine-tuning (up to {MAX_EPOCHS} epochs, patience={PATIENCE})")
print("=" * 70)
print(f"{'Epoch':>5} {'Train Loss':>11} {'Val Loss':>11} {'Val Acc':>9} {'Val F1':>9}")
sys.stdout.flush()

for epoch in range(1, MAX_EPOCHS + 1):
    global_epoch = 5 + epoch
    t0 = time.time()
    tr_loss = train_one_epoch(model, train_loader, criterion, optimizer2)
    vl_loss, vl_acc, vl_f1 = evaluate_epoch(model, val_loader, criterion)
    scheduler.step()
    elapsed = time.time() - t0

    history["epoch"].append(global_epoch)
    history["stage"].append(2)
    history["train_loss"].append(tr_loss)
    history["val_loss"].append(vl_loss)
    history["val_acc"].append(vl_acc)
    history["val_f1"].append(vl_f1)

    note = ""
    if vl_f1 > best_val_f1:
        best_val_f1 = vl_f1
        best_model_state = copy.deepcopy(model.state_dict())
        best_epoch = global_epoch
        note += " *best_f1*"

    if vl_loss < best_val_loss:
        best_val_loss = vl_loss
        patience_counter = 0
        note += " +loss"
    else:
        patience_counter += 1

    print(f"{global_epoch:>5} {tr_loss:>11.4f} {vl_loss:>11.4f} {vl_acc:>8.1%} {vl_f1:>8.4f}  ({elapsed:.0f}s){note}")
    sys.stdout.flush()

    if patience_counter >= PATIENCE:
        print(f"\nEarly stopping at epoch {global_epoch} (patience={PATIENCE}).")
        break

model.load_state_dict(best_model_state)
print(f"\nBest model from epoch {best_epoch} (val macro-F1 = {best_val_f1:.4f})")
torch.save(best_model_state, "baseline_efficientnet_b0.pth")
print("Model saved to baseline_efficientnet_b0.pth")
sys.stdout.flush()

# ---- Training curves ----
fig, axes = plt.subplots(1, 3, figsize=(16, 4))
epochs = history["epoch"]
stage1_end = 5

axes[0].plot(epochs, history["train_loss"], label="Train")
axes[0].plot(epochs, history["val_loss"], label="Val")
axes[0].axvline(stage1_end, color="gray", ls="--", alpha=0.5, label="Stage 1→2")
axes[0].set_title("Loss"); axes[0].set_xlabel("Epoch"); axes[0].legend()

axes[1].plot(epochs, history["val_acc"])
axes[1].axvline(stage1_end, color="gray", ls="--", alpha=0.5)
axes[1].set_title("Val Accuracy"); axes[1].set_xlabel("Epoch")

axes[2].plot(epochs, history["val_f1"])
axes[2].axvline(stage1_end, color="gray", ls="--", alpha=0.5)
axes[2].set_title("Val Macro F1"); axes[2].set_xlabel("Epoch")

plt.tight_layout()
Path("baseline_visuals").mkdir(exist_ok=True)
plt.savefig("baseline_visuals/training_curves.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved training_curves.png")
sys.stdout.flush()

# ==============================================================
# STEP 4 — Baseline Evaluation
# ==============================================================
def full_evaluation(model, loader, split_name):
    model.eval()
    all_preds, all_labels, all_confs = [], [], []
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            out = model(imgs)
            probs = torch.softmax(out, dim=1)
            confs, preds = probs.max(dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_confs.extend(confs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_confs = np.array(all_confs)

    acc = accuracy_score(all_labels, all_preds)
    f1_mac = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    f1_wt  = f1_score(all_labels, all_preds, average="weighted", zero_division=0)
    report = classification_report(all_labels, all_preds,
                                   target_names=classes, zero_division=0)
    cm = confusion_matrix(all_labels, all_preds)

    print(f"\n{'='*60}")
    print(f"  {split_name.upper()} SET EVALUATION")
    print(f"{'='*60}")
    print(f"  Accuracy:    {acc:.4f}")
    print(f"  Macro F1:    {f1_mac:.4f}")
    print(f"  Weighted F1: {f1_wt:.4f}")
    print(f"\n{report}")
    sys.stdout.flush()

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=classes, yticklabels=classes, ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion Matrix — {split_name.capitalize()}")
    plt.tight_layout()
    cm_path = f"baseline_visuals/cm_{split_name}.png"
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()

    return {
        "accuracy": acc, "macro_f1": f1_mac, "weighted_f1": f1_wt,
        "report": report, "cm": cm, "cm_path": cm_path,
        "preds": all_preds, "labels": all_labels, "confs": all_confs,
    }


val_results  = full_evaluation(model, val_loader, "validation")
test_results = full_evaluation(model, test_loader, "test")

# ==============================================================
# STEP 5 — Visual Prediction Analysis & Report
# ==============================================================
val_image_names = val_df["image_name"].values
preds  = val_results["preds"]
labels = val_results["labels"]
confs  = val_results["confs"]

correct_idx = np.where(preds == labels)[0]
wrong_idx   = np.where(preds != labels)[0]

rng = np.random.RandomState(42)
n_correct = min(10, len(correct_idx))
n_wrong   = min(10, len(wrong_idx))
sample_correct = rng.choice(correct_idx, n_correct, replace=False) if n_correct else []
sample_wrong   = rng.choice(wrong_idx,   n_wrong,   replace=False) if n_wrong else []

print(f"\nCorrect predictions available: {len(correct_idx)} — sampled {n_correct}")
print(f"Wrong   predictions available: {len(wrong_idx)} — sampled {n_wrong}")
sys.stdout.flush()


def save_prediction_grid(indices, title, filename):
    if len(indices) == 0:
        print(f"No samples for '{title}'")
        return
    n = len(indices)
    cols = min(5, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 5 * rows))
    if rows == 1 and cols == 1:
        axes = np.array([axes])
    axes = np.atleast_2d(axes)

    for pos, idx in enumerate(indices):
        r, c = divmod(pos, cols)
        ax = axes[r, c]
        img = Image.open(IMAGE_DIR / val_image_names[idx]).convert("RGB")
        ax.imshow(img)
        true_lbl = idx2label[labels[idx]]
        pred_lbl = idx2label[preds[idx]]
        conf = confs[idx]
        color = "green" if true_lbl == pred_lbl else "red"
        ax.set_title(f"{val_image_names[idx]}\nTrue: {true_lbl}\nPred: {pred_lbl}\nConf: {conf:.1%}",
                     fontsize=8, color=color)
        ax.axis("off")

    for pos in range(len(indices), rows * cols):
        r, c = divmod(pos, cols)
        axes[r, c].axis("off")

    fig.suptitle(title, fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {filename}")
    sys.stdout.flush()


save_prediction_grid(sample_correct,
                     "Correct Predictions (Validation)",
                     "baseline_visuals/correct_predictions.png")

save_prediction_grid(sample_wrong,
                     "Misclassified Samples (Validation)",
                     "baseline_visuals/misclassified_samples.png")

# ==============================================================
# Generate Markdown Report
# ==============================================================
lines = []
lines.append("# Baseline Transfer Learning Report")
lines.append("")
lines.append("---")
lines.append("")

lines.append("## 1. Training Summary")
lines.append("")
lines.append("| Parameter | Value |")
lines.append("|-----------|-------|")
lines.append("| Model | EfficientNet-B0 (ImageNet pretrained) |")
lines.append(f"| Total parameters | {total_params:,} |")
lines.append("| Input size | 224 x 224 |")
lines.append("| Preprocessing | Pad to square (white), resize, ImageNet normalize |")
lines.append("| Train augmentations | Rotation +/-2 deg, Brightness +/-10%, Contrast +/-10% |")
lines.append(f"| Batch size | {BATCH_SIZE} |")
lines.append("| Loss | Weighted CrossEntropyLoss (inverse-frequency) |")
lines.append("| Stage 1 | Frozen backbone, 5 epochs, LR=1e-3 |")
lines.append("| Stage 2 | Full fine-tune, backbone LR=1e-4, head LR=1e-3, CosineAnnealing |")
lines.append(f"| Early stopping | Patience={PATIENCE} on val loss |")
lines.append(f"| Best epoch | {best_epoch} |")
lines.append(f"| Best val macro F1 | {best_val_f1:.4f} |")
lines.append(f"| Total epochs trained | {history['epoch'][-1]} |")
lines.append("")
lines.append("### Training Curves")
lines.append("")
lines.append("![Training Curves](baseline_visuals/training_curves.png)")
lines.append("")

lines.append("## 2. Validation Metrics")
lines.append("")
lines.append("| Metric | Value |")
lines.append("|--------|------:|")
lines.append(f"| Accuracy | {val_results['accuracy']:.4f} |")
lines.append(f"| Macro F1 | {val_results['macro_f1']:.4f} |")
lines.append(f"| Weighted F1 | {val_results['weighted_f1']:.4f} |")
lines.append("")
lines.append("### Classification Report (Validation)")
lines.append("")
lines.append("```")
lines.append(val_results["report"].strip())
lines.append("```")
lines.append("")
lines.append("### Confusion Matrix (Validation)")
lines.append("")
lines.append(f"![Confusion Matrix — Validation]({val_results['cm_path']})")
lines.append("")

lines.append("## 3. Test Metrics")
lines.append("")
lines.append("| Metric | Value |")
lines.append("|--------|------:|")
lines.append(f"| Accuracy | {test_results['accuracy']:.4f} |")
lines.append(f"| Macro F1 | {test_results['macro_f1']:.4f} |")
lines.append(f"| Weighted F1 | {test_results['weighted_f1']:.4f} |")
lines.append("")
lines.append("### Classification Report (Test)")
lines.append("")
lines.append("```")
lines.append(test_results["report"].strip())
lines.append("```")
lines.append("")
lines.append("### Confusion Matrix (Test)")
lines.append("")
lines.append(f"![Confusion Matrix — Test]({test_results['cm_path']})")
lines.append("")

lines.append("## 4. Visual Evaluation")
lines.append("")
lines.append("### Correct Predictions")
lines.append("")
lines.append("![Correct Predictions](baseline_visuals/correct_predictions.png)")
lines.append("")
if n_correct > 0:
    lines.append("| # | Image | True Label | Predicted | Confidence |")
    lines.append("|---|-------|-----------|-----------|-----------|")
    for i, idx in enumerate(sample_correct, 1):
        lines.append(f"| {i} | {val_image_names[idx]} | {idx2label[labels[idx]]} | {idx2label[preds[idx]]} | {confs[idx]:.1%} |")
    lines.append("")

lines.append("### Misclassified Samples")
lines.append("")
if n_wrong > 0:
    lines.append("![Misclassified Samples](baseline_visuals/misclassified_samples.png)")
    lines.append("")
    lines.append("| # | Image | True Label | Predicted | Confidence |")
    lines.append("|---|-------|-----------|-----------|-----------|")
    for i, idx in enumerate(sample_wrong, 1):
        lines.append(f"| {i} | {val_image_names[idx]} | {idx2label[labels[idx]]} | {idx2label[preds[idx]]} | {confs[idx]:.1%} |")
    lines.append("")
else:
    lines.append("No misclassified samples in the validation set.")
    lines.append("")

lines.append("---")
lines.append("")
lines.append("*Report generated by BASELINE_TRAINING — no dataset modifications made.*")

report_path = Path("BASELINE_TRANSFER_LEARNING_REPORT.md")
report_path.write_text("\n".join(lines), encoding="utf-8")
print(f"\nReport saved to {report_path}")

print(f"\n{'='*60}")
print(f"  FINAL SUMMARY")
print(f"{'='*60}")
print(f"  Model:       EfficientNet-B0")
print(f"  Best epoch:  {best_epoch}")
print(f"  Val  — Acc: {val_results['accuracy']:.4f}  Macro-F1: {val_results['macro_f1']:.4f}  Weighted-F1: {val_results['weighted_f1']:.4f}")
print(f"  Test — Acc: {test_results['accuracy']:.4f}  Macro-F1: {test_results['macro_f1']:.4f}  Weighted-F1: {test_results['weighted_f1']:.4f}")
print(f"{'='*60}")
print("\nDONE.")
