import os
import h5py
import numpy as np
import cv2
import matplotlib.pyplot as plt
from glob import glob
from sklearn.model_selection import train_test_split

import albumentations as A
from albumentations.pytorch import ToTensorV2

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import segmentation_models_pytorch as smp

SyntaxError: invalid syntax
DATA_DIR = "/kaggle/input/brats2020-training-data/BraTS2020_training_data"

# Collect all H5 files
all_h5_files = sorted(glob(os.path.join(DATA_DIR, "**", "*.h5"), recursive=True))
print("Total H5 files found:", len(all_h5_files))

# Small subset for ~1–2 hr run
small_train = all_h5_files[:12000]
small_val = all_h5_files[-2000:]

train_files, val_files = train_test_split(small_train, test_size=0.2, random_state=42)

SyntaxError: multiple statements found while compiling a single statement
class BraTSDatasetH5(Dataset):
    def __init__(self, h5_files, transform=None):
        self.h5_files = h5_files
        self.transform = transform

    def __len__(self):
        return len(self.h5_files)

    def __getitem__(self, idx):
        file_path = self.h5_files[idx]
        with h5py.File(file_path, 'r') as f:
            image = np.array(f['image'])
            mask = np.array(f['mask'])

        if image.ndim == 3:
            image = image[:,:,0]
        if mask.ndim == 3:
            mask = mask[:,:,0]

        image = cv2.resize(image, (128, 128))
        mask = cv2.resize(mask, (128, 128), interpolation=cv2.INTER_NEAREST)

        image = np.expand_dims(image, axis=-1)
        mask = np.expand_dims(mask, axis=-1)

        if self.transform:
            aug = self.transform(image=image, mask=mask)
            image = aug['image']
            mask = aug['mask']

        return image, mask

Traceback (most recent call last):
  File "<pyshell#2>", line 1, in <module>
    class BraTSDatasetH5(Dataset):
NameError: name 'Dataset' is not defined

train_transform = A.Compose([
    A.Normalize(mean=0, std=1),
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Normalize(mean=0, std=1),
    ToTensorV2()
])

train_dataset = BraTSDatasetH5(train_files, transform=train_transform)
val_dataset = BraTSDatasetH5(val_files, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=0)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

print("Train batches:", len(train_loader), "Val batches:", len(val_loader))

SyntaxError: multiple statements found while compiling a single statement
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    in_channels=1,
    classes=2  # binary mask
).to(device)

dice_loss = smp.losses.DiceLoss(mode='multiclass')
ce_loss = nn.CrossEntropyLoss()

def loss_fn(pred, target):
    return dice_loss(pred, target) + ce_loss(pred, target)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
scaler = torch.cuda.amp.GradScaler()

SyntaxError: multiple statements found while compiling a single statement
def dice_score(preds, targets, eps=1e-6):
    preds = torch.argmax(preds, dim=1)
    targets = targets.squeeze(1)
    intersection = ((preds == 1) & (targets == 1)).float().sum((1,2))
    union = ((preds == 1) | (targets == 1)).float().sum((1,2))
    return ((2 * intersection + eps) / (union + intersection + eps)).mean().item()

def iou_score(preds, targets, eps=1e-6):
    preds = torch.argmax(preds, dim=1)
    targets = targets.squeeze(1)
    intersection = ((preds == 1) & (targets == 1)).float().sum((1,2))
    union = ((preds == 1) | (targets == 1)).float().sum((1,2))
    return ((intersection + eps) / (union + eps)).mean().item()

SyntaxError: invalid syntax
best_dice = 0.0

def train_fn(loader):
    model.train()
    total_loss = 0
    for batch_idx, (imgs, masks) in enumerate(loader):
        imgs = imgs.float().to(device)
        masks = masks.squeeze(-1).long().to(device)
        optimizer.zero_grad()
        with torch.amp.autocast(device_type='cuda'):
            preds = model(imgs)
            loss = loss_fn(preds, masks)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        total_loss += loss.item()
        if batch_idx % 50 == 0:
            print(f"Train Batch {batch_idx}/{len(loader)} Loss: {loss.item():.4f}")
    return total_loss / len(loader)

def val_fn(loader):
    model.eval()
    total_loss = 0
    total_dice = 0
    total_iou = 0
    with torch.no_grad():
        for imgs, masks in loader:
            imgs = imgs.float().to(device)
            masks = masks.squeeze(-1).long().to(device)
            with torch.amp.autocast(device_type='cuda'):
                preds = model(imgs)
                loss = loss_fn(preds, masks)
            total_loss += loss.item()
            total_dice += dice_score(preds, masks)
            total_iou += iou_score(preds, masks)
    return total_loss / len(loader), total_dice / len(loader), total_iou / len(loader)

EPOCHS = 10
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    train_loss = train_fn(train_loader)
    val_loss, val_dice, val_iou = val_fn(val_loader)
    print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Dice: {val_dice:.4f} | IoU: {val_iou:.4f}")

    import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def visualize_predictions(model, loader, num_samples=3):
    model.eval()
    imgs, masks = next(iter(loader))
    imgs = imgs.float().to(device)

    with torch.no_grad(), torch.amp.autocast(device_type='cuda'):
        preds = model(imgs)

    preds = torch.argmax(preds, dim=1).cpu().numpy()
    masks = masks.squeeze(1).cpu().numpy()
    imgs = imgs.squeeze(1).cpu().numpy()

    # Create colormaps
    true_cmap = ListedColormap(['black', 'green'])
    pred_cmap = ListedColormap(['black', 'red'])

    plt.figure(figsize=(16, num_samples * 4))
    
    for i in range(num_samples):
        # Dice score for sample
        intersection = ((preds[i] == 1) & (masks[i] == 1)).sum()
        union = ((preds[i] == 1) | (masks[i] == 1)).sum()
        dice_sample = (2 * intersection) / (union + intersection + 1e-6)
 
         plt.subplot(num_samples, 4, 4*i + 1)
         plt.imshow(imgs[i], cmap='gray')
         plt.title("MRI")
         plt.axis('off')

         plt.subplot(num_samples, 4, 4*i + 2)
         plt.imshow(masks[i], cmap=true_cmap)
         plt.title("True Mask")
         plt.axis('off')
 
         plt.subplot(num_samples, 4, 4*i + 3)
         plt.imshow(preds[i], cmap=pred_cmap)
         plt.title(f"Predicted Mask\nDice={dice_sample:.2f}")
         plt.axis('off')
 
         plt.subplot(num_samples, 4, 4*i + 4)
         plt.imshow(imgs[i], cmap='gray')
         plt.imshow(preds[i], alpha=0.5, cmap=pred_cmap)
         plt.title("Overlay")
         plt.axis('off')
 
     plt.tight_layout()
     plt.savefig("/kaggle/working/visual_results.png", dpi=150)
     plt.show()

 visualize_predictions(model, val_loader)

 # Evaluate model on a few batches to calculate summary metrics
 model.eval()
 val_loss, val_dice, val_iou = val_fn(val_loader)
 print(f"Validation Dice: {val_dice:.4f}, IoU: {val_iou:.4f}")
 
 def get_visual_samples(model, loader, num_samples=6):
    model.eval()
    imgs, masks = next(iter(loader))
    imgs = imgs.float().to(device)
    with torch.no_grad(), torch.amp.autocast(device_type='cuda'):
        preds = model(imgs)
    preds = torch.argmax(preds, dim=1).cpu().numpy()
    masks = masks.squeeze(1).cpu().numpy()
    imgs = imgs.squeeze(1).cpu().numpy()
    return imgs[:num_samples], masks[:num_samples], preds[:num_samples]

imgs, masks, preds = get_visual_samples(model, val_loader)
SyntaxError: multiple statements found while compiling a single statement
SyntaxError: multiple statements found while compiling a single statement
SyntaxError: invalid syntax
def generate_interpretive_report(model, loader, dice, iou, filename="/kaggle/working/interpretive_report.pdf"):
    model.eval()
    imgs, masks = next(iter(loader))
    imgs = imgs.float().to(device)

    with torch.no_grad(), torch.amp.autocast(device_type='cuda'):
        preds = model(imgs)

    preds = torch.argmax(preds, dim=1).cpu().numpy()
    masks = masks.squeeze(1).cpu().numpy()
    imgs = imgs.squeeze(1).cpu().numpy()

    with PdfPages(filename) as pdf:
        # Page 1: Summary
        plt.figure(figsize=(8,6))
        plt.text(0.1, 0.9, "🧠 Brain Tumor Segmentation Interpretive Report", fontsize=16, fontweight="bold")
        plt.text(0.1, 0.8, f"Validation Dice: {dice:.4f}", fontsize=12)
        plt.text(0.1, 0.75, f"Validation IoU: {iou:.4f}", fontsize=12)
        plt.text(0.1, 0.7, "Model: U-Net (ResNet34 Encoder)", fontsize=12)
        plt.text(0.1, 0.6, "Interpretation: Model successfully detects tumors but may miss smaller regions.", fontsize=10)
        plt.axis("off")
        pdf.savefig()
        plt.close()

        # Pages: Each sample with explanation
        for i in range(5):  # first 5 samples
            plt.figure(figsize=(12,6))
            plt.subplot(1,3,1)
            plt.imshow(imgs[i], cmap='gray')
            plt.title("MRI")
            plt.axis('off')

            plt.subplot(1,3,2)
            plt.imshow(masks[i], cmap='gray')
            plt.title("True Mask")
            plt.axis('off')

            plt.subplot(1,3,3)
            plt.imshow(imgs[i], cmap='gray')
            plt.imshow(preds[i], alpha=0.5, cmap='autumn')
            plt.title("Prediction Overlay")
            plt.axis('off')

            # Interpretation text
            intersection = ((preds[i] == 1) & (masks[i] == 1)).sum()
            union = ((preds[i] == 1) | (masks[i] == 1)).sum()
            dice_sample = (2 * intersection) / (union + intersection + 1e-6)

            if dice_sample > 0.7:
                comment = f"Case {i+1}: Tumor detected with good overlap (Dice={dice_sample:.2f}). Prediction closely matches ground truth."
            elif dice_sample > 0.4:
                comment = f"Case {i+1}: Tumor detected partially (Dice={dice_sample:.2f}). Some boundaries missed."
            else:
                comment = f"Case {i+1}: Tumor detection weak (Dice={dice_sample:.2f}). Model missed significant areas or predicted false positives."

            plt.figtext(0.1, -0.05, comment, wrap=True, fontsize=10)
            pdf.savefig()
            plt.close()

        # Final page: Conclusion
        plt.figure(figsize=(8,6))
        plt.text(0.1, 0.9, "📌 Conclusion & Recommendations", fontsize=14, fontweight="bold")
        plt.text(0.1, 0.8, "✅ Strengths:", fontsize=12, fontweight="bold")
        plt.text(0.1, 0.75, "- Detects major tumor regions with decent overlap", fontsize=10)
        plt.text(0.1, 0.7, "- Works well on medium/large lesions", fontsize=10)

        plt.text(0.1, 0.6, "⚠️ Weaknesses:", fontsize=12, fontweight="bold")
        plt.text(0.1, 0.55, "- May miss small tumors or edges", fontsize=10)
        plt.text(0.1, 0.5, "- Occasional false positives outside tumor area", fontsize=10)

        plt.text(0.1, 0.4, "💡 Recommendations:", fontsize=12, fontweight="bold")
        plt.text(0.1, 0.35, "- Train with more data or augmentations", fontsize=10)
        plt.text(0.1, 0.3, "- Use 3D U-Net for better spatial context", fontsize=10)
        plt.axis("off")
        pdf.savefig()
        plt.close()

    print("✅ Interpretive report saved at:", filename)

# Generate the interpretive PDF
generate_interpretive_report(model, val_loader, val_dice, val_iou)
SyntaxError: invalid syntax
def generate_radiology_report(model, loader, filename="/kaggle/working/mri_radiology_reports.txt"):
    model.eval()
    imgs, masks = next(iter(loader))
    imgs = imgs.float().to(device)

    with torch.no_grad(), torch.amp.autocast(device_type='cuda'):
        preds = model(imgs)

    preds = torch.argmax(preds, dim=1).cpu().numpy()
    masks = masks.squeeze(1).cpu().numpy()
    imgs = imgs.squeeze(1).cpu().numpy()

    with open(filename, "w") as f:
        for i in range(5):  # Generate report for first 5 patients
            tumor_area = np.sum(preds[i] == 1)
            total_area = preds[i].size
            tumor_percentage = (tumor_area / total_area) * 100

            # Location analysis (left vs right hemisphere)
            tumor_coords = np.argwhere(preds[i] == 1)
            if len(tumor_coords) > 0:
                mean_x = np.mean(tumor_coords[:, 1])
                hemisphere = "Right hemisphere" if mean_x > preds[i].shape[1] / 2 else "Left hemisphere"
            else:
                hemisphere = "No significant lesion detected"

            # Size description
            if tumor_percentage < 1:
                size_desc = "Minimal lesion volume"
            elif tumor_percentage < 5:
                size_desc = "Small lesion"
            elif tumor_percentage < 15:
                size_desc = "Moderate-sized lesion"
            else:
                size_desc = "Large lesion occupying significant brain volume"

            # Shape description
            if tumor_area > 0:
                edges = np.count_nonzero(cv2.Canny(preds[i].astype(np.uint8)*255, 100, 200))
                if edges / tumor_area > 0.1:
                    shape_desc = "Irregular margins suggestive of infiltrative growth"
                else:
                    shape_desc = "Well-defined borders suggestive of localized mass"
            else:
                shape_desc = "No abnormal growth pattern observed"

            # Write radiology-style report
            f.write(f"--- Patient {i+1} MRI Report ---\n")
            f.write(f"Findings: {hemisphere}. {size_desc}.\n")
            f.write(f"Tumor Characteristics: {shape_desc}.\n")
            if tumor_area > 0:
                f.write(f"Assessment: Likely primary brain neoplasm. Recommend correlation with clinical history and possible surgical consultation.\n")
            else:
                f.write(f"Assessment: No evidence of significant tumor. Continue routine follow-up.\n")
            f.write("\n")

    print(f"✅ Radiology-style reports saved to: {filename}")

# Generate reports
generate_radiology_report(model, val_loader)
