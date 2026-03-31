import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from new_dataset import GIDDataset
from models import UNet, SwinUNet
from utils import get_miou, calculate_gaf, save_checkpoint, load_checkpoint
import pandas as pd
import argparse
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm  # <--- THIS IS THE MAGIC VISUALIZER

# ---------------- SETUP ----------------
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="swin")
parser.add_argument("--resume", action="store_true")

args = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

RUN_TAG = f"{args.model}_gpu" if use_cuda else f"{args.model}_cpu"
CSV_PATH = f"results/{RUN_TAG}_GID_model.csv"

# --- POINT TO THE NEW PREPROCESSED FOLDERS ---
TRAIN_DIR = "/home/jayadeepj/Desktop/Urbanlens/gid_dataset/preprocessed_224/train_images"
TRAIN_MASK = "/home/jayadeepj/Desktop/Urbanlens/gid_dataset/preprocessed_224/train_masks"
VAL_DIR = "/home/jayadeepj/Desktop/Urbanlens/gid_dataset/preprocessed_224/val_images"
VAL_MASK = "/home/jayadeepj/Desktop/Urbanlens/gid_dataset/preprocessed_224/val_masks"

EPOCHS = 50
BATCH_SIZE = 8
PATIENCE = 3 # Early stopping patience
ACCUMULATION_STEPS = 4 # Acts like Batch Size 32 (8 * 4)

# ---------------- AUGMENTATIONS ----------------
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# ---------------- TRAIN ----------------
def run_train():
    if args.model == "swin":
        model = SwinUNet()
    else:
        model = UNet()

    model.to(device)

    # Increased starting LR so the scheduler has room to reduce it
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3 if args.model == "swin" else 1e-3, weight_decay=1e-4)
    
    # Scheduler: Cuts LR in half if validation loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=1, verbose=True)
    
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(enabled=use_cuda)

    train_loader = DataLoader(
        GIDDataset(TRAIN_DIR, TRAIN_MASK, transform=train_transform),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False
    )

    val_loader = DataLoader(
        GIDDataset(VAL_DIR, VAL_MASK, transform=val_transform),
        batch_size=BATCH_SIZE, num_workers=0, pin_memory=False
    )

    start_epoch = load_checkpoint(model, optimizer, RUN_TAG, args.resume)
    history = []
    
    best_iou = 0.0
    patience_counter = 0

    print(f"Starting training for {args.model.upper()}...")

    for epoch in range(start_epoch, EPOCHS):
        # -------- TRAIN --------
        model.train()
        t_loss, t_iou = 0.0, 0.0

        # Wrap train_loader in tqdm for a progress bar
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1:02d}/{EPOCHS} [Train]")
        
        for i, (imgs, masks, _) in enumerate(train_pbar):
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            with torch.amp.autocast("cuda", enabled=use_cuda):
                out = model(imgs)
                loss = criterion(out, masks)
                # Normalize loss for accumulation
                loss = loss / ACCUMULATION_STEPS 

            scaler.scale(loss).backward()

            # Step optimizer only every ACCUMULATION_STEPS
            if (i + 1) % ACCUMULATION_STEPS == 0 or (i + 1) == len(train_loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            # Un-normalize for logging
            step_loss = loss.item() * ACCUMULATION_STEPS
            t_loss += step_loss
            t_iou += get_miou(out, masks)
            
            # Update the progress bar with the current loss
            train_pbar.set_postfix(loss=f"{step_loss:.4f}")

        # -------- VALID --------
        model.eval()
        v_loss, v_iou, v_gaf = 0.0, 0.0, 0.0

        # Wrap val_loader in tqdm
        val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1:02d}/{EPOCHS} [Valid]")
        
        with torch.no_grad():
            for imgs, masks, _ in val_pbar:
                imgs = imgs.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=use_cuda):
                    out = model(imgs)
                    batch_loss = criterion(out, masks).item()
                    v_loss += batch_loss
                    v_iou += get_miou(out, masks)
                    v_gaf += calculate_gaf(out)
                    
                val_pbar.set_postfix(loss=f"{batch_loss:.4f}")

        # -------- METRICS & SCHEDULER --------
        metrics = {
            "epoch": epoch + 1,
            "train_loss": t_loss / len(train_loader),
            "val_loss": v_loss / len(val_loader),
            "train_mIoU": t_iou / len(train_loader),
            "val_mIoU": v_iou / len(val_loader),
            "avg_gaf": v_gaf / len(val_loader),
        }
        history.append(metrics)

        print(
            f"\n[{RUN_TAG}] Ep {epoch+1:02d} Summary | "
            f"T-Loss: {metrics['train_loss']:.4f} | "
            f"V-mIoU: {metrics['val_mIoU']:.4f} | "
            f"GAF: {metrics['avg_gaf']:.4f}"
        )

        pd.DataFrame(history).to_csv(CSV_PATH, index=False)

        # ---------------- IMPORTANT: STEP THE SCHEDULER ----------------
        scheduler.step(metrics['val_loss'])

        # -------- EARLY STOPPING LOGIC --------
        current_iou = metrics['val_mIoU']
        if current_iou > best_iou:
            best_iou = current_iou
            patience_counter = 0
            save_checkpoint({'epoch': epoch + 1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, RUN_TAG)
            print(f"  -> Best model saved! (mIoU: {best_iou:.4f})\n")
        else:
            patience_counter += 1
            print(f"  -> No improvement. Patience: {patience_counter}/{PATIENCE}\n")

        if patience_counter >= PATIENCE:
            print(f"Early stopping triggered at epoch {epoch+1}! Best mIoU was {best_iou:.4f}")
            break

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    run_train()