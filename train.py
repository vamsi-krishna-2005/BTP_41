import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import UrbanLensDataset
from models import UNet, SwinUNet
from utils import get_miou, calculate_gaf, save_checkpoint, load_checkpoint
import pandas as pd
import argparse
import os
import albumentations as A
from albumentations.pytorch import ToTensorV2

torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
torch.cuda.empty_cache()

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="swin")
parser.add_argument("--resume", action="store_true", default=False)
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print(f"Device name: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")

# --- PATHS ---
TRAIN_DIR = "/home/jayadeepj/Desktop/Urbanlens/data/train"
VAL_DIR = "/home/jayadeepj/Desktop/Urbanlens/data/valid"

EPOCHS = 30
BATCH_SIZE = 1
ACCUMULATION_STEPS = 8   # effective batch = 8
num_workers = 0
pin_memory = True

# BATCH_SIZE = 4 # Efficient for H100
use_cuda = torch.cuda.is_available()
use_amp = use_cuda

scaler = torch.amp.GradScaler("cuda") if use_amp else None


train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.VerticalFlip(p=0.5),
    A.RandomRotate90(p=0.5),

    A.ShiftScaleRotate(
        shift_limit=0.05,
        scale_limit=0.1,
        rotate_limit=15,
        p=0.5
    ),

    A.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.05,
        p=0.5
    ),

    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),

    ToTensorV2()
])

val_transform = A.Compose([
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])


def run_train():
    model = SwinUNet().to(device) if args.model == "swin" else UNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    train_loader = DataLoader(UrbanLensDataset(TRAIN_DIR, transform=train_transform), batch_size=BATCH_SIZE, shuffle=True, num_workers=0 if not use_cuda else 2, pin_memory=use_cuda, persistent_workers=use_cuda)
    val_loader = DataLoader(UrbanLensDataset(VAL_DIR, transform=val_transform), batch_size=BATCH_SIZE, num_workers=0 if not use_cuda else 2, pin_memory=use_cuda, persistent_workers=use_cuda)

    start_epoch = load_checkpoint(model, optimizer, args.model, args.resume)
    history = []

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        t_loss, t_iou = 0, 0
        for imgs, masks, _ in train_loader:
            imgs, masks = imgs.to(device, non_blocking=True), masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with torch.amp.autocast("cuda"):
                    out = model(imgs)
                    loss = criterion(out, masks)
                    loss = loss/ACCUMULATION_STEPS
                scaler.scale(loss).backward()
                if (epoch + 1) % ACCUMULATION_STEPS == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
            else:
                out = model(imgs)
                loss = criterion(out, masks)
                loss.backward()
                optimizer.step()
            t_loss += loss.item()
            t_iou += get_miou(out, masks)

        model.eval()
        v_loss, v_iou, v_gaf = 0, 0, 0
        with torch.no_grad():
            for imgs, masks, _ in val_loader:
                imgs, masks = imgs.to(device, non_blocking=True), masks.to(device, non_blocking=True)
                if use_amp:
                    with torch.amp.autocast("cuda"):
                        out = model(imgs)
                        v_loss += criterion(out, masks).item()
                        v_iou += get_miou(out, masks)
                        v_gaf += calculate_gaf(out)
                else:
                    out = model(imgs)
                    v_loss += criterion(out, masks).item()
                    v_iou += get_miou(out, masks)
                    v_gaf += calculate_gaf(out)


        metrics = {
            "epoch": epoch + 1,
            "train_loss": t_loss/len(train_loader),
            "val_loss": v_loss/len(val_loader),
            "train_mIoU": t_iou/len(train_loader),
            "val_mIoU": v_iou/len(val_loader),
            "avg_gaf": v_gaf/len(val_loader)
        }
        history.append(metrics)
        print(f"[{args.model}] Ep {epoch+1} | T-Loss: {metrics['train_loss']:.4f} | V-mIoU: {metrics['val_mIoU']:.4f} | GAF: {metrics['avg_gaf']:.4f}")
        
        save_checkpoint({'epoch': epoch+1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, args.model)
        pd.DataFrame(history).to_csv(f"results/Albumented_{args.model}_metrics.csv", index=False)

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    run_train()