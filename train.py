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

# ---------------- SETUP ----------------
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="swin")
parser.add_argument("--resume", action="store_true")

args = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

RUN_TAG = f"{args.model}_gpu" if use_cuda else f"{args.model}_cpu"
CSV_PATH = f"results/{RUN_TAG}_GID_model.csv"

TRAIN_DIR = "/home/jayadeepj/Desktop/Urbanlens/gid_dataset/data/data_for_keras_aug/train_images/train"
TRAIN_MASK = "/home/jayadeepj/Desktop/Urbanlens/gid_dataset/data/data_for_keras_aug/train_masks/train"
VAL_DIR = "/home/jayadeepj/Desktop/Urbanlens/gid_dataset/data/data_for_keras_aug/val_images/val"
VAL_MASK = "/home/jayadeepj/Desktop/Urbanlens/gid_dataset/data/data_for_keras_aug/val_masks/val"

EPOCHS = 30
BATCH_SIZE = 8   #NO gradient accumulation now

# ---------------- AUGMENTATIONS ----------------
train_transform = A.Compose([
    A.HorizontalFlip(p=0.5),
    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ),
    ToTensorV2()
])

val_transform = A.Compose([
    A.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225)
    ),
    ToTensorV2()
])

# ---------------- TRAIN ----------------
def run_train():
    if args.model == "swin":
        model = SwinUNet()
    else:
        model = UNet()

    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=3e-5 if args.model == "swin" else 1e-4,
        weight_decay=1e-4
    )

    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(enabled=use_cuda)

    train_loader = DataLoader(
        GIDDataset(TRAIN_DIR, TRAIN_MASK, size=224, transform=train_transform),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False
    )

    val_loader = DataLoader(
        GIDDataset(VAL_DIR, VAL_MASK, size=224, transform=val_transform),
        batch_size=BATCH_SIZE,
        num_workers=0,
        pin_memory=False
    )

    start_epoch = load_checkpoint(model, optimizer, RUN_TAG, args.resume)
    history = []


    for epoch in range(start_epoch, EPOCHS):
        # -------- TRAIN --------
        model.train()
        t_loss, t_iou = 0.0, 0.0

        for imgs, masks, _ in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast("cuda", enabled=use_cuda):
                out = model(imgs)
                loss = criterion(out, masks)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            t_loss += loss.item()
            t_iou += get_miou(out, masks)

        # -------- VALID --------
        model.eval()
        v_loss, v_iou, v_gaf = 0.0, 0.0, 0.0

        with torch.no_grad():
            for imgs, masks, _ in val_loader:
                imgs = imgs.to(device, non_blocking=True)
                masks = masks.to(device, non_blocking=True)

                with torch.amp.autocast("cuda", enabled=use_cuda):
                    out = model(imgs)
                    v_loss += criterion(out, masks).item()
                    v_iou += get_miou(out, masks)
                    v_gaf += calculate_gaf(out)

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
            f"[{RUN_TAG}] Ep {epoch+1:02d} | "
            f"T-Loss: {metrics['train_loss']:.4f} | "
            f"V-mIoU: {metrics['val_mIoU']:.4f} | "
            f"GAF: {metrics['avg_gaf']:.4f}"
        )

        save_checkpoint(
            {
                "epoch": epoch + 1,
                "state_dict": model.state_dict(),
                "optimizer": optimizer.state_dict(),
            },
            RUN_TAG,
        )

        pd.DataFrame(history).to_csv(CSV_PATH, index=False)



if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    run_train()
