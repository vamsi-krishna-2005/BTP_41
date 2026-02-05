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

# ---------------- SETUP ----------------
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="swin")
parser.add_argument("--resume", action="store_true")
args = parser.parse_args()

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

RUN_TAG = f"{args.model}_{'gpu' if use_cuda else 'cpu'}"
CSV_PATH = f"results/Albumented_{RUN_TAG}_metrics.csv"

TRAIN_DIR = "/home/jayadeepj/Desktop/Urbanlens/data/train"
VAL_DIR = "/home/jayadeepj/Desktop/Urbanlens/data/valid"

EPOCHS = 30
BATCH_SIZE = 2
ACCUMULATION_STEPS = 8  # effective batch = 16

# ---------------- AUGMENTATIONS ----------------
train_transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(
            shift_limit=0.05,
            scale_limit=0.1,
            rotate_limit=15,
            interpolation=0,
            border_mode=0,
            p=0.5,
        ),
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ],
    additional_targets={"mask": "mask"},
)

val_transform = A.Compose(
    [
        A.Normalize(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ],
    additional_targets={"mask": "mask"},
)

# ---------------- TRAIN ----------------
def run_train():
    model = SwinUNet() if args.model == "swin" else UNet()
    model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler(enabled=use_cuda)

    train_loader = DataLoader(
        UrbanLensDataset(TRAIN_DIR, transform=train_transform),
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=2 if use_cuda else 0,
        pin_memory=use_cuda,
    )

    val_loader = DataLoader(
        UrbanLensDataset(VAL_DIR, transform=val_transform),
        batch_size=BATCH_SIZE,
        num_workers=2 if use_cuda else 0,
        pin_memory=use_cuda,
    )

    start_epoch = load_checkpoint(model, optimizer, RUN_TAG, args.resume)
    history = []

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        optimizer.zero_grad(set_to_none=True)

        t_loss, t_iou = 0, 0

        for step, (imgs, masks, _) in enumerate(train_loader):
            imgs, masks = imgs.to(device), masks.to(device)

            with torch.cuda.amp.autocast(enabled=use_cuda):
                out = model(imgs)
                loss = criterion(out, masks) / ACCUMULATION_STEPS

            scaler.scale(loss).backward()

            if (step + 1) % ACCUMULATION_STEPS == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

            t_loss += loss.item() * ACCUMULATION_STEPS
            t_iou += get_miou(out, masks)

        model.eval()
        v_loss, v_iou, v_gaf = 0, 0, 0

        with torch.no_grad():
            for imgs, masks, _ in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                with torch.cuda.amp.autocast(enabled=use_cuda):
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
        print(f"[{RUN_TAG}] Ep {epoch+1} |T-Loss: {metrics['train_loss']:.4f} | V-mIoU: {metrics['val_mIoU']:.4f} | V-GAF: {metrics['avg_gaf']:.4f}")

        save_checkpoint(
            {"epoch": epoch + 1,
             "state_dict": model.state_dict(),
             "optimizer": optimizer.state_dict()},
            RUN_TAG,
        )

        pd.DataFrame(history).to_csv(CSV_PATH, index=False)


if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    run_train()
