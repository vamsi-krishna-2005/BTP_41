import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import UrbanLensDataset
from models import UNet, SwinUNet
from utils import get_miou, calculate_gaf, save_checkpoint, load_checkpoint
import pandas as pd
import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="swin")
args = parser.parse_args()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- PATHS ---
TRAIN_DIR = "/home/jayadeepj/Desktop/Urbanlens/data/train"
VAL_DIR = "/home/jayadeepj/Desktop/Urbanlens/data/valid"

EPOCHS = 30
BATCH_SIZE = 16 # Efficient for H100

def run_train():
    model = SwinUNet().to(device) if args.model == "swin" else UNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    train_loader = DataLoader(UrbanLensDataset(TRAIN_DIR), batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=False)
    val_loader = DataLoader(UrbanLensDataset(VAL_DIR), batch_size=BATCH_SIZE, num_workers=0, pin_memory=False)

    start_epoch = load_checkpoint(model, optimizer, args.model)
    history = []

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        t_loss, t_iou = 0, 0
        for imgs, masks, _ in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
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
                imgs, masks = imgs.to(device), masks.to(device)
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
        pd.DataFrame(history).to_csv(f"results/{args.model}_metrics.csv", index=False)

if __name__ == "__main__":
    os.makedirs("results", exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)
    run_train()