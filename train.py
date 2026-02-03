import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset import DeepGlobeDataset
from models import UNet, SwinUNet
from utils import get_miou, calculate_gaf, save_checkpoint, load_checkpoint
import pandas as pd

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
EPOCHS = 30
BATCH_SIZE = 8

def run_train(model_type='swin'):
    model = SwinUNet().to(device) if model_type == 'swin' else UNet().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    # LOAD DATA (Update these paths)
    train_ds = DeepGlobeDataset("data/train/images", "data/train/masks")
    val_ds = DeepGlobeDataset("data/val/images", "data/val/masks")
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE)

    start_epoch = load_checkpoint(model, optimizer, model_type)
    history = []

    for epoch in range(start_epoch, EPOCHS):
        model.train()
        train_loss, train_iou = 0, 0
        for imgs, masks, _ in train_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            train_iou += get_miou(out, masks)

        # Validation
        model.eval()
        val_loss, val_iou, val_gaf = 0, 0, 0
        with torch.no_grad():
            for imgs, masks, _ in val_loader:
                imgs, masks = imgs.to(device), masks.to(device)
                out = model(imgs)
                val_loss += criterion(out, masks).item()
                val_iou += get_miou(out, masks)
                # Calculate GAF for first image in batch as sample
                val_gaf += calculate_gaf(out[0])

        metrics = {
            "epoch": epoch + 1,
            "train_loss": train_loss/len(train_loader),
            "val_loss": val_loss/len(val_loader),
            "train_mIoU": train_iou/len(train_loader),
            "val_mIoU": val_iou/len(val_loader),
            "avg_gaf": val_gaf/len(val_loader)
        }
        history.append(metrics)
        print(f"Epoch {epoch+1} | Val mIoU: {metrics['val_mIoU']:.4f} | GAF: {metrics['avg_gaf']:.4f}")
        
        save_checkpoint({'epoch': epoch+1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}, model_type)
        pd.DataFrame(history).to_csv(f"results/{model_type}_metrics.csv", index=False)

if __name__ == "__main__":
    run_train('unet')
    run_train('swin')