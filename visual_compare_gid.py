import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
from new_dataset import GIDDataset
from models import UNet, SwinUNet
from utils import calculate_gaf, get_miou
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import random
import os

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- NEW GID-15 PATHS ---
TEST_IMG_DIR = "/home/jayadeepj/Desktop/Urbanlens/gid_dataset/preprocessed_224/test_images"
TEST_MASK_DIR = "/home/jayadeepj/Desktop/Urbanlens/gid_dataset/preprocessed_224/test_masks"

# --- 16 CLASSES FOR GID-15 ---
CLASS_NAMES = {
    0: "Background/Unlabeled", 1: "Industrial", 2: "Urban Res.", 3: "Rural Res.", 4: "Traffic Land",
    5: "Paddy Field", 6: "Irrigated Land", 7: "Dry Cropland", 8: "Garden Plot", 9: "Arbor Woodland",
    10: "Shrub Land", 11: "Natural Grass", 12: "Art. Grassland", 13: "River", 14: "Lake", 15: "Pond"
}

# 16 Distinct Colors
COLORS = [
    "#000000", "#555555", "#cc0000", "#ff6666", "#999999", # 0-4 (Urban/Background)
    "#ffcc00", "#ffff99", "#d2b48c", "#ff99cc", "#006600", # 5-9 (Agri/Trees)
    "#66aa00", "#b3ff66", "#99ff99", "#0000ff", "#0099ff", "#00ffff" # 10-15 (Grass/Water)
]

cmap = ListedColormap(COLORS)
norm = BoundaryNorm(range(len(COLORS) + 1), cmap.N)

# Must use the exact validation transform used in training
val_transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

def generate_report(idx=0):
    # Load Dataset
    ds = GIDDataset(TEST_IMG_DIR, TEST_MASK_DIR, transform=val_transform)
    
    # Also load the raw image for plotting and resize to match model input/mask size
    raw_img = Image.open(os.path.join(TEST_IMG_DIR, ds.images[idx]))
    raw_img = raw_img.resize((224, 224), Image.BILINEAR)
    
    img_tensor, mask, name = ds[idx]

    # Initialize Models (Ensure n_classes=16 in models.py!)
    unet = UNet().to(device)
    swin = SwinUNet().to(device)

    # Load New GID Checkpoints
    ckpt_unet = torch.load("checkpoints/Albumented_unet_gpu_gid_latest.pth", map_location=device)
    ckpt_swin = torch.load("checkpoints/Albumented_swin_gpu_gid_latest.pth", map_location=device)

    unet.load_state_dict(ckpt_unet["state_dict"])
    swin.load_state_dict(ckpt_swin["state_dict"])

    unet.eval()
    swin.eval()

    with torch.no_grad():
        inp = img_tensor.unsqueeze(0).to(device)
        out_u = unet(inp)
        out_s = swin(inp)

    # Metrics Calculation
    gaf_gt = calculate_gaf(mask)
    gaf_u = calculate_gaf(out_u)
    gaf_s = calculate_gaf(out_s)

    # Note: mask needs to be on device for mIoU
    iou_u = get_miou(out_u, mask.unsqueeze(0).to(device))
    iou_s = get_miou(out_s, mask.unsqueeze(0).to(device))

    # --- Visualization ---
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    
    axes[0].imshow(raw_img)
    axes[0].set_title(f"Input: {name}")

    axes[1].imshow(mask.cpu().numpy(), cmap=cmap, norm=norm)
    axes[1].set_title(f"Ground Truth\nGAF: {gaf_gt:.4f}")

    pred_u = torch.argmax(out_u[0], dim=0).cpu().numpy()
    axes[2].imshow(pred_u, cmap=cmap, norm=norm)
    axes[2].set_title(f"U-Net\nmIoU: {iou_u:.3f} | GAF: {gaf_u:.4f}")

    pred_s = torch.argmax(out_s[0], dim=0).cpu().numpy()
    axes[3].imshow(pred_s, cmap=cmap, norm=norm)
    axes[3].set_title(f"Swin-UNet\nmIoU: {iou_s:.3f} | GAF: {gaf_s:.4f}")

    # Fixed legend with all 16 classes for consistent presentation
    handles = [mpatches.Patch(color=COLORS[c], label=f"{c}: {CLASS_NAMES[c]}") for c in range(16)]

    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.23))
    plt.tight_layout()
    plt.savefig(f"results/GID_comparison_{idx}.png", bbox_inches='tight')
    plt.close()
    print(f"Report saved for {name}")

if __name__ == "__main__":
    # Get actual number of test samples dynamically
    test_images = [f for f in os.listdir(TEST_IMG_DIR) if f.endswith(('.png', '.jpg', '.tif'))]
    num_samples = len(test_images)
    
    if num_samples == 0:
        print(f"Error: No test images found in {TEST_IMG_DIR}")
       
    else:
        print(f"Found {num_samples} test samples\n")
        random.seed()  
        
        for i in range(5): 
            random_idx = random.randint(0, num_samples - 1)
            print(f"Sample {i+1}/5 - Testing random image at index {random_idx}...")
            generate_report(random_idx)