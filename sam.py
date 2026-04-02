import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm

# Import your custom model
from models import SwinUNet
from segment_anything import sam_model_registry, SamPredictor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- PATHS ---
DEEPGLOBE_TEST_IMG = "/home/jayadeepj/Desktop/Urbanlens/data/test/848780_sat.jpg" 
DEEPGLOBE_TEST_MASK = DEEPGLOBE_TEST_IMG.replace("_sat.jpg", "_mask.png") 

SWIN_CKPT = "checkpoints/Albumented_swin_gpu_gid_latest.pth"
SAM_CKPT = "sam_vit_h_4b8939.pth"

# Hardcoded for Water (GID-15 Classes: River=13, Lake=14, Pond=15)
TARGET_CLASSES = [5, 6, 7] 

# ==========================================
# GID-15 CLASSES & COLORS (For Swin-Unet)
# ==========================================
GID_CLASSES = {
    0: "Background", 1: "Industrial", 2: "Urban Res.", 3: "Rural Res.", 4: "Traffic Land",
    5: "Paddy Field", 6: "Irrigated Land", 7: "Dry Cropland", 8: "Garden Plot", 9: "Arbor Woodland",
    10: "Shrub Land", 11: "Natural Grass", 12: "Art. Grassland", 13: "River", 14: "Lake", 15: "Pond"
}

GID_COLORS = [
    "#000000", "#555555", "#cc0000", "#ff6666", "#999999", 
    "#ffcc00", "#ffff99", "#d2b48c", "#ff99cc", "#006600", 
    "#66aa00", "#b3ff66", "#99ff99", "#0000ff", "#0099ff", "#00ffff" 
]

gid_cmap = ListedColormap(GID_COLORS)
gid_norm = BoundaryNorm(range(len(GID_COLORS) + 1), gid_cmap.N)

# ==========================================
# DEEPGLOBE CLASSES & COLORS (For Ground Truth)
# Standard DeepGlobe RGB map
# ==========================================
DEEPGLOBE_CLASSES = {
    "Urban": "#00FFFF",       # Cyan
    "Agriculture": "#FFFF00", # Yellow
    "Rangeland": "#FF00FF",   # Magenta
    "Forest": "#00FF00",      # Green
    "Water": "#0000FF",       # Blue
    "Barren": "#FFFFFF",      # White
    "Unknown": "#000000"      # Black
}

transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

def show_mask(mask, ax, color):
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0,0,0,0), lw=3))

def run_generalization():
    print("[1] Loading Models...")
    swin = SwinUNet().to(device)
    swin.load_state_dict(torch.load(SWIN_CKPT, map_location=device)["state_dict"])
    swin.eval()

    sam = sam_model_registry["vit_h"](checkpoint=SAM_CKPT).to(device)
    sam_predictor = SamPredictor(sam)

    print("[2] Processing Images...")
    raw_img_pil = Image.open(DEEPGLOBE_TEST_IMG).convert("RGB").resize((224, 224))
    raw_img_np = np.array(raw_img_pil)
    
    # Load DeepGlobe GT Mask as RGB Image
    if os.path.exists(DEEPGLOBE_TEST_MASK):
        gt_img_np = np.array(Image.open(DEEPGLOBE_TEST_MASK).convert("RGB").resize((224, 224)))
    else:
        gt_img_np = np.zeros_like(raw_img_np)

    input_tensor = transform(image=raw_img_np)["image"].unsqueeze(0).to(device)

    print("[3] Swin-Unet Prediction...")
    with torch.no_grad():
        swin_out = swin(input_tensor)
    swin_pred = torch.argmax(swin_out[0], dim=0).cpu().numpy()

    # Isolate the target class (Water)
    target_mask = np.isin(swin_pred, TARGET_CLASSES).astype(np.uint8)

    if np.sum(target_mask) == 0:
        print("\n[!] Swin-Unet could not find Agriculture in this image. Try an image with agricultural land!")
        return

    print("[4] Extracting Bounding Box & Running SAM...")
    contours, _ = cv2.findContours(target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    input_box = np.array([x, y, x+w, y+h])

    sam_predictor.set_image(raw_img_np)
    masks, _, _ = sam_predictor.predict(
        point_coords=None, point_labels=None, box=input_box[None, :], multimask_output=False,
    )
    sam_mask = masks[0]

    print("[5] Generating the 2x3 Storyboard...")
    fig, axes = plt.subplots(2, 3, figsize=(24, 15))
    fig.suptitle("DeepGlobe Generalization & Boundary Refinement (Target: Agriculture)", fontsize=26, fontweight='bold', y=0.98)

    # --- ROW 1: THE DATA & SEMANTICS ---
    axes[0, 0].imshow(raw_img_np)
    axes[0, 0].set_title("1. Raw DeepGlobe Image", fontsize=18, fontweight='bold', pad=15)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(gt_img_np)
    axes[0, 1].set_title("2. DeepGlobe Ground Truth", fontsize=18, fontweight='bold', pad=15)
    axes[0, 1].axis('off')

    axes[0, 2].imshow(swin_pred, cmap=gid_cmap, norm=gid_norm)
    axes[0, 2].set_title("3. Full Swin-Unet Prediction", fontsize=18, fontweight='bold', pad=15)
    axes[0, 2].axis('off')

    # --- ROW 2: THE AUTOMATION PIPELINE ---
    axes[1, 0].imshow(target_mask, cmap='gray')
    axes[1, 0].set_title("4. Isolated Prediction (Water Only)", fontsize=18, fontweight='bold', pad=15)
    axes[1, 0].axis('off')

    axes[1, 1].imshow(raw_img_np)
    show_mask(target_mask, axes[1, 1], color=np.array([1, 0, 0, 0.5])) # Red Mask
    show_box(input_box, axes[1, 1])
    axes[1, 1].set_title("5. Swin-Unet Spatial Prompt (Bounding Box)", fontsize=18, fontweight='bold', pad=15)
    axes[1, 1].axis('off')

    axes[1, 2].imshow(raw_img_np)
    show_mask(sam_mask, axes[1, 2], color=np.array([0, 1, 0, 0.5])) # Green Mask
    axes[1, 2].set_title("6. Final SAM Pixel-Perfect Mask", fontsize=18, fontweight='bold', pad=15)
    axes[1, 2].axis('off')

    # ==========================================
    # ADDING THE SPECIFIC LEGENDS
    # ==========================================
    
    # 1. DeepGlobe Legend (Placed under Panel 2)
    dg_handles = [mpatches.Patch(color=color, label=label) for label, color in DEEPGLOBE_CLASSES.items()]
    axes[0, 1].legend(handles=dg_handles, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4, fontsize=12, title="DeepGlobe Classes (7)", title_fontsize=14)

    # 2. GID-15 Legend (Placed under Panel 3)
    gid_handles = [mpatches.Patch(color=GID_COLORS[i], label=f"{i}: {GID_CLASSES[i]}") for i in range(16)]
    axes[0, 2].legend(handles=gid_handles, loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=4, fontsize=10, title="GID-15 Classes (16)", title_fontsize=14)

    # 3. Pipeline Legend (Placed under Row 2)
    red_patch = mpatches.Patch(color=(1, 0, 0, 0.5), label='Swin-Unet Rough Location')
    box_patch = mpatches.Patch(edgecolor='green', facecolor='none', linewidth=3, label='Automated SAM Prompt')
    green_patch = mpatches.Patch(color=(0, 1, 0, 0.5), label='SAM Refined Boundary')
    fig.legend(handles=[red_patch, box_patch, green_patch], loc="lower center", ncol=3, fontsize=16, bbox_to_anchor=(0.5, 0.02))

    plt.tight_layout(h_pad=8) # Added extra padding so legends don't overlap images
    plt.savefig(f"results/DeepGlobe_Storyboard_Agriculture.png", dpi=300, bbox_inches='tight')
    plt.close()
    print("Done! Check out results/DeepGlobe_Storyboard_Agriculture.png")

if __name__ == "__main__":
    run_generalization()