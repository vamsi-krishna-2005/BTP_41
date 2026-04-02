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

# Import SAM
from segment_anything import sam_model_registry, SamPredictor

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- PATHS ---
DEEPGLOBE_TEST_IMG = "/home/jayadeepj/Desktop/Urbanlens/data/test/926551_sat.jpg" 
DEEPGLOBE_TEST_MASK = DEEPGLOBE_TEST_IMG.replace("_sat.jpg", "_mask.png") 

SWIN_CKPT = "checkpoints/Albumented_swin_gpu_gid_latest.pth"
SAM_CKPT = "sam_vit_h_4b8939.pth"

# =========================================================================
# THE DYNAMIC FEATURE SELECTOR (Change this to test different things!)
# Options: "Water", "Shrub Land", "Urban Res.", "Arbor Woodland", etc.
# =========================================================================
TARGET_FEATURE = "Shrub Land" 

# --- GID-15 CLASSES & COLORS ---
CLASS_NAMES = {
    0: "Background/Unlabeled", 1: "Industrial", 2: "Urban Res.", 3: "Rural Res.", 4: "Traffic Land",
    5: "Paddy Field", 6: "Irrigated Land", 7: "Dry Cropland", 8: "Garden Plot", 9: "Arbor Woodland",
    10: "Shrub Land", 11: "Natural Grass", 12: "Art. Grassland", 13: "River", 14: "Lake", 15: "Pond"
}

# Map features to their GID-15 IDs
FEATURE_MAP = {
    "Water": [13, 14, 15],
    "Shrub Land": [10],
    "Urban Res.": [2],
    "Arbor Woodland": [9],
    "Agriculture": [5, 6, 7]
}

TARGET_CLASSES = FEATURE_MAP.get(TARGET_FEATURE, [10]) # Default to Shrub if not found

COLORS = [
    "#000000", "#555555", "#cc0000", "#ff6666", "#999999", 
    "#ffcc00", "#ffff99", "#d2b48c", "#ff99cc", "#006600", 
    "#66aa00", "#b3ff66", "#99ff99", "#0000ff", "#0099ff", "#00ffff" 
]

cmap = ListedColormap(COLORS)
norm = BoundaryNorm(range(len(COLORS) + 1), cmap.N)

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
    print(f"[1] Hunting for '{TARGET_FEATURE}' in DeepGlobe image...")
    swin = SwinUNet().to(device)
    swin.load_state_dict(torch.load(SWIN_CKPT, map_location=device)["state_dict"])
    swin.eval()

    sam = sam_model_registry["vit_h"](checkpoint=SAM_CKPT).to(device)
    sam_predictor = SamPredictor(sam)

    raw_img_pil = Image.open(DEEPGLOBE_TEST_IMG).convert("RGB").resize((224, 224))
    raw_img_np = np.array(raw_img_pil)
    
    if os.path.exists(DEEPGLOBE_TEST_MASK):
        gt_img_np = np.array(Image.open(DEEPGLOBE_TEST_MASK).convert("RGB").resize((224, 224)))
    else:
        gt_img_np = np.zeros_like(raw_img_np)

    input_tensor = transform(image=raw_img_np)["image"].unsqueeze(0).to(device)

    with torch.no_grad():
        swin_out = swin(input_tensor)
    swin_pred = torch.argmax(swin_out[0], dim=0).cpu().numpy()

    # Isolate the target class
    target_mask = np.isin(swin_pred, TARGET_CLASSES).astype(np.uint8)

    if np.sum(target_mask) == 0:
        print(f"\n[!] Swin-Unet could not find any '{TARGET_FEATURE}' in this image. Try another image or feature!")
        return

    print("[2] Extracting Bounding Box & Running SAM...")
    contours, _ = cv2.findContours(target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    input_box = np.array([x, y, x+w, y+h])

    sam_predictor.set_image(raw_img_np)
    masks, _, _ = sam_predictor.predict(
        point_coords=None, point_labels=None, box=input_box[None, :], multimask_output=False,
    )
    sam_mask = masks[0]

    print("[3] Generating the Ultimate 2x3 Storyboard...")
    fig, axes = plt.subplots(2, 3, figsize=(24, 14))
    fig.suptitle(f"Zero-Shot Feature Extraction & Boundary Refinement: Focus on '{TARGET_FEATURE}'", fontsize=26, fontweight='bold', y=1.03)

    # --- ROW 1: The Context ---
    axes[0, 0].imshow(raw_img_np)
    axes[0, 0].set_title("1. Raw DeepGlobe Image", fontsize=18, fontweight='bold', pad=15)
    axes[0, 0].axis('off')

    axes[0, 1].imshow(gt_img_np)
    axes[0, 1].set_title("2. DeepGlobe GT (7 Broad Classes)", fontsize=18, fontweight='bold', pad=15)
    axes[0, 1].axis('off')

    axes[0, 2].imshow(swin_pred, cmap=cmap, norm=norm)
    axes[0, 2].set_title("3. Full Swin-Unet Prediction (16 GID Classes)", fontsize=18, fontweight='bold', pad=15)
    axes[0, 2].axis('off')

    # --- ROW 2: The Target Isolation ---
    # Show isolated binary mask (Black/White)
    axes[1, 0].imshow(target_mask, cmap='gray')
    axes[1, 0].set_title(f"4. Isolated AI Prediction: {TARGET_FEATURE}", fontsize=18, fontweight='bold', pad=15)
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

    # --- LEGENDS ---
    handles1 = [mpatches.Patch(color=COLORS[i], label=f"{i}: {CLASS_NAMES[i]}") for i in range(16)]
    fig.legend(handles=handles1, loc="lower center", ncol=8, fontsize=12, bbox_to_anchor=(0.5, 0.48)) # Middle legend for Row 1

    red_patch = mpatches.Patch(color=(1, 0, 0, 0.5), label=f'Swin-Unet Rough Location')
    box_patch = mpatches.Patch(edgecolor='green', facecolor='none', linewidth=3, label='Automated SAM Prompt')
    green_patch = mpatches.Patch(color=(0, 1, 0, 0.5), label='SAM Refined Boundary')
    fig.legend(handles=[red_patch, box_patch, green_patch], loc="lower center", ncol=3, fontsize=16, bbox_to_anchor=(0.5, -0.02)) # Bottom legend for Row 2

    plt.tight_layout(h_pad=6) 
    plt.savefig(f"results/Ultimate_Storyboard_{TARGET_FEATURE.replace(' ', '_')}.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Done! Check out results/Ultimate_Storyboard_{TARGET_FEATURE.replace(' ', '_')}.png")

if __name__ == "__main__":
    run_generalization()