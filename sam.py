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
DEEPGLOBE_TEST_IMG = "/home/jayadeepj/Desktop/Urbanlens/data/test/855_sat.jpg" 
# DeepGlobe masks usually share the same ID but end in _mask.png
DEEPGLOBE_TEST_MASK = DEEPGLOBE_TEST_IMG.replace("_sat.jpg", "_mask.png") 

SWIN_CKPT = "checkpoints/Albumented_swin_gpu_gid_latest.pth"
SAM_CKPT = "sam_vit_h_4b8939.pth"

# Target classes for SAM refinement (Water = 13, 14, 15)
TARGET_CLASSES = [13, 14, 15] 

# --- GID-15 16 CLASSES & COLORS ---
CLASS_NAMES = {
    0: "Background/Unlabeled", 1: "Industrial", 2: "Urban Res.", 3: "Rural Res.", 4: "Traffic Land",
    5: "Paddy Field", 6: "Irrigated Land", 7: "Dry Cropland", 8: "Garden Plot", 9: "Arbor Woodland",
    10: "Shrub Land", 11: "Natural Grass", 12: "Art. Grassland", 13: "River", 14: "Lake", 15: "Pond"
}

COLORS = [
    "#000000", "#555555", "#cc0000", "#ff6666", "#999999", 
    "#ffcc00", "#ffff99", "#d2b48c", "#ff99cc", "#006600", 
    "#66aa00", "#b3ff66", "#99ff99", "#0000ff", "#0099ff", "#00ffff" 
]

cmap = ListedColormap(COLORS)
norm = BoundaryNorm(range(len(COLORS) + 1), cmap.N)

# Same transform used during training
transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

def show_mask(mask, ax, color=np.array([30/255, 144/255, 255/255, 0.6])):
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

    print("[2] Processing DeepGlobe Image & GT...")
    raw_img_pil = Image.open(DEEPGLOBE_TEST_IMG).convert("RGB").resize((224, 224))
    raw_img_np = np.array(raw_img_pil)
    
    # Load DeepGlobe GT Mask if it exists
    if os.path.exists(DEEPGLOBE_TEST_MASK):
        gt_img_pil = Image.open(DEEPGLOBE_TEST_MASK).convert("RGB").resize((224, 224))
        gt_img_np = np.array(gt_img_pil)
    else:
        gt_img_np = np.zeros_like(raw_img_np) # Blank if not found
        print(f"Warning: GT mask not found at {DEEPGLOBE_TEST_MASK}")

    input_tensor = transform(image=raw_img_np)["image"].unsqueeze(0).to(device)

    print("[3] Swin-Unet Semantic Prediction...")
    with torch.no_grad():
        swin_out = swin(input_tensor)
    
    swin_pred = torch.argmax(swin_out[0], dim=0).cpu().numpy()

    # =========================================================================
    # IMAGE 1: SEMANTIC COMPARISON (Zero-Shot DeepGlobe)
    # =========================================================================
    print("[4] Generating Slide 1: Semantic Comparison...")
    fig1, axes1 = plt.subplots(1, 3, figsize=(22, 7))
    fig1.suptitle("Zero-Shot Semantic Inference on DeepGlobe Dataset", fontsize=22, fontweight='bold', y=1.02)

    # Panel 1: Raw Image
    axes1[0].imshow(raw_img_np)
    axes1[0].set_title("1. Raw DeepGlobe Image", fontsize=16, fontweight='bold', pad=15)
    axes1[0].axis('off')

    # Panel 2: DeepGlobe GT
    axes1[1].imshow(gt_img_np)
    axes1[1].set_title("2. DeepGlobe Ground Truth (7 Classes)", fontsize=16, fontweight='bold', pad=15)
    axes1[1].axis('off')

    # Panel 3: Swin-Unet Prediction
    axes1[2].imshow(swin_pred, cmap=cmap, norm=norm)
    axes1[2].set_title("3. Swin-Unet Prediction (16 GID Classes)", fontsize=16, fontweight='bold', pad=15)
    axes1[2].axis('off')

    # Add massive 16-class GID legend
    handles1 = [mpatches.Patch(color=COLORS[i], label=f"{i}: {CLASS_NAMES[i]}") for i in range(16)]
    fig1.legend(handles=handles1, loc="lower center", ncol=8, fontsize=12, bbox_to_anchor=(0.5, -0.1))
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.2) 
    plt.savefig("results/DeepGlobe_Slide1_Semantic.png", dpi=300, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # IMAGE 2: SAM BOUNDARY REFINEMENT
    # =========================================================================
    # Create a binary mask of just the target classes (e.g., Water)
    target_mask = np.isin(swin_pred, TARGET_CLASSES).astype(np.uint8)

    if np.sum(target_mask) == 0:
        print("\nSwin-Unet did not find the target class (Water) in this image. Skipping SAM slide.")
        return

    print("[5] Extracting Bounding Box & Running SAM...")
    contours, _ = cv2.findContours(target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)
    input_box = np.array([x, y, x+w, y+h])

    sam_predictor.set_image(raw_img_np)
    masks, _, _ = sam_predictor.predict(
        point_coords=None, point_labels=None, box=input_box[None, :], multimask_output=False,
    )
    sam_mask = masks[0]

    print("[6] Generating Slide 2: SAM Refinement...")
    fig2, axes2 = plt.subplots(1, 3, figsize=(22, 7))
    fig2.suptitle("Automated Boundary Refinement via Swin-Unet & SAM", fontsize=22, fontweight='bold', y=1.02)

    axes2[0].imshow(raw_img_np)
    axes2[0].set_title("1. Raw DeepGlobe Image", fontsize=16, fontweight='bold', pad=15)
    axes2[0].axis('off')

    axes2[1].imshow(raw_img_np)
    show_mask(target_mask, axes2[1], color=np.array([1, 0, 0, 0.5])) # Red mask
    show_box(input_box, axes2[1])
    axes2[1].set_title("2. Swin-Unet Spatial Intel (Water)", fontsize=16, fontweight='bold', pad=15)
    axes2[1].axis('off')

    axes2[2].imshow(raw_img_np)
    show_mask(sam_mask, axes2[2], color=np.array([0, 1, 0, 0.5])) # Green mask
    axes2[2].set_title("3. SAM Zero-Shot Refinement", fontsize=16, fontweight='bold', pad=15)
    axes2[2].axis('off')

    red_patch = mpatches.Patch(color=(1, 0, 0, 0.5), label='Swin-Unet Prediction (Class 13/14/15)')
    green_patch = mpatches.Patch(color=(0, 1, 0, 0.5), label='SAM Pixel-Perfect Boundary')
    box_patch = mpatches.Patch(edgecolor='green', facecolor='none', linewidth=3, label='Swin-Unet Bounding Box Prompt')
    fig2.legend(handles=[red_patch, box_patch, green_patch], loc="lower center", ncol=3, fontsize=14, bbox_to_anchor=(0.5, -0.05))

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.15)
    plt.savefig("results/DeepGlobe_Slide2_SAM.png", dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\nSUCCESS! Both slides saved to the 'results' folder.")

if __name__ == "__main__":
    run_generalization()