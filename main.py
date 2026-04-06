from fastapi import FastAPI, File, UploadFile, Form
from fastapi.middleware.cors import CORSMiddleware
import torch
import numpy as np
import cv2
from PIL import Image
import io
import base64
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Import your models
from models import SwinUNet
from utils import calculate_gaf
from segment_anything import sam_model_registry, SamPredictor

app = FastAPI(title="UrbanLens BTP API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- GID-15 Colors ---
COLORS = np.array([
    [0,0,0], [85,85,85], [204,0,0], [255,102,102], [153,153,153], 
    [255,204,0], [255,255,153], [210,180,140], [255,153,204], [0,102,0], 
    [102,170,0], [179,255,102], [153,255,153], [0,0,255], [0,153,255], [0,255,255]
], dtype=np.uint8)

# --- Feature Mapping ---
FEATURE_MAP = {
    "Water": [13, 14, 15],
    "Shrub Land": [10],
    "Urban Res.": [2],
    "Agriculture": [5, 6, 7]
}

# --- LOAD MODELS GLOBALLY ---
print("Loading Swin-Unet...")
swin = SwinUNet().to(device)
swin.load_state_dict(torch.load("checkpoints/Albumented_swin_gpu_gid_latest.pth", map_location=device, weights_only=False)["state_dict"])
swin.eval()

print("Loading SAM (This takes a few seconds)...")
sam = sam_model_registry["vit_h"](checkpoint="sam_vit_h_4b8939.pth").to(device)
sam_predictor = SamPredictor(sam)
print("Models Loaded!")

transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

def image_to_base64(img_array):
    """Helper to convert numpy arrays to base64 strings for the UI"""
    pil_img = Image.fromarray(img_array.astype('uint8'))
    buffered = io.BytesIO()
    pil_img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode("utf-8")

@app.get("/")
def read_root():
    return {"message": "UrbanLens API is LIVE! Send POST requests to /predict"}

@app.post("/predict")
async def predict(file: UploadFile = File(...), target_feature: str = Form("Water")):
    # 1. Read Image
    contents = await file.read()
    raw_img_pil = Image.open(io.BytesIO(contents)).convert("RGB").resize((224, 224))
    raw_img_np = np.array(raw_img_pil)

    # 2. Swin-Unet Inference & GAF
    input_tensor = transform(image=raw_img_np)["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        swin_out = swin(input_tensor)
    
    swin_pred = torch.argmax(swin_out[0], dim=0).cpu().numpy()
    gaf_score = float(calculate_gaf(swin_out))
    swin_color_mask = COLORS[swin_pred]

    # 3. SAM Generalization 
    target_classes = FEATURE_MAP.get(target_feature, [13, 14, 15])
    target_mask = np.isin(swin_pred, target_classes).astype(np.uint8)
    
    sam_mask_b64 = None
    if np.sum(target_mask) > 0:
        # Find Bounding Box
        contours, _ = cv2.findContours(target_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        input_box = np.array([x, y, x+w, y+h])

        # Run SAM
        sam_predictor.set_image(raw_img_np)
        masks, _, _ = sam_predictor.predict(
            point_coords=None, point_labels=None, box=input_box[None, :], multimask_output=False
        )
        # Create a Green Mask overlay for the UI
        sam_overlay = raw_img_np.copy()
        sam_overlay[masks[0]] = sam_overlay[masks[0]] * 0.5 + np.array([0, 255, 0]) * 0.5
        sam_mask_b64 = image_to_base64(sam_overlay)

    return {
        "gaf_score": round(gaf_score, 4),
        "swin_mask_base64": image_to_base64(swin_color_mask),
        "sam_mask_base64": sam_mask_b64,
        "message": f"SAM successfully isolated {target_feature}!" if sam_mask_b64 else f"Target '{target_feature}' not found in image."
    }