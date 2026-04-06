from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import torch
import numpy as np
from PIL import Image
import io
import base64
import albumentations as A
from albumentations.pytorch import ToTensorV2

from models import SwinUNet
from utils import calculate_gaf

COLORS = np.array([
    [0,0,0], [85,85,85], [204,0,0], [255,102,102], [153,153,153], 
    [255,204,0], [255,255,153], [210,180,140], [255,153,204], [0,102,0], 
    [102,170,0], [179,255,102], [153,255,153], [0,0,255], [0,153,255], [0,255,255]
], dtype=np.uint8)

app = FastAPI(title="GAF Calculator API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Loading Swin-Unet...")
model = SwinUNet().to(device)
model.load_state_dict(torch.load("checkpoints/Albumented_swin_gpu_gid_latest.pth", map_location=device)["state_dict"])
model.eval()

transform = A.Compose([
    A.Resize(224, 224),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

@app.post("/predict_gaf")
async def predict_gaf(file: UploadFile = File(...)):
    # 1. Read the image 
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB").resize((224, 224))
    image_np = np.array(image)

    # 2. Run Inference
    input_tensor = transform(image=image_np)["image"].unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(input_tensor)
    
    # 3. Calculate metrics
    pred_mask = torch.argmax(out[0], dim=0).cpu().numpy()
    gaf_score = float(calculate_gaf(out))

    # 4. Convert the math array back into a colorful Base64 Image string
    color_mask = COLORS[pred_mask]
    mask_pil = Image.fromarray(color_mask)
    
    buffered = io.BytesIO()
    mask_pil.save(buffered, format="PNG")
    mask_base64 = base64.b64encode(buffered.getvalue()).decode("utf-8")

    # 5. Send JSON back to the UI
    return {
        "gaf_score": round(gaf_score, 4),
        "predicted_mask_base64": mask_base64
    }