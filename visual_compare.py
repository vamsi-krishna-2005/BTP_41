import torch
import matplotlib.pyplot as plt
from dataset import UrbanLensDataset
from models import UNet, SwinUNet
from train import VAL_DIR
from utils import calculate_gaf, get_miou

device = torch.device("cuda")
TEST_DIR = "/home/jayadeepj/Desktop/Urbanlens/data/test"

def generate_report(idx=0):
    ds = UrbanLensDataset(TEST_DIR)
    img, mask, name = ds[idx]
    
    # Load Models
    unet = UNet().to(device)
    swin = SwinUNet().to(device)
    
    unet.load_state_dict(torch.load("checkpoints/unet_latest.pth")['state_dict'])
    swin.load_state_dict(torch.load("checkpoints/swin_latest.pth")['state_dict'])
    
    unet.eval(); swin.eval()
    with torch.no_grad():
        inp = img.unsqueeze(0).to(device)
        out_u = unet(inp)
        out_s = swin(inp)
        
    # Metrics
    gaf_gt = calculate_gaf(mask)
    gaf_u = calculate_gaf(out_u[0])
    gaf_s = calculate_gaf(out_s[0])
    
    iou_u = get_miou(out_u, mask.unsqueeze(0).to(device))
    iou_s = get_miou(out_s, mask.unsqueeze(0).to(device))

    # Visualization
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 6))
    axes[0].imshow(img.permute(1,2,0)); axes[0].set_title(f"Input: {name}")
    axes[1].imshow(mask, cmap='tab10'); axes[1].set_title(f"Ground Truth\nGAF: {gaf_gt:.2f}")
    
    pred_u = torch.argmax(out_u[0], 0).cpu()
    axes[2].imshow(pred_u, cmap='tab10'); axes[2].set_title(f"U-Net Baseline\nmIoU: {iou_u:.2f} | GAF: {gaf_u:.2f}")
    
    pred_s = torch.argmax(out_s[0], 0).cpu()
    axes[3].imshow(pred_s, cmap='tab10'); axes[3].set_title(f"Swin-Unet\nmIoU: {iou_s:.2f} | GAF: {gaf_s:.2f}")
    
    plt.tight_layout()
    plt.savefig(f"results/final_comparison_{idx}.png")
    print(f"Report saved for {name}")

if __name__ == "__main__":
    for i in range(5): # Generate 5 sample comparisons
        generate_report(i)