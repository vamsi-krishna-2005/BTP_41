import torch
import matplotlib.pyplot as plt
from dataset import DeepGlobeDataset
from models import UNet, SwinUNet
from utils import calculate_gaf, get_miou

device = torch.device("cuda")

def show_comparison(idx=5):
    ds = DeepGlobeDataset("data/val/images", "data/val/masks")
    img, mask, name = ds[idx]
    
    # Load Models
    unet = UNet().to(device); swin = SwinUNet().to(device)
    unet.load_state_dict(torch.load("checkpoints/unet_latest.pth")['state_dict'])
    swin.load_state_dict(torch.load("checkpoints/swin_latest.pth")['state_dict'])
    
    unet.eval(); swin.eval()
    with torch.no_grad():
        inp = img.unsqueeze(0).to(device)
        p_unet = unet(inp)
        p_swin = swin(inp)
        
    gaf_gt = calculate_gaf(mask)
    gaf_u = calculate_gaf(p_unet[0])
    gaf_s = calculate_gaf(p_swin[0])
    
    iou_u = get_miou(p_unet, mask.unsqueeze(0).to(device))
    iou_s = get_miou(p_swin, mask.unsqueeze(0).to(device))

    # Plot
    fig, axes = plt.subplots(1, 4, figsize=(20, 5))
    axes[0].imshow(img.permute(1,2,0)); axes[0].set_title("Input")
    axes[1].imshow(mask, cmap='tab10'); axes[1].set_title(f"GT (GAF: {gaf_gt:.2f})")
    axes[2].imshow(torch.argmax(p_unet[0],0).cpu(), cmap='tab10'); axes[2].set_title(f"U-Net\nmIoU:{iou_u:.2f} GAF:{gaf_u:.2f}")
    axes[3].imshow(torch.argmax(p_swin[0],0).cpu(), cmap='tab10'); axes[3].set_title(f"Swin\nmIoU:{iou_s:.2f} GAF:{gaf_s:.2f}")
    plt.savefig(f"results/compare_{name}.png")
    plt.show()

if __name__ == "__main__":
    show_comparison()