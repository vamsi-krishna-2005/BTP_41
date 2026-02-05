import torch
import matplotlib.pyplot as plt
from dataset import UrbanLensDataset
from models import UNet, SwinUNet
from train import VAL_DIR
from utils import calculate_gaf, get_miou
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm


device = torch.device("cpu")
TEST_DIR = "/home/jayadeepj/Desktop/Urbanlens/data/test"


CLASS_NAMES = {
    0: "Urban",
    1: "Agriculture",
    2: "Rangeland",
    3: "Forest",
    4: "Water",
    5: "Barren",
    6: "Unknown"
}


COLORS = [
    "#e41a1c",  # 0 Urban (red)
    "#ffd92f",  # 1 Agriculture (yellow)
    "#4dd2ff",  # 2 Rangeland (cyan)
    "#4daf4a",  # 3 Forest (green)
    "#377eb8",  # 4 Water (blue)
    "#8c510a",  # 5 Barren (brown)
    "#999999",  # 6 Unknown (gray)
]

cmap = ListedColormap(COLORS)
norm = BoundaryNorm(range(len(COLORS) + 1), cmap.N)


def get_present_classes(mask):
    unique = torch.unique(mask).cpu().numpy().tolist()
    return [(cls, CLASS_NAMES[cls]) for cls in unique]



def generate_report(idx=0):
    ds = UrbanLensDataset(TEST_DIR, transform=None)
    img, mask, name = ds[idx]

    # Load models on CPU first
    unet = UNet()
    swin = SwinUNet()

    ckpt_unet = torch.load("checkpoints/Albumented_unet_gpu_latest.pth", map_location="cpu")
    ckpt_swin = torch.load("checkpoints/Albumented_swin_gpu_latest.pth", map_location="cpu")

    unet.load_state_dict(ckpt_unet["state_dict"])
    swin.load_state_dict(ckpt_swin["state_dict"])

    # Move to GPU AFTER loading
    unet = unet.to(device)
    swin = swin.to(device)

    unet.eval()
    swin.eval()

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
    axes[0].imshow(img.permute(1,2,0))
    axes[0].set_title(f"Input: {name}")

    axes[1].imshow(mask, cmap=cmap, norm=norm)
    axes[1].set_title(f"Ground Truth\nGAF: {gaf_gt:.4f}")

    pred_u = torch.argmax(out_u[0], 0).cpu()
    axes[2].imshow(pred_u, cmap=cmap, norm=norm)
    axes[2].set_title(f"U-Net\nmIoU: {iou_u:.3f} | GAF: {gaf_u:.4f}")

    pred_s = torch.argmax(out_s[0], 0).cpu()
    axes[3].imshow(pred_s, cmap=cmap, norm=norm)
    axes[3].set_title(f"Swin-UNet\nmIoU: {iou_s:.3f} | GAF: {gaf_s:.4f}")

    present_classes = get_present_classes(mask)

    handles = [
        mpatches.Patch(color=COLORS[c], label=f"{c}: {CLASS_NAMES[c]}")
        for c in torch.unique(mask).cpu().tolist()
    ]

    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=4,
        bbox_to_anchor=(0.5, -0.05)
    )


    plt.tight_layout()
    plt.savefig(f"results/final_comparison_Albumentation_{idx}.png")
    plt.close()
    print(f"Report saved for {name}")


if __name__ == "__main__":
    for i in range(5): # Generate 5 sample comparisons
        generate_report(i)
    torch.cuda.empty_cache()
