import torch
import matplotlib.pyplot as plt
from dataset import UrbanLensDataset
from models import UNet, SwinUNet
from train import VAL_DIR
from utils import calculate_gaf, get_miou
import matplotlib.colors as mcolors
from matplotlib.patches import Patch


device = torch.device("cuda")
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

CLASS_COLORS = {
    0: (220/255, 20/255, 60/255),    # Urban - red
    1: (255/255, 215/255, 0/255),    # Agriculture - yellow
    2: (0/255, 255/255, 255/255),    # Rangeland - cyan
    3: (34/255, 139/255, 34/255),    # Forest - green
    4: (30/255, 144/255, 255/255),   # Water - blue
    5: (139/255, 69/255, 19/255),    # Barren - brown
    6: (128/255, 128/255, 128/255)   # Unknown - gray
}


def get_present_classes(mask):
    unique = torch.unique(mask).cpu().numpy().tolist()
    return [(cls, CLASS_NAMES[cls]) for cls in unique]

CLASS_CMAP = mcolors.ListedColormap(
    [CLASS_COLORS[i] for i in range(len(CLASS_COLORS))]
)


def create_class_legend(present_classes):
    legend_elements = []
    for cls, name in present_classes:
        legend_elements.append(
            Patch(facecolor=CLASS_COLORS[cls], label=f"{cls}: {name}")
        )
    return legend_elements


def generate_report(idx=0):
    ds = UrbanLensDataset(TEST_DIR)
    img, mask, name = ds[idx]

    # Load models on CPU first
    unet = UNet()
    swin = SwinUNet()

    ckpt_unet = torch.load("checkpoints/unet_latest.pth", map_location="cpu")
    ckpt_swin = torch.load("checkpoints/swin_latest.pth", map_location="cpu")

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

    axes[1].imshow(mask, cmap=CLASS_CMAP)
    axes[1].set_title(f"Ground Truth\nGAF: {gaf_gt:.4f}")

    pred_u = torch.argmax(out_u[0], 0).cpu()
    axes[2].imshow(pred_u, cmap=CLASS_CMAP)
    axes[2].set_title(f"U-Net\nmIoU: {iou_u:.3f} | GAF: {gaf_u:.4f}")

    pred_s = torch.argmax(out_s[0], 0).cpu()
    axes[3].imshow(pred_s, cmap=CLASS_CMAP)
    axes[3].set_title(f"Swin-UNet\nmIoU: {iou_s:.3f} | GAF: {gaf_s:.4f}")

    present_classes = get_present_classes(mask)
    legend_elements = create_class_legend(present_classes)

    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=len(legend_elements),
        fontsize=10,
        frameon=True
    )


    plt.tight_layout()
    plt.savefig(f"results/final_comparison_No_Albumentation_{idx}.png")
    plt.close()
    print(f"Report saved for {name}")


if __name__ == "__main__":
    for i in range(5): # Generate 5 sample comparisons
        generate_report(i)