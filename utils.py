import torch
import numpy as np
import pandas as pd
import os

# GAF_WEIGHTS = {0: 0.0, 1: 0.5, 2: 0.7, 3: 1.0, 4: 1.0, 5: 0.0, 6: 0.0} ## This is the old weights for deepglobe dataset, we will need to change it for GID dataset
GAF_WEIGHTS = {
    0: 0.0, 1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0,      # Background / Urban
    5: 1.5, 6: 1.5, 7: 1.5,                      # Agriculture / Cultivation
    8: 1.0,                                      # Garden Plot / Floral
    9: 1.80,                                     # Arbor Woodland / Large Trees
    10: 0.20,                                    # Shrub land
    11: 1.80,                                    # Natural Grassland
    12: 1.20,                                    # Artificial Grassland
    13: 1.0, 14: 1.0, 15: 1.0                    # River, Lake, Pond
}

def get_miou(pred, target, n_classes=7):
    pred = torch.argmax(pred, dim=1).view(-1)
    target = target.view(-1)
    iou_list = []
    for cls in range(n_classes):
        intersection = ((pred == cls) & (target == cls)).sum().item()
        union = ((pred == cls) | (target == cls)).sum().item()
        if union > 0: iou_list.append(intersection / union)
    return np.mean(iou_list) if iou_list else 0

def calculate_gaf(mask):
    if torch.is_tensor(mask):
        if len(mask.shape) == 3: # (C, H, W)
            mask = torch.argmax(mask, dim=0).cpu().numpy()
        else:
            mask = mask.cpu().numpy()
    total = mask.size
    score = sum((np.sum(mask == cls) * w) for cls, w in GAF_WEIGHTS.items())
    return score / total

def save_checkpoint(state, model_name):
    path = f"checkpoints/Albumented_{model_name}_latest.pth" # this has been changed with respect to albumentaions version
    torch.save(state, path)

def load_checkpoint(model, optimizer, model_name , resume=True):
    path = f"checkpoints/Albumented_{model_name}_latest.pth" # this can be changed with respect to albumentaions version
    if os.path.exists(path) and resume:
        ckpt = torch.load(path, weights_only=False)
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        return ckpt['epoch']
    return 0