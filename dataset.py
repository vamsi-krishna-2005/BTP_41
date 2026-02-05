import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2


class UrbanLensDataset(Dataset):
    def __init__(self, data_dir, size=224, transform=None):
        self.data_dir = data_dir
        self.size = size
        self.transform = transform

        all_files = set(os.listdir(data_dir))
        self.img_names = [
            f for f in all_files
            if f.endswith("_sat.jpg") and f.replace("_sat.jpg", "_mask.png") in all_files
        ]
        self.img_names.sort()

        self.color_map = {
            (0, 255, 255): 0,   # Urban
            (255, 255, 0): 1,   # Agriculture
            (255, 0, 255): 2,   # Rangeland
            (0, 255, 0): 3,     # Forest
            (0, 0, 255): 4,     # Water
            (255, 255, 255): 5,# Barren
            (0, 0, 0): 6        # Unknown
        }

    def _rgb_to_mask(self, rgb):
        mask = np.zeros((rgb.shape[0], rgb.shape[1]), dtype=np.uint8)
        for color, cls in self.color_map.items():
            mask[np.all(rgb == np.array(color), axis=-1)] = cls
        return mask

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        mask_name = img_name.replace("_sat.jpg", "_mask.png")

        image = np.array(
            Image.open(os.path.join(self.data_dir, img_name))
            .convert("RGB")
            .resize((self.size, self.size))
        )

        mask_rgb = np.array(
            Image.open(os.path.join(self.data_dir, mask_name))
            .convert("RGB")
            .resize((self.size, self.size), Image.NEAREST)
        )

        mask = self._rgb_to_mask(mask_rgb)

        if self.transform:
            augmented = self.transform(image=image, mask=mask)
            image = augmented["image"]
            mask = augmented["mask"].long()
        else:
            image = ToTensorV2()(image=image)["image"]
            mask = torch.from_numpy(mask).long()

        return image, mask, img_name
