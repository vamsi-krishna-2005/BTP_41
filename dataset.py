import os
from streamlit import image
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from albumentations.pytorch import ToTensorV2


class UrbanLensDataset(Dataset):
    def __init__(self, data_dir, size=224, transform=None):
        self.data_dir = data_dir
        self.size = size
        self.transform = transform
        
        # 1. Get all files in the directory once
        all_files = set(os.listdir(data_dir))
        
        # 2. Only include satellite images if the corresponding mask exists
        self.img_names = []
        for f in all_files:
            if f.endswith('_sat.jpg'):
                mask_name = f.replace('_sat.jpg', '_mask.png')
                if mask_name in all_files:
                    self.img_names.append(f)
        
        self.img_names.sort()
        
        if len(self.img_names) == 0:
            print(f"Warning: No valid image-mask pairs found in {data_dir}")

        self.color_map = {
            (0, 255, 255): 0,   # Urban
            (255, 255, 0): 1,   # Agriculture
            (255, 0, 255): 2,   # Rangeland
            (0, 255, 0): 3,     # Forest
            (0, 0, 255): 4,     # Water
            (255, 255, 255): 5, # Barren
            (0, 0, 0): 6        # Unknown
        }

    def _rgb_to_mask(self, rgb_mask):
        mask = np.zeros((rgb_mask.shape[0], rgb_mask.shape[1]), dtype=np.uint8)
        for color, class_id in self.color_map.items():
            match = np.all(np.abs(rgb_mask - np.array(color)) < 128, axis=-1)
            mask[match] = class_id
        return mask

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        mask_name = img_name.replace('_sat.jpg', '_mask.png')

        img_path = os.path.join(self.data_dir, img_name)
        mask_path = os.path.join(self.data_dir, mask_name)

        # Load with PIL
        image = Image.open(img_path).convert("RGB").resize((self.size, self.size))
        mask_rgb = Image.open(mask_path).convert("RGB").resize(
            (self.size, self.size), resample=Image.NEAREST
        )

        # Convert to numpy (CRITICAL for Albumentations)
        image = np.array(image)
        mask_rgb = np.array(mask_rgb)

        if self.transform:
            augmented = self.transform(image=image, mask=mask_rgb)
            image = augmented["image"]          # Tensor (C,H,W)
            mask = augmented["mask"]            # Still RGB numpy
            mask = torch.from_numpy(self._rgb_to_mask(mask)).long()
        else:
            image = transforms.ToTensor()(image)
            mask = torch.from_numpy(self._rgb_to_mask(mask_rgb)).long()

        return image, mask, img_name
