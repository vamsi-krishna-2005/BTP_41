import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class UrbanLensDataset(Dataset):
    def __init__(self, data_dir, size=224):
        self.data_dir = data_dir
        self.size = size
        
        # Identify satellite images by suffix
        self.img_names = sorted([f for f in os.listdir(data_dir) if f.endswith('_sat.jpg')])
        
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
        # Match mask: replace _sat.jpg with _mask.png
        mask_name = img_name.replace('_sat.jpg', '_mask.png')
        
        img_path = os.path.join(self.data_dir, img_name)
        mask_path = os.path.join(self.data_dir, mask_name)

        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Missing mask: {mask_path}")

        image = Image.open(img_path).convert("RGB").resize((self.size, self.size))
        mask_rgb = Image.open(mask_path).convert("RGB").resize((self.size, self.size), resample=Image.NEAREST)
        
        image = transforms.ToTensor()(image)
        mask = torch.from_numpy(self._rgb_to_mask(np.array(mask_rgb))).long()

        return image, mask, img_name