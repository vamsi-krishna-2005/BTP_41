import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class DeepGlobeDataset(Dataset):
    def __init__(self, img_dir, mask_dir, size=224):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_names = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
        self.size = size
        
        # Color Mapping for DeepGlobe 7 classes
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
            # Thresholding at 128 as per DeepGlobe standard to handle compression artifacts
            match = np.all(np.abs(rgb_mask - np.array(color)) < 128, axis=-1)
            mask[match] = class_id
        return mask

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        name = self.img_names[idx]
        img_path = os.path.join(self.img_dir, name)
        # DeepGlobe masks usually have '_mask.png' suffix
        mask_path = os.path.join(self.mask_dir, name.replace('_sat.jpg', '_mask.png'))

        image = Image.open(img_path).convert("RGB").resize((self.size, self.size))
        mask_rgb = Image.open(mask_path).convert("RGB").resize((self.size, self.size), resample=Image.NEAREST)
        
        image = transforms.ToTensor()(image)
        mask = torch.from_numpy(self._rgb_to_mask(np.array(mask_rgb))).long()

        return image, mask, name