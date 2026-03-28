import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import warnings

class GIDDataset(Dataset):
    def __init__(self, img_dir, mask_dir, size=224, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.size = size
        self.transform = transform # Added transform support
        
        # Filter: keep only images that have corresponding masks
        all_images = sorted(os.listdir(img_dir))
        self.images = []
        
        for img_name in all_images:
            mask_path = os.path.join(mask_dir, img_name)
            if os.path.exists(mask_path):
                self.images.append(img_name)
            else:
                warnings.warn(f"Mask not found for {img_name}, skipping this sample")
        
        if len(self.images) == 0:
            raise ValueError(f"No matching image-mask pairs found in {img_dir} and {mask_dir}")
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        mask_name = img_name # Assuming mask has the exact same name
        
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        # Albumentations requires NumPy arrays, NOT PIL Images
        image = np.array(Image.open(img_path).convert("RGB").resize((self.size, self.size)))
        mask = np.array(Image.open(mask_path).resize((self.size, self.size), resample=Image.NEAREST))
        
        # Apply Albumentations
        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            # Fallback if no transform is provided
            image = transforms.ToTensor()(image)
            mask = torch.from_numpy(mask).long()

        return image, mask.long(), img_name