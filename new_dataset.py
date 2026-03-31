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
        # Handle naming mismatch: masks have _15label_ in their name
        all_images = sorted(os.listdir(img_dir))
        all_masks = sorted(os.listdir(mask_dir))
        self.images = []
        self.mask_mapping = {}
        
        for img_name in all_images:
            # Try direct match first
            mask_path = os.path.join(mask_dir, img_name)
            if os.path.exists(mask_path):
                self.images.append(img_name)
                self.mask_mapping[img_name] = img_name
            else:
                # Try to find mask with _15label_ pattern
                # e.g., GF2_PMS2__L1A0000958146-MSS2_98.tif -> GF2_PMS2__L1A0000958146-MSS2_15label_98.tif
                base_name = img_name.rsplit('.', 1)[0]  # remove extension
                matching_mask = None
                for mask_name in all_masks:
                    mask_base = mask_name.rsplit('.', 1)[0]
                    # Check if mask contains the same identifier
                    if base_name in mask_base or mask_base.replace('_15label_', '_') == base_name:
                        matching_mask = mask_name
                        break
                
                if matching_mask:
                    self.images.append(img_name)
                    self.mask_mapping[img_name] = matching_mask
                else:
                    warnings.warn(f"Mask not found for {img_name}, skipping this sample")
        
        if len(self.images) == 0:
            raise ValueError(f"No matching image-mask pairs found in {img_dir} and {mask_dir}")
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        mask_name = self.mask_mapping[img_name]  # Use mapping to get correct mask name
        
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        # Albumentations requires NumPy arrays, NOT PIL Images
        image = np.array(Image.open(img_path).convert("RGB").resize((self.size, self.size)))
        mask = np.array(Image.open(mask_path).resize((self.size, self.size), resample=Image.NEAREST))
        
        # --- THE FIX IS HERE ---
        # If the mask loads with 3 color channels (RGB), keep only the first channel
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]
        # -----------------------

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