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
        self.transform = transform 
        
        print(f"[*] Scanning directory: {img_dir}...")
        all_images = sorted(os.listdir(img_dir))
        all_masks = sorted(os.listdir(mask_dir))
        self.images = []
        self.mask_mapping = {}
        
        print(f"[*] Indexing {len(all_masks)} masks for fast lookup...")
        mask_dict = {}
        for m in all_masks:
            mask_dict[m] = m  # Exact match
            base = m.rsplit('.', 1)[0]
            mask_dict[base] = m  # Base match
            mask_dict[base.replace('_15label_', '_')] = m # Alt base match
            
        # Now we match images instantly using the dictionary
        print(f"[*] Pairing images with masks...")
        for img_name in all_images:
            base_name = img_name.rsplit('.', 1)[0]
            
            if img_name in mask_dict:
                self.images.append(img_name)
                self.mask_mapping[img_name] = mask_dict[img_name]
            elif base_name in mask_dict:
                self.images.append(img_name)
                self.mask_mapping[img_name] = mask_dict[base_name]
            else:
                pass # Skip if no mask found silently to avoid spamming the console
                
        if len(self.images) == 0:
            raise ValueError(f"No matching image-mask pairs found in {img_dir} and {mask_dir}")
            
        print(f"[SUCCESS] Loaded {len(self.images)} image-mask pairs ready for training!\n")
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        mask_name = self.mask_mapping[img_name] 
        
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = np.array(Image.open(img_path).convert("RGB").resize((self.size, self.size)))
        mask = np.array(Image.open(mask_path).resize((self.size, self.size), resample=Image.NEAREST))
        
        # --- IRONCLAD FIX ---
        if mask.ndim >= 3:
            mask = mask[:, :, 0]
        mask = mask.astype(np.int64)
        # --------------------

        if self.transform is not None:
            augmented = self.transform(image=image, mask=mask)
            image = augmented['image']
            mask = augmented['mask']
        else:
            image = transforms.ToTensor()(image)
            mask = torch.from_numpy(mask).long()

        if hasattr(mask, 'squeeze'):
            mask = mask.squeeze()

        return image, mask.long(), img_name