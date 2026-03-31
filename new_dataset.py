import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class GIDDataset(Dataset):
    def __init__(self, img_dir, mask_dir, transform=None):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.transform = transform 
        
        # Images and masks now have the exact same names and are pre-filtered
        self.images = sorted(os.listdir(img_dir))
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, img_name) # Name is identical now

        # Lightning fast read - NO RESIZING needed!
        image = np.array(Image.open(img_path))
        mask = np.array(Image.open(mask_path))
        
        # Ensure mask is 2D
        if mask.ndim >= 3:
            mask = mask[:, :, 0]
        mask = mask.astype(np.int64)

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