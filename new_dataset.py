import os
import torch
import numpy as np
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class GIDDataset(Dataset):
    def __init__(self, img_dir, mask_dir, size=224):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.size = size
        
        # List all images
        self.images = sorted(os.listdir(img_dir))
        
    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        # Assuming the mask has the exact same name or a standard suffix
        # Adjust the replace string if your masks have a '_mask' suffix
        mask_name = img_name 
        
        img_path = os.path.join(self.img_dir, img_name)
        mask_path = os.path.join(self.mask_dir, mask_name)

        image = Image.open(img_path).convert("RGB").resize((self.size, self.size))
        
        # Masks in GID are usually 1-channel arrays with values 0-15
        mask = Image.open(mask_path).resize((self.size, self.size), resample=Image.NEAREST)
        
        image = transforms.ToTensor()(image)
        mask = torch.from_numpy(np.array(mask)).long()

        return image, mask, img_name