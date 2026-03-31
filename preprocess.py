import os
import numpy as np
from PIL import Image
import warnings

# Original paths
SRC_TRAIN_IMG = "/home/jayadeepj/Desktop/Urbanlens/gid_dataset/data/data_for_keras_aug/train_images/train"
SRC_TRAIN_MASK = "/home/jayadeepj/Desktop/Urbanlens/gid_dataset/data/data_for_keras_aug/train_masks/train"
SRC_VAL_IMG = "/home/jayadeepj/Desktop/Urbanlens/gid_dataset/data/data_for_keras_aug/val_images/val"
SRC_VAL_MASK = "/home/jayadeepj/Desktop/Urbanlens/gid_dataset/data/data_for_keras_aug/val_masks/val"

# New ultra-fast preprocessed paths
DEST_BASE = "/home/jayadeepj/Desktop/Urbanlens/gid_dataset/preprocessed_224"
os.makedirs(os.path.join(DEST_BASE, "train_images"), exist_ok=True)
os.makedirs(os.path.join(DEST_BASE, "train_masks"), exist_ok=True)
os.makedirs(os.path.join(DEST_BASE, "val_images"), exist_ok=True)
os.makedirs(os.path.join(DEST_BASE, "val_masks"), exist_ok=True)

def preprocess_folder(img_dir, mask_dir, dest_img_dir, dest_mask_dir):
    all_images = sorted(os.listdir(img_dir))
    all_masks = sorted(os.listdir(mask_dir))
    
    # Hash map for fast matching (same as we did in dataset.py)
    mask_dict = {}
    for m in all_masks:
        mask_dict[m] = m
        base = m.rsplit('.', 1)[0]
        mask_dict[base] = m
        mask_dict[base.replace('_15label_', '_')] = m
        
    print(f"Processing {len(all_images)} images from {img_dir}...")
    
    processed_count = 0
    for img_name in all_images:
        base_name = img_name.rsplit('.', 1)[0]
        
        mask_name = None
        if img_name in mask_dict: mask_name = mask_dict[img_name]
        elif base_name in mask_dict: mask_name = mask_dict[base_name]
        
        if mask_name:
            img_path = os.path.join(img_dir, img_name)
            mask_path = os.path.join(mask_dir, mask_name)
            
            # Load and resize ONCE
            img = Image.open(img_path).convert("RGB").resize((224, 224))
            mask = Image.open(mask_path).resize((224, 224), resample=Image.NEAREST)
            
            # Save as fast-loading PNGs (Lossless)
            new_img_name = base_name + ".png"
            new_mask_name = base_name + ".png" # Keep names identical in new folder
            
            img.save(os.path.join(dest_img_dir, new_img_name))
            mask.save(os.path.join(dest_mask_dir, new_mask_name))
            
            processed_count += 1
            if processed_count % 1000 == 0:
                print(f"  -> Processed {processed_count} pairs...")

    print(f"Done! Saved {processed_count} 224x224 pairs to {dest_img_dir}\n")

if __name__ == "__main__":
    print("Starting pre-processing for Train set...")
    preprocess_folder(SRC_TRAIN_IMG, SRC_TRAIN_MASK, os.path.join(DEST_BASE, "train_images"), os.path.join(DEST_BASE, "train_masks"))
    
    print("Starting pre-processing for Val set...")
    preprocess_folder(SRC_VAL_IMG, SRC_VAL_MASK, os.path.join(DEST_BASE, "val_images"), os.path.join(DEST_BASE, "val_masks"))