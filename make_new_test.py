import os
import random
import shutil

# Paths based on your current directory structure
BASE_DIR = "/home/jayadeepj/Desktop/Urbanlens/gid_dataset/data/data_for_keras_aug"

VAL_IMG_DIR = os.path.join(BASE_DIR, "val_images")
VAL_MASK_DIR = os.path.join(BASE_DIR, "val_masks")

TEST_IMG_DIR = os.path.join(BASE_DIR, "test_images")
TEST_MASK_DIR = os.path.join(BASE_DIR, "test_masks")

# Create test directories
os.makedirs(TEST_IMG_DIR, exist_ok=True)
os.makedirs(TEST_MASK_DIR, exist_ok=True)

# Get all validation images
val_images = [f for f in os.listdir(VAL_IMG_DIR) if f.endswith(('.png', '.jpg', '.tif'))]

# Shuffle randomly to ensure a fair split
random.seed(42) # For reproducibility
random.shuffle(val_images)

# Take exactly 50% of the validation set for testing
split_idx = len(val_images) // 2
test_images = val_images[:split_idx]

print(f"Total Validation Images found: {len(val_images)}")
print(f"Moving {len(test_images)} images to the new Test Set...")

moved_count = 0
for img_name in test_images:
    # Assuming masks have the exact same name. 
    # (If they have a suffix like _mask, change this line: mask_name = img_name.replace('.png', '_mask.png'))
    mask_name = img_name 
    
    # Source paths
    src_img = os.path.join(VAL_IMG_DIR, img_name)
    src_mask = os.path.join(VAL_MASK_DIR, mask_name)
    
    # Destination paths
    dst_img = os.path.join(TEST_IMG_DIR, img_name)
    dst_mask = os.path.join(TEST_MASK_DIR, mask_name)
    
    # Move files
    if os.path.exists(src_mask):
        shutil.move(src_img, dst_img)
        shutil.move(src_mask, dst_mask)
        moved_count += 1
    else:
        print(f"Warning: Mask for {img_name} not found. Skipping.")

print(f"Success! {moved_count} image-mask pairs moved to the Test directory.")