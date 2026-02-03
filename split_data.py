import os
import random
import shutil

# Configuration
BASE_DIR = "/home/jayadeepj/Desktop/Urbanlens/data"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
VALID_DIR = os.path.join(BASE_DIR, "valid")
TEST_DIR = os.path.join(BASE_DIR, "test")

# Create directories if they don't exist
os.makedirs(VALID_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

# 1. Identify all satellite images that HAVE corresponding masks in the train folder
all_files = os.listdir(TRAIN_DIR)
sat_images = [f for f in all_files if f.endswith('_sat.jpg')]
labeled_images = []

for img in sat_images:
    mask_name = img.replace('_sat.jpg', '_mask.png')
    if os.path.exists(os.path.join(TRAIN_DIR, mask_name)):
        labeled_images.append(img)

print(f"Found {len(labeled_images)} images with masks.")

# 2. Shuffle and Split (80% Train, 10% Valid, 10% Test)
random.shuffle(labeled_images)
num_val = int(len(labeled_images) * 0.10)
num_test = int(len(labeled_images) * 0.10)

val_subset = labeled_images[:num_val]
test_subset = labeled_images[num_val : num_val + num_test]

def move_files(file_list, target_dir):
    for img_name in file_list:
        mask_name = img_name.replace('_sat.jpg', '_mask.png')
        # Move Image
        shutil.move(os.path.join(TRAIN_DIR, img_name), os.path.join(target_dir, img_name))
        # Move Mask
        shutil.move(os.path.join(TRAIN_DIR, mask_name), os.path.join(target_dir, mask_name))

move_files(val_subset, VALID_DIR)
move_files(test_subset, TEST_DIR)

print(f"Moved {len(val_subset)} files to {VALID_DIR}")
print(f"Moved {len(test_subset)} files to {TEST_DIR}")