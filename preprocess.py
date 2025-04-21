import os
import random
from PIL import Image

# Replace with your actual directories
dirs = ['1/1', '1/2', '1/3']  # e.g., ['images/part1', 'images/part2', 'images/part3']

# Output directories
left_output_dir = 'left_patches'
right_output_dir = 'right_patches'

# Create output directories
os.makedirs(left_output_dir, exist_ok=True)
os.makedirs(right_output_dir, exist_ok=True)

# Parameters
sample_size = 350
patch_width = 64
patch_height = 128

# Gather all .pmg files from the directories
all_images = []
for d in dirs:
    files = [os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith('.pgm')]
    all_images.extend(files)

print(f"Found {len(all_images)} images in total.")

# Sample 350 uniformly
sampled_images = random.sample(all_images, sample_size)

# Process images
for idx, img_path in enumerate(sampled_images):
    try:
        with Image.open(img_path) as img:
            width, height = img.size

            if height < patch_height or width < 2 * patch_width:
                print(f"Skipping {img_path}: too small ({width}x{height})")
                continue

            middle_y = (height - patch_height) // 2

            # Left patch: center of the left half
            left_x = (width // 4) - (patch_width // 2)
            left_box = (left_x, middle_y, left_x + patch_width, middle_y + patch_height)
            left_patch = img.crop(left_box)
            left_patch.save(os.path.join(left_output_dir, f"left_{idx:04d}.png"))

            # Right patch: center of the right half
            right_x = (3 * width // 4) - (patch_width // 2)
            right_box = (right_x, middle_y, right_x + patch_width, middle_y + patch_height)
            right_patch = img.crop(right_box)
            right_patch.save(os.path.join(right_output_dir, f"right_{idx:04d}.png"))
    
    except Exception as e:
        print(f"Error processing {img_path}: {e}")

print("Done.")
