import os
import shutil
import random
from pathlib import Path

# Define constants - modify these as needed
SOURCE_HUMAN_DIR = "dataset/human"
SOURCE_NONHUMAN_DIR = "dataset/non-human"
OUTPUT_DIR = "dataset"
TRAIN_COUNT = 500
TEST_COUNT = 200
RANDOM_SEED = 42

def create_directory(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

def split_dataset():
    """Split images from source directories into training and testing sets"""
    
    # Set random seed for reproducibility
    random.seed(RANDOM_SEED)
    
    # Create output directories
    train_human_dir = "dataset/human_train"
    train_nonhuman_dir = "dataset/non-human_train"
    test_human_dir = "dataset/human_test"
    test_nonhuman_dir = "dataset/non-human_test"
    
    create_directory(train_human_dir)
    create_directory(train_nonhuman_dir)
    create_directory(test_human_dir)
    create_directory(test_nonhuman_dir)
    
    # Get list of image files
    human_files = [f for f in os.listdir(SOURCE_HUMAN_DIR) 
                  if os.path.isfile(os.path.join(SOURCE_HUMAN_DIR, f))
                  and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    nonhuman_files = [f for f in os.listdir(SOURCE_NONHUMAN_DIR) 
                     if os.path.isfile(os.path.join(SOURCE_NONHUMAN_DIR, f))
                     and f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    
    # Check if we have enough images
    if len(human_files) < TRAIN_COUNT + TEST_COUNT:
        print(f"Warning: Not enough human images. Found {len(human_files)}, needed {TRAIN_COUNT + TEST_COUNT}")
        return
    
    if len(nonhuman_files) < TRAIN_COUNT + TEST_COUNT:
        print(f"Warning: Not enough non-human images. Found {len(nonhuman_files)}, needed {TRAIN_COUNT + TEST_COUNT}")
        return
    
    # Shuffle files for random selection
    random.shuffle(human_files)
    random.shuffle(nonhuman_files)
    
    # Split human images
    train_human_files = human_files[:TRAIN_COUNT]
    test_human_files = human_files[TRAIN_COUNT:TRAIN_COUNT+TEST_COUNT]
    
    # Split non-human images
    train_nonhuman_files = nonhuman_files[:TRAIN_COUNT]
    test_nonhuman_files = nonhuman_files[TRAIN_COUNT:TRAIN_COUNT+TEST_COUNT]
    
    # Copy human images
    print(f"Copying {len(train_human_files)} human images to training set...")
    for file in train_human_files:
        src = os.path.join(SOURCE_HUMAN_DIR, file)
        dst = os.path.join(train_human_dir, file)
        shutil.copy2(src, dst)
    
    print(f"Copying {len(test_human_files)} human images to test set...")
    for file in test_human_files:
        src = os.path.join(SOURCE_HUMAN_DIR, file)
        dst = os.path.join(test_human_dir, file)
        shutil.copy2(src, dst)
    
    # Copy non-human images
    print(f"Copying {len(train_nonhuman_files)} non-human images to training set...")
    for file in train_nonhuman_files:
        src = os.path.join(SOURCE_NONHUMAN_DIR, file)
        dst = os.path.join(train_nonhuman_dir, file)
        shutil.copy2(src, dst)
    
    print(f"Copying {len(test_nonhuman_files)} non-human images to test set...")
    for file in test_nonhuman_files:
        src = os.path.join(SOURCE_NONHUMAN_DIR, file)
        dst = os.path.join(test_nonhuman_dir, file)
        shutil.copy2(src, dst)
    
    print("\nDataset split complete!")
    print(f"Training set: {len(train_human_files)} human, {len(train_nonhuman_files)} non-human")
    print(f"Test set: {len(test_human_files)} human, {len(test_nonhuman_files)} non-human")

if __name__ == "__main__":
    split_dataset()