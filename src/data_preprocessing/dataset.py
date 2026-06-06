# -*- coding: utf-8 -*-
import os
import random
import argparse
from pathlib import Path
from PIL import Image, ImageFilter, ImageDraw
import numpy as np
import torch
from torch.utils.data import Dataset

# Dataset for training
class ImagePairDataset(Dataset):
    def __init__(self, manifest_file, raw_root, corrupted_root, transform=None):
        self.raw_root = Path(raw_root)
        self.corrupted_root = Path(corrupted_root)
        self.transform = transform
        self.image_files = []

        if os.path.exists(manifest_file):
            with open(manifest_file, 'r') as f:
                self.image_files = [line.strip() for line in f.readlines() if line.strip()]
        else:
            print(f"⚠️ Warning: Manifest file {manifest_file} not found. Run with --create_splits first.")

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]
        raw_path = self.raw_root / img_name
        corrupted_path = self.corrupted_root / img_name

        raw_image = Image.open(raw_path).convert("RGB")
        corrupted_image = Image.open(corrupted_path).convert("RGB")

        # --- SYNCHRONIZED DYNAMIC AUGMENTATION ---
        # 50% chance to flip BOTH images exactly the same way to prevent spatial mismatch
        if random.random() < 0.5:
            raw_image = raw_image.transpose(Image.FLIP_LEFT_RIGHT)
            corrupted_image = corrupted_image.transpose(Image.FLIP_LEFT_RIGHT)

        if self.transform:
            raw_image = self.transform(raw_image)
            corrupted_image = self.transform(corrupted_image)

        return corrupted_image, raw_image

def apply_corruption(img):
    """Applies random blur, digital noise, or black masks to an image."""
    img = img.copy()
    w, h = img.size
    
    # 1. Random Blur
    if random.random() < 0.7:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(1.0, 3.5)))
        
    # 2. Random Noise
    if random.random() < 0.5:
        img_array = np.array(img)
        noise = np.random.normal(0, 25, img_array.shape).astype(np.float32)
        img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)
        img = Image.fromarray(img_array)
        
    # 3. Random Masking (Black boxes simulating data loss)
    if random.random() < 0.3:
        draw = ImageDraw.Draw(img)
        for _ in range(random.randint(1, 3)):
            x1 = random.randint(0, int(w * 0.6))
            y1 = random.randint(0, int(h * 0.6))
            x2 = x1 + random.randint(20, int(w * 0.3))
            y2 = y1 + random.randint(20, int(h * 0.3))
            draw.rectangle([x1, y1, x2, y2], fill=(0, 0, 0))
            
    return img

def generate_corrupted_dataset(raw_dir, corrupted_dir):
    print(f"\nGenerating corrupted dataset...")
    os.makedirs(corrupted_dir, exist_ok=True)
    
    raw_files = [f for f in os.listdir(raw_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    if not raw_files:
        print(" Error: No images found in data/raw! Run your scraper first.")
        return
        
    count = 0
    for filename in raw_files:
        raw_path = os.path.join(raw_dir, filename)
        corr_path = os.path.join(corrupted_dir, filename)
        
        try:
            # Open, standard size, and save the corrupted version
            img = Image.open(raw_path).convert("RGB")
            img = img.resize((256, 256), Image.Resampling.LANCZOS)
            
            corrupted_img = apply_corruption(img)
            corrupted_img.save(corr_path)
            count += 1
            
            if count % 50 == 0:
                print(f" -> Processed {count} / {len(raw_files)} images...")
        except Exception as e:
            print(f"Skipping {filename} due to error: {e}")
            
    print(f"Generated {count} corrupted images in {corrupted_dir}/")

def create_splits(raw_dir, processed_dir, train_ratio=0.9):
    print(f"\nCreating Train/Validation splits...")
    os.makedirs(processed_dir, exist_ok=True)
    
    raw_files = [f for f in os.listdir(raw_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    random.shuffle(raw_files)
    
    split_idx = int(len(raw_files) * train_ratio)
    train_files = raw_files[:split_idx]
    val_files = raw_files[split_idx:]
    
    with open(os.path.join(processed_dir, 'train_pairs.txt'), 'w') as f:
        for name in train_files: f.write(f"{name}\n")
            
    with open(os.path.join(processed_dir, 'val_pairs.txt'), 'w') as f:
        for name in val_files: f.write(f"{name}\n")
            
    print(f"Splits created. Train: {len(train_files)} | Val: {len(val_files)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset Generator")
    parser.add_argument('--create_dataset', action='store_true', help="Generate corrupted images")
    parser.add_argument('--create_splits', action='store_true', help="Generate train/val text files")
    args = parser.parse_args()
    
    RAW_DIR = "data/raw"
    CORRUPTED_DIR = "data/corrupted"
    PROCESSED_DIR = "data/processed"
    
    if args.create_dataset:
        generate_corrupted_dataset(RAW_DIR, CORRUPTED_DIR)
        
    if args.create_splits:
        create_splits(RAW_DIR, PROCESSED_DIR)
        
    if not args.create_dataset and not args.create_splits:
        print("Please provide an argument: --create_dataset, --create_splits, or both.")