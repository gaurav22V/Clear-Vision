
"""
Data Preprocessing Pipeline 
"""

import os
import torch
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path

class ImagePairDataset(Dataset):
    """
    A clean, lightweight PyTorch dataset that loads pairs of ground-truth
    and synthetically degraded images. Handles path cross-compatibility natively.
    """
    def __init__(self, manifest_file, raw_root, corrupted_root, transform=None):
        """
        Args:
            manifest_file (str): Path to the txt file containing the comma-separated image pairings.
            raw_root (str): Root directory path for clean target images.
            corrupted_root (str): Root directory path for degraded input images.
            transform (callable, optional): PyTorch transform pipeline applied to both image sets.
        """
        self.raw_root = Path(raw_root)
        self.corrupted_root = Path(corrupted_root)
        self.transform = transform
        self.pairs = []

        if not os.path.exists(manifest_file):
            raise FileNotFoundError(f"Manifest mapping file missing: {manifest_file}")

        # Read manifest files 
        with open(manifest_file, "r") as f:
            for line in f:
                line = line.strip()
                if not line or ',' not in line:
                    continue
                
                # Split paths and strip Windows backslash
                raw_rel, corrupted_rel = line.split(',')
                raw_name = os.path.basename(raw_rel.strip().replace("\\", "/"))
                corrupted_name = os.path.basename(corrupted_rel.strip().replace("\\", "/"))
                
                self.pairs.append((raw_name, corrupted_name))

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        raw_name, corrupted_name = self.pairs[idx]
        
        raw_path = self.raw_root / raw_name
        corrupted_path = self.corrupted_root / corrupted_name

        try:
            raw_img = Image.open(raw_path).convert("RGB")
            corrupted_img = Image.open(corrupted_path).convert("RGB")
        except Exception as e:
            print(f"⚠️ Index [{idx}] Data Loading Error: Could not resolve paths.\n -> Raw: {raw_path}\n -> Corrupted: {corrupted_path}")
            raise e

        if self.transform:
            corrupted_tensor = self.transform(corrupted_img)
            raw_tensor = self.transform(raw_img)
            return corrupted_tensor, raw_tensor

        return corrupted_img, raw_img


if __name__ == "__main__":
    from torchvision import transforms
    from torch.utils.data import DataLoader

    print("Verifying mock dataset loader pipeline configurations...")
    
    mock_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    # Testing
    try:
        dataset = ImagePairDataset(
            manifest_file="data/splits/train_pairs.txt",
            raw_root="data/raw",
            corrupted_root="data/corrupted",
            transform=mock_transform
        )
        print(f"✓ Dataset parsed successfully! Detected sample pairs: {len(dataset)}")
    except FileNotFoundError:
        print("ℹ Manifest missing. Skipping runtime iteration validation (Expected default test behavior outside execution roots).")