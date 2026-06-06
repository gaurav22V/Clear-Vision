
"""
Image Corruption Utility
"""

import os
import cv2
import random
import yaml
import numpy as np
import io
from PIL import Image
from pathlib import Path
from tqdm import tqdm

class ImageCorruption:
    def __init__(self, config=None, config_path="config/gan_config.yaml"):
        if config is None:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        else:
            self.config = config
            
        self.output_dir = Path(self.config.get('corrupted_image_root', 'data/corrupted'))
        os.makedirs(self.output_dir, exist_ok=True)
        
    def apply_corruptions(self, image_paths, save_corrupted=True, corruption_types=None):
        corruption_mapping = {}
        # Fallback to standard types if not explicitly specified in a config sub-key
        corruption_types = corruption_types or self.config.get('corruption_types', ['noise', 'jpeg', 'blur', 'mask', 'combined'])
        
        for img_path in tqdm(image_paths, desc="Corrupting images"):
            try:
                img = cv2.imread(img_path)
                if img is None:
                    continue
                    
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                c_type = random.choice(corruption_types)
                corrupted_img = self._apply_corruption(img, c_type)
                
                if save_corrupted:
                    filename = os.path.basename(img_path)
                    base, ext = os.path.splitext(filename)
                    out_filename = f"{base}_corrupted_{c_type}{ext}"
                    out_path = str(self.output_dir / out_filename)
                    
                    cv2.imwrite(out_path, cv2.cvtColor(corrupted_img, cv2.COLOR_RGB2BGR))
                    corruption_mapping[img_path] = out_path
                else:
                    corruption_mapping[img_path] = corrupted_img
                    
            except Exception as e:
                print(f"Error processing {img_path}: {e}")
                
        return corruption_mapping
    
    def _apply_corruption(self, img, corruption_type):
        if corruption_type == "noise": return self._add_noise(img)
        if corruption_type == "jpeg": return self._jpeg_compression(img)
        if corruption_type == "blur": return self._add_blur(img)
        if corruption_type == "mask": return self._add_mask(img)
        if corruption_type == "combined":
            corrupted = img.copy()
            for c_type in random.sample(["noise", "jpeg", "blur", "mask"], random.randint(2, 3)):
                corrupted = self._apply_corruption(corrupted, c_type)
            return corrupted
        return img
            
    def _add_noise(self, img):
        if random.choice(["gaussian", "salt_pepper"]) == "gaussian":
            std = random.uniform(0.01, 0.05)
            noise = np.random.normal(0, std, img.shape).astype(np.float32)
            return np.clip(img.astype(np.float32) + noise * 255, 0, 255).astype(np.uint8)
        else:
            prob = random.uniform(0.01, 0.05)
            mask = np.random.random(img.shape[:2])
            noisy_img = img.copy()
            noisy_img[mask < prob / 2] = 255
            noisy_img[mask > 1 - prob / 2] = 0
            return noisy_img
    
    def _jpeg_compression(self, img):
        quality = random.randint(10, 50)
        pil_img = Image.fromarray(img)
        buffer = io.BytesIO()
        pil_img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return np.array(Image.open(buffer))
        
    def _add_blur(self, img):
        kernel_size = random.choice([3, 5, 7])
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), random.uniform(0.5, 1.5))
        
    def _add_mask(self, img):
        masked_img = img.copy()
        h, w = img.shape[:2]
        
        for _ in range(random.randint(1, 3)):
            ratio = random.uniform(0.05, 0.15)
            box_w, box_h = int(w * ratio), int(h * ratio)
            x = random.randint(0, w - box_w)
            y = random.randint(0, h - box_h)
            
            color = random.choice([(0,0,0), (255,255,255), tuple(random.randint(0,255) for _ in range(3))])
            masked_img[y:y+box_h, x:x+box_w] = color
            
        return masked_img