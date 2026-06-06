
"""
Model Evaluation 
"""

import os
import torch
import yaml
import numpy as np
import argparse
from datetime import datetime
from PIL import Image
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import lpips

from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torchvision import transforms

# Adaptive imports based on your exact module tree
from src.models.gan_model import UNetGenerator, SimpleUNetGenerator
from src.data_preprocessing.dataset import ImageRestorationDataset


def load_model(model_path, model_type="unet", device="cpu"):
    """Loads the trained generator with fallback handling for all saved checkpoint styles."""
    try:
        if model_type == "simple_unet":
            model = SimpleUNetGenerator().to(device)
        else:
            model = UNetGenerator().to(device)
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Checkpoint not found at: {model_path}")
            
        print(f"Loading weights from: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        
        # Safely extract weights depending on how it was saved
        if isinstance(checkpoint, dict):
            if 'generator_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['generator_state_dict'])
                print(f"✓ Loaded full checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
            elif 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                model.load_state_dict(checkpoint)
                print("✓ Loaded direct generator state dictionary")
        else:
            model.load_state_dict(checkpoint)
            print("✓ Loaded direct state dict weights")
                
        model.eval()
        return model
    except Exception as e:
        print(f"Extraction Error: {e}")
        raise


def tensor_to_numpy(tensor):
    """Converts a standardized torch tensor back to a valid channel-last numpy image."""
    if tensor.min() < 0 or tensor.max() > 1:
        tensor = torch.clamp((tensor + 1) / 2, 0, 1)  
    
    array = tensor.detach().cpu().numpy()
    if len(array.shape) == 4:
        array = array[0]
    if len(array.shape) == 3:
        array = np.transpose(array, (1, 2, 0))  
    return array


def prepare_tensor_for_lpips(tensor):
    """Guarantees that input tensors match the strict [-1, 1] requirement for LPIPS networks."""
    if tensor.min() >= 0:
        tensor = tensor * 2.0 - 1.0
    return tensor


def write_metrics_report(metrics, file_path, resolution_str):
    """Writes a clean evaluation log report to disk."""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        f.write("="*60 + "\n")
        f.write(f"IMAGE RESTORATION EVALUATION REPORT ({resolution_str})\n")
        f.write("="*60 + "\n")
        f.write(f"Timestamp:   {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model Path:  {metrics.get('model_path')}\n")
        f.write(f"Model Type:  {metrics.get('model_type')}\n")
        f.write(f"Total Test Samples: {metrics.get('num_samples')}\n")
        f.write("-" * 60 + "\n\n")
        
        for name, key in [("STRUCTURAL SIMILARITY (SSIM)", "ssim"), 
                          ("PEAK SIGNAL-TO-NOISE RATIO (PSNR)", "psnr"), 
                          ("LEARNED PERCEPTUAL SIMILARITY (LPIPS)", "lpips")]:
            scores = metrics.get(f"{key}_scores", [])
            if scores:
                f.write(f"{name}\n" + "-"*40 + "\n")
                unit = " dB" if key == "psnr" else ""
                f.write(f"  Mean:   {np.mean(scores):.6f}{unit}\n")
                f.write(f"  Std:    {np.std(scores):.6f}{unit}\n")
                f.write(f"  Median: {np.median(scores):.6f}{unit}\n\n")


def run_evaluation(config_path="config/gan_config.yaml", model_path=None, lpips_backbone='alex'):
    """Executes a full pipeline quantitative assessment using the configured validation pair split."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
        
    # Dynamically extract and normalize resolution 
    raw_size = config.get("image_size", [128, 128])
    img_size = raw_size if isinstance(raw_size, int) else raw_size[0]
    resolution_str = f"{img_size}x{img_size}"
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using Device Context: {device} | Resolution Target: {resolution_str}")
    
    try:
        lpips_metric = lpips.LPIPS(net=lpips_backbone).to(device)
    except Exception as e:
        print(f"Skipping LPIPS setup. Dependency or download missing: {e}")
        lpips_metric = None

    if model_path is None:
        chk_dir = config.get("checkpoint_dir", "checkpoints")
        model_path = os.path.join(chk_dir, "G_epoch99.pth") 

    model = load_model(model_path, model_type=config.get("model_type", "unet"), device=device)
    eval_transforms = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    dataset = ImageRestorationDataset(config["val_pairs_path"], transform=eval_transforms, target_transform=eval_transforms)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    print(f"Loaded {len(dataset)} evaluation pairs successfully.")

    ssim_list, psnr_list, lpips_list = [], [], []
    out_img_dir = os.path.join(config.get("results_dir", "results"), "eval_outputs")
    os.makedirs(out_img_dir, exist_ok=True)

    print("\n--- Computing Benchmarks ---")
    with torch.no_grad():
        for idx, (corrupted, clean) in enumerate(loader):
            corrupted, clean = corrupted.to(device), clean.to(device)
            restored = model(corrupted)
            
            save_image(restored * 0.5 + 0.5, os.path.join(out_img_dir, f"restored_{idx:04d}.png"))
            save_image(corrupted * 0.5 + 0.5, os.path.join(out_img_dir, f"input_{idx:04d}.png"))
            save_image(clean * 0.5 + 0.5, os.path.join(out_img_dir, f"target_{idx:04d}.png"))

            # Calculations
            np_restored = np.clip(tensor_to_numpy(restored), 0, 1)
            np_clean = np.clip(tensor_to_numpy(clean), 0, 1)
            
            current_ssim = ssim(np_clean, np_restored, channel_axis=-1, data_range=1.0)
            current_psnr = psnr(np_clean, np_restored, data_range=1.0)
            
            ssim_list.append(current_ssim)
            psnr_list.append(current_psnr)
            
            if lpips_metric is not None:
                lpips_score = lpips_metric(prepare_tensor_for_lpips(restored), prepare_tensor_for_lpips(clean)).item()
                lpips_list.append(lpips_score)

    # Bundle summary statistics
    summary = {
        'model_path': model_path,
        'model_type': config.get("model_type", "unet"),
        'num_samples': len(ssim_list),
        'ssim_scores': ssim_list,
        'psnr_scores': psnr_list,
        'lpips_scores': lpips_list
    }
    
    # Save text 
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = os.path.join(config.get("results_dir", "results"), f"evaluation_metrics_{resolution_str}_{timestamp}.txt")
    write_metrics_report(summary, report_file, resolution_str)
    
    print(f"\nEvaluation Completed Successfully!\nReport generated at: {report_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dynamic GAN Quantitative Evaluation Utility")
    parser.add_argument("--config", default="config/gan_config.yaml", help="Path to config file")
    parser.add_argument("--model", default=None, help="Force override specific model checkpoint path")
    parser.add_argument("--backbone", choices=["alex", "vgg", "squeeze"], default="alex", help="LPIPS backbone architecture")
    args = parser.parse_args()
    
    run_evaluation(config_path=args.config, model_path=args.model, lpips_backbone=args.backbone)