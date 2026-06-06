# -*- coding: utf-8 -*-
"""
Pix2Pix GAN Training Pipeline
"""

import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms

# Import the custom classes from gan_model.py
from src.data_preprocessing.dataset import ImagePairDataset
from src.models.gan_model import UNetGenerator, PatchDiscriminator

def train():
    config_path = "config/gan_config.yaml"
    if not os.path.exists(config_path):
        print("Error: config/gan_config.yaml not found!")
        return

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    # FORCE PyTorch to use the GPU, or crash immediately.
    assert torch.cuda.is_available(), "CRASH: PyTorch is refusing to see the GPU!"
    device = torch.device("cuda:0")
    print(f"Initializing Training Pipeline...")
    print(f"Hardware detected: {device.type.upper()}")

    transform = transforms.Compose([
        transforms.Resize((config.get('size', 256), config.get('size', 256))),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    dataset = ImagePairDataset(
        manifest_file=config["train_pairs_path"],
        raw_root=config["raw_image_root"],
        corrupted_root=config["corrupted_image_root"],
        transform=transform
    )

    if len(dataset) == 0:
        print("Error: Dataset is empty!")
        return

    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)
    print(f" -> Dataset loaded successfully: {len(dataset)} image pairs.")

    netG = UNetGenerator().to(device)
    netD = PatchDiscriminator().to(device)

    optG = optim.Adam(netG.parameters(), lr=config['learning_rate_g'], betas=(config['beta1'], 0.999))
    optD = optim.Adam(netD.parameters(), lr=config['learning_rate_d'], betas=(config['beta1'], 0.999))

    criterionGAN = nn.BCEWithLogitsLoss()
    criterionL1 = nn.L1Loss()
    lambda_L1 = config['lambda_L1']

    os.makedirs(config['checkpoint_dir'], exist_ok=True)

    epochs = config['epochs']
    print(f"\nStarting training for {epochs} epochs...\n")

    for epoch in range(epochs):
        loss_D_val = 0.0
        loss_G_val = 0.0

        for i, (corrupted, real) in enumerate(dataloader):
            corrupted = corrupted.to(device)
            real = real.to(device)

            # Train Discriminator
            optD.zero_grad()
            fake = netG(corrupted)
            pred_fake = netD(fake.detach(), corrupted)
            loss_D_fake = criterionGAN(pred_fake, torch.zeros_like(pred_fake))
            pred_real = netD(real, corrupted)
            loss_D_real = criterionGAN(pred_real, torch.ones_like(pred_real))
            loss_D = (loss_D_fake + loss_D_real) * 0.5
            loss_D.backward()
            optD.step()

            # Train Generator
            optG.zero_grad()
            pred_fake_G = netD(fake, corrupted)
            loss_G_GAN = criterionGAN(pred_fake_G, torch.ones_like(pred_fake_G))
            loss_G_L1 = criterionL1(fake, real) * lambda_L1
            loss_G = loss_G_GAN + loss_G_L1
            loss_G.backward()
            optG.step()

            loss_D_val = loss_D.item()
            loss_G_val = loss_G.item()

        print(f"Epoch [{epoch+1}/{epochs}] | Critic Loss: {loss_D_val:.4f} | Restorer Loss: {loss_G_val:.4f}")

        if (epoch + 1) % 5 == 0 or (epoch + 1) == epochs:
            checkpoint_path = os.path.join(config['checkpoint_dir'], f"G_epoch_{epoch+1}.pth")
            torch.save(netG.state_dict(), checkpoint_path)
            print(f" 💾 Saved brain to: {checkpoint_path}")

    print("\nTraining Complete!")

if __name__ == "__main__":
    train()