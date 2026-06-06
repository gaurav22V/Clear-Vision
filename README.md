# Clear Vision GAN: Image Restoration Pipeline

A PyTorch Generative Adversarial Network (GAN) designed to restore heavily corrupted, blurry, and noisy images. This repository contains the complete deep learning pipeline, from the custom dataset generator to the Dockerized Streamlit web deployment.

## Tech Stack
* **Deep Learning:** PyTorch, Torchvision
* **Web UI:** Streamlit
* **Image Processing:** Pillow (PIL), NumPy
* **Cloud & Deployment:** Docker, Hugging Face Spaces, Google Drive API (gdown)

## Architecture & Features
* **Model:** Custom U-Net Generator with a PatchGAN Discriminator.
* **Dynamic Augmentation:** Synchronized spatial transformations to prevent input mismatch during training.
* **Cloud Deployment:** Dockerized Streamlit app configured for Hugging Face Spaces.
* **Weight Management:** Automatically bypasses 100MB Git repository limits by downloading `.pth` checkpoints directly from Google Drive on server boot.

## Project Structure
```text
clear-vision-gan/
 |- app.py                  # Main Streamlit web application
 |- dataset.py              # Custom PyTorch Dataset and augmentation logic
 |- requirements.txt        # Production dependencies
 |- Dockerfile              # Python 3.10 deployment container
 |- src/
    |- models/
       |- gan_model.py      # UNet & PatchGAN architectures
       |- gan_train.py      # Training loop and loss functions
       |- gan_eval.py       # Evaluate model 