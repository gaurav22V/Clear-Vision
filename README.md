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

## Dataset & Training Details
* **Dataset Base:** 250 custom-scraped images standardized to 256x256 via Lanczos resampling.
* **Automated Degradation Pipeline (`corruption.py`):**
  * Gaussian Blur (radius 1.0 - 3.5) with 70% probability.
  * Synthetic Digital Noise (mean = 0, sigma = 25) with 50% probability.
  * Random black structural masks (1 to 3 blocks) with 30% probability.
* **Hyperparameters:** 100 Epochs, Adam Optimizer (LR = 0.0002, Beta1 = 0.5).

## Project Structure
```text
clear-vision-gan/       
 |- requirements.txt        # Production dependencies
 |- Dockerfile              # Python 3.10 deployment container
 |- src/
    |- app.py               # Main Streamlit web application
    |- models/
       |- gan_model.py      # UNet & PatchGAN architectures
       |- gan_train.py      # Training loop and loss functions
       |- gan_eval.py       # Evaluate model 
    |- data_preprocessing/
        |- dataset.py       # Custom PyTorch Dataset and augmentation logic
        |- corruption.py    # Adding blur, noise, and masks to raw data
    |- scraper/
        |- scraper.py       # web scraper to build the raw image dataset
```

## Quick Start
```bash
git clone (<repository url>)
cd Clear-Vision
pip install -r requirements.txt
streamlit run src/app.py