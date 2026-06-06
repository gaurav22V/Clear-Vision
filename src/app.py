import os
import time
import tempfile
import gdown
import torch
import streamlit as st
import gdown
from PIL import Image
from torchvision import transforms
from src.models.gan_model import UNetGenerator

st.set_page_config(
    page_title="ClearVision",
    page_icon="✨",
    layout="centered"
)

CONFIG = {
    "image_size": 256,
    "gdrive_id": "1sv_NFc6Cvfj9LA0kdzm8xgPaRQi9wM4d", 
    "checkpoint_path": "G_epoch_100.pth"
}

@st.cache_resource
def load_restoration_model(device="cpu"):
    try:
        # Download from Google-Drive
        if not os.path.exists(CONFIG["checkpoint_path"]):
            with st.spinner("Server initializing: Downloading weights from cloud..."):
                gdown.download(id=CONFIG["gdrive_id"], output=CONFIG["checkpoint_path"], quiet=False)

        # Load the Model
        model = UNetGenerator().to(device)
        checkpoint = torch.load(CONFIG["checkpoint_path"], map_location=device)
        
        if isinstance(checkpoint, dict):
            if 'generator_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['generator_state_dict'])
            elif 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            else:
                model.load_state_dict(checkpoint)
        else:
            model.load_state_dict(checkpoint)
            
        model.eval()
        return model
    except Exception as e:
        st.error(f"Initialization failure: {e}")
        st.stop()

device = "cuda" if torch.cuda.is_available() else "cpu"

model = load_restoration_model(device)

st.title("ClearVision Studio")
st.caption("AI-powered image restoration and artifact removal pipeline")
st.divider()

uploaded_file = st.file_uploader(
    "Upload a degraded image asset (JPG, JPEG, PNG)", 
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    # Read image dimensions
    input_image = Image.open(uploaded_file).convert("RGB")
    original_size = input_image.size
    
    # Process through pipeline
    start_time = time.time()
    
    preprocess = transforms.Compose([
        transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    
    input_tensor = preprocess(input_image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output_tensor = model(input_tensor)[0].cpu().clamp(-1, 1)
        output_tensor = (output_tensor + 1) / 2
        
    restored_image = transforms.ToPILImage()(output_tensor).resize(original_size)
    latency = time.time() - start_time

    tab1, tab2 = st.tabs(["Comparison View", "Export Asset"])
    
    with tab1:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Source Input")
            st.image(input_image, use_container_width=True)
        with col2:
            st.subheader("Restored Output")
            st.image(restored_image, use_container_width=True)
            
        st.divider()
        m1, m2, m3 = st.columns(3)
        m1.metric("Inference Latency", f"{latency:.3f}s")
        m2.metric("Target Resolution", f"{original_size[0]} × {original_size[1]}")
        m3.metric("Model Input Space", f"{CONFIG['image_size']}³")

    with tab2:
        st.subheader("Save Processed Media")
        
        # Buffer conversion
        from io import BytesIO
        buffer = BytesIO()
        restored_image.save(buffer, format="JPEG", quality=95)
        buffer.seek(0)
        
        st.download_button(
            label="Download High-Fidelity Restored JPG",
            data=buffer,
            file_name="clearvision_restored.jpg",
            mime="image/jpeg",
            use_container_width=True
        )

else:
    # Placeholder
    st.info("Drop a corrupted or noisy image above to begin restoration.")