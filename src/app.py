import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
from models.gan_model import UNetGenerator
from models.gan_model import SimpleUNetGenerator
import os
import gdown
import tempfile
import time

st.set_page_config(
    page_title="Neural Image Restoration",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styles */
    .stApp {
        background: linear-gradient(135deg, #0f0f23 0%, #1a1a2e 50%, #16213e 100%);
        font-family: 'Inter', sans-serif;
    }
    
    /* Main container */
    .main-container {
        background: rgba(255, 255, 255, 0.02);
        backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 20px;
        padding: 2.5rem;
        margin: 2rem 0;
        box-shadow: 0 25px 50px rgba(0, 0, 0, 0.3);
    }
    
    /* Header styling */
    .header-container {
        text-align: center;
        margin-bottom: 3rem;
        padding: 2rem 0;
        background: linear-gradient(135deg, rgba(64, 224, 208, 0.1) 0%, rgba(138, 43, 226, 0.1) 100%);
        border-radius: 15px;
        border: 1px solid rgba(64, 224, 208, 0.2);
    }
    
    .main-title {
        font-size: 3.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #40E0D0 0%, #8A2BE2 50%, #FF6B6B 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 1rem;
        text-shadow: 0 0 30px rgba(64, 224, 208, 0.3);
    }
    
    .subtitle {
        font-size: 1.3rem;
        color: rgba(255, 255, 255, 0.8);
        font-weight: 400;
        letter-spacing: 0.5px;
    }
    
    /* Upload section */
    .upload-section {
        background: rgba(255, 255, 255, 0.03);
        border: 2px dashed rgba(64, 224, 208, 0.4);
        border-radius: 15px;
        padding: 2rem;
        margin: 2rem 0;
        text-align: center;
        transition: all 0.3s ease;
    }
    
    .upload-section:hover {
        border-color: rgba(64, 224, 208, 0.7);
        background: rgba(64, 224, 208, 0.05);
        transform: translateY(-2px);
    }
    
    /* Image display containers */
    .image-container {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 15px;
        padding: 1.5rem;
        margin: 1rem 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
    }
    
    .image-label {
        font-size: 1.2rem;
        font-weight: 600;
        color: #40E0D0;
        margin-bottom: 1rem;
        text-align: center;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Stats and info boxes */
    .stats-container {
        display: flex;
        justify-content: space-around;
        margin: 2rem 0;
        gap: 1rem;
    }
    
    .stat-box {
        background: linear-gradient(135deg, rgba(64, 224, 208, 0.1) 0%, rgba(138, 43, 226, 0.1) 100%);
        border: 1px solid rgba(64, 224, 208, 0.3);
        border-radius: 12px;
        padding: 1.5rem;
        text-align: center;
        flex: 1;
    }
    
    .stat-value {
        font-size: 2rem;
        font-weight: 700;
        color: #40E0D0;
        display: block;
    }
    
    .stat-label {
        font-size: 0.9rem;
        color: rgba(255, 255, 255, 0.7);
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Download button */
    .download-container {
        text-align: center;
        margin: 2rem 0;
    }
    
    .download-btn {
        background: linear-gradient(135deg, #40E0D0 0%, #8A2BE2 100%);
        border: none;
        border-radius: 25px;
        padding: 1rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        color: white;
        cursor: pointer;
        transition: all 0.3s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .download-btn:hover {
        transform: translateY(-3px);
        box-shadow: 0 15px 30px rgba(64, 224, 208, 0.4);
    }
    
    /* Progress and loading */
    .loading-container {
        text-align: center;
        padding: 2rem;
        background: rgba(255, 255, 255, 0.03);
        border-radius: 15px;
        margin: 2rem 0;
    }
    
    .loading-text {
        color: #40E0D0;
        font-size: 1.2rem;
        font-weight: 500;
        margin-bottom: 1rem;
    }
    
    /* Tech grid background */
    .tech-grid {
        position: fixed;
        top: 0;
        left: 0;
        width: 100%;
        height: 100%;
        opacity: 0.03;
        z-index: -1;
        background-image: 
            linear-gradient(rgba(64, 224, 208, 0.1) 1px, transparent 1px),
            linear-gradient(90deg, rgba(64, 224, 208, 0.1) 1px, transparent 1px);
        background-size: 50px 50px;
    }
    
    /* Success/Error messages */
    .success-msg {
        background: rgba(0, 255, 127, 0.1);
        border: 1px solid rgba(0, 255, 127, 0.3);
        border-radius: 10px;
        padding: 1rem;
        color: #00FF7F;
        margin: 1rem 0;
    }
    
    .error-msg {
        background: rgba(255, 99, 71, 0.1);
        border: 1px solid rgba(255, 99, 71, 0.3);
        border-radius: 10px;
        padding: 1rem;
        color: #FF6347;
        margin: 1rem 0;
    }
    
    /* Override Streamlit defaults */
    .stFileUploader > div > div {
        background: transparent !important;
        border: none !important;
    }
    
    .stDownloadButton > button {
        background: linear-gradient(135deg, #40E0D0 0%, #8A2BE2 100%) !important;
        border: none !important;
        border-radius: 25px !important;
        padding: 0.75rem 2rem !important;
        font-weight: 600 !important;
        text-transform: uppercase !important;
        letter-spacing: 1px !important;
        transition: all 0.3s ease !important;
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 10px 25px rgba(64, 224, 208, 0.3) !important;
    }
    
    /* Hide streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>

<div class="tech-grid"></div>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-container">
    <div class="header-container">
        <h1 class="main-title">⚡ NEURAL IMAGE RESTORATION</h1>
        <p class="subtitle">Advanced GAN-powered image enhancement technology</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Load config with Google Drive file ID
config = {
    "image_size": 256,
    "gdrive_file_id": "1tPw9pYdMAcGmQnpHzCYrLKhxp_DdsNc9",  # Replace with your actual file ID
    "checkpoint_filename": "checkpoint_epoch_99.pth"
}

def download_model_from_gdrive(file_id, filename):
    """
    Download model from Google Drive using gdown
    
    Args:
        file_id: Google Drive file ID
        filename: Name to save the file as
    
    Returns:
        Path to downloaded file
    """
    try:
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, filename)
        
        # Construct Google Drive URL
        url = f"https://drive.google.com/uc?id={file_id}"
        
        # Download the file
        st.markdown('<div class="loading-container"><div class="loading-text">🔄 Downloading neural network model...</div></div>', unsafe_allow_html=True)
        gdown.download(url, file_path, quiet=False)
        
        if os.path.exists(file_path):
            st.markdown('<div class="success-msg">✅ Model downloaded successfully!</div>', unsafe_allow_html=True)
            return file_path
        else:
            raise FileNotFoundError("Failed to download model file")
            
    except Exception as e:
        st.markdown(f'<div class="error-msg">❌ Error downloading model: {e}<br>Please check your Google Drive file ID and make sure the file is publicly accessible</div>', unsafe_allow_html=True)
        raise

# Load model
@st.cache_resource
def load_model(gdrive_file_id, filename, model_type="unet", device="cpu"):
    """
    Load the trained model from Google Drive with proper error handling
    Handles your specific saving format:
    - G_epoch{epoch}.pth (generator only)
    - checkpoint_epoch_{epoch}.pth (full checkpoint)
    
    Args:
        gdrive_file_id: Google Drive file ID
        filename: Name of the checkpoint file
        model_type: Type of model ("unet" or "simple_unet")
        device: Device to load model on
    """
    try:
        # Download model from Google Drive
        model_path = download_model_from_gdrive(gdrive_file_id, filename)
        
        # Initialize the model
        if model_type == "simple_unet":
            model = SimpleUNetGenerator().to(device)
        else:
            model = UNetGenerator().to(device)
        
        # Load the state dict
        print(f"Loading model from: {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        
        # Handle different checkpoint formats based on your saving convention
        if isinstance(checkpoint, dict):
            if 'generator_state_dict' in checkpoint:
                # Full checkpoint format: checkpoint_epoch_{epoch}.pth
                model.load_state_dict(checkpoint['generator_state_dict'])
                epoch = checkpoint.get('epoch', 'unknown')
                print(f"✓ Loaded full checkpoint from epoch {epoch}")
            elif 'model_state_dict' in checkpoint:
                model.load_state_dict(checkpoint['model_state_dict'])
            elif 'state_dict' in checkpoint:
                model.load_state_dict(checkpoint['state_dict'])
            else:
                # Assume the entire dict is the state dict (G_epoch{epoch}.pth format)
                model.load_state_dict(checkpoint)
                print("✓ Loaded generator-only checkpoint")
        else:
            # Direct state dict (G_epoch{epoch}.pth format)
            model.load_state_dict(checkpoint)
            print("✓ Loaded generator-only checkpoint")
                
        model.eval()
        return model
        
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Make sure the model architecture matches the saved checkpoint.")
        print("Available checkpoint types:")
        print("  - G_epoch{N}.pth (generator only)")
        print("  - checkpoint_epoch_{N}.pth (full checkpoint)")
        raise

device = "cpu"
try:
    model = load_model(
        config["gdrive_file_id"], 
        config["checkpoint_filename"], 
        device=device
    )
except Exception as e:
    st.markdown('<div class="error-msg">❌ Failed to load model. Please check the Google Drive configuration.</div>', unsafe_allow_html=True)
    st.stop()

st.markdown("""
<div class="main-container">
    <div class="upload-section">
        <h3 style="color: #40E0D0; margin-bottom: 1rem;">📁 UPLOAD CORRUPTED IMAGE</h3>
        <p style="color: rgba(255, 255, 255, 0.7);">Supported formats: JPG, JPEG, PNG</p>
    </div>
</div>
""", unsafe_allow_html=True)

# Image upload
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded_file is not None:
    start_time = time.time()
    
    # Create two columns for before/after
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        <div class="main-container">
            <div class="image-container">
                <div class="image-label">🔧 CORRUPTED INPUT</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.image(uploaded_file, use_container_width=True)

    # Load and preprocess
    img = Image.open(uploaded_file).convert("RGB")
    original_size = img.size

    transform = transforms.Compose([
        transforms.Resize((config["image_size"], config["image_size"])),
        transforms.ToTensor()
    ])
    
    input_tensor = transform(img).unsqueeze(0).to(device)

    # Processing indicator
    st.markdown("""
    <div class="main-container">
        <div class="loading-container">
            <div class="loading-text">🧠 Neural network processing...</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Inference
    with torch.no_grad():
        output = model(input_tensor)[0].cpu().clamp(0, 1)
    
    inference_latency = time.time() - start_time
    
    # Convert to image and resize back to original size
    restored_img = transforms.ToPILImage()(output).resize(original_size)

    with col2:
        st.markdown("""
        <div class="main-container">
            <div class="image-container">
                <div class="image-label">✨ RESTORED OUTPUT</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.image(restored_img, use_container_width=True)

    # Statistics
    st.markdown(f"""
    <div class="main-container">
        <div class="stats-container">
            <div class="stat-box">
                <span class="stat-value">{inference_latency:.3f}s</span>
                <span class="stat-label">Processing Time</span>
            </div>
            <div class="stat-box">
                <span class="stat-value">{original_size[0]}x{original_size[1]}</span>
                <span class="stat-label">Resolution</span>
            </div>
            <div class="stat-box">
                <span class="stat-value">{config["image_size"]}x{config["image_size"]}</span>
                <span class="stat-label">Model Input</span>
            </div>
            <div class="stat-box">
                <span class="stat-value">GAN</span>
                <span class="stat-label">Neural Network</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    from io import BytesIO

    # Convert to JPG in memory
    buffer = BytesIO()
    restored_img.save(buffer, format="JPEG")
    buffer.seek(0)

    # Download section
    st.markdown("""
    <div class="main-container">
        <div class="download-container">
            <h3 style="color: #40E0D0; margin-bottom: 1.5rem;">💾 DOWNLOAD ENHANCED IMAGE</h3>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Download button for JPG
    st.download_button(
        label="⬇️ Download Restored Image (JPG)",
        data=buffer,
        file_name="neural_restored.jpg",
        mime="image/jpeg"
    )

else:
    # Instructions when no file is uploaded
    st.markdown("""
    <div class="main-container">
        <div style="text-align: center; padding: 3rem 2rem; color: rgba(255, 255, 255, 0.6);">
            <h3 style="color: #40E0D0; margin-bottom: 1.5rem;">🚀 HOW IT WORKS</h3>
            <p style="font-size: 1.1rem; line-height: 1.8; margin-bottom: 1.5rem;">
                Our advanced Generative Adversarial Network (GAN) uses deep learning to intelligently restore and enhance corrupted images.
            </p>
            <div style="display: flex; justify-content: space-around; margin: 2rem 0; flex-wrap: wrap; gap: 1rem;">
                <div style="flex: 1; min-width: 200px; padding: 1rem;">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">📤</div>
                    <strong style="color: #40E0D0;">Upload</strong><br>
                    Select your corrupted image
                </div>
                <div style="flex: 1; min-width: 200px; padding: 1rem;">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">🧠</div>
                    <strong style="color: #40E0D0;">Process</strong><br>
                    AI analyzes and restores
                </div>
                <div style="flex: 1; min-width: 200px; padding: 1rem;">
                    <div style="font-size: 2rem; margin-bottom: 0.5rem;">💾</div>
                    <strong style="color: #40E0D0;">Download</strong><br>
                    Get your enhanced image
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)