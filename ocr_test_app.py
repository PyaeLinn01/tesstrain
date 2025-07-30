import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pytesseract
import os
import tempfile
import shutil
from pathlib import Path

# Configure page
st.set_page_config(
    page_title="Myanmar OCR Tester",
    page_icon="🔍",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #2c3e50;
        margin-bottom: 1rem;
    }
    .result-box {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin: 1rem 0;
    }
    .upload-area {
        border: 2px dashed #ccc;
        border-radius: 0.5rem;
        padding: 2rem;
        text-align: center;
        background-color: #fafafa;
    }
    .metric-card {
        background-color: white;
        padding: 1rem;
        border-radius: 0.5rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def setup_tesseract():
    """Setup Tesseract with the trained model"""
    # Check if the trained model exists
    model_path = "data/alg.traineddata"
    if not os.path.exists(model_path):
        st.error(f"❌ Trained model not found at {model_path}")
        st.info("Please make sure you have completed the training process first.")
        return False
    
    # Set the TESSDATA_PREFIX environment variable to the data directory
    data_dir = os.path.join(os.getcwd(), "data")
    os.environ['TESSDATA_PREFIX'] = data_dir
    
    # Configure pytesseract to use the trained model
    custom_config = r'--oem 1 --psm 6 -l alg'
    
    return custom_config

def preprocess_image(image):
    """Preprocess image for better OCR results"""
    # Convert PIL to OpenCV format
    if isinstance(image, Image.Image):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply different preprocessing techniques
    processed_images = {}
    
    # Original grayscale
    processed_images['Original'] = gray
    
    # Binary threshold
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    processed_images['Binary'] = binary
    
    # Adaptive threshold
    adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    processed_images['Adaptive'] = adaptive
    
    # Gaussian blur + threshold
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, blurred_binary = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    processed_images['Blurred'] = blurred_binary
    
    return processed_images

def perform_ocr(image, config):
    """Perform OCR on the image"""
    try:
        # Convert OpenCV image to PIL for pytesseract
        if len(image.shape) == 3:
            image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            image_pil = Image.fromarray(image)
        
        # Perform OCR
        text = pytesseract.image_to_string(image_pil, config=config)
        
        # Get bounding boxes
        boxes = pytesseract.image_to_boxes(image_pil, config=config)
        
        # Get detailed data
        data = pytesseract.image_to_data(image_pil, config=config, output_type=pytesseract.Output.DICT)
        
        return text.strip(), boxes, data
    except Exception as e:
        st.error(f"OCR Error: {str(e)}")
        return "", "", ""

def draw_boxes(image, boxes):
    """Draw bounding boxes on the image"""
    if len(image.shape) == 2:
        image_color = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        image_color = image.copy()
    
    # Parse boxes
    for box in boxes.split('\n'):
        if box:
            parts = box.split()
            if len(parts) >= 6:
                char = parts[0]
                x1, y1, x2, y2 = int(parts[1]), int(parts[2]), int(parts[3]), int(parts[4])
                # OpenCV coordinates are different from Tesseract
                cv2.rectangle(image_color, (x1, image_color.shape[0] - y1), (x2, image_color.shape[0] - y2), (0, 255, 0), 2)
                cv2.putText(image_color, char, (x1, image_color.shape[0] - y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
    
    return image_color

def main():
    # Header
    st.markdown('<h1 class="main-header">🔍 Myanmar OCR Tester</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Test your fine-tuned Myanmar OCR model</p>', unsafe_allow_html=True)
    
    # Setup Tesseract
    config = setup_tesseract()
    if not config:
        return
    
    # Sidebar
    st.sidebar.markdown("## ⚙️ Settings")
    
    # Preprocessing options
    st.sidebar.markdown("### Image Preprocessing")
    preprocessing_method = st.sidebar.selectbox(
        "Choose preprocessing method:",
        ["Original", "Binary", "Adaptive", "Blurred"],
        help="Different preprocessing methods can improve OCR accuracy"
    )
    
    # OCR settings
    st.sidebar.markdown("### OCR Settings")
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=100.0,
        value=60.0,
        help="Minimum confidence for text detection"
    )
    
    # File upload
    st.markdown('<h2 class="sub-header">📤 Upload Image</h2>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg', 'tiff', 'bmp'],
        help="Upload an image containing Myanmar text to test OCR"
    )
    
    if uploaded_file is not None:
        # Display original image
        st.markdown('<h3 class="sub-header">📷 Original Image</h3>', unsafe_allow_html=True)
        
        # Load image
        image = Image.open(uploaded_file)
        st.image(image, caption="Original Image", use_column_width=True)
        
        # Preprocess image
        processed_images = preprocess_image(image)
        
        # Display preprocessed image
        st.markdown(f'<h3 class="sub-header">🔧 Preprocessed Image ({preprocessing_method})</h3>', unsafe_allow_html=True)
        
        selected_image = processed_images[preprocessing_method]
        st.image(selected_image, caption=f"Preprocessed: {preprocessing_method}", use_column_width=True)
        
        # Perform OCR
        st.markdown('<h3 class="sub-header">📝 OCR Results</h3>', unsafe_allow_html=True)
        
        with st.spinner("Performing OCR..."):
            text, boxes, data = perform_ocr(selected_image, config)
        
        if text:
            # Display results in columns
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                st.markdown("**Extracted Text:**")
                st.text_area("OCR Result", text, height=200, key="ocr_result")
                st.markdown('</div>', unsafe_allow_html=True)
            
            with col2:
                # Statistics
                st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                st.markdown("**📊 Statistics**")
                
                # Count characters and words
                char_count = len(text.replace(' ', ''))
                word_count = len(text.split())
                line_count = len(text.split('\n'))
                
                st.metric("Characters", char_count)
                st.metric("Words", word_count)
                st.metric("Lines", line_count)
                
                # Confidence metrics if available
                if data and 'conf' in data:
                    confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                    if confidences:
                        avg_confidence = sum(confidences) / len(confidences)
                        st.metric("Avg Confidence", f"{avg_confidence:.1f}%")
                st.markdown('</div>', unsafe_allow_html=True)
            
            # Show image with bounding boxes
            st.markdown('<h3 class="sub-header">🎯 Character Detection</h3>', unsafe_allow_html=True)
            
            if boxes:
                annotated_image = draw_boxes(selected_image, boxes)
                st.image(annotated_image, caption="Image with character bounding boxes", use_column_width=True)
            
            # Download results
            st.markdown('<h3 class="sub-header">💾 Download Results</h3>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Download text
                st.download_button(
                    label="📄 Download Text",
                    data=text,
                    file_name="ocr_result.txt",
                    mime="text/plain"
                )
            
            with col2:
                # Download annotated image
                if boxes:
                    annotated_pil = Image.fromarray(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
                    img_buffer = tempfile.NamedTemporaryFile(delete=False, suffix='.png')
                    annotated_pil.save(img_buffer.name)
                    
                    with open(img_buffer.name, 'rb') as f:
                        st.download_button(
                            label="🖼️ Download Annotated Image",
                            data=f.read(),
                            file_name="annotated_image.png",
                            mime="image/png"
                        )
                    
                    os.unlink(img_buffer.name)
        
        else:
            st.warning("⚠️ No text was detected in the image. Try adjusting the preprocessing method or upload a different image.")
    
    # Instructions
    with st.expander("ℹ️ How to use this app"):
        st.markdown("""
        ### Instructions:
        1. **Upload an image** containing Myanmar text
        2. **Choose preprocessing method** - Different methods work better for different image types:
           - **Original**: No preprocessing
           - **Binary**: Good for high contrast images
           - **Adaptive**: Good for varying lighting conditions
           - **Blurred**: Good for noisy images
        3. **Adjust confidence threshold** if needed
        4. **View results** - The app will show:
           - Extracted text
           - Character bounding boxes
           - Statistics
        5. **Download results** as text file or annotated image
        
        ### Tips for better results:
        - Use clear, high-resolution images
        - Ensure good contrast between text and background
        - Try different preprocessing methods
        - The model was trained on Myanmar text, so it works best with Myanmar characters
        """)
    
    # Model information
    with st.expander("🔬 Model Information"):
        st.markdown("""
        ### Trained Model Details:
        - **Model Name**: alg
        - **Base Model**: Myanmar (mya)
        - **Training Data**: 1,406 images
        - **Final Error Rate**: 0.111% (BCER)
        - **Training Iterations**: 10,000
        
        ### Technical Details:
        - **OCR Engine**: Tesseract 4.x
        - **Model Type**: LSTM-based neural network
        - **Language**: Myanmar (alg)
        - **Preprocessing**: Line-level text recognition
        """)

if __name__ == "__main__":
    main() 