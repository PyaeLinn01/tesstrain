import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pytesseract
import os
import tempfile
import shutil
from pathlib import Path
from ultralytics import YOLO
import cvzone

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

# Class labels for detection
class_labels = {
    0: 'dob',
    1: 'father',
    2: 'id',
    3: 'name'
}

def detect_boxes_and_ocr(image_cv, preprocess_image, config):
    model_path = "v5.pt"
    if not os.path.exists(model_path):
        return []
    yolo_model = YOLO(model_path)
    results = yolo_model(image_cv)
    detected = []
    for r in results:
        boxes = r.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cls = int(box.cls[0])
            confidence = float(box.conf[0])
            class_name = class_labels.get(cls, "Unknown")
            roi = image_cv[y1:y2, x1:x2]
            # Convert ROI to grayscale
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            # OCR on grayscale crop
            text, _, _ = perform_ocr(roi_gray, config)
            detected.append({
                'class': class_name,
                'confidence': confidence,
                'box': (x1, y1, x2, y2),
                'text': text.strip(),
                'roi_gray': roi_gray
            })
    return detected

def main():
    # Header
    st.markdown('<h1 class="main-header">🔍 Myanmar OCR Tester</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666;">Test your fine-tuned Myanmar OCR model</p>', unsafe_allow_html=True)
    
    # Setup Tesseract
    config = setup_tesseract()
    if not config:
        return
    
    # File upload
    st.markdown('<h2 class="sub-header">📤 Upload Image</h2>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg', 'tiff', 'bmp'],
        help="Upload an image containing Myanmar text to test OCR"
    )
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        if os.path.exists("v5.pt"):
            st.markdown('<h3 class="sub-header">🟩 NRC Field Detection (YOLOv5)</h3>', unsafe_allow_html=True)
            with st.spinner("Detecting NRC fields with YOLOv5..."):
                detections = detect_boxes_and_ocr(image_cv, preprocess_image, config)
            if detections:
                image_disp = image_cv.copy()
                for det in detections:
                    x1, y1, x2, y2 = det['box']
                    class_name = det['class']
                    confidence = det['confidence']
                    cvzone.putTextRect(image_disp, f'{class_name} ({confidence:.2f})', (x1, y1 - 10), scale=1, thickness=2)
                    cvzone.cornerRect(image_disp, (x1, y1, x2 - x1, y2 - y1))
                st.subheader("Detected NRC Textboxes")
                st.image(cv2.cvtColor(image_disp, cv2.COLOR_BGR2RGB), use_column_width=True)
                for det in detections:
                    st.markdown(f"**Class:** {det['class']} | **Confidence:** {det['confidence']*100:.1f}%")
                    st.image(det['roi_gray'], caption=f"Detected: {det['class']} (grayscale)", use_column_width=True, channels="GRAY")
                    st.text_area("OCR Result", det['text'], height=100, key=f"ocr_{det['box'][0]}_{det['box'][1]}")
        else:
            st.warning("YOLOv5 model (v5.pt) not found. Please add the model to enable NRC field detection.")

if __name__ == "__main__":
    main() 