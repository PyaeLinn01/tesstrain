import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pytesseract
import os
import tempfile
import shutil
from pathlib import Path
import json
import re
from difflib import SequenceMatcher

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
    model_path = "/Users/pyaelinn/tessFinetune/tesstrain/data/nameV3.traineddata"
    if not os.path.exists(model_path):
        st.error(f"❌ Trained model not found at {model_path}")
        st.info("Please make sure you have completed the training process first.")
        return False
    
    # Set the TESSDATA_PREFIX environment variable to the data directory
    data_dir = os.path.join(os.getcwd(), "data")
    os.environ['TESSDATA_PREFIX'] = data_dir
    
    # Configure pytesseract to use the trained model
    custom_config = r'--oem 1 --psm 6 -l nameV3'
    
    return custom_config

def remove_background_canvas_style(image):
    """Remove background and return a clean grayscale image on white background for better OCR.

    Accepts either PIL.Image or OpenCV BGR/gray numpy array and returns a uint8 grayscale numpy array.
    """
    # Normalize input to OpenCV BGR/gray numpy array
    if isinstance(image, Image.Image):
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    else:
        img = image

    # Convert to grayscale if needed
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()

    # Slight blur for robustness
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Otsu threshold to separate foreground/background
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Invert so text/foreground = white, background = black
    mask = cv2.bitwise_not(thresh)

    # Clean mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # Compose on white background in grayscale
    white_bg = np.full_like(gray, 255, dtype=np.uint8)
    result = np.where(mask == 255, gray, white_bg).astype(np.uint8)
    return result

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

def _load_nrc_my_list(nrc_path: Path) -> list[str]:
    """Load all Myanmar NRC prefix strings from nrc.json into a flat list.

    Each entry looks like: "၁၂/သလန(နိုင်)". We'll match against these.
    """
    try:
        with open(nrc_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        result = []
        # nrc.json is a dict of id -> list of {my,en}
        for _, arr in data.items():
            for item in arr:
                my = item.get("my")
                if my:
                    result.append(my)
        return result
    except Exception as e:
        st.warning(f"Failed to load NRC list: {e}")
        return []

def _similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio()

def _split_serial(line: str) -> tuple[str, str]:
    """Split line into prefix (non-serial) and trailing serial digits (Myanmar or ASCII).
    If no trailing digits, serial is empty and prefix is the whole line.
    """
    # Match trailing run of ASCII or Myanmar digits
    m = re.search(r"([0-9\u1040-\u1049]+)$", line)
    if m:
        start = m.start(1)
        return line[:start], line[start:]
    return line, ""

def _extract_marker_and_body(prefix: str) -> tuple[str, str, str]:
    """Extract leading marker like 'အမှတ်' and its separator, and the rest body.
    Returns (marker, sep, body). If no marker, returns ("", "", prefix).
    """
    m = re.match(r"^(အမှတ်)([_\s\-]?)", prefix)
    if m:
        marker = m.group(1)
        sep = m.group(2) or "_"
        body = prefix[m.end():]
        return marker, sep, body
    return "", "", prefix

def _normalize_for_match(s: str) -> str:
    """Normalize a string to improve fuzzy matching by removing spaces and underscores.
    Keep Myanmar letters, digits, slash and parentheses.
    """
    s = s.replace(" ", "").replace("_", "")
    return s

def _normalize_prefix_for_compare(s: str) -> str:
    """Normalize a candidate prefix for strict equality check.
    - Map '-' to '/' so '၉-မရတ(နိုင်)' equals '၉/မရတ(နိုင်)'
    - Keep only Myanmar letters, '/', '()'
    - Remove spaces/underscores
    """
    s = s.replace("-", "/")
    s = re.sub(r"[^\u1000-\u109F/()]+", "", s)
    return s.replace(" ", "").replace("_", "")

MY_DIGITS_MAP = {
    "0": "၀", "1": "၁", "2": "၂", "3": "၃", "4": "၄", "5": "၅", "6": "၆", "7": "၇", "8": "၈", "9": "၉",
    "၀": "၀", "၁": "၁", "၂": "၂", "၃": "၃", "၄": "၄", "၅": "၅", "၆": "၆", "၇": "၇", "၈": "၈", "၉": "၉",
}

def _to_myanmar_digits(digits: str) -> str:
    return "".join(MY_DIGITS_MAP.get(ch, ch) for ch in digits)

def _format_candidate_to_prefix(cand: str) -> str:
    """Keep prefix with slash: '၁၂/သလန(နိုင်)' remains unchanged."""
    return cand

def _ensure_six_myanmar_digits(digits: str) -> str:
    md = _to_myanmar_digits(digits)
    if len(md) >= 6:
        return md[:6]
    return ("၀" * (6 - len(md))) + md

def correct_id_line(line: str, nrc_list: list[str], min_ratio: float = 0.6) -> str:
    """Force NRC ID into exact format: 'အမှတ်_xx-yyy(z)aaaaaa'.
    - Always output leading 'အမှတ်_'
    - Choose nearest prefix from nrc.json, rendered as 'xx-yyy(z)'
    - Serial must be exactly 6 Myanmar digits
    - Skip lines starting with 'မွေး'
    """
    if line.strip().startswith("မွေး"):
        return line

    # Remove duplicated marker fragments and underscores
    raw = re.sub(r"အမှတ်+", "အမှတ်", line)
    raw = raw.replace("__", "_")
    prefix, serial = _split_serial(raw.strip())
    # Ignore any existing marker; force 'အမှတ်_'
    _, _, body = _extract_marker_and_body(prefix)
    if not body:
        body = prefix
    # Sanitize body: keep Myanmar letters, '/', '()' and allow '-' for equality check
    body = re.sub(r"[^\u1000-\u109F/()\-]+", "", body)
    # Early accept: if OCR body exactly matches an NRC prefix (allow '-' vs '/') and serial is 6 digits, do not change
    body_cmp = _normalize_prefix_for_compare(body)
    nrc_cmp_set = {_normalize_prefix_for_compare(c) for c in nrc_list}
    serial_digits = re.sub(r"[^0-9\u1040-\u1049]", "", serial)
    if body_cmp in nrc_cmp_set and len(_to_myanmar_digits(serial_digits)) == 6:
        return line.strip()
    # Otherwise proceed with fuzzy matching
    body_norm = _normalize_for_match(body)
    best = None
    best_score = 0.0
    for cand in nrc_list:
        score = _similarity(body_norm, _normalize_for_match(cand))
        if score > best_score:
            best_score = score
            best = cand

    if not best or best_score < min_ratio:
        return line

    fixed_body = _format_candidate_to_prefix(best)
    serial6 = _ensure_six_myanmar_digits(serial)
    return f"အမှတ်_{fixed_body}{serial6}"

def postprocess_text(text: str, nrc_list: list[str]) -> str:
    """Produce a single-line corrected NRC in format 'အမှတ်_xx/yyy(z)aaaaaa'."""
    blob = " ".join(text.split())  # collapse whitespace/newlines
    return correct_id_line(blob, nrc_list)

# ========================= DOB Postprocessing ============================
ASCII_FROM_MY = {
    "၀": "0", "၁": "1", "၂": "2", "၃": "3", "၄": "4",
    "၅": "5", "၆": "6", "၇": "7", "၈": "8", "၉": "9",
}

def _my_to_ascii_digits(s: str) -> str:
    return "".join(ASCII_FROM_MY.get(ch, ch) for ch in s)

def _clamp_int(val: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, val))

def _extract_digits_groups(s: str) -> list[str]:
    cleaned = re.sub(r"[^0-9\u1040-\u1049\.\-_/]", " ", s)
    return re.findall(r"[0-9\u1040-\u1049]{1,4}", cleaned)

def _dob_from_groups(groups: list[str]) -> tuple[str, str, str] | None:
    g_ascii = [_my_to_ascii_digits(g) for g in groups]
    year_idx = None
    for i, g in enumerate(g_ascii):
        if len(g) == 4 and g.isdigit() and 1900 <= int(g) <= 2099:
            year_idx = i
    dd = mm = yyyy = None
    if year_idx is not None:
        yyyy = g_ascii[year_idx]
        prev = [g for g in g_ascii[:year_idx] if g.isdigit()]
        if len(prev) >= 2:
            dd, mm = prev[0], prev[1]
    if not (dd and mm and yyyy):
        all_digits = re.sub(r"[^0-9]", "", "".join(g_ascii))
        # Build dd, mm, yyyy with defaults if insufficient digits
        dd = (all_digits[0:2] if len(all_digits) >= 2 else "01")
        mm = (all_digits[2:4] if len(all_digits) >= 4 else (all_digits[2:3] if len(all_digits) >= 3 else "01"))
        yyyy = (all_digits[4:8] if len(all_digits) >= 8 else "2000")
    # Dedupe duplicated month only if the two-digit value is > 12 (e.g., '88' -> '8'),
    # but preserve valid '11' or '12'.
    if len(mm) >= 2 and mm[0] == mm[1]:
        try:
            mm_val = int(mm[:2])
        except ValueError:
            mm_val = 99
        if mm_val > 12:
            mm = mm[0]
    # Fix impossible day like '86' by dropping the second digit -> '8' before clamping
    if len(dd) >= 2:
        try:
            dd_val = int(dd[:2])
        except ValueError:
            dd_val = 99
        if dd_val > 31:
            dd = dd[0]
    dd_i = _clamp_int(int(dd[:2] or 0), 1, 31)
    mm_i = _clamp_int(int(mm[:2] or 0), 1, 12)
    # Ensure year starts with 1 or 2; otherwise default to 2000
    yyyy_str = yyyy[:4] if len(yyyy) >= 4 else "2000"
    if not (yyyy_str and yyyy_str[0] in ("1", "2")):
        yyyy_str = "2000"
    yyyy_i = int(yyyy_str)
    return f"{dd_i:02d}", f"{mm_i:02d}", f"{yyyy_i:04d}"

def correct_dob_line(line: str) -> str:
    """Force DOB into 'မွေးသက္ကရာဇ်_dd.mm.yyyy'. If text is ID (starts with 'အမှတ်'), leave as-is."""
    s = line.strip()
    if s.startswith("အမှတ်"):
        return line
    groups = _extract_digits_groups(s)
    parsed = _dob_from_groups(groups)
    if not parsed:
        return line
    dd, mm, yyyy = parsed
    return f"မွေးသက္ကရာဇ်_{_to_myanmar_digits(dd)}.{_to_myanmar_digits(mm)}.{_to_myanmar_digits(yyyy)}"
# ========================================================================

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
    
    # Load NRC list for postprocessing
    nrc_path = Path("/Users/pyaelinn/tessFinetune/tesstrain/nrc.json")
    nrc_list = _load_nrc_my_list(nrc_path)

    # Sidebar
    st.sidebar.markdown("## ⚙️ Settings")
    
    # Preprocessing info
    st.sidebar.markdown("### Image Preprocessing")
    st.sidebar.info("Background removal is applied automatically before OCR.")
    
    # OCR settings
    st.sidebar.markdown("### OCR Settings")
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=100.0,
        value=60.0,
        help="Minimum confidence for text detection"
    )
    
    # Set up upload change tracking
    if "upload_counter" not in st.session_state:
        st.session_state.upload_counter = 0

    def _on_upload_change():
        st.session_state.upload_counter += 1

    run_id = st.session_state.upload_counter

    # File upload
    st.markdown('<h2 class="sub-header">📤 Upload Image</h2>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg', 'tiff', 'bmp'],
        help="Upload an image containing Myanmar text to test OCR",
        key="uploader_file",
        on_change=_on_upload_change
    )
    
    if uploaded_file is not None:
        # Display original image
        st.markdown('<h3 class="sub-header">📷 Original Image</h3>', unsafe_allow_html=True)
        
        # Load image from fresh bytes to avoid any internal pointer issues
        uploaded_file.seek(0)
        img_bytes = uploaded_file.read()
        from io import BytesIO
        image = Image.open(BytesIO(img_bytes))
        st.image(image, caption="Original Image", use_column_width=True)
        
        # Preprocess image (background removal)
        preprocessed_image = remove_background_canvas_style(image)
        
        # Display preprocessed image
        st.markdown('<h3 class="sub-header">🔧 Preprocessed Image (Background Removed)</h3>', unsafe_allow_html=True)
        st.image(preprocessed_image, caption="Preprocessed: Background Removed", use_column_width=True)
        
        # Perform OCR
        st.markdown('<h3 class="sub-header">📝 OCR Results</h3>', unsafe_allow_html=True)
        
        with st.spinner("Performing OCR..."):
            text, boxes, data = perform_ocr(preprocessed_image, config)
            corrected = postprocess_text(text, nrc_list) if nrc_list else text
        
        if text:
            # Display results in columns
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.markdown('<div class="result-box">', unsafe_allow_html=True)
                st.markdown("**Extracted Text:**")
                st.text_area("OCR Result", text, height=200, key=f"ocr_result_{run_id}")
                st.markdown("**Corrected Text (ID postprocessed):**")
                st.text_area("Corrected Result", corrected, height=200, key=f"ocr_corrected_{run_id}")
                # DOB corrected (if applicable)
                dob_corrected = correct_dob_line(text)
                st.markdown("**Corrected DOB (if detected):**")
                st.text_area("DOB Result", dob_corrected, height=100, key=f"ocr_dob_corrected_{run_id}")
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
                annotated_image = draw_boxes(preprocessed_image, boxes)
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
            st.warning("⚠️ No text was detected in the image. Try a clearer image, crop to the text region, or adjust confidence threshold.")
    
    # Instructions
    with st.expander("ℹ️ How to use this app"):
        st.markdown("""
        ### Instructions:
        1. **Upload an image** containing Myanmar text
        2. **Preprocessing** is applied automatically: background is removed and text is composed on a clean white background for better OCR.
        3. **Adjust confidence threshold** if needed
        4. **View results** - The app will show:
           - Extracted text
           - Character bounding boxes
           - Statistics
        5. **Download results** as text file or annotated image
        
        ### Tips for better results:
        - Use clear, high-resolution images
        - Ensure reasonable contrast between text and background
        - If results are poor, try cropping to the text area before upload
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