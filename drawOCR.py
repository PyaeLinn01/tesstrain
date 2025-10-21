import os
from pathlib import Path
import numpy as np
import streamlit as st
from PIL import Image
import cv2
import pytesseract

def setup_tesseract():
    iddob_model_path = "/Users/pyaelinn/tessFinetune/tesstrain/data/id_bdV3.traineddata"
    name_model_path = "/Users/pyaelinn/tessFinetune/tesstrain/data/nameV3.traineddata"
    missing = []
    if not os.path.exists(iddob_model_path):
        missing.append(iddob_model_path)
    if not os.path.exists(name_model_path):
        missing.append(name_model_path)
    if missing:
        st.error("❌ Trained model(s) not found:")
        for p in missing:
            st.write(f"- {p}")
        st.info("Please make sure you have completed the training process and placed the files correctly.")
        return False
    data_dir = os.path.join(os.getcwd(), "data")
    os.environ['TESSDATA_PREFIX'] = data_dir
    configs = {
        'iddob': r'--oem 1 --psm 6 -l id_bdV3',
        'name': r'--oem 1 --psm 6 -l nameV3',
    }
    return configs

def remove_bg(image):
    if isinstance(image, Image.Image):
        img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    else:
        img = image
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img.copy()
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.bitwise_not(thresh)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    white_bg = np.full_like(gray, 255, dtype=np.uint8)
    result = np.where(mask == 255, gray, white_bg).astype(np.uint8)
    return result

def perform_ocr(image, config):
    if len(image.shape) == 3:
        image_pil = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    else:
        image_pil = Image.fromarray(image)
    text = pytesseract.image_to_string(image_pil, config=config)
    return text.strip()

def to_pil_from_canvas(canvas_data: np.ndarray) -> Image.Image | None:
    if canvas_data is None:
        return None
    arr = canvas_data.astype(np.uint8)
    if arr.shape[-1] == 4:
        bg = np.ones_like(arr[..., :3], dtype=np.uint8) * 255
        alpha = arr[..., 3:4] / 255.0
        rgb = (arr[..., :3] * alpha + bg * (1 - alpha)).astype(np.uint8)
    else:
        rgb = arr
    return Image.fromarray(rgb)

def main():
    st.set_page_config(page_title="Draw OCR", page_icon="✍️", layout="wide")
    st.markdown('<h1 class="main-header">✍️ Draw and OCR</h1>', unsafe_allow_html=True)
    configs = setup_tesseract()
    if not configs:
        return
    with st.sidebar:
        st.markdown("### Settings")
        target = st.selectbox("Model", options=["name", "iddob"], index=0, help="Choose OCR model for the drawing")
        stroke_width = st.slider("Pen width", 1, 40, 8)
        stroke_color = st.color_picker("Pen color", "#000000")
        bg_color = st.color_picker("Background", "#FFFFFF")
        canvas_w = st.number_input("Canvas width", 256, 1024, 640, step=32)
        canvas_h = st.number_input("Canvas height", 128, 1024, 320, step=32)
        realtime = st.checkbox("Realtime OCR while drawing", value=False)
    try:
        from streamlit_drawable_canvas import st_canvas
    except Exception:
        st.error("streamlit-drawable-canvas is required. Install with: pip install streamlit-drawable-canvas")
        return
    col1, col2 = st.columns([3, 2])
    with col1:
        canvas_result = st_canvas(
            fill_color=bg_color + "00",
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_color=bg_color,
            update_streamlit=realtime,
            height=int(canvas_h),
            width=int(canvas_w),
            drawing_mode="freedraw",
            key="canvas",
        )
        c1, c2 = st.columns(2)
        submitted = c1.button("Submit to OCR", use_container_width=True)
        cleared = c2.button("Clear", use_container_width=True)
        if cleared:
            # Streamlit >= 1.25 uses st.rerun; older versions used st.experimental_rerun
            if hasattr(st, "rerun"):
                st.rerun()
            elif hasattr(st, "experimental_rerun"):
                st.experimental_rerun()
            else:
                # Fallback: no-op; user can manually rerun
                pass
    ocr_text = ""
    if realtime or submitted:
        pil_img = to_pil_from_canvas(canvas_result.image_data if canvas_result else None)
        if pil_img is not None:
            cleaned = remove_bg(pil_img)
            cfg = configs.get(target)
            ocr_text = perform_ocr(cleaned, cfg)
        else:
            st.info("Draw something first.")
    with col2:
        st.markdown("### OCR Result")
        st.text_area("Text", ocr_text, height=260)
        st.markdown("### Preview")
        if canvas_result and canvas_result.image_data is not None:
            st.image(canvas_result.image_data.astype(np.uint8), caption="Canvas", use_column_width=True)
        if ocr_text:
            st.download_button("Download OCR Text", data=ocr_text, file_name="draw_ocr.txt")

if __name__ == "__main__":
    main()

