import streamlit as st
from PIL import Image
import numpy as np
import cv2
import os
from ultralytics import YOLO

st.set_page_config(page_title="NRC Detector (YOLOv8)", page_icon="🪪", layout="wide")

def load_model(model_path: str) -> YOLO | None:
    if not os.path.exists(model_path):
        st.error(f"Model not found: {model_path}")
        return None
    try:
        return YOLO(model_path)
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None

def to_bgr(image: Image.Image) -> np.ndarray:
    return cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

def annotate(img_bgr: np.ndarray, boxes, confs) -> np.ndarray:
    out = img_bgr.copy()
    for (x1, y1, x2, y2), conf in zip(boxes, confs):
        x1, y1, x2, y2 = map(int, [x1, y1, x2, y2])
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(out, f"nrc {conf:.2f}", (x1, max(0, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return out

def main():
    st.title("NRC or Not NRC (YOLOv8)")
    model_path = st.sidebar.text_input("Model path", value="nrcDetect.pt")
    conf_thresh = st.sidebar.slider("Confidence threshold", 0.0, 1.0, 0.5, 0.01)

    uploaded = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg", "bmp", "tiff"]) 
    if uploaded is None:
        return

    image = Image.open(uploaded).convert("RGB")
    st.image(image, caption="Input Image", use_column_width=True)

    model = load_model(model_path)
    if model is None:
        return

    img_bgr = to_bgr(image)
    with st.spinner("Running detection..."):
        results = model(img_bgr)

    found = False
    annotated = img_bgr
    for r in results:
        if r.boxes is None:
            continue
        confs = r.boxes.conf.cpu().numpy() if hasattr(r.boxes.conf, 'cpu') else np.array(r.boxes.conf)
        xyxy = r.boxes.xyxy.cpu().numpy() if hasattr(r.boxes.xyxy, 'cpu') else np.array(r.boxes.xyxy)
        keep = confs >= conf_thresh
        if np.any(keep):
            found = True
            annotated = annotate(img_bgr, xyxy[keep], confs[keep])

    st.subheader("Result")
    if found:
        st.success("NRC detected")
    else:
        st.warning("Not NRC")

    st.subheader("Detections")
    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Annotated", use_column_width=True)

if __name__ == "__main__":
    main()
