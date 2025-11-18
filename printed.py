import streamlit as st
from PIL import Image
import pytesseract
import os

st.set_page_config(page_title="Printed Myanmar OCR (mya)", page_icon="🖨️", layout="wide")

def setup_tesseract():
    """Setup Tesseract with the trained model"""
    # Check if the trained model exists
    model_path = "/Users/pyaelinn/tessFinetune/tesstrain/data/mya.traineddata"
    if not os.path.exists(model_path):
        st.error(f"❌ Trained model not found at {model_path}")
        st.info("Please make sure you have completed the training process first.")
        return False
    
    # Set the TESSDATA_PREFIX environment variable to the data directory
    data_dir = os.path.join(os.getcwd(), "data")
    os.environ['TESSDATA_PREFIX'] = data_dir
    
    # Configure pytesseract to use the trained model
    custom_config = r'--oem 1 --psm 6 -l mya'
    
    return custom_config

def main():
    config = setup_tesseract()
    if not config:
        return

    st.header("Printed Myanmar OCR (no preprocessing/post-processing)")

    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=["png", "jpg", "jpeg", "tiff", "bmp"],
        help="Upload an image to test the mya model"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Original Image", use_column_width=True)

        with st.spinner("Running OCR..."):
            text = pytesseract.image_to_string(image, config=config).strip()

        st.subheader("OCR Result")
        st.text_area("Text", text, height=300)

if __name__ == "__main__":
    main()
