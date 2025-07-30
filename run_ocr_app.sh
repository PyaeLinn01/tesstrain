#!/bin/bash

# Check if the trained model exists
if [ ! -f "data/alg.traineddata" ]; then
    echo "❌ Error: Trained model not found at data/alg.traineddata"
    echo "Please make sure you have completed the training process first."
    exit 1
fi

# Check if required packages are installed
echo "🔍 Checking dependencies..."

# Check if streamlit is installed
if ! python3 -c "import streamlit" 2>/dev/null; then
    echo "📦 Installing required packages..."
    pip3 install -r requirements.txt
fi

# Check if tesseract is installed
if ! command -v tesseract &> /dev/null; then
    echo "❌ Error: Tesseract is not installed"
    echo "Please install Tesseract first:"
    echo "  macOS: brew install tesseract"
    echo "  Ubuntu: sudo apt-get install tesseract-ocr"
    exit 1
fi

echo "✅ All dependencies are ready!"
echo "🚀 Starting Myanmar OCR Tester..."

# Run the Streamlit app
streamlit run ocr_test_app.py --server.port 8501 --server.address 0.0.0.0 