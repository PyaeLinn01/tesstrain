# 🔍 Myanmar OCR Tester

A Streamlit web application for testing your fine-tuned Myanmar OCR model.

## 🚀 Quick Start

### Option 1: Using the provided script (Recommended)
```bash
./run_ocr_app.sh
```

### Option 2: Manual setup
```bash
# Install dependencies
pip3 install -r requirements.txt

# Run the app
streamlit run ocr_test_app.py
```

## 📋 Prerequisites

1. **Trained Model**: Make sure you have completed the training process and have `data/alg.traineddata`
2. **Tesseract**: Install Tesseract OCR engine
   - macOS: `brew install tesseract`
   - Ubuntu: `sudo apt-get install tesseract-ocr`
3. **Python Dependencies**: The app will automatically install required packages

## 🎯 Features

### Core Features
- **Image Upload**: Upload PNG, JPG, JPEG, TIFF, or BMP images
- **Multiple Preprocessing**: Choose from 4 different image preprocessing methods
- **Real-time OCR**: Perform OCR with your trained model
- **Character Detection**: Visualize character bounding boxes
- **Statistics**: View character count, word count, and confidence metrics
- **Download Results**: Download extracted text and annotated images

### Preprocessing Methods
1. **Original**: No preprocessing (good for clean images)
2. **Binary**: Otsu thresholding (good for high contrast)
3. **Adaptive**: Adaptive thresholding (good for varying lighting)
4. **Blurred**: Gaussian blur + thresholding (good for noisy images)

### Settings
- **Confidence Threshold**: Adjust minimum confidence for text detection
- **Preprocessing Method**: Choose the best method for your image type

## 🖥️ Usage

1. **Start the app**: Run `./run_ocr_app.sh`
2. **Open browser**: Navigate to `http://localhost:8501`
3. **Upload image**: Click "Browse files" and select an image
4. **Adjust settings**: Use the sidebar to choose preprocessing and confidence settings
5. **View results**: The app will display:
   - Original image
   - Preprocessed image
   - Extracted text
   - Character bounding boxes
   - Statistics
6. **Download results**: Use the download buttons to save text and annotated images

## 📊 Model Information

- **Model Name**: alg
- **Base Model**: Myanmar (mya)
- **Training Data**: 1,406 images
- **Final Error Rate**: 0.111% (BCER)
- **Training Iterations**: 10,000

## 🛠️ Technical Details

- **OCR Engine**: Tesseract 4.x with LSTM
- **Model Type**: Fine-tuned LSTM-based neural network
- **Language**: Myanmar (alg)
- **Framework**: Streamlit
- **Image Processing**: OpenCV
- **Text Processing**: pytesseract

## 💡 Tips for Better Results

1. **Image Quality**: Use clear, high-resolution images
2. **Contrast**: Ensure good contrast between text and background
3. **Preprocessing**: Try different preprocessing methods for different image types
4. **Myanmar Text**: The model is specifically trained for Myanmar characters
5. **Confidence**: Adjust confidence threshold based on your needs

## 🔧 Troubleshooting

### Common Issues

1. **"Trained model not found"**
   - Make sure you've completed the training process
   - Check that `data/alg.traineddata` exists

2. **"Tesseract not found"**
   - Install Tesseract: `brew install tesseract` (macOS)
   - Or: `sudo apt-get install tesseract-ocr` (Ubuntu)

3. **"No text detected"**
   - Try different preprocessing methods
   - Check image quality and contrast
   - Ensure the image contains Myanmar text

4. **"Import errors"**
   - Install dependencies: `pip3 install -r requirements.txt`

### Performance Tips

- **Large Images**: The app can handle large images, but processing may take longer
- **Multiple Images**: Upload one image at a time for best results
- **Browser**: Use a modern browser (Chrome, Firefox, Safari) for best experience

## 📁 File Structure

```
tesstrain/
├── ocr_test_app.py          # Main Streamlit application
├── requirements.txt          # Python dependencies
├── run_ocr_app.sh          # Quick start script
├── README_OCR_App.md       # This file
└── data/
    └── alg.traineddata     # Your trained model
```

## 🤝 Contributing

Feel free to modify the app for your specific needs:
- Add new preprocessing methods
- Customize the UI
- Add batch processing capabilities
- Integrate with other OCR engines

## 📄 License

This project is part of your Myanmar OCR training workflow. Use it to test and validate your trained model. 