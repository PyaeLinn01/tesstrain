# 🎯 How to Fine-tune Tesseract OCR for Myanmar Language

This guide documents the complete process of fine-tuning Tesseract OCR for Myanmar language, from initial setup to final testing.

## 📋 Prerequisites

### System Requirements
- macOS (tested on macOS 23.6.0)
- Homebrew package manager
- Python 3.x
- At least 4GB RAM (8GB+ recommended)

### Required Software
- GNU Make 4.2+ (we upgraded from 3.81 to 4.4.1)
- wget (for downloading language data)
- Tesseract OCR engine
- Python with required packages

## 🚀 Step-by-Step Process

### 1. Environment Setup

#### 1.1 Upgrade GNU Make
The training process requires GNU Make 4.2 or newer. We upgraded from version 3.81:

```bash
# Install newer version via Homebrew
brew install make

# Add to PATH (add to ~/.zshrc for permanent)
export PATH="/opt/homebrew/opt/make/libexec/gnubin:$PATH"

# Verify version
make --version  # Should show 4.4.1
```

#### 1.2 Install Required Tools
```bash
# Install wget for downloading language data
brew install wget

# Install Tesseract OCR
brew install tesseract
```

### 2. Project Structure Setup

#### 2.1 Directory Structure
```
tessFinetune/
├── tesstrain/
│   ├── data/
│   │   ├── alg-ground-truth/     # Training images and ground truth
│   │   ├── alg/                  # Generated training files
│   │   └── langdata/             # Language data files
│   ├── usr/share/tessdata/
│   │   └── mya.traineddata      # Base Myanmar model
│   ├── Makefile                  # Training configuration
│   └── [other training files]
```

#### 2.2 Download Language Data
```bash
cd tesstrain
make tesseract-langdata
```
This downloads 30+ language unicharset files to `data/langdata/`.

### 3. Training Data Preparation

#### 3.1 Original Data Format
Our training data was in label files format:
- `test_labels.txt`
- `train_labels.txt` 
- `val_labels.txt`

Format: `file\test\image.jpg\ttext_label`

#### 3.2 Convert to Tesseract Format
Created `convert_labels.py` to convert label files to individual `.gt.txt` files:

```python
#!/usr/bin/env python3
import os
import sys

def convert_labels_to_gt_files(label_file, output_dir):
    """Convert label file to individual .gt.txt files"""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    with open(label_file, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            # Split by tab or space
            parts = line.split('\t')
            if len(parts) != 2:
                parts = line.split('  ')  # Try double space
                if len(parts) != 2:
                    print(f"Warning: Could not parse line: {line}")
                    continue
            
            image_path, text_label = parts
            
            # Extract filename from path
            filename = os.path.basename(image_path)
            base_name = os.path.splitext(filename)[0]
            
            # Create .gt.txt file
            gt_file = os.path.join(output_dir, f"{base_name}.gt.txt")
            with open(gt_file, 'w', encoding='utf-8') as gt_f:
                gt_f.write(text_label + '\n')

# Convert all label files
convert_labels_to_gt_files("data/langdata/alg-ground-truth/train_labels.txt", "data/alg-ground-truth")
convert_labels_to_gt_files("data/langdata/alg-ground-truth/test_labels.txt", "data/alg-ground-truth")
convert_labels_to_gt_files("data/langdata/alg-ground-truth/val_labels.txt", "data/alg-ground-truth")
```

#### 3.3 Organize Training Data
```bash
# Create ground truth directory
mkdir -p data/alg-ground-truth

# Copy image files to match ground truth files
find data/langdata/alg-ground-truth/file/ -name "*.jpg" -exec cp {} data/alg-ground-truth/ \;

# Clean up file names (remove backslashes)
cd data/alg-ground-truth
for file in *; do 
    if [[ "$file" == *"\\"* ]]; then 
        newname=$(echo "$file" | sed 's/\\/\//g' | sed 's/.*\///'); 
        mv "$file" "$newname"; 
    fi; 
done
cd ../..
```

### 4. Create Training Files

#### 4.1 Generate Box and LSTM Files
Created `process_files.py` to convert images to required training format:

```python
#!/usr/bin/env python3
import os
import glob
import subprocess
import sys

def process_files():
    """Process all image files to create box and lstmf files"""
    ground_truth_dir = "data/alg-ground-truth"
    
    # Get all image files
    image_files = []
    for ext in ['*.jpg', '*.png', '*.tif', '*.tiff']:
        image_files.extend(glob.glob(os.path.join(ground_truth_dir, ext)))
    
    print(f"Found {len(image_files)} image files")
    
    for i, image_file in enumerate(image_files):
        if i % 100 == 0:
            print(f"Processing file {i+1}/{len(image_files)}: {os.path.basename(image_file)}")
        
        # Get base name without extension
        base_name = os.path.splitext(image_file)[0]
        gt_file = base_name + ".gt.txt"
        box_file = base_name + ".box"
        lstmf_file = base_name + ".lstmf"
        
        # Skip if lstmf already exists
        if os.path.exists(lstmf_file):
            continue
        
        # Create box file
        if os.path.exists(gt_file):
            try:
                subprocess.run([
                    "python3", "generate_line_box.py",
                    "-i", image_file,
                    "-t", gt_file
                ], stdout=open(box_file, 'w'), check=True)
            except subprocess.CalledProcessError:
                print(f"Failed to create box file for {image_file}")
                continue
        
        # Create lstmf file
        if os.path.exists(box_file):
            try:
                subprocess.run([
                    "tesseract", image_file, base_name,
                    "--psm", "13", "lstm.train"
                ], check=True)
            except subprocess.CalledProcessError:
                print(f"Failed to create lstmf file for {image_file}")
                continue

if __name__ == "__main__":
    process_files()
```

#### 4.2 Run the Processing Script
```bash
python3 process_files.py
```

This created:
- 1406 box files (character bounding boxes)
- 1406 lstmf files (LSTM training format)

### 5. Training Configuration

#### 5.1 Training Parameters
```bash
MODEL_NAME=alg           # Output model name
START_MODEL=mya          # Base Myanmar model
FINETUNE_TYPE=Impact     # Fine-tuning approach
```

#### 5.2 Training Command
```bash
make training MODEL_NAME=alg START_MODEL=mya FINETUNE_TYPE=Impact
```

### 6. Training Process

#### 6.1 Training Steps
The training process automatically:
1. Extracts LSTM components from base model (`mya.traineddata`)
2. Creates unicharset from training data
3. Generates training lists
4. Runs LSTM training for 10,000 iterations
5. Saves checkpoints and final model

#### 6.2 Training Results
```
Final BCER (Best Character Error Rate): 0.111%
Final BWER (Best Word Error Rate): 1.2%
Training completed: 10,000 iterations
Best checkpoint: data/alg/checkpoints/alg_0.111_1143_9200.checkpoint
```

### 7. Model Files Generated

#### 7.1 Output Files
- `data/alg.traineddata` - Final trained model
- `data/alg/checkpoints/` - Training checkpoints
- `data/alg/alg_0.111_1143_9200.checkpoint` - Best checkpoint
- `data/alg/alg_checkpoint` - Latest checkpoint
- `data/alg/alg_best.traineddata` - Best model

### 8. Testing the Model

#### 8.1 Create Streamlit Testing App
Created `ocr_test_app.py` with features:
- Image upload (PNG, JPG, JPEG, TIFF, BMP)
- Multiple preprocessing methods
- Real-time OCR with trained model
- Character detection visualization
- Confidence metrics
- Download results

#### 8.2 Fix TESSDATA_PREFIX Issue
The app needed to set the correct path for the trained model:

```python
def setup_tesseract():
    """Setup Tesseract with the trained model"""
    model_path = "data/alg.traineddata"
    if not os.path.exists(model_path):
        st.error(f"❌ Trained model not found at {model_path}")
        return False
    
    # Set TESSDATA_PREFIX to data directory
    data_dir = os.path.join(os.getcwd(), "data")
    os.environ['TESSDATA_PREFIX'] = data_dir
    
    custom_config = r'--oem 1 --psm 6 -l alg'
    return custom_config
```

#### 8.3 Run the Testing App
```bash
# Install dependencies
pip3 install -r requirements.txt

# Run the app
streamlit run ocr_test_app.py
```

## 📊 Results Summary

### Training Performance
- **Starting Model**: `mya.traineddata` (base Myanmar model)
- **Training Data**: 1,406 images with ground truth
- **Training Time**: ~30-60 minutes (depending on hardware)
- **Final BCER**: 0.111% (excellent character accuracy)
- **Final BWER**: 1.2% (excellent word accuracy)

### Model Quality
- **Character Error Rate**: 0.111% (99.89% accuracy)
- **Word Error Rate**: 1.2% (98.8% accuracy)
- **Training Convergence**: Successful (10,000 iterations)

## 🔧 Troubleshooting

### Common Issues and Solutions

#### 1. GNU Make Version Error
```
Error: GNU Make version too low. Need 4.2 or newer.
```
**Solution**: Upgrade GNU Make via Homebrew and update PATH.

#### 2. Missing wget
```
Error: wget: command not found
```
**Solution**: Install wget via Homebrew.

#### 3. Missing Training Data
```
Error: found no data/alg-ground-truth/*.gt.txt
```
**Solution**: Convert label files to individual .gt.txt files.

#### 4. Missing Image Files
```
Error: No lstmf files found
```
**Solution**: Process images to create box and lstmf files.

#### 5. TESSDATA_PREFIX Error
```
Error: Could not initialize tesseract
```
**Solution**: Set TESSDATA_PREFIX to the data directory containing the trained model.

## 📁 File Structure After Training

```
tessFinetune/
├── tesstrain/
│   ├── data/
│   │   ├── alg-ground-truth/          # Training images + .gt.txt files
│   │   ├── alg/                       # Training artifacts
│   │   │   ├── checkpoints/           # Training checkpoints
│   │   │   ├── alg.traineddata        # Final trained model
│   │   │   └── [other training files]
│   │   └── langdata/                  # Language data
│   ├── convert_labels.py              # Label conversion script
│   ├── process_files.py               # File processing script
│   ├── ocr_test_app.py               # Streamlit testing app
│   ├── requirements.txt               # Python dependencies
│   ├── run_ocr_app.sh                # App runner script
│   └── HOW_TO_FINETUNE.md            # This guide
```

## 🎯 Next Steps

1. **Test with Real Data**: Use the Streamlit app to test with new Myanmar text images
2. **Model Optimization**: Fine-tune parameters for specific use cases
3. **Data Augmentation**: Add more training data for better generalization
4. **Production Deployment**: Deploy the model for production use

## 📚 Additional Resources

- [Tesseract Training Documentation](https://github.com/tesseract-ocr/tesseract/wiki/TrainingTesseract)
- [tesstrain Repository](https://github.com/tesseract-ocr/tesstrain)
- [Myanmar Language Support](https://github.com/tesseract-ocr/tessdata)

---

**Note**: This guide documents the specific process used for fine-tuning Tesseract OCR for Myanmar language. The same approach can be adapted for other languages by changing the base model and training data. 