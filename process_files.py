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
        if not os.path.exists(box_file):
            try:
                result = subprocess.run([
                    "python3", "generate_line_box.py", 
                    "-i", image_file, 
                    "-t", gt_file
                ], capture_output=True, text=True)
                
                if result.returncode == 0:
                    with open(box_file, 'w') as f:
                        f.write(result.stdout)
                else:
                    print(f"Error creating box file for {image_file}: {result.stderr}")
                    continue
            except Exception as e:
                print(f"Error processing {image_file}: {e}")
                continue
        
        # Create lstmf file
        if not os.path.exists(lstmf_file):
            try:
                result = subprocess.run([
                    "tesseract", image_file, base_name, 
                    "--psm", "13", "lstm.train"
                ], capture_output=True, text=True)
                
                if result.returncode != 0:
                    print(f"Error creating lstmf file for {image_file}: {result.stderr}")
            except Exception as e:
                print(f"Error processing {image_file}: {e}")
    
    print("Processing complete!")

if __name__ == "__main__":
    process_files() 