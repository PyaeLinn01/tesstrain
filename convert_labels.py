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
            
            # Extract filename without extension
            filename = os.path.splitext(os.path.basename(image_path))[0]
            
            # Create .gt.txt file
            gt_file_path = os.path.join(output_dir, f"{filename}.gt.txt")
            
            with open(gt_file_path, 'w', encoding='utf-8') as gt_file:
                gt_file.write(text_label.strip())
            
            print(f"Created: {gt_file_path}")

def main():
    # Define paths
    base_dir = "data/langdata/alg-ground-truth"
    output_dir = "data/alg-ground-truth"
    
    # Convert train labels
    train_label_file = os.path.join(base_dir, "train_labels.txt")
    if os.path.exists(train_label_file):
        print("Converting train labels...")
        convert_labels_to_gt_files(train_label_file, output_dir)
    
    # Convert test labels
    test_label_file = os.path.join(base_dir, "test_labels.txt")
    if os.path.exists(test_label_file):
        print("Converting test labels...")
        convert_labels_to_gt_files(test_label_file, output_dir)
    
    # Convert val labels
    val_label_file = os.path.join(base_dir, "val_labels.txt")
    if os.path.exists(val_label_file):
        print("Converting validation labels...")
        convert_labels_to_gt_files(val_label_file, output_dir)
    
    print("Conversion completed!")

if __name__ == "__main__":
    main() 