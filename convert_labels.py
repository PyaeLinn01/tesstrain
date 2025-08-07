#!/usr/bin/env python3
import os


def convert_labels_to_gt_files(label_file, images_dir, output_dir):
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
                parts = line.split('  ')
                if len(parts) != 2:
                    print(f"Warning: Could not parse line: {line}")
                    continue
            image_rel_path, text_label = parts
            # Get the image filename without extension
            filename = os.path.splitext(os.path.basename(image_rel_path))[0]
            # Check if the image exists in the images_dir
            image_path = os.path.join(images_dir, os.path.basename(image_rel_path))
            if not os.path.exists(image_path):
                print(f"Warning: Image not found: {image_path}")
                continue
            # Create .gt.txt file
            gt_file_path = os.path.join(output_dir, f"{filename}.gt.txt")
            with open(gt_file_path, 'w', encoding='utf-8') as gt_file:
                gt_file.write(text_label.strip())
            print(f"Created: {gt_file_path}")

def main():
    label_file = "data/langdata/alg-ground-truth/labels.txt"
    images_dir = "data/langdata/alg-ground-truth/PL_processed_images"
    output_dir = "data/alg-ground-truth"
    convert_labels_to_gt_files(label_file, images_dir, output_dir)
    print("Conversion completed!")

if __name__ == "__main__":
    main() 