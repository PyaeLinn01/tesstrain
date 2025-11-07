import os
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from pathlib import Path

# Class labels for detection (same as in onlybox.py)
class_labels = {0: "dob", 1: "father", 2: "id", 3: "name"}

# Configuration - Change this to your input folder path
INPUT_FOLDER = "/Users/pyaelinn/tessFinetune/tesstrain/nrc-photo"  # Change this to your folder path
OUTPUT_FOLDER = "cropped_output14_10_25"  # Output folder will be created automatically
MODEL_PATH = "/Users/pyaelinn/tessFinetune/tesstrain/v5.pt"


def detect_and_crop_images():
    """
    Detect textboxes in images and crop the detected areas.
    """

    # Check if model exists
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file '{MODEL_PATH}' not found.")
        return

    # Check if input folder exists
    if not os.path.exists(INPUT_FOLDER):
        print(f"Error: Input folder '{INPUT_FOLDER}' does not exist.")
        return

    # Create output folder if it doesn't exist
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Load YOLO model
    print(f"Loading model from {MODEL_PATH}...")
    yolo_model = YOLO(MODEL_PATH)

    # Get all image files from input folder
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]
    image_files = []

    for ext in image_extensions:
        image_files.extend(Path(INPUT_FOLDER).glob(f"*{ext}"))
        image_files.extend(Path(INPUT_FOLDER).glob(f"*{ext.upper()}"))

    if not image_files:
        print(f"No image files found in {INPUT_FOLDER}")
        return

    print(f"Found {len(image_files)} images to process...")

    # Process each image
    for img_path in image_files:
        print(f"Processing: {img_path.name}")

        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"Could not load image: {img_path}")
            continue

        # Run detection
        results = yolo_model(image)

        # Process detections
        crop_count = 0
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls = int(box.cls[0])
                confidence = box.conf[0]
                class_name = class_labels.get(cls, "Unknown")

                # Crop the detected area
                cropped = image[y1:y2, x1:x2]

                # Create filename for cropped image
                base_name = img_path.stem
                crop_filename = f"{base_name}_{class_name}_{crop_count:02d}.jpg"
                crop_path = os.path.join(OUTPUT_FOLDER, crop_filename)

                # Save cropped image
                cv2.imwrite(crop_path, cropped)
                print(f"  Saved crop: {crop_filename} (confidence: {confidence:.2f})")
                crop_count += 1

        if crop_count == 0:
            print(f"  No detections found in {img_path.name}")

    print(f"\nProcessing complete! Cropped images saved to: {OUTPUT_FOLDER}")


if __name__ == "__main__":
    detect_and_crop_images()
