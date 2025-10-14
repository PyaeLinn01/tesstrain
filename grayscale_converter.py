import os
import cv2
from pathlib import Path
from PIL import Image
import numpy as np

# Configuration
INPUT_FOLDER = "/Users/pyaelinn/tessFinetune/tesstrain/cropped_output14_10_25"  # Folder containing the cropped images
OUTPUT_FOLDER = "grayscale_output"  # Output folder for grayscale images


def convert_to_grayscale():
    """
    Convert all images in the cropped_output folder to grayscale.
    """

    # Check if input folder exists
    if not os.path.exists(INPUT_FOLDER):
        print(f"Error: Input folder '{INPUT_FOLDER}' does not exist.")
        return

    # Create output folder if it doesn't exist
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Get all image files from input folder
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]
    image_files = []

    for ext in image_extensions:
        image_files.extend(Path(INPUT_FOLDER).glob(f"*{ext}"))
        image_files.extend(Path(INPUT_FOLDER).glob(f"*{ext.upper()}"))

    if not image_files:
        print(f"No image files found in {INPUT_FOLDER}")
        return

    print(f"Found {len(image_files)} images to convert to grayscale...")

    # Process each image
    for img_path in image_files:
        print(f"Converting: {img_path.name}")

        try:
            # Method 1: Using OpenCV
            image = cv2.imread(str(img_path))
            if image is not None:
                # Convert to grayscale
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # Create output filename
                base_name = img_path.stem
                output_filename = f"{base_name}_grayscale.jpg"
                output_path = os.path.join(OUTPUT_FOLDER, output_filename)

                # Save grayscale image
                cv2.imwrite(output_path, gray_image)
                print(f"  Saved: {output_filename}")

            else:
                # Method 2: Using PIL if OpenCV fails
                try:
                    with Image.open(img_path) as img:
                        # Convert to grayscale
                        gray_img = img.convert("L")

                        # Create output filename
                        base_name = img_path.stem
                        output_filename = f"{base_name}_grayscale.jpg"
                        output_path = os.path.join(OUTPUT_FOLDER, output_filename)

                        # Save grayscale image
                        gray_img.save(output_path, "JPEG", quality=95)
                        print(f"  Saved: {output_filename}")

                except Exception as e:
                    print(f"  Error processing {img_path.name}: {e}")

        except Exception as e:
            print(f"  Error processing {img_path.name}: {e}")

    print(f"\nConversion complete! Grayscale images saved to: {OUTPUT_FOLDER}")


def convert_to_grayscale_pil_only():
    """
    Alternative method using only PIL for grayscale conversion.
    """

    # Check if input folder exists
    if not os.path.exists(INPUT_FOLDER):
        print(f"Error: Input folder '{INPUT_FOLDER}' does not exist.")
        return

    # Create output folder if it doesn't exist
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Get all image files from input folder
    image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"]
    image_files = []

    for ext in image_extensions:
        image_files.extend(Path(INPUT_FOLDER).glob(f"*{ext}"))
        image_files.extend(Path(INPUT_FOLDER).glob(f"*{ext.upper()}"))

    if not image_files:
        print(f"No image files found in {INPUT_FOLDER}")
        return

    print(f"Found {len(image_files)} images to convert to grayscale using PIL...")

    # Process each image
    for img_path in image_files:
        print(f"Converting: {img_path.name}")

        try:
            with Image.open(img_path) as img:
                # Convert to grayscale
                gray_img = img.convert("L")

                # Create output filename
                base_name = img_path.stem
                output_filename = f"{base_name}_grayscale.jpg"
                output_path = os.path.join(OUTPUT_FOLDER, output_filename)

                # Save grayscale image
                gray_img.save(output_path, "JPEG", quality=95)
                print(f"  Saved: {output_filename}")

        except Exception as e:
            print(f"  Error processing {img_path.name}: {e}")

    print(f"\nConversion complete! Grayscale images saved to: {OUTPUT_FOLDER}")


if __name__ == "__main__":
    # Use the main conversion function (OpenCV + PIL fallback)
    convert_to_grayscale()

    # Uncomment the line below if you prefer to use only PIL
    # convert_to_grayscale_pil_only()
