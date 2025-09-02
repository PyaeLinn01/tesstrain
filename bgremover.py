import cv2
import numpy as np
import os
from pathlib import Path

# Hardcoded paths (edit these if you want different folders)
INPUT_DIR = "/Users/pyaelinn/tessFinetune/tesstrain/data/langdata/alg-ground-truth/PL_processed_images"
OUTPUT_DIR = "/Users/pyaelinn/tessFinetune/tesstrain/data/langdata/alg-ground-truth/bgremove"
SUPPORTED_EXTS = [".jpg", ".jpeg", ".png", ".bmp", ".tiff"]


class BackgroundRemover:
    """
    Background removal using a canvas-style method (Otsu + morphology).
    Replaces background with solid white while preserving foreground strokes.
    """

    def load_image(self, image_path):
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Could not read image: {image_path}")
        return img

    def remove_background_canvas_style(self, img):
        # Convert to grayscale if needed
        if len(img.shape) == 3:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        else:
            gray = img.copy()

        # Slight blur for robustness
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)

        # Otsu threshold to separate foreground/background
        _, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        # Invert so text/foreground = white, background = black
        mask = cv2.bitwise_not(thresh)

        # Clean mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

        # Compose on white background in grayscale
        white_bg = np.full_like(gray, 255, dtype=np.uint8)
        result = np.where(mask == 255, gray, white_bg).astype(np.uint8)
        return result

    def process_image_file(self, image_path, output_dir):
        img = self.load_image(image_path)
        result = self.remove_background_canvas_style(img)
        base_name = Path(image_path).stem
        os.makedirs(output_dir, exist_ok=True)
        out_path = os.path.join(
            output_dir, f"{base_name}.png"
        )
        cv2.imwrite(out_path, result)
        print(f"Saved: {out_path}")

    def process_folder(self, input_dir, output_dir):
        if not os.path.isdir(input_dir):
            print(f"Input directory not found: {input_dir}")
            return 1

        files = [
            os.path.join(input_dir, f)
            for f in os.listdir(input_dir)
            if any(f.lower().endswith(ext) for ext in SUPPORTED_EXTS)
        ]

        if not files:
            print("No images found to process.")
            return 0

        print(f"Processing {len(files)} images from: {input_dir}")
        print(f"Output will be saved to: {output_dir}")

        for idx, fp in enumerate(files, 1):
            print(f"[{idx}/{len(files)}] {os.path.basename(fp)}")
            try:
                self.process_image_file(fp, output_dir)
            except Exception as exc:
                print(f"  Failed: {fp} -> {exc}")
        return 0


def main():
    remover = BackgroundRemover()
    return remover.process_folder(INPUT_DIR, OUTPUT_DIR)


if __name__ == "__main__":
    exit(main())
