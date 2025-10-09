import argparse
import json
import os
from typing import List, Dict, Any

# Optional deps
try:
    import fitz  # PyMuPDF
except Exception:
    fitz = None

try:
    from PyPDF2 import PdfReader
except Exception:
    PdfReader = None

try:
    from PIL import Image
except Exception:
    Image = None

try:
    import pytesseract
except Exception:
    pytesseract = None


def page_to_image_via_fitz(page, zoom: float = 2.0):
    """Render a fitz page to a PIL Image if possible."""
    if fitz is None or Image is None:
        return None
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    mode = "RGB" if pix.alpha == 0 else "RGBA"
    img = Image.frombytes(mode, [pix.width, pix.height], pix.samples)
    if img.mode == "RGBA":
        img = img.convert("RGB")
    return img


def ocr_image(img, lang: str = "eng") -> str:
    if pytesseract is None:
        return ""
    try:
        return pytesseract.image_to_string(img, config=f"--oem 1 --psm 6 -l {lang}").strip()
    except Exception:
        return ""


def extract_with_fitz(pdf_path: str, lang: str) -> Dict[str, Any]:
    doc = fitz.open(pdf_path)
    output = {
        "file": os.path.abspath(pdf_path),
        "pages": []
    }
    for i, page in enumerate(doc):
        blocks = []
        text_blocks = page.get_text("blocks") or []  # list of tuples: (x0,y0,x1,y1, text, block_no, block_type)
        page_text = "".join([b[4] for b in text_blocks if len(b) >= 5 and isinstance(b[4], str)])
        for b in text_blocks:
            if len(b) >= 5:
                x0, y0, x1, y1, txt = b[0], b[1], b[2], b[3], b[4]
                if isinstance(txt, str) and txt.strip():
                    blocks.append({
                        "bbox": [x0, y0, x1, y1],
                        "text": txt
                    })
        source = "text"
        # Fallback to OCR if no text extracted
        if not page_text.strip():
            img = page_to_image_via_fitz(page)
            if img is not None:
                page_text = ocr_image(img, lang=lang)
                blocks = [{"bbox": [0, 0, img.width, img.height], "text": page_text}] if page_text else []
                source = "ocr"
        mediabox = page.mediabox
        output["pages"].append({
            "page_number": i + 1,
            "width": mediabox.width,
            "height": mediabox.height,
            "text": page_text,
            "blocks": blocks,
            "source": source
        })
    return output


def extract_with_pypdf2(pdf_path: str) -> Dict[str, Any]:
    reader = PdfReader(pdf_path)
    output = {
        "file": os.path.abspath(pdf_path),
        "pages": []
    }
    for i, page in enumerate(reader.pages):
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""
        # PyPDF2 doesn't provide layout bboxes; just dump text
        output["pages"].append({
            "page_number": i + 1,
            "width": None,
            "height": None,
            "text": txt,
            "blocks": [{"bbox": None, "text": txt}] if txt else [],
            "source": "text" if txt.strip() else "unknown"
        })
    return output


def merge_sources(primary: Dict[str, Any], fallback: Dict[str, Any]) -> Dict[str, Any]:
    # Prefer primary page content; if empty, use fallback
    pages: List[Dict[str, Any]] = []
    for i in range(max(len(primary.get("pages", [])), len(fallback.get("pages", [])))):
        p = primary["pages"][i] if i < len(primary.get("pages", [])) else None
        f = fallback["pages"][i] if i < len(fallback.get("pages", [])) else None
        if p and (p.get("text", "").strip() or p.get("blocks")):
            pages.append(p)
        elif f:
            pages.append(f)
        else:
            pages.append({
                "page_number": i + 1,
                "width": None,
                "height": None,
                "text": "",
                "blocks": [],
                "source": "unknown"
            })
    return {"file": primary.get("file") or fallback.get("file"), "pages": pages}


def convert_pdf_to_json(pdf_path: str, out_path: str, lang: str = "eng"):
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(pdf_path)

    if fitz is not None:
        primary = extract_with_fitz(pdf_path, lang=lang)
    else:
        primary = {"file": os.path.abspath(pdf_path), "pages": []}

    if not primary.get("pages") and PdfReader is not None:
        fallback = extract_with_pypdf2(pdf_path)
        result = merge_sources(primary, fallback)
    else:
        # If primary has pages but some are empty text, try to enrich with PyPDF2 text
        if PdfReader is not None:
            fallback = extract_with_pypdf2(pdf_path)
            result = merge_sources(primary, fallback)
        else:
            result = primary

    # Ensure directory exists
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Convert a PDF to structured JSON with text blocks; fallback to OCR for scanned pages.")
    parser.add_argument("pdf", help="Path to input PDF")
    parser.add_argument("output", nargs="?", help="Path to output JSON; defaults to <pdf>.json next to input")
    parser.add_argument("--lang", default="eng", help="Tesseract language(s) for OCR, e.g., 'eng', 'mya', or 'eng+mya'")
    args = parser.parse_args()

    pdf_path = args.pdf
    out_path = args.output or os.path.splitext(pdf_path)[0] + ".json"

    convert_pdf_to_json(pdf_path, out_path, lang=args.lang)
    print(f"Saved JSON to: {out_path}")


if __name__ == "__main__":
    main()
