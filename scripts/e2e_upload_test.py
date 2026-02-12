"""
End-to-end upload test: create an image-only PDF in-memory and run
`load_document_from_bytes` to exercise OCR fallback paths.

Run: python scripts/e2e_upload_test.py
"""
from io import BytesIO
from PIL import Image, ImageDraw, ImageFont
import sys

from utils.file_loader import load_document_from_bytes


def make_scanned_pdf_bytes(text: str = "Test OCR from image", width: int = 800, height: int = 1000) -> bytes:
    # Create a simple white image with black text
    img = Image.new("RGB", (width, height), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("arial.ttf", 48)
    except Exception:
        font = ImageFont.load_default()
    draw.text((50, 200), text, fill=(0, 0, 0), font=font)

    bio = BytesIO()
    # Save image as a single-page PDF (image embedded) — this simulates a scanned PDF
    img.save(bio, format="PDF")
    return bio.getvalue()


def main():
    pdf_bytes = make_scanned_pdf_bytes()
    print(f"Generated PDF bytes: {len(pdf_bytes)} bytes")
    doc, err = load_document_from_bytes("scanned_test.pdf", pdf_bytes, source="e2e_test", use_ocr=True)
    if err:
        print("ERROR:", err)
        sys.exit(2)
    print("DOC:")
    print("name:", doc.get("name"))
    print("ocr_used:", doc.get("ocr_used"))
    text = doc.get("text", "")
    print(f"Extracted text length: {len(text)}")
    print("--- snippet ---")
    print(text[:400])


if __name__ == "__main__":
    main()
