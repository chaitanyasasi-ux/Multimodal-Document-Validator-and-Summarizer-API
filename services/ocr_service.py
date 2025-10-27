import easyocr
import numpy as np
from PIL import Image
import io

# Initialize a dictionary to cache the reader instance
_ocr_reader_cache = {}

def get_ocr_reader(language: list = ['en']):
    """Initializes and returns the EasyOCR reader, caching the instance."""
    cache_key = tuple(sorted(language))
    if cache_key not in _ocr_reader_cache:
        # NOTE: This is where the heavy model loading happens
        print("Initializing EasyOCR Reader (Memory Intensive)...")
        _ocr_reader_cache[cache_key] = easyocr.Reader(language)
        print("EasyOCR Initialization Complete.")
    return _ocr_reader_cache[cache_key]

def image_to_text(image_bytes: bytes) -> str:
    """Extracts text from image bytes using EasyOCR."""
    try:
        # 1. Get the reader (initializes only on first call)
        reader = get_ocr_reader() 
        
        # 2. Convert bytes to a PIL Image object
        image = Image.open(io.BytesIO(image_bytes))
        image_np = np.array(image)
        
        # 3. Perform OCR
        result = reader.readtext(image_np)
        
        extracted_text = " ".join([text for (bbox, text, prob) in result])
        
        return extracted_text if extracted_text else "No readable text found."
    
    except Exception as e:
        # Check specifically for the memory error again just in case, but usually delayed init fixes it.
        print(f"OCR Error during request processing: {e}")
        return f"OCR processing failed: {str(e)}"