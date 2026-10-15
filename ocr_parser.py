from functools import lru_cache
from PIL import Image
import numpy as np

@lru_cache(maxsize=1)
def _get_reader():
    """Create EasyOCR reader only when an image is actually processed."""
    import easyocr
    return easyocr.Reader(["en"], gpu=False, verbose=False)


def extract_text_from_image(uploaded_file):
    uploaded_file.seek(0)
    image = Image.open(uploaded_file).convert("RGB")
    image_array = np.array(image)
    reader = _get_reader()
    results = reader.readtext(image_array, detail=0, paragraph=False)
    return "\n".join(str(x).strip() for x in results if str(x).strip())
