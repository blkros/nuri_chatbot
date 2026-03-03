import logging

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

_ocr_engine = None


def get_ocr_engine():
    """PaddleOCR 3.x 엔진 싱글턴 로드."""
    global _ocr_engine
    if _ocr_engine is None:
        from paddleocr import PaddleOCR

        _ocr_engine = PaddleOCR(
            lang="korean",
            use_textline_orientation=True,
            enable_mkldnn=False,
        )
        logger.info("PaddleOCR 한국어 엔진 로드 완료")
    return _ocr_engine


def extract_text(image: Image.Image) -> str:
    """페이지 이미지에서 한국어 텍스트 추출 (PaddleOCR 3.x API)."""
    ocr = get_ocr_engine()
    img_array = np.array(image)
    results = ocr.predict(input=img_array)

    lines = []
    for res in results:
        texts = res.json.get("rec_texts", [])
        lines.extend(texts)

    if not lines:
        return ""

    extracted = "\n".join(lines)
    logger.debug("OCR 추출: %d 글자", len(extracted))
    return extracted
