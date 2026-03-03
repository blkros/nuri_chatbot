import logging

from PIL import Image

logger = logging.getLogger(__name__)

_ocr_engine = None


def get_ocr_engine():
    """PaddleOCR 엔진 싱글턴 로드."""
    global _ocr_engine
    if _ocr_engine is None:
        from paddleocr import PaddleOCR

        _ocr_engine = PaddleOCR(use_angle_cls=True, lang="korean")
        logger.info("PaddleOCR 한국어 엔진 로드 완료")
    return _ocr_engine


def extract_text(image: Image.Image) -> str:
    """페이지 이미지에서 한국어 텍스트 추출."""
    import numpy as np

    ocr = get_ocr_engine()
    img_array = np.array(image)
    results = ocr.ocr(img_array)

    if not results or not results[0]:
        return ""

    lines = []
    for line in results[0]:
        text = line[1][0]
        lines.append(text)

    extracted = "\n".join(lines)
    logger.debug("OCR 추출: %d 글자", len(extracted))
    return extracted
