import logging

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

_ocr_engine = None

# 같은 행으로 간주할 Y좌표 허용 오차 (픽셀)
_ROW_GROUP_THRESHOLD = 15


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
    """페이지 이미지에서 한국어 텍스트 추출 (PaddleOCR 3.x API).

    바운딩 박스 좌표를 활용하여 공간 순서(위→아래, 왼→오른)로 정렬하고,
    같은 행의 텍스트를 ' | '로 연결하여 표 구조를 보존.
    """
    ocr = get_ocr_engine()
    img_array = np.array(image)
    results = ocr.predict(input=img_array)

    # (text, score, y1, x1) 수집
    items = []
    for res in results:
        inner = res.json.get("res", {})
        texts = inner.get("rec_texts", [])
        scores = inner.get("rec_scores", [])
        boxes = inner.get("rec_boxes", [])

        for text, score, box in zip(texts, scores, boxes):
            if not text.strip() or score < 0.5:
                continue
            # box = [x1, y1, x2, y2]
            x1, y1 = box[0], box[1]
            items.append((text, y1, x1))

    if not items:
        return ""

    # Y좌표로 정렬 후 행 그룹핑
    items.sort(key=lambda t: (t[1], t[2]))

    rows = []
    current_row = [items[0]]
    current_y = items[0][1]

    for item in items[1:]:
        if abs(item[1] - current_y) <= _ROW_GROUP_THRESHOLD:
            current_row.append(item)
        else:
            # X좌표로 정렬 후 행 완성
            current_row.sort(key=lambda t: t[2])
            rows.append(" | ".join(t[0] for t in current_row))
            current_row = [item]
            current_y = item[1]

    # 마지막 행
    current_row.sort(key=lambda t: t[2])
    rows.append(" | ".join(t[0] for t in current_row))

    extracted = "\n".join(rows)
    logger.debug("OCR 추출: %d행, %d글자", len(rows), len(extracted))
    return extracted
