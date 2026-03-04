import logging
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)


def extract_texts_from_pdf(pdf_path: Path) -> list[str]:
    """PDF에서 페이지별 텍스트 직접 추출 (pdfplumber).

    OCR 없이 PDF 내부 텍스트 레이어를 읽으므로 정확도가 높고 빠르다.
    LibreOffice로 변환된 Office 문서(HWP, Excel, Word 등)도 동일하게 처리.
    """
    import pdfplumber

    texts = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text() or ""
            texts.append(text)
            if text.strip():
                logger.debug("PDF p.%d 텍스트 추출: %d글자", i + 1, len(text))
            else:
                logger.debug("PDF p.%d 텍스트 없음 (스캔 문서일 수 있음)", i + 1)

    logger.info("PDF 텍스트 추출 완료: %s (%d 페이지)", pdf_path.name, len(texts))
    return texts


def extract_text_from_image(image: Image.Image) -> str:
    """이미지 파일 직접 업로드 시 PaddleOCR 폴백."""
    from app.ingest.ocr import extract_text

    return extract_text(image)
