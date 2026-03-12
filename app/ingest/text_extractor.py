import logging
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)


def _format_table(table: list[list[str | None]]) -> str:
    """pdfplumber 표를 구조화 텍스트로 변환.

    첫 행을 헤더로 사용하여 각 행을 `키: 값` 형태로 구성.
    빈 셀은 스킵하고, 헤더가 없거나 행이 부족하면 빈 문자열 반환.
    """
    if not table or len(table) < 2:
        return ""

    # 헤더 행 (None → 빈 문자열)
    headers = [str(h).strip() if h else "" for h in table[0]]

    rows_text = []
    for row in table[1:]:
        cells = []
        for j, cell in enumerate(row):
            val = str(cell).strip() if cell else ""
            if not val:
                continue
            header = headers[j] if j < len(headers) and headers[j] else f"열{j+1}"
            cells.append(f"{header}: {val}")
        if cells:
            rows_text.append(" | ".join(cells))

    if not rows_text:
        return ""

    return "\n".join(rows_text)


def _extract_page_text_with_tables(page) -> str:
    """페이지에서 표와 일반 텍스트를 모두 추출하여 합친다.

    표가 있으면 구조화 텍스트로 변환하고, 표 외 영역의 텍스트도 함께 포함.
    표가 없으면 기존 extract_text()와 동일하게 동작.
    """
    tables = page.extract_tables()

    if not tables:
        return page.extract_text() or ""

    parts = []

    # 표 영역의 bbox 수집 (표 외 텍스트 추출용)
    table_bboxes = []
    for table_obj in page.find_tables():
        table_bboxes.append(table_obj.bbox)

    # 표 외 영역 텍스트 추출
    if table_bboxes:
        non_table_page = page
        for bbox in table_bboxes:
            # 표 영역을 crop out — pdfplumber의 .outside_bbox() 사용 불가 시 폴백
            try:
                non_table_page = non_table_page.outside_bbox(bbox)
            except Exception:
                pass
        non_table_text = non_table_page.extract_text() or ""
        if non_table_text.strip():
            parts.append(non_table_text.strip())

    # 각 표를 구조화 텍스트로 변환
    for i, table in enumerate(tables):
        formatted = _format_table(table)
        if formatted:
            parts.append(f"[표 {i+1}]\n{formatted}")

    return "\n\n".join(parts) if parts else (page.extract_text() or "")


def extract_texts_from_pdf(pdf_path: Path) -> list[str]:
    """PDF에서 페이지별 텍스트 직접 추출 (pdfplumber).

    표가 있는 페이지는 표 구조를 보존한 구조화 텍스트로 변환.
    OCR 없이 PDF 내부 텍스트 레이어를 읽으므로 정확도가 높고 빠르다.
    LibreOffice로 변환된 Office 문서(HWP, Excel, Word 등)도 동일하게 처리.
    """
    import pdfplumber

    texts = []
    with pdfplumber.open(str(pdf_path)) as pdf:
        for i, page in enumerate(pdf.pages):
            text = _extract_page_text_with_tables(page)
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
