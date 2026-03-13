import logging
import re
from pathlib import Path

from PIL import Image

logger = logging.getLogger(__name__)


def _clean_cell(val: str | None) -> str:
    """셀 값 정리: None/공백 처리 + 줄바꿈을 공백으로."""
    if not val:
        return ""
    return re.sub(r"\s+", " ", str(val)).strip()


def _is_meaningful_header(headers: list[str]) -> bool:
    """헤더 행이 의미 있는지 판단 (빈 헤더가 절반 이상이면 무의미)."""
    non_empty = sum(1 for h in headers if h)
    return non_empty >= len(headers) * 0.4 and non_empty >= 2


def _format_table(table: list[list[str | None]], label: str = "") -> str:
    """pdfplumber 표를 구조화 텍스트로 변환.

    헤더가 의미 있으면 `키: 값` 형태, 아니면 `|`로 셀 나열.
    label이 있으면 표 앞에 섹션 제목으로 붙인다.
    """
    if not table or len(table) < 2:
        return ""

    # 헤더 행 정리
    headers = [_clean_cell(h) for h in table[0]]
    use_headers = _is_meaningful_header(headers)

    rows_text = []
    for row in table[1:]:
        cells = []
        for j, cell in enumerate(row):
            val = _clean_cell(cell)
            if not val:
                continue
            if use_headers and j < len(headers) and headers[j]:
                cells.append(f"{headers[j]}: {val}")
            else:
                cells.append(val)
        if cells:
            rows_text.append(" | ".join(cells))

    if not rows_text:
        return ""

    result = ""
    if label:
        result += f"[{label}]\n"
    if use_headers:
        header_line = " | ".join(h for h in headers if h)
        result += f"({header_line})\n"
    result += "\n".join(rows_text)

    return result


def _find_section_label(page, table_bbox, all_table_bboxes) -> str:
    """표 바로 위 영역에서 섹션 제목을 찾는다.

    표의 상단 bbox 위쪽 영역을 crop하여 텍스트를 추출하고,
    마지막 줄(표 바로 위 텍스트)을 섹션 제목으로 사용.
    """
    table_top = table_bbox[1]  # y0 (표 상단 y좌표)

    # 이 표 바로 위에 있는 다른 표의 하단 찾기 (겹치지 않게)
    search_top = 0
    for other_bbox in all_table_bboxes:
        other_bottom = other_bbox[3]  # y1
        if other_bottom < table_top and other_bottom > search_top:
            search_top = other_bottom

    # 표 위 영역이 너무 좁으면 스킵
    if table_top - search_top < 10:
        return ""

    try:
        # 표 위쪽 영역 crop
        above_area = page.within_bbox((
            0,                   # x0: 페이지 왼쪽
            search_top,          # y0: 이전 표 하단 또는 페이지 상단
            page.width,          # x1: 페이지 오른쪽
            table_top,           # y1: 현재 표 상단
        ))
        text = above_area.extract_text() or ""
        text = text.strip()
        if not text:
            return ""

        # 마지막 몇 줄에서 섹션 제목 추출
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        if not lines:
            return ""

        # 마지막 줄이 섹션 제목일 가능성이 높음
        # 너무 긴 텍스트는 제목이 아님
        candidate = lines[-1]
        if len(candidate) > 80:
            return ""

        return candidate

    except Exception:
        return ""


def _extract_page_text_with_tables(page) -> str:
    """페이지에서 표와 일반 텍스트를 위치 순서대로 추출.

    각 표 앞에 섹션 제목(예: "산업용전력(갑)")을 붙여
    VLM이 어떤 카테고리의 표인지 구분할 수 있게 한다.
    """
    table_objects = page.find_tables()

    if not table_objects:
        return page.extract_text() or ""

    # 표 bounding box 수집
    table_bboxes = [t.bbox for t in table_objects]
    tables_data = [t.extract() for t in table_objects]

    # 모든 요소를 (y좌표, 타입, 내용) 리스트로 수집
    elements = []

    # 1. 각 표 + 섹션 라벨
    for i, (tdata, bbox) in enumerate(zip(tables_data, table_bboxes)):
        label = _find_section_label(page, bbox, table_bboxes)
        formatted = _format_table(tdata, label=label)
        if formatted:
            elements.append((bbox[1], formatted))  # y0 기준 정렬

    # 2. 표 외 영역 텍스트 (표에 포함되지 않은 독립 텍스트)
    # 페이지 상단/하단의 일반 텍스트 추출
    non_table_page = page
    for bbox in table_bboxes:
        try:
            non_table_page = non_table_page.outside_bbox(bbox)
        except Exception:
            pass
    non_table_text = non_table_page.extract_text() or ""

    # 표 라벨에 이미 포함된 텍스트는 중복이므로,
    # 섹션 라벨로 사용되지 않은 독립 텍스트만 추가
    if non_table_text.strip():
        # 페이지 최상단 텍스트 (첫 표 위)
        if table_bboxes:
            first_table_top = min(b[1] for b in table_bboxes)
            last_table_bottom = max(b[3] for b in table_bboxes)
        else:
            first_table_top = page.height
            last_table_bottom = 0

        # 페이지 제목/머리글 (첫 표 위 전체)
        try:
            top_area = page.within_bbox((0, 0, page.width, first_table_top))
            top_text = (top_area.extract_text() or "").strip()
            if top_text and len(top_text) > 5:
                elements.append((0, top_text))
        except Exception:
            pass

        # 페이지 하단 (마지막 표 아래)
        try:
            bottom_area = page.within_bbox((
                0, last_table_bottom, page.width, page.height
            ))
            bottom_text = (bottom_area.extract_text() or "").strip()
            if bottom_text and len(bottom_text) > 5:
                elements.append((last_table_bottom, bottom_text))
        except Exception:
            pass

    # y좌표 기준 정렬 (페이지 위→아래 순서)
    elements.sort(key=lambda x: x[0])

    result = "\n\n".join(content for _, content in elements)
    return result if result.strip() else (page.extract_text() or "")


def extract_texts_from_pdf(pdf_path: Path) -> list[str]:
    """PDF에서 페이지별 텍스트 직접 추출 (pdfplumber).

    표가 있는 페이지는 표 구조를 보존한 구조화 텍스트로 변환.
    각 표 앞에 섹션 제목을 붙여 카테고리를 구분한다.
    OCR 없이 PDF 내부 텍스트 레이어를 읽으므로 정확도가 높고 빠르다.
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
