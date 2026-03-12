import logging
import shutil
import subprocess
from pathlib import Path

from pdf2image import convert_from_path
from PIL import Image

logger = logging.getLogger(__name__)

# Magic bytes로 파일 무결성 사전 검증
_MAGIC_SIGNATURES: dict[str, list[bytes]] = {
    ".pdf": [b"%PDF"],
    ".png": [b"\x89PNG"],
    ".jpg": [b"\xff\xd8\xff"],
    ".jpeg": [b"\xff\xd8\xff"],
    ".gif": [b"GIF87a", b"GIF89a"],
    ".bmp": [b"BM"],
    ".tiff": [b"II\x2a\x00", b"MM\x00\x2a"],
    ".webp": [b"RIFF"],
    ".xlsx": [b"PK\x03\x04"],   # ZIP 기반
    ".docx": [b"PK\x03\x04"],
    ".pptx": [b"PK\x03\x04"],
    ".hwpx": [b"PK\x03\x04"],
    ".xls": [b"\xd0\xcf\x11\xe0"],  # OLE2
    ".doc": [b"\xd0\xcf\x11\xe0"],
    ".ppt": [b"\xd0\xcf\x11\xe0"],
    ".hwp": [b"\xd0\xcf\x11\xe0"],
}


def validate_file(file_path: Path) -> None:
    """파일 무결성 사전 검증 (빈 파일, magic bytes 불일치 탐지).

    Raises:
        ValueError: 파일이 비어있거나 magic bytes가 일치하지 않을 때
    """
    if not file_path.exists():
        raise ValueError(f"파일이 존재하지 않습니다: {file_path}")

    size = file_path.stat().st_size
    if size == 0:
        raise ValueError(f"빈 파일입니다: {file_path.name}")

    suffix = file_path.suffix.lower()
    signatures = _MAGIC_SIGNATURES.get(suffix)
    if not signatures:
        return  # 시그니처 미등록 포맷은 검증 스킵 (CSV, RTF 등)

    with open(file_path, "rb") as f:
        header = f.read(16)

    if not any(header.startswith(sig) for sig in signatures):
        raise ValueError(
            f"파일 손상 또는 확장자 불일치: {file_path.name} "
            f"(확장자: {suffix}, 실제 헤더: {header[:8].hex()})"
        )


def convert_hwp_to_pdf(hwp_path: Path) -> Path:
    """HWP 파일을 pyhwp(HTML) → LibreOffice(PDF) 2단계로 변환."""
    html_dir = hwp_path.parent / f"{hwp_path.stem}_html"

    # 1단계: HWP → HTML (hwp5html)
    result = subprocess.run(
        ["hwp5html", str(hwp_path), "--output", str(html_dir)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"HWP→HTML 변환 실패: {result.stderr}")

    xhtml_path = html_dir / "index.xhtml"
    if not xhtml_path.exists():
        raise FileNotFoundError(f"변환된 XHTML을 찾을 수 없음: {xhtml_path}")

    # 2단계: XHTML → PDF (LibreOffice)
    result = subprocess.run(
        [
            "libreoffice",
            "--headless",
            "--convert-to",
            "pdf",
            "--outdir",
            str(hwp_path.parent),
            str(xhtml_path),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )

    # LibreOffice가 index.pdf로 생성하므로 원래 파일명으로 변경
    index_pdf = hwp_path.parent / "index.pdf"
    pdf_path = hwp_path.parent / f"{hwp_path.stem}.pdf"

    if index_pdf.exists():
        index_pdf.rename(pdf_path)

    # HTML 임시 디렉토리 정리
    shutil.rmtree(html_dir, ignore_errors=True)

    _validate_pdf_output(pdf_path, hwp_path.name)
    logger.info("HWP→PDF 변환 완료: %s → %s", hwp_path.name, pdf_path.name)
    return pdf_path


def pdf_to_page_images(pdf_path: Path, dpi: int = 200) -> list[Image.Image]:
    """PDF를 페이지별 PIL 이미지로 변환."""
    images = convert_from_path(str(pdf_path), dpi=dpi)
    logger.info("PDF→이미지 변환 완료: %s (%d 페이지)", pdf_path.name, len(images))
    return images


def convert_office_to_pdf(file_path: Path) -> Path:
    """DOCX/HWPX 등 오피스 파일을 LibreOffice로 PDF 변환."""
    output_dir = file_path.parent
    result = subprocess.run(
        [
            "libreoffice",
            "--headless",
            "--convert-to",
            "pdf",
            "--outdir",
            str(output_dir),
            str(file_path),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )

    pdf_path = output_dir / f"{file_path.stem}.pdf"
    if not pdf_path.exists():
        raise FileNotFoundError(
            f"LibreOffice 변환 실패: {file_path.name} → PDF (stderr: {result.stderr})"
        )
    _validate_pdf_output(pdf_path, file_path.name)

    logger.info("오피스→PDF 변환 완료: %s → %s", file_path.name, pdf_path.name)
    return pdf_path


MIN_TEXT_LENGTH = 30  # 이 글자수 미만이면 OCR 폴백 실행


def _align_images_texts(
    images: list[Image.Image], texts: list[str],
) -> tuple[list[Image.Image], list[str]]:
    """이미지와 텍스트 리스트 길이가 다르면 짧은 쪽을 패딩."""
    if len(images) == len(texts):
        return images, texts
    logger.warning(
        "페이지 수 불일치: 이미지=%d, 텍스트=%d → 패딩 적용",
        len(images), len(texts),
    )
    while len(texts) < len(images):
        texts.append("")
    # 텍스트가 더 많으면 (드묾) 이미지를 None-safe하게 자름
    if len(texts) > len(images):
        texts = texts[:len(images)]
    return images, texts


def _ocr_fallback_for_empty_pages(
    texts: list[str], images: list[Image.Image],
) -> list[str]:
    """pdfplumber 텍스트가 부족한 페이지에 PaddleOCR 폴백 실행.

    스캔 PDF나 이미지 기반 PDF에서 텍스트 추출 품질을 보장한다.
    """
    from app.ingest.ocr import extract_text

    ocr_count = 0
    for i, text in enumerate(texts):
        if len(text.strip()) < MIN_TEXT_LENGTH and i < len(images):
            try:
                ocr_text = extract_text(images[i])
                if len(ocr_text.strip()) > len(text.strip()):
                    texts[i] = ocr_text
                    ocr_count += 1
            except Exception as e:
                logger.warning("OCR 폴백 실패 (p.%d): %s", i + 1, e)

    if ocr_count:
        logger.info("OCR 폴백: %d/%d 페이지에 OCR 텍스트 적용", ocr_count, len(texts))
    return texts


def _validate_pdf_output(pdf_path: Path, source_name: str) -> None:
    """LibreOffice 변환 결과 PDF가 정상인지 검증."""
    if not pdf_path.exists():
        raise FileNotFoundError(f"변환된 PDF를 찾을 수 없음: {source_name}")
    if pdf_path.stat().st_size < 100:  # 정상 PDF는 최소 수백 바이트
        pdf_path.unlink(missing_ok=True)
        raise ValueError(f"변환 결과가 비어있습니다 (PDF 크기 < 100B): {source_name}")


def process_document(
    file_path: Path,
) -> tuple[list[Image.Image], Path | None, list[str], list[dict] | None]:
    """문서 파일을 페이지 이미지 + 텍스트로 변환.

    Returns:
        (page_images, temp_pdf_path, page_texts, chunk_metas)
        - temp_pdf_path: 변환된 임시 PDF 경로 (정리 필요, 원본 PDF면 None)
        - page_texts: 페이지별 추출 텍스트 (pdfplumber 또는 OCR)
        - chunk_metas: Excel 전용 — 청크별 메타 (sheet, section). 그 외 None
    """
    from app.ingest.text_extractor import (
        extract_text_from_image,
        extract_texts_from_pdf,
    )

    # 파일 무결성 사전 검증
    validate_file(file_path)

    suffix = file_path.suffix.lower()

    if suffix in (".xlsx", ".xls"):
        from app.ingest.excel_parser import parse_excel

        texts, chunk_metas = parse_excel(file_path)
        return [], None, texts, chunk_metas

    elif suffix == ".hwp":
        pdf_path = convert_hwp_to_pdf(file_path)
        images = pdf_to_page_images(pdf_path)
        texts = extract_texts_from_pdf(pdf_path)
        images, texts = _align_images_texts(images, texts)
        texts = _ocr_fallback_for_empty_pages(texts, images)
        return images, pdf_path, texts, None
    elif suffix == ".pdf":
        images = pdf_to_page_images(file_path)
        texts = extract_texts_from_pdf(file_path)
        images, texts = _align_images_texts(images, texts)
        texts = _ocr_fallback_for_empty_pages(texts, images)
        return images, None, texts, None
    elif suffix in (".docx", ".doc", ".hwpx", ".pptx", ".ppt",
                     ".csv", ".odt", ".ods", ".odp", ".rtf"):
        pdf_path = convert_office_to_pdf(file_path)
        images = pdf_to_page_images(pdf_path)
        texts = extract_texts_from_pdf(pdf_path)
        images, texts = _align_images_texts(images, texts)
        texts = _ocr_fallback_for_empty_pages(texts, images)
        return images, pdf_path, texts, None
    elif suffix in (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif", ".webp"):
        with Image.open(file_path) as _img:
            img = _img.convert("RGB")
        text = extract_text_from_image(img)
        return [img], None, [text], None
    else:
        raise ValueError(f"지원하지 않는 파일 형식: {suffix}")
