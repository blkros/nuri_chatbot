import logging
import shutil
import subprocess
from pathlib import Path

from pdf2image import convert_from_path
from PIL import Image

logger = logging.getLogger(__name__)


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
    elif not pdf_path.exists():
        raise FileNotFoundError(f"변환된 PDF를 찾을 수 없음: {pdf_path}")

    # HTML 임시 디렉토리 정리
    shutil.rmtree(html_dir, ignore_errors=True)

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

    logger.info("오피스→PDF 변환 완료: %s → %s", file_path.name, pdf_path.name)
    return pdf_path


def process_document(
    file_path: Path,
) -> tuple[list[Image.Image], Path | None, list[str]]:
    """문서 파일을 페이지 이미지 + 텍스트로 변환.

    Returns:
        (page_images, temp_pdf_path, page_texts)
        - temp_pdf_path: 변환된 임시 PDF 경로 (정리 필요, 원본 PDF면 None)
        - page_texts: 페이지별 추출 텍스트 (pdfplumber 또는 OCR)
    """
    from app.ingest.text_extractor import (
        extract_text_from_image,
        extract_texts_from_pdf,
    )

    suffix = file_path.suffix.lower()

    if suffix in (".xlsx", ".xls"):
        from app.ingest.excel_parser import parse_excel

        texts = parse_excel(file_path)
        return [], None, texts

    elif suffix == ".hwp":
        pdf_path = convert_hwp_to_pdf(file_path)
        images = pdf_to_page_images(pdf_path)
        texts = extract_texts_from_pdf(pdf_path)
        return images, pdf_path, texts
    elif suffix == ".pdf":
        images = pdf_to_page_images(file_path)
        texts = extract_texts_from_pdf(file_path)
        return images, None, texts
    elif suffix in (".docx", ".doc", ".hwpx", ".pptx", ".ppt",
                     ".csv", ".odt", ".ods", ".odp", ".rtf"):
        pdf_path = convert_office_to_pdf(file_path)
        images = pdf_to_page_images(pdf_path)
        texts = extract_texts_from_pdf(pdf_path)
        return images, pdf_path, texts
    elif suffix in (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif", ".webp"):
        img = Image.open(file_path).convert("RGB")
        text = extract_text_from_image(img)
        return [img], None, [text]
    else:
        raise ValueError(f"지원하지 않는 파일 형식: {suffix}")
