import logging
import subprocess
import tempfile
from pathlib import Path

from pdf2image import convert_from_path
from PIL import Image

logger = logging.getLogger(__name__)


def convert_hwp_to_pdf(hwp_path: Path) -> Path:
    """HWP 파일을 LibreOffice headless로 PDF 변환."""
    output_dir = hwp_path.parent
    result = subprocess.run(
        [
            "libreoffice",
            "--headless",
            "--convert-to",
            "pdf",
            "--outdir",
            str(output_dir),
            str(hwp_path),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(f"LibreOffice 변환 실패: {result.stderr}")

    pdf_path = output_dir / f"{hwp_path.stem}.pdf"
    if not pdf_path.exists():
        raise FileNotFoundError(f"변환된 PDF를 찾을 수 없음: {pdf_path}")

    logger.info("HWP→PDF 변환 완료: %s → %s", hwp_path.name, pdf_path.name)
    return pdf_path


def pdf_to_page_images(pdf_path: Path, dpi: int = 200) -> list[Image.Image]:
    """PDF를 페이지별 PIL 이미지로 변환."""
    images = convert_from_path(str(pdf_path), dpi=dpi)
    logger.info("PDF→이미지 변환 완료: %s (%d 페이지)", pdf_path.name, len(images))
    return images


def process_document(file_path: Path) -> tuple[list[Image.Image], Path]:
    """문서 파일을 페이지 이미지 리스트로 변환.

    Returns:
        (page_images, pdf_path) - 원본이 HWP인 경우 변환된 PDF 경로 포함
    """
    suffix = file_path.suffix.lower()

    if suffix == ".hwp":
        pdf_path = convert_hwp_to_pdf(file_path)
        images = pdf_to_page_images(pdf_path)
        return images, pdf_path
    elif suffix == ".pdf":
        images = pdf_to_page_images(file_path)
        return images, file_path
    elif suffix in (".png", ".jpg", ".jpeg", ".tiff", ".bmp"):
        img = Image.open(file_path).convert("RGB")
        return [img], file_path
    else:
        raise ValueError(f"지원하지 않는 파일 형식: {suffix}")
