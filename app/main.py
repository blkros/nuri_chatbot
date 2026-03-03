import logging
import shutil
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse

from app.config import settings
from app.ingest.converter import process_document
from app.ingest.embedder import embed_images, embed_texts
from app.ingest.ocr import extract_text
from app.vectordb.qdrant_client import ensure_collection, upsert_page

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Nuri RAG Chatbot API")

SUPPORTED_EXTENSIONS = {".hwp", ".hwpx", ".pdf", ".docx", ".doc", ".pptx", ".xlsx",
                        ".png", ".jpg", ".jpeg", ".tiff", ".bmp"}


@app.on_event("startup")
async def startup():
    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
    ensure_collection()
    logger.info("서버 시작 완료")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.post("/ingest")
async def ingest_document(
    file: UploadFile = File(...),
    department: str = Form(""),
    doc_type: str = Form(""),
):
    """문서 업로드 → 변환 → OCR → 임베딩 → Qdrant 저장."""
    # 확장자 검증
    suffix = Path(file.filename).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"지원하지 않는 파일 형식: {suffix} (지원: {', '.join(sorted(SUPPORTED_EXTENSIONS))})",
        )

    # 파일 크기 검증
    contents = await file.read()
    if len(contents) > settings.max_upload_size_mb * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail=f"파일 크기 초과: 최대 {settings.max_upload_size_mb}MB",
        )

    upload_dir = Path(settings.upload_dir)
    file_path = upload_dir / file.filename
    temp_pdf_path = None

    # 파일 저장
    with open(file_path, "wb") as f:
        f.write(contents)
    logger.info("파일 업로드: %s (%d bytes)", file.filename, len(contents))

    try:
        # 1. 문서 → 페이지 이미지
        page_images, temp_pdf_path = process_document(file_path)

        # 2. OCR 텍스트 추출
        ocr_texts = [extract_text(img) for img in page_images]

        # 3. 임베딩
        image_vectors = embed_images(page_images)
        text_vectors = embed_texts(
            [t if t.strip() else file.filename for t in ocr_texts]
        )

        # 4. Qdrant 저장
        metadata = {}
        if department:
            metadata["department"] = department
        if doc_type:
            metadata["doc_type"] = doc_type

        point_ids = []
        for i, (img_vec, txt_vec, ocr_text) in enumerate(
            zip(image_vectors, text_vectors, ocr_texts)
        ):
            pid = upsert_page(
                file_name=file.filename,
                page_number=i + 1,
                image_vectors=img_vec,
                text_vector=txt_vec,
                ocr_text=ocr_text,
                metadata=metadata,
            )
            point_ids.append(pid)

        return {
            "status": "success",
            "file_name": file.filename,
            "pages": len(page_images),
            "point_ids": point_ids,
        }

    except Exception as e:
        logger.exception("인제스트 실패: %s", file.filename)
        return JSONResponse(status_code=500, content={"status": "error", "detail": str(e)})

    finally:
        # 임시 파일 정리
        if file_path.exists():
            file_path.unlink()
        if temp_pdf_path and temp_pdf_path.exists():
            temp_pdf_path.unlink()
