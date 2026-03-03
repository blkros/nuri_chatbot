import logging
import shutil
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile
from fastapi.responses import JSONResponse

from app.config import settings
from app.ingest.converter import process_document
from app.ingest.embedder import embed_images, embed_texts
from app.ingest.ocr import extract_text
from app.vectordb.qdrant_client import ensure_collection, upsert_page

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Nuri RAG Chatbot API")


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
    upload_dir = Path(settings.upload_dir)
    file_path = upload_dir / file.filename

    # 1. 파일 저장
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    logger.info("파일 업로드: %s", file.filename)

    try:
        # 2. 문서 → 페이지 이미지
        page_images, pdf_path = process_document(file_path)

        # 3. OCR 텍스트 추출
        ocr_texts = [extract_text(img) for img in page_images]

        # 4. 임베딩
        image_vectors = embed_images(page_images)
        text_vectors = embed_texts(
            [t if t.strip() else file.filename for t in ocr_texts]
        )

        # 5. Qdrant 저장
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
