import logging
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image

from app.config import settings
from app.ingest.converter import process_document
from app.ingest.embedder import (
    embed_images,
    embed_query_for_images,
    embed_query_text,
    embed_texts,
)
from app.ingest.ocr import extract_text
from app.search.reranker import rerank
from app.search.vllm_client import generate_answer
from app.vectordb.qdrant_client import ensure_collection, search_pages, upsert_page

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Nuri RAG Chatbot API")

SUPPORTED_EXTENSIONS = {".hwp", ".hwpx", ".pdf", ".docx", ".doc", ".pptx", ".xlsx",
                        ".png", ".jpg", ".jpeg", ".tiff", ".bmp"}


@app.on_event("startup")
async def startup():
    Path(settings.upload_dir).mkdir(parents=True, exist_ok=True)
    Path(settings.page_images_dir).mkdir(parents=True, exist_ok=True)
    ensure_collection()
    logger.info("서버 시작 완료")


@app.get("/health")
async def health():
    return {"status": "ok"}


# ─── 문서 인제스트 ───────────────────────────────────────────

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

        # 2. 페이지 이미지를 디스크에 저장
        doc_id = str(uuid4())
        img_dir = Path(settings.page_images_dir) / doc_id
        img_dir.mkdir(parents=True, exist_ok=True)

        image_paths = []
        for i, img in enumerate(page_images):
            img_path = img_dir / f"page_{i + 1}.jpg"
            img.save(str(img_path), "JPEG", quality=90)
            image_paths.append(str(img_path))

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
        for i, (img_vec, txt_vec, ocr_text, img_path) in enumerate(
            zip(image_vectors, text_vectors, ocr_texts, image_paths)
        ):
            pid = upsert_page(
                file_name=file.filename,
                page_number=i + 1,
                image_vectors=img_vec,
                text_vector=txt_vec,
                ocr_text=ocr_text,
                image_path=img_path,
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
        # 임시 파일 정리 (페이지 이미지는 유지)
        if file_path.exists():
            file_path.unlink()
        if temp_pdf_path and temp_pdf_path.exists():
            temp_pdf_path.unlink()


# ─── 검색 + 답변 ────────────────────────────────────────────

@app.post("/search")
def search_documents(
    question: str = Form(...),
    department: str = Form(""),
    doc_type: str = Form(""),
    limit: int = Form(10),
):
    """질문으로 문서 검색 (하이브리드 검색 + 리랭킹, VLM 답변 없음)."""
    try:
        # 1. 쿼리 임베딩
        text_vector = embed_query_text(question)
        image_vectors = embed_query_for_images(question)

        # 2. Qdrant 하이브리드 검색
        results = search_pages(
            text_query_vector=text_vector,
            image_query_vectors=image_vectors,
            limit=limit,
            department=department or None,
            doc_type=doc_type or None,
        )

        if not results:
            return {"results": [], "message": "검색 결과가 없습니다."}

        # 3. 리랭킹
        ocr_texts = [r["ocr_text"] for r in results]
        ranked = rerank(question, ocr_texts, top_k=limit)

        # 리랭킹 순서로 결과 재정렬
        reranked_results = []
        for idx, score in ranked:
            result = results[idx].copy()
            result["rerank_score"] = score
            reranked_results.append(result)

        return {"results": reranked_results}

    except Exception as e:
        logger.exception("검색 실패")
        return JSONResponse(status_code=500, content={"status": "error", "detail": str(e)})


@app.post("/ask")
def ask_question(
    question: str = Form(...),
    department: str = Form(""),
    doc_type: str = Form(""),
    top_k: int = Form(3),
):
    """질문 → 검색 → 리랭킹 → VLM 답변 생성 (전체 RAG 파이프라인)."""
    try:
        # 1. 쿼리 임베딩
        text_vector = embed_query_text(question)
        image_vectors = embed_query_for_images(question)

        # 2. Qdrant 하이브리드 검색
        results = search_pages(
            text_query_vector=text_vector,
            image_query_vectors=image_vectors,
            limit=max(top_k * 2, 10),
            department=department or None,
            doc_type=doc_type or None,
        )

        if not results:
            return {
                "answer": "관련 문서를 찾을 수 없습니다. 다른 질문을 시도해주세요.",
                "sources": [],
            }

        # 3. 리랭킹
        ocr_texts = [r["ocr_text"] for r in results]
        ranked = rerank(question, ocr_texts, top_k=top_k)

        # 4. 상위 페이지 이미지 로드
        top_results = [results[idx] for idx, _ in ranked]
        page_images = []
        valid_results = []

        for r in top_results:
            img_path = r.get("image_path", "")
            if img_path and Path(img_path).exists():
                page_images.append(Image.open(img_path).convert("RGB"))
                valid_results.append(r)
            else:
                logger.warning("이미지 없음: %s p.%d", r["file_name"], r["page_number"])

        if not page_images:
            return {
                "answer": "검색된 문서의 페이지 이미지를 로드할 수 없습니다. 문서를 재인제스트해주세요.",
                "sources": [{"file_name": r["file_name"], "page_number": r["page_number"]} for r in top_results],
            }

        # 5. VLM 답변 생성
        source_info = [
            {"file_name": r["file_name"], "page_number": r["page_number"]}
            for r in valid_results
        ]
        ocr_for_vlm = [r["ocr_text"] for r in valid_results]

        result = generate_answer(
            question=question,
            page_images=page_images,
            ocr_texts=ocr_for_vlm,
            source_info=source_info,
        )

        return result

    except Exception as e:
        logger.exception("답변 생성 실패")
        return JSONResponse(status_code=500, content={"status": "error", "detail": str(e)})
