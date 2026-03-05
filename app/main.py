import logging
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from app.config import settings
from app.ingest.classifier import classify_document_vlm
from app.ingest.converter import process_document
from app.ingest.embedder import (
    embed_images,
    embed_query_for_images,
    embed_query_text,
    embed_texts,
)
from app.search.reranker import rerank
from app.search.vllm_client import generate_answer
from app.vectordb.qdrant_client import ensure_collection, search_pages, upsert_page

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = FastAPI(title="Nuri RAG Chatbot API")

# Static 파일 서빙
STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/")
async def root():
    return FileResponse(str(STATIC_DIR / "index.html"))


# 문서 포맷
OFFICE_EXTENSIONS = {".hwp", ".hwpx", ".docx", ".doc", ".pptx", ".ppt",
                     ".xlsx", ".xls", ".csv", ".odt", ".ods", ".odp", ".rtf"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".gif", ".webp"}
SUPPORTED_EXTENSIONS = OFFICE_EXTENSIONS | IMAGE_EXTENSIONS | {".pdf"}


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
):
    """문서 업로드 → 변환 → 텍스트 추출 → 임베딩 → Qdrant 저장."""
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
        # 1. 문서 → 페이지 이미지 + 텍스트 추출
        page_images, temp_pdf_path, page_texts = process_document(file_path)

        # 2. 페이지 이미지를 디스크에 저장
        doc_id = str(uuid4())
        img_dir = Path(settings.page_images_dir) / doc_id
        img_dir.mkdir(parents=True, exist_ok=True)

        image_paths = []
        for i, img in enumerate(page_images):
            img_path = img_dir / f"page_{i + 1}.jpg"
            img.save(str(img_path), "JPEG", quality=90)
            image_paths.append(str(img_path))

        # 3. VLM 자동 분류 (임베딩 전에 실행 → prefix에 활용)
        all_text = "\n".join(page_texts[:3])
        classification = classify_document_vlm(
            page_image=page_images[0],
            text_content=all_text,
            file_name=file.filename,
        )
        department = classification.get("department", "기타")
        doc_type = classification.get("doc_type", "기타")
        summary = classification.get("summary", "")

        # 4. 임베딩 (텍스트에 메타데이터 prefix 추가)
        prefix = f"[{department}/{doc_type}] {file.filename} | "
        image_vectors = embed_images(page_images)
        text_vectors = embed_texts(
            [prefix + t if t.strip() else prefix + file.filename for t in page_texts]
        )

        # 5. Qdrant 저장
        metadata = {
            "department": department,
            "doc_type": doc_type,
            "summary": summary,
        }

        point_ids = []
        for i, (img_vec, txt_vec, page_text, img_path) in enumerate(
            zip(image_vectors, text_vectors, page_texts, image_paths)
        ):
            pid = upsert_page(
                file_name=file.filename,
                page_number=i + 1,
                image_vectors=img_vec,
                text_vector=txt_vec,
                ocr_text=page_text,
                image_path=img_path,
                metadata=metadata,
            )
            point_ids.append(pid)

        return {
            "status": "success",
            "file_name": file.filename,
            "pages": len(page_images),
            "department": department,
            "doc_type": doc_type,
            "summary": summary,
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
        )

        if not results:
            return {"results": [], "message": "검색 결과가 없습니다."}

        # 3. 리랭킹 (전체 결과 대상)
        ocr_texts = [r["ocr_text"] for r in results]
        ranked = rerank(question, ocr_texts, top_k=len(results))

        # 4. 검색 순위 + 리랭크 순위 RRF 결합
        rerank_order = {idx: rank for rank, (idx, _) in enumerate(ranked)}
        rerank_scores = {idx: score for idx, score in ranked}
        RRF_K = 60
        SEARCH_WEIGHT = 1.5
        RERANK_WEIGHT = 1.0

        fused = []
        for search_rank in range(len(results)):
            rr_rank = rerank_order.get(search_rank, len(results))
            score = (
                SEARCH_WEIGHT / (RRF_K + search_rank)
                + RERANK_WEIGHT / (RRF_K + rr_rank)
            )
            fused.append((search_rank, score))

        fused.sort(key=lambda x: x[1], reverse=True)

        reranked_results = []
        for idx, fused_score in fused[:limit]:
            result = results[idx].copy()
            result["rerank_score"] = rerank_scores.get(idx, 0.0)
            result["fused_score"] = fused_score
            reranked_results.append(result)

        return {"results": reranked_results}

    except Exception as e:
        logger.exception("검색 실패")
        return JSONResponse(status_code=500, content={"status": "error", "detail": str(e)})


@app.post("/ask")
def ask_question(
    question: str = Form(...),
    top_k: int = Form(5),
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
            limit=max(top_k * 3, 15),
        )

        if not results:
            return {
                "answer": "관련 문서를 찾을 수 없습니다. 다른 질문을 시도해주세요.",
                "sources": [],
            }

        # 3. 리랭킹 (전체 결과 대상)
        ocr_texts = [r["ocr_text"] for r in results]
        ranked = rerank(question, ocr_texts, top_k=len(results))

        # 4. 검색 순위 + 리랭크 순위 RRF 결합
        #    벡터 검색(이미지+텍스트)과 리랭커 양쪽을 반영
        #    검색 가중치를 높여 리랭커 truncation 문제 보완
        rerank_order = {idx: rank for rank, (idx, _) in enumerate(ranked)}
        RRF_K = 60
        SEARCH_WEIGHT = 1.5
        RERANK_WEIGHT = 1.0

        fused = []
        for search_rank in range(len(results)):
            rr_rank = rerank_order.get(search_rank, len(results))
            score = (
                SEARCH_WEIGHT / (RRF_K + search_rank)
                + RERANK_WEIGHT / (RRF_K + rr_rank)
            )
            fused.append((search_rank, score))

        fused.sort(key=lambda x: x[1], reverse=True)
        top_results = [results[idx] for idx, _ in fused[:top_k]]
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
