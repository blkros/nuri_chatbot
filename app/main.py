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
from app.search.vllm_client import describe_image_for_search, generate_answer
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
        page_images, temp_pdf_path, page_texts, chunk_metas = process_document(file_path)

        # 2. 분류 (이미지 있으면 VLM, 없으면 키워드 폴백)
        all_text = "\n".join(page_texts[:3])
        classification = classify_document_vlm(
            page_image=page_images[0] if page_images else None,
            text_content=all_text,
            file_name=file.filename,
        )
        department = classification.get("department", "기타")
        doc_type = classification.get("doc_type", "기타")
        summary = classification.get("summary", "")
        prefix = f"[{department}/{doc_type}] {file.filename} | "

        metadata = {
            "department": department,
            "doc_type": doc_type,
            "summary": summary,
        }

        # 이미지 파일: VLM 설명을 OCR 텍스트 앞에 붙여 검색 품질 향상
        # (OCR은 레이아웃/컨텍스트 키워드를 놓치므로 VLM이 보완)
        if suffix in IMAGE_EXTENSIONS and page_images:
            try:
                description = describe_image_for_search(page_images[0])
                page_texts = [f"[이미지 설명] {description}\n\n{t}" for t in page_texts]
                logger.info("이미지 설명 추가 완료: %s", file.filename)
            except Exception as e:
                logger.warning("이미지 설명 생성 실패 (OCR만 사용): %s", e)

        point_ids = []

        if page_images:
            # ── 이미지 + 텍스트 경로 (PDF, HWP, 오피스, 이미지) ──
            doc_id = str(uuid4())
            img_dir = Path(settings.page_images_dir) / doc_id
            img_dir.mkdir(parents=True, exist_ok=True)

            image_paths = []
            for i, img in enumerate(page_images):
                img_path = img_dir / f"page_{i + 1}.jpg"
                img.save(str(img_path), "JPEG", quality=90)
                image_paths.append(str(img_path))

            img_vectors_list = embed_images(page_images)
            text_vectors = embed_texts(
                [prefix + t if t.strip() else prefix + file.filename for t in page_texts]
            )

            for i, (img_vec, txt_vec, page_text, img_path) in enumerate(
                zip(img_vectors_list, text_vectors, page_texts, image_paths)
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
        else:
            # ── 텍스트 전용 경로 (Excel 등) ──
            text_vectors = embed_texts(
                [prefix + t if t.strip() else prefix + file.filename for t in page_texts]
            )

            for i, (txt_vec, page_text) in enumerate(
                zip(text_vectors, page_texts)
            ):
                # 청크별 메타 (sheet, section) 병합
                chunk_meta = metadata.copy()
                if chunk_metas and i < len(chunk_metas):
                    chunk_meta.update(chunk_metas[i])

                pid = upsert_page(
                    file_name=file.filename,
                    page_number=i + 1,
                    image_vectors=None,
                    text_vector=txt_vec,
                    ocr_text=page_text,
                    image_path="",
                    metadata=chunk_meta,
                )
                point_ids.append(pid)

        return {
            "status": "success",
            "file_name": file.filename,
            "pages": len(page_images) or len(page_texts),
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


# ─── Adaptive top_k ──────────────────────────────────────────


def _compute_adaptive_k(rerank_scores_sorted: list[float]) -> int:
    """리랭커 점수 분포 기반 adaptive top_k 결정.

    1) 절대 점수 하한 필터 — 노이즈(비관련 문서) 제거
    2) Gap-knee 탐지 — 점수 급락 지점에서 관련/비관련 경계 식별
    3) MIN_K ~ MAX_K 클램프
    """
    if not rerank_scores_sorted:
        return settings.adaptive_min_k

    # 1. 절대 점수 하한 필터
    valid_count = sum(
        1 for s in rerank_scores_sorted if s >= settings.rerank_score_min
    )

    # 2. Gap-knee 탐지: 점수 급락 지점 찾기
    knee_k = len(rerank_scores_sorted)
    for i in range(len(rerank_scores_sorted) - 1):
        gap = rerank_scores_sorted[i] - rerank_scores_sorted[i + 1]
        if gap > settings.rerank_gap_threshold:
            knee_k = i + 1
            break

    # 3. 두 기준 중 더 보수적인 값 (절대 점수 + gap 모두 통과해야 포함)
    k = min(knee_k, valid_count)

    # 4. MIN_K ~ MAX_K 클램프
    k = max(settings.adaptive_min_k, min(k, settings.adaptive_max_k))

    logger.info(
        "Adaptive top_k=%d (valid=%d, knee=%d, scores=[%s])",
        k,
        valid_count,
        knee_k,
        ", ".join(f"{s:.3f}" for s in rerank_scores_sorted[: min(k + 2, len(rerank_scores_sorted))]),
    )
    return k


def _expand_anchor_document(
    results: list[dict],
    fused: list[tuple[int, float]],
    initial_k: int,
) -> list[int]:
    """앵커 문서 확장: top-1 결과의 동일 문서에서 추가 페이지 포함.

    리랭커가 개요 페이지만 높게 평가하고 상세 페이지를 낮게 평가하는 경우,
    같은 문서 내 페이지를 추가로 포함하여 상세 컨텍스트를 확보한다.
    다른 문서의 노이즈 유입 없이 안전하게 확장.
    """
    top_indices = [idx for idx, _ in fused[:initial_k]]

    if not top_indices:
        return top_indices

    # Top-1 결과의 파일 = 앵커 문서 후보
    anchor_file = results[fused[0][0]]["file_name"]

    # 앵커 문서가 검색 결과에 3개 이상 존재할 때만 확장
    # (단일/소수 페이지 문서는 확장 불필요)
    anchor_count = sum(1 for r in results if r["file_name"] == anchor_file)
    if anchor_count < 3:
        return top_indices

    # 나머지 결과에서 앵커 문서 페이지 추가
    # (확장 페이지는 VLM에 텍스트만 전달되므로 이미지 토큰 부담 없음)
    for idx, _ in fused[initial_k:]:
        if len(top_indices) >= settings.adaptive_max_k:
            break
        if results[idx]["file_name"] == anchor_file:
            top_indices.append(idx)

    if len(top_indices) > initial_k:
        # 페이지 번호순 정렬 (자연스러운 읽기 순서)
        top_indices.sort(key=lambda i: results[i].get("page_number", 0))
        logger.info(
            "앵커 문서 확장: %s (%d→%d 페이지)",
            anchor_file, initial_k, len(top_indices),
        )

    return top_indices


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
    top_k: int = Form(0),  # 0 = adaptive (리랭커 점수 기반 자동 결정)
):
    """질문 → 검색 → 리랭킹 → VLM 답변 생성 (전체 RAG 파이프라인)."""
    try:
        # 1. 쿼리 임베딩
        text_vector = embed_query_text(question)
        image_vectors = embed_query_for_images(question)

        # 2. Qdrant 하이브리드 검색 (넓은 풀에서 후보 확보)
        results = search_pages(
            text_query_vector=text_vector,
            image_query_vectors=image_vectors,
            limit=15,
        )

        if not results:
            return {
                "answer": "관련 문서를 찾을 수 없습니다. 다른 질문을 시도해주세요.",
                "sources": [],
            }

        # 3. 리랭킹 (전체 결과 대상)
        ocr_texts = [r["ocr_text"] for r in results]
        ranked = rerank(question, ocr_texts, top_k=len(results))

        # 4. Adaptive top_k 결정 (리랭커 점수 분포 기반)
        if top_k <= 0:
            rerank_scores_sorted = [score for _, score in ranked]
            effective_k = _compute_adaptive_k(rerank_scores_sorted)
        else:
            effective_k = top_k

        # 5. 검색 순위 + 리랭크 순위 RRF 결합
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

        # 6. 앵커 문서 확장 (같은 문서 내 페이지 추가)
        top_indices = _expand_anchor_document(results, fused, effective_k)
        top_results = [results[idx] for idx in top_indices]

        # 선택된 결과에 리랭크 점수 기록 (디버깅용)
        for idx in top_indices:
            results[idx]["rerank_score"] = rerank_scores.get(idx, 0.0)

        # 이미지 전송: 텍스트 충분도 기반 결정
        # - 이미지 원본 파일 (.png, .jpg 등) → 항상 이미지 전송 (OCR은 레이아웃 손실)
        # - 텍스트가 충분한 페이지 → 텍스트만 (눈 불필요)
        # - 텍스트가 부족한 페이지 → 이미지 필요 (스캔 문서 등)
        page_images = []
        img_count = 0
        for r in top_results:
            ocr_text = r.get("ocr_text", "")
            img_path = r.get("image_path", "")
            file_ext = Path(r.get("file_name", "")).suffix.lower()
            is_image_file = file_ext in IMAGE_EXTENSIONS
            text_sufficient = (
                not is_image_file  # 이미지 파일은 항상 "눈 필요"
                and len(ocr_text.strip()) >= settings.text_sufficient_length
            )
            if not text_sufficient and img_path and Path(img_path).exists():
                page_images.append(Image.open(img_path).convert("RGB"))
                img_count += 1
            else:
                page_images.append(None)  # 텍스트 충분 → 이미지 불필요
        logger.info("VLM 전송: 이미지 %d장 (눈 필요) + 텍스트 전용 %d장", img_count, len(top_results) - img_count)

        # 7. VLM 답변 생성 (이미지/텍스트 혼합 지원)
        source_info = [
            {"file_name": r["file_name"], "page_number": r["page_number"]}
            for r in top_results
        ]
        ocr_for_vlm = [r["ocr_text"] for r in top_results]

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
