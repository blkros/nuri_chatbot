import json
import logging
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
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
from app.search.vllm_client import (
    describe_image_for_search,
    generate_answer,
    generate_answer_stream,
    rewrite_query,
)
from app.vectordb.qdrant_client import (
    delete_document_pages,
    ensure_collection,
    get_document_pages,
    list_documents,
    search_pages,
    upsert_page,
)

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
    """서비스 상태 확인 (Qdrant + vLLM 연결 포함)."""
    import httpx as _httpx

    checks = {"qdrant": "ok", "vllm": "ok"}

    # Qdrant 연결 확인
    try:
        from app.vectordb.qdrant_client import get_client
        get_client().get_collections()
    except Exception as e:
        checks["qdrant"] = f"error: {e}"

    # vLLM 연결 확인
    try:
        async with _httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(f"{settings.vllm_base_url}/models")
            if resp.status_code != 200:
                checks["vllm"] = f"error: HTTP {resp.status_code}"
    except Exception as e:
        checks["vllm"] = f"error: {e}"

    all_ok = all(v == "ok" for v in checks.values())
    return {"status": "ok" if all_ok else "degraded", **checks}


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
    # 경로 순회 방지: 파일명에서 디렉토리 구분자 제거
    safe_filename = Path(file.filename).name
    if not safe_filename or safe_filename.startswith("."):
        raise HTTPException(status_code=400, detail="잘못된 파일명입니다.")
    file_path = upload_dir / safe_filename
    temp_pdf_path = None

    # 파일 저장
    with open(file_path, "wb") as f:
        f.write(contents)
    logger.info("파일 업로드: %s (%d bytes)", safe_filename, len(contents))

    try:
        # 1. 문서 → 페이지 이미지 + 텍스트 추출
        page_images, temp_pdf_path, page_texts, chunk_metas = process_document(file_path)

        # 페이지 수 상한 제한 (임베딩 비용 폭발 방지)
        max_pages = settings.max_pages_per_document
        total_pages = len(page_images) if page_images else len(page_texts)
        if total_pages > max_pages:
            logger.warning(
                "페이지 수 제한: %s (%d → %d 페이지)", safe_filename, total_pages, max_pages
            )
            if page_images:
                page_images = page_images[:max_pages]
            page_texts = page_texts[:max_pages]
            if chunk_metas:
                chunk_metas = chunk_metas[:max_pages]

        # 2. 분류 (이미지 있으면 VLM, 없으면 키워드 폴백)
        all_text = "\n".join(page_texts[:3])
        classification = classify_document_vlm(
            page_image=page_images[0] if page_images else None,
            text_content=all_text,
            file_name=safe_filename,
        )
        department = classification.get("department", "기타")
        doc_type = classification.get("doc_type", "기타")
        summary = classification.get("summary", "")
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
                logger.info("이미지 설명 추가 완료: %s", safe_filename)
            except Exception as e:
                logger.warning("이미지 설명 생성 실패 (OCR만 사용): %s", e)

        # 빈 페이지 필터링 (텍스트 없고 이미지도 없는 페이지 제거)
        MIN_MEANINGFUL_TEXT = 10
        if page_images:
            filtered = [
                (img, txt, meta)
                for i, (img, txt) in enumerate(zip(page_images, page_texts))
                for meta in [chunk_metas[i] if chunk_metas and i < len(chunk_metas) else None]
                if len(txt.strip()) >= MIN_MEANINGFUL_TEXT or img is not None
            ]
            if filtered:
                page_images = [f[0] for f in filtered]
                page_texts = [f[1] for f in filtered]
                if chunk_metas:
                    chunk_metas = [f[2] for f in filtered]
                if len(filtered) < total_pages:
                    logger.info("빈 페이지 필터링: %d → %d 페이지", total_pages, len(filtered))
        else:
            # 텍스트 전용 (Excel 등): 빈 청크 제거
            filtered_txt = []
            filtered_meta = []
            for i, txt in enumerate(page_texts):
                if len(txt.strip()) >= MIN_MEANINGFUL_TEXT:
                    filtered_txt.append(txt)
                    if chunk_metas and i < len(chunk_metas):
                        filtered_meta.append(chunk_metas[i])
            if filtered_txt:
                if len(filtered_txt) < len(page_texts):
                    logger.info("빈 청크 필터링: %d → %d", len(page_texts), len(filtered_txt))
                page_texts = filtered_txt
                if chunk_metas:
                    chunk_metas = filtered_meta

        if not page_images and not page_texts:
            return JSONResponse(
                status_code=400,
                content={"status": "error", "detail": "문서에서 텍스트를 추출할 수 없습니다."},
            )

        # 컬렉션 보장 (삭제 후 재인제스트 시 필요)
        ensure_collection()
        # 기존 동일 파일의 벡터 삭제 (중복 방지)
        delete_document_pages(safe_filename)

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
                [t if t.strip() else safe_filename for t in page_texts]
            )

            for i, (img_vec, txt_vec, page_text, img_path) in enumerate(
                zip(img_vectors_list, text_vectors, page_texts, image_paths)
            ):
                pid = upsert_page(
                    file_name=safe_filename,
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
                [t if t.strip() else safe_filename for t in page_texts]
            )

            for i, (txt_vec, page_text) in enumerate(
                zip(text_vectors, page_texts)
            ):
                # 청크별 메타 (sheet, section) 병합
                chunk_meta = metadata.copy()
                if chunk_metas and i < len(chunk_metas):
                    chunk_meta.update(chunk_metas[i])

                pid = upsert_page(
                    file_name=safe_filename,
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
            "file_name": safe_filename,
            "pages": len(page_images) or len(page_texts),
            "department": department,
            "doc_type": doc_type,
            "summary": summary,
            "point_ids": point_ids,
        }

    except Exception as e:
        logger.exception("인제스트 실패: %s", safe_filename)
        return JSONResponse(status_code=500, content={"status": "error", "detail": str(e)})

    finally:
        # 임시 파일 정리 (페이지 이미지는 유지)
        if file_path.exists():
            file_path.unlink()
        if temp_pdf_path and temp_pdf_path.exists():
            temp_pdf_path.unlink()


# ─── 문서 관리 ───────────────────────────────────────────────


@app.get("/documents")
def get_documents():
    """인제스트된 문서 목록 조회."""
    try:
        docs = list_documents()
        return {"documents": docs, "total": len(docs)}
    except Exception as e:
        logger.exception("문서 목록 조회 실패")
        return JSONResponse(status_code=500, content={"status": "error", "detail": str(e)})


@app.delete("/documents/{file_name}")
def delete_document(file_name: str):
    """특정 문서 삭제 (Qdrant 포인트 + 저장된 이미지)."""
    import shutil

    try:
        # 1. 저장된 이미지 경로 조회 후 삭제
        pages = get_document_pages(file_name, limit=500)
        if not pages:
            raise HTTPException(status_code=404, detail=f"문서를 찾을 수 없습니다: {file_name}")

        deleted_dirs = set()
        for page in pages:
            img_path = page.get("image_path", "")
            if img_path and Path(img_path).exists():
                deleted_dirs.add(Path(img_path).parent)

        # 2. Qdrant에서 포인트 삭제
        delete_document_pages(file_name)

        # 3. 이미지 디렉토리 삭제
        for d in deleted_dirs:
            shutil.rmtree(d, ignore_errors=True)
            logger.info("이미지 디렉토리 삭제: %s", d)

        return {
            "status": "success",
            "file_name": file_name,
            "deleted_pages": len(pages),
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("문서 삭제 실패: %s", file_name)
        return JSONResponse(status_code=500, content={"status": "error", "detail": str(e)})


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


def _expand_with_doc_concentration(
    results: list[dict],
    fused: list[tuple[int, float]],
    initial_k: int,
    search_query: str,
    rerank_scores: dict[int, float] | None = None,
) -> list[dict]:
    """문서 집중도 기반 컨텍스트 확장 (확장 페이지도 리랭킹 적용).

    검색 결과를 doc_id(file_name)별로 집계하여:
    1) 집중도 높은 경우 (같은 문서가 결과의 50%+) → Qdrant scroll로 문서 전체 페이지 조회
    2) 집중도 낮은 경우 → 기존처럼 검색 결과 내에서만 확장

    Returns:
        최종 컨텍스트에 포함할 결과 리스트 (page_number순 정렬)
    """
    top_indices = [idx for idx, _ in fused[:initial_k]]
    if not top_indices:
        return []

    # ── 문서별 집중도 분석 ──
    from collections import Counter
    doc_counts = Counter(results[idx]["file_name"] for idx, _ in fused[:initial_k])
    total = len(top_indices)
    anchor_file, anchor_count = doc_counts.most_common(1)[0]
    concentration = anchor_count / total

    logger.info(
        "문서 집중도: %s = %d/%d (%.0f%%)",
        anchor_file, anchor_count, total, concentration * 100,
    )

    # anchor 문서의 최고 rerank score 확인 — 낮으면 확장 스킵
    # (무관한 문서가 여러 페이지 검색된 것일 뿐, 진짜 집중이 아님)
    if rerank_scores:
        anchor_best_score = max(
            (rerank_scores.get(idx, 0.0) for idx in top_indices
             if results[idx]["file_name"] == anchor_file),
            default=0.0,
        )
    else:
        anchor_best_score = 1.0  # rerank_scores 없으면 확장 허용

    # ── 집중도 높지만 anchor 점수가 낮음 → 확장 없이 초기 결과만 반환 ──
    if concentration >= settings.doc_concentration_threshold and anchor_best_score < settings.rerank_score_min:
        logger.info(
            "집중도 높으나 앵커 점수 낮음 (%.3f < %.3f), 확장 스킵",
            anchor_best_score, settings.rerank_score_min,
        )
        return [results[idx] for idx in top_indices]

    # ── 집중도 높음 + anchor 문서가 실제로 관련 있음 → 문서 확장 ──
    if concentration >= settings.doc_concentration_threshold and anchor_best_score >= settings.rerank_score_min:
        all_pages = get_document_pages(anchor_file, limit=200)
        doc_is_small = len(all_pages) <= settings.max_doc_expansion_pages

        other_results = [
            results[idx] for idx in top_indices
            if results[idx]["file_name"] != anchor_file
        ]

        # 검색에 이미 걸린 페이지 번호
        existing_pages = {results[idx]["page_number"] for idx in top_indices
                         if results[idx]["file_name"] == anchor_file}

        if doc_is_small:
            candidate_pages = all_pages
        else:
            # 문서가 크면 검색에 걸린 페이지 + 인접 페이지(±1)
            candidate_pages = []
            for page in all_pages:
                pn = page["page_number"]
                if pn in existing_pages or any(abs(pn - ep) <= 1 for ep in existing_pages):
                    candidate_pages.append(page)

        # 새로 추가된 페이지(검색에 없던 것)를 리랭커로 필터링
        new_pages = [p for p in candidate_pages if p["page_number"] not in existing_pages]
        kept_pages = [p for p in candidate_pages if p["page_number"] in existing_pages]

        if new_pages:
            new_texts = [p["ocr_text"] for p in new_pages]
            ranked_new = rerank(search_query, new_texts, top_k=len(new_texts))
            # 리랭크 점수 하한 필터 적용
            for idx, score in ranked_new:
                if score >= settings.rerank_score_min:
                    kept_pages.append(new_pages[idx])

        # max_doc_expansion_pages로 cap + 정렬
        kept_pages.sort(key=lambda p: p["page_number"])
        if len(kept_pages) > settings.max_doc_expansion_pages:
            kept_pages = kept_pages[:settings.max_doc_expansion_pages]

        logger.info(
            "문서 확장: %s → %d 페이지 (검색 히트: %s, 리랭크 통과: %d)",
            anchor_file, len(kept_pages),
            sorted(existing_pages),
            len(kept_pages) - len(existing_pages),
        )
        return kept_pages + other_results[:2]

    # ── 집중도 낮음 → 기존 방식: 검색 결과 내에서만 확장 ──
    for idx, _ in fused[initial_k:]:
        if len(top_indices) >= settings.adaptive_max_k:
            break
        if results[idx]["file_name"] == anchor_file:
            top_indices.append(idx)

    if len(top_indices) > initial_k:
        top_indices.sort(key=lambda i: results[i].get("page_number", 0))
        logger.info(
            "앵커 문서 확장 (기존): %s (%d→%d 페이지)",
            anchor_file, initial_k, len(top_indices),
        )

    return [results[idx] for idx in top_indices]


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

        fused = []
        for search_rank in range(len(results)):
            rr_rank = rerank_order.get(search_rank, len(results))
            score = (
                settings.rrf_search_weight / (settings.rrf_k + search_rank)
                + settings.rrf_rerank_weight / (settings.rrf_k + rr_rank)
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


def _prepare_rag_context(question: str, top_k: int = 0, history: list[dict] | None = None) -> dict:
    """검색 → 리랭킹 → 컨텍스트 준비 (ask / ask/stream 공용).

    Returns:
        {"page_images", "ocr_for_vlm", "source_info"} 또는
        {"early_return": dict} (결과 없을 때)
    """
    # 0. 쿼리 리라이팅 (멀티턴 히스토리 반영)
    try:
        search_query = rewrite_query(question, history=history)
    except Exception as e:
        logger.warning("쿼리 리라이팅 실패 (원본 사용): %s", e)
        search_query = question

    # 1. 쿼리 임베딩
    text_vector = embed_query_text(search_query)
    image_vectors = embed_query_for_images(search_query)

    # 2. Qdrant 하이브리드 검색
    results = search_pages(
        text_query_vector=text_vector,
        image_query_vectors=image_vectors,
        limit=15,
    )

    if not results:
        return {"early_return": {
            "answer": "관련 문서를 찾을 수 없습니다. 다른 질문을 시도해주세요.",
            "sources": [],
        }}

    # 2.5 중복 제거 (text/image prefetch에서 같은 페이지가 중복될 수 있음)
    seen = set()
    deduped = []
    for r in results:
        key = (r["file_name"], r["page_number"])
        if key not in seen:
            seen.add(key)
            deduped.append(r)
    if len(deduped) < len(results):
        logger.info("검색 결과 중복 제거: %d → %d", len(results), len(deduped))
        results = deduped

    # 3. 리랭킹 (실패 시 검색 점수 순서 폴백)
    try:
        ocr_texts = [r["ocr_text"] for r in results]
        ranked = rerank(search_query, ocr_texts, top_k=len(results))
        rerank_failed = False
    except Exception as e:
        logger.warning("리랭킹 실패, 검색 점수 순서 사용: %s", e)
        ranked = [(i, 0.5) for i in range(len(results))]
        rerank_failed = True

    # 4. Adaptive top_k
    if top_k <= 0:
        if rerank_failed:
            effective_k = min(settings.adaptive_max_k, len(results))
        else:
            rerank_scores_sorted = [score for _, score in ranked]
            effective_k = _compute_adaptive_k(rerank_scores_sorted)
    else:
        effective_k = top_k

    # 5. RRF 결합
    rerank_order = {idx: rank for rank, (idx, _) in enumerate(ranked)}
    rerank_scores = {idx: score for idx, score in ranked}

    fused = []
    for search_rank in range(len(results)):
        rr_rank = rerank_order.get(search_rank, len(results))
        score = (
            settings.rrf_search_weight / (settings.rrf_k + search_rank)
            + settings.rrf_rerank_weight / (settings.rrf_k + rr_rank)
        )
        fused.append((search_rank, score))

    fused.sort(key=lambda x: x[1], reverse=True)

    # 6. 문서 집중도 기반 컨텍스트 확장
    top_results = _expand_with_doc_concentration(results, fused, effective_k, search_query, rerank_scores)

    # VLM 컨텍스트 예산 제한 (max_model_len=8192 초과 방지)
    if len(top_results) > settings.max_context_pages:
        logger.warning("컨텍스트 페이지 수 제한: %d → %d", len(top_results), settings.max_context_pages)
        top_results = top_results[:settings.max_context_pages]

    # 이미지 전송: 텍스트 충분도 기반 결정 + 최대 이미지 수 제한
    page_images = []
    img_count = 0
    for r in top_results:
        ocr_text = r.get("ocr_text", "")
        img_path = r.get("image_path", "")
        file_ext = Path(r.get("file_name", "")).suffix.lower()
        is_image_file = file_ext in IMAGE_EXTENSIONS
        text_sufficient = (
            not is_image_file
            and len(ocr_text.strip()) >= settings.text_sufficient_length
        )
        if (
            not text_sufficient
            and img_path
            and Path(img_path).exists()
            and img_count < settings.max_context_images
        ):
            with Image.open(img_path) as _img:
                page_images.append(_img.convert("RGB"))
            img_count += 1
        else:
            page_images.append(None)
    logger.info("VLM 전송: 이미지 %d장 (눈 필요) + 텍스트 전용 %d장", img_count, len(top_results) - img_count)

    source_info = [
        {"file_name": r["file_name"], "page_number": r["page_number"]}
        for r in top_results
    ]
    ocr_for_vlm = [r["ocr_text"] for r in top_results]

    return {
        "page_images": page_images,
        "ocr_for_vlm": ocr_for_vlm,
        "source_info": source_info,
    }


@app.post("/ask")
def ask_question(
    question: str = Form(...),
    top_k: int = Form(0),
    history: str = Form("[]"),
):
    """질문 → 검색 → 리랭킹 → VLM 답변 생성 (비스트리밍)."""
    try:
        conv_history = json.loads(history) if history else []
    except (json.JSONDecodeError, TypeError):
        conv_history = []

    try:
        ctx = _prepare_rag_context(question, top_k, history=conv_history or None)
        if "early_return" in ctx:
            return ctx["early_return"]

        result = generate_answer(
            question=question,
            page_images=ctx["page_images"],
            ocr_texts=ctx["ocr_for_vlm"],
            source_info=ctx["source_info"],
            history=conv_history or None,
        )
        return result

    except Exception as e:
        logger.exception("답변 생성 실패")
        err_str = str(e).lower()
        if any(kw in err_str for kw in ("connect", "timeout", "refused", "unreachable")):
            user_msg = "AI 모델 서버에 연결할 수 없습니다. 잠시 후 다시 시도해주세요."
        else:
            user_msg = "답변 생성 중 오류가 발생했습니다. 다시 시도해주세요."
        return JSONResponse(status_code=503, content={"status": "error", "detail": user_msg})


@app.post("/ask/stream")
def ask_question_stream(
    question: str = Form(...),
    top_k: int = Form(0),
    history: str = Form("[]"),
):
    """질문 → 검색 → 리랭킹 → VLM SSE 스트리밍 답변."""
    try:
        conv_history = json.loads(history) if history else []
    except (json.JSONDecodeError, TypeError):
        conv_history = []

    def event_stream():
        try:
            ctx = _prepare_rag_context(question, top_k, history=conv_history or None)

            if "early_return" in ctx:
                data = ctx["early_return"]
                yield f"data: {json.dumps(data, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
                return

            # 출처 정보 먼저 전송 (프론트에서 미리 표시 가능)
            yield f"data: {json.dumps({'sources': ctx['source_info']}, ensure_ascii=False)}\n\n"

            # VLM 스트리밍 답변 (멀티턴 히스토리 전달)
            for token in generate_answer_stream(
                question=question,
                page_images=ctx["page_images"],
                ocr_texts=ctx["ocr_for_vlm"],
                source_info=ctx["source_info"],
                history=conv_history or None,
            ):
                yield f"data: {json.dumps({'token': token}, ensure_ascii=False)}\n\n"

            yield "data: [DONE]\n\n"

        except Exception as e:
            logger.exception("스트리밍 답변 생성 실패")
            # 사용자 친화적 에러 메시지 (VLM 연결 실패 vs 기타)
            err_str = str(e).lower()
            if any(kw in err_str for kw in ("connect", "timeout", "refused", "unreachable")):
                user_msg = "AI 모델 서버에 연결할 수 없습니다. 잠시 후 다시 시도해주세요."
            else:
                user_msg = "답변 생성 중 오류가 발생했습니다. 다시 시도해주세요."
            yield f"data: {json.dumps({'token': user_msg}, ensure_ascii=False)}\n\n"
            yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )
