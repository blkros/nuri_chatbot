import logging
from uuid import uuid4

from qdrant_client import QdrantClient
from qdrant_client.http import models

from app.config import settings

logger = logging.getLogger(__name__)

_client = None


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        _client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port)
        logger.info("Qdrant 연결: %s:%d", settings.qdrant_host, settings.qdrant_port)
    return _client


def ensure_collection():
    """documents 컬렉션이 없으면 생성 (Named Vectors + MultiVector)."""
    client = get_client()
    collections = [c.name for c in client.get_collections().collections]

    if settings.collection_name in collections:
        logger.info("컬렉션 '%s' 이미 존재", settings.collection_name)
        return

    client.create_collection(
        collection_name=settings.collection_name,
        vectors_config={
            "image_vector": models.VectorParams(
                size=settings.nemotron_dim,
                distance=models.Distance.COSINE,
                multivector_config=models.MultiVectorConfig(
                    comparator=models.MultiVectorComparator.MAX_SIM,
                ),
            ),
            "text_vector": models.VectorParams(
                size=settings.bge_m3_ko_dim,
                distance=models.Distance.COSINE,
            ),
        },
    )
    logger.info("컬렉션 '%s' 생성 완료", settings.collection_name)


def upsert_page(
    file_name: str,
    page_number: int,
    image_vectors: list[list[float]] | None,
    text_vector: list[float],
    ocr_text: str,
    image_path: str = "",
    metadata: dict | None = None,
):
    """단일 페이지를 Qdrant에 저장. image_vectors=None이면 텍스트 벡터만 저장."""
    client = get_client()
    point_id = str(uuid4())

    payload = {
        "file_name": file_name,
        "page_number": page_number,
        "ocr_text": ocr_text,
        "image_path": image_path,
    }
    if metadata:
        payload.update(metadata)

    vectors = {"text_vector": text_vector}
    if image_vectors is not None:
        vectors["image_vector"] = image_vectors

    client.upsert(
        collection_name=settings.collection_name,
        points=[
            models.PointStruct(
                id=point_id,
                vector=vectors,
                payload=payload,
            )
        ],
    )
    logger.debug("페이지 저장: %s p.%d (id=%s)", file_name, page_number, point_id)
    return point_id


def search_pages(
    text_query_vector: list[float],
    image_query_vectors: list[list[float]],
    limit: int = 10,
    department: str | None = None,
    doc_type: str | None = None,
) -> list[dict]:
    """하이브리드 검색: text_vector + image_vector를 RRF로 융합."""
    client = get_client()

    # 메타데이터 필터 구성
    must_conditions = []
    if department:
        must_conditions.append(
            models.FieldCondition(
                key="department", match=models.MatchValue(value=department)
            )
        )
    if doc_type:
        must_conditions.append(
            models.FieldCondition(
                key="doc_type", match=models.MatchValue(value=doc_type)
            )
        )
    query_filter = models.Filter(must=must_conditions) if must_conditions else None

    prefetch_limit = max(limit * 2, 20)

    results = client.query_points(
        collection_name=settings.collection_name,
        prefetch=[
            models.Prefetch(
                query=text_query_vector,
                using="text_vector",
                limit=prefetch_limit,
                filter=query_filter,
            ),
            models.Prefetch(
                query=image_query_vectors,
                using="image_vector",
                limit=prefetch_limit,
                filter=query_filter,
            ),
        ],
        query=models.FusionQuery(fusion=models.Fusion.RRF),
        limit=limit,
        with_payload=True,
    )

    pages = []
    for point in results.points:
        pages.append({
            "id": point.id,
            "score": point.score,
            "file_name": point.payload.get("file_name", ""),
            "page_number": point.payload.get("page_number", 0),
            "ocr_text": point.payload.get("ocr_text", ""),
            "image_path": point.payload.get("image_path", ""),
            "department": point.payload.get("department", ""),
            "doc_type": point.payload.get("doc_type", ""),
            "sheet": point.payload.get("sheet", ""),
            "section": point.payload.get("section", ""),
        })

    logger.info("하이브리드 검색 완료: %d 결과", len(pages))
    return pages
