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
    image_vectors: list[list[float]],
    text_vector: list[float],
    ocr_text: str,
    metadata: dict | None = None,
):
    """단일 페이지를 Qdrant에 저장."""
    client = get_client()
    point_id = str(uuid4())

    payload = {
        "file_name": file_name,
        "page_number": page_number,
        "ocr_text": ocr_text,
    }
    if metadata:
        payload.update(metadata)

    client.upsert(
        collection_name=settings.collection_name,
        points=[
            models.PointStruct(
                id=point_id,
                vector={
                    "image_vector": image_vectors,
                    "text_vector": text_vector,
                },
                payload=payload,
            )
        ],
    )
    logger.debug("페이지 저장: %s p.%d (id=%s)", file_name, page_number, point_id)
    return point_id
