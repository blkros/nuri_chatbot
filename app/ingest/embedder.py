import logging

import torch
from PIL import Image

from app.config import settings

logger = logging.getLogger(__name__)

# Lazy-loaded model singletons
_nemotron_model = None
_bge_model = None

# Qdrant multivector 제한: 1,048,576 floats / 2560 dim = 409 tokens
MAX_TOKENS_PER_IMAGE = 400
EMBED_MAX_IMAGE_SIZE = 1280


def get_nemotron_model():
    """Nemotron ColEmbed V2 4B 모델 로드 (CPU, ColBERT multi-vector)."""
    global _nemotron_model
    if _nemotron_model is None:
        from transformers import AutoModel

        logger.info("Nemotron ColEmbed V2 로딩 중... (약 8GB RAM)")
        _nemotron_model = AutoModel.from_pretrained(
            settings.nemotron_model_path,
            trust_remote_code=True,
            torch_dtype=torch.float32,
        ).eval()
        logger.info("Nemotron ColEmbed V2 로드 완료")
    return _nemotron_model


def get_bge_model():
    """BGE-m3-ko 텍스트 임베딩 모델 로드 (CPU, 1024차원)."""
    global _bge_model
    if _bge_model is None:
        from sentence_transformers import SentenceTransformer

        logger.info("BGE-m3-ko 로딩 중... (약 1.5GB RAM)")
        _bge_model = SentenceTransformer(
            settings.bge_m3_ko_model_path, device="cpu"
        )
        logger.info("BGE-m3-ko 로드 완료")
    return _bge_model


def _resize_for_embedding(image: Image.Image) -> Image.Image:
    """임베딩용 이미지 리사이즈 (토큰 수 제한)."""
    w, h = image.size
    if max(w, h) <= EMBED_MAX_IMAGE_SIZE:
        return image
    scale = EMBED_MAX_IMAGE_SIZE / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    return image.resize((new_w, new_h), Image.LANCZOS)


def embed_images(images: list[Image.Image]) -> list[list[list[float]]]:
    """페이지 이미지들을 Nemotron ColEmbed multi-vector로 변환.

    Returns:
        각 이미지에 대해 [num_tokens x 2560] 형태의 벡터 리스트
    """
    model = get_nemotron_model()
    resized = [_resize_for_embedding(img) for img in images]

    with torch.no_grad():
        embeddings = model.forward_images(resized, batch_size=2)

    result = []
    for emb in embeddings:
        if isinstance(emb, torch.Tensor):
            if emb.shape[0] > MAX_TOKENS_PER_IMAGE:
                emb = emb[:MAX_TOKENS_PER_IMAGE]
            result.append(emb.cpu().tolist())
        else:
            result.append(emb[:MAX_TOKENS_PER_IMAGE])

    logger.info("이미지 임베딩 완료: %d 페이지 (토큰: %s)",
                len(result), [len(r) for r in result])
    return result


def embed_texts(texts: list[str]) -> list[list[float]]:
    """텍스트들을 BGE-m3-ko 1024차원 벡터로 변환."""
    model = get_bge_model()
    embeddings = model.encode(texts, normalize_embeddings=True)
    logger.info("텍스트 임베딩 완료: %d 건", len(texts))
    return embeddings.tolist()


def embed_query_text(query: str) -> list[float]:
    """단일 텍스트 쿼리를 BGE-m3-ko 1024차원 벡터로 변환."""
    return embed_texts([query])[0]


def embed_query_for_images(query: str) -> list[list[float]]:
    """텍스트 쿼리를 Nemotron ColEmbed multi-vector로 변환 (이미지 벡터 검색용)."""
    model = get_nemotron_model()
    with torch.no_grad():
        embeddings = model.forward_queries([query], batch_size=1)

    emb = embeddings[0]
    if isinstance(emb, torch.Tensor):
        return emb.cpu().tolist()
    return emb
