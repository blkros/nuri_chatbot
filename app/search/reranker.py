import logging

import torch

from app.config import settings

logger = logging.getLogger(__name__)

_reranker_model = None
_reranker_tokenizer = None


def get_reranker():
    """bge-reranker-v2-m3 모델 + 토크나이저 로드 (CPU)."""
    global _reranker_model, _reranker_tokenizer
    if _reranker_model is None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        logger.info("bge-reranker-v2-m3 로딩 중... (약 1.5GB RAM)")
        _reranker_tokenizer = AutoTokenizer.from_pretrained(
            settings.reranker_model_path
        )
        _reranker_model = AutoModelForSequenceClassification.from_pretrained(
            settings.reranker_model_path
        ).eval()
        logger.info("bge-reranker-v2-m3 로드 완료")
    return _reranker_model, _reranker_tokenizer


def rerank(
    query: str, passages: list[str], top_k: int = 5
) -> list[tuple[int, float]]:
    """쿼리-패시지 쌍을 리랭킹하여 점수순 정렬.

    Returns:
        [(원본_인덱스, 점수), ...] 점수 내림차순, 최대 top_k개
    """
    if not passages:
        return []

    model, tokenizer = get_reranker()
    pairs = [[query, passage] for passage in passages]

    with torch.no_grad():
        inputs = tokenizer(
            pairs,
            padding=True,
            truncation=True,
            return_tensors="pt",
            max_length=512,
        )
        scores = model(**inputs, return_dict=True).logits.view(-1).float()
        scores = torch.sigmoid(scores).tolist()

    ranked = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)
    logger.info(
        "리랭킹 완료: %d개 중 상위 %d개 (최고: %.4f)",
        len(passages),
        min(top_k, len(passages)),
        ranked[0][1] if ranked else 0,
    )
    return ranked[:top_k]
