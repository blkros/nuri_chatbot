from pathlib import Path

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Model paths (mounted from /ddrive/models)
    nemotron_model_path: str = "/models/nemotron-colembed-vl-4b-v2"
    bge_m3_ko_model_path: str = "/models/bge-m3-korean"
    reranker_model_path: str = "/models/bge-reranker-v2-m3"

    # Qdrant
    qdrant_host: str = "qdrant"
    qdrant_port: int = 6333
    collection_name: str = "documents"

    # vLLM
    vllm_base_url: str = "http://vllm:8000/v1"
    vllm_model: str = "/models/Qwen3-VL-32B-Instruct-FP8"

    # Document processing
    upload_dir: str = "/tmp/uploads"
    page_images_dir: str = "/data/page_images"
    max_upload_size_mb: int = 100
    max_pages_per_document: int = 200  # 페이지 수 상한 (임베딩 비용 폭발 방지)

    # Embedding dimensions
    nemotron_dim: int = 2560
    bge_m3_ko_dim: int = 1024

    # Adaptive top_k (리랭커 점수 기반 동적 결정)
    adaptive_min_k: int = 3
    adaptive_max_k: int = 8
    rerank_score_min: float = 0.3
    rerank_gap_threshold: float = 0.15
    text_sufficient_length: int = 150  # 이 글자수 이상이면 텍스트만으로 충분 (이미지 스킵)
    max_context_images: int = 5  # VLM에 전달 가능한 최대 이미지 수 (토큰 예산)
    max_context_pages: int = 15  # VLM 컨텍스트 최대 페이지 수 (max_model_len 예산)
    doc_concentration_threshold: float = 0.5  # 검색 결과의 50%+ 같은 문서면 문서 확장
    max_doc_expansion_pages: int = 12  # 문서 확장 시 최대 페이지 수

    # Noise filter (absolute floor + gap detection)
    noise_score_floor: float = 0.02  # rerank 절대 최소 점수 (이하 제거)
    noise_gap_ratio: float = 2.0  # 평균 갭의 N배 이상이면 자연 끊김으로 판단

    # RRF (Reciprocal Rank Fusion)
    rrf_k: int = 60  # RRF smoothing 파라미터
    rrf_search_weight: float = 1.0  # 검색 순위 가중치
    rrf_rerank_weight: float = 1.5  # 리랭크 순위 가중치

    model_config = {"env_prefix": "APP_"}


settings = Settings()
