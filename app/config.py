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

    # Embedding dimensions
    nemotron_dim: int = 2560
    bge_m3_ko_dim: int = 1024

    # Adaptive top_k (리랭커 점수 기반 동적 결정)
    adaptive_min_k: int = 3
    adaptive_max_k: int = 8
    rerank_score_min: float = 0.3
    rerank_gap_threshold: float = 0.15

    model_config = {"env_prefix": "APP_"}


settings = Settings()
