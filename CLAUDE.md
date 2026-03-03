# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Internal Korean-language RAG chatbot for corporate document Q&A (~30-50 concurrent users). Supports HWP (Korean .hwp), PDF, images, and scanned documents. The project is in pre-implementation phase — see `RAG_CHATBOT_DESIGN.md` for the full technical design document (in Korean).

## Target Server

- **Host:** jhko@dl380gen11 (172.16.10.30, SSH port 7980)
- **GPU:** NVIDIA H100 PCIe 80GB × 1
- **Coexisting service:** PV solar forecast pipeline (CPU-only, port 5000) — no GPU conflict

## Architecture

```
[Browser] → [Web UI (Open WebUI / Gradio)]
                    │
              [FastAPI Server]  ← CPU: embedding, reranking, OCR, doc conversion
                    │
        ┌───────────┴───────────┐
   [Qdrant]                [vLLM :8000]
   (vector DB,             (GPU, Qwen3-VL-32B FP8)
    Docker :6333)
```

**Document ingestion:** Upload → HWP→PDF (LibreOffice) → PDF→page images (pdf2image) → dual-path embedding: Nemotron ColEmbed V2 for image vectors + PaddleOCR→BGE-m3-ko for text vectors → Qdrant (Named Vectors + metadata).

**Query pipeline:** Question → BGE-m3-ko query vector → Qdrant hybrid search (image+text) with metadata filters → bge-reranker reranking → top-K page images + text → Qwen3-VL-32B answer generation → response with source attribution.

## Technology Stack

| Component | Model / Tool | Runs on | Notes |
|-----------|-------------|---------|-------|
| VLM | Qwen3-VL-32B-Instruct-FP8 | GPU (32.5GB) | vLLM served, OpenAI-compatible API |
| Image embedding | Nemotron ColEmbed V2 4B | CPU (~8GB RAM) | ViDoRe V3 #1 |
| Text embedding | BGE-m3-ko (568M) | CPU (~1.5GB RAM) | Korean-optimized |
| Reranker | bge-reranker-v2-m3 | CPU (~1.5GB RAM) | |
| OCR | PaddleOCR v5 Korean | CPU (~1GB RAM) | Auxiliary to VLM |
| Vector DB | Qdrant | Docker, CPU (~2GB+ RAM) | Named Vectors, hybrid search |
| HWP conversion | LibreOffice headless | CPU | `apt install libreoffice` |
| API server | FastAPI + Uvicorn | CPU | |

## Key Design Decisions

- **FP8 over 4-bit AWQ:** 4-bit quantization causes Korean grammar particle and numerical errors — unacceptable for Korean document RAG. FP8 is H100-native with minimal quality loss.
- **Dense over MoE:** Quality over speed. MoE models (e.g., 30B with 3B active) sacrifice quality despite large parameter counts.
- **GPU exclusively for VLM:** All embeddings, reranking, and OCR run on CPU to maximize KV cache for concurrent request handling.
- **`--gpu-memory-utilization 0.80`:** Must not exceed this — prevents OOM when coexisting with other processes. Yields 64GB: 32.5GB model + 31.5GB KV cache.

## Infrastructure Commands

```bash
# vLLM server
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-VL-32B-Instruct-FP8 \
  --gpu-memory-utilization 0.80 \
  --max-model-len 8192 \
  --port 8000

# Qdrant
docker run -d --name qdrant \
  -p 6333:6333 -p 6334:6334 \
  -v /path/to/qdrant/storage:/qdrant/storage \
  qdrant/qdrant

# Download models
huggingface-cli download Qwen/Qwen3-VL-32B-Instruct-FP8
huggingface-cli download nvidia/nemotron-colembed-vl-4b-v2
huggingface-cli download upskyy/bge-m3-ko
huggingface-cli download BAAI/bge-reranker-v2-m3
```

## Port Allocation

| Service | Port |
|---------|------|
| PV forecast API (existing) | 5000 |
| vLLM (OpenAI-compatible) | 8000 |
| Qdrant REST | 6333 |
| Qdrant gRPC | 6334 |

## Implementation Phases

1. **Infrastructure:** vLLM + Qdrant + FastAPI skeleton
2. **Document ingestion:** LibreOffice HWP→PDF, pdf2image, dual embedding → Qdrant
3. **Search + answer:** Hybrid search, reranking, VLM answer generation with source pages
4. **UI + operations:** Web UI, metadata filter UI, upload, logging/monitoring

## Language

All user-facing content, documentation, and LLM prompts are in **Korean**. Code comments and variable names may use English.
