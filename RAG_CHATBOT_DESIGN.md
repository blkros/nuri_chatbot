# 사내 RAG Chatbot 설계 문서

> 작성일: 2026-03-03
> 서버: NVIDIA H100 PCIe 80GB (dl380gen11)
> 목적: 사내 문서 기반 질의응답 챗봇 구축

---

## 1. 프로젝트 개요

### 1.1 목표
- 사내 문서(HWP, PDF, 이미지, 스캔문서 등)를 기반으로 한 RAG 질의응답 챗봇
- 30~50명 규모 사내 사용자 동시 서비스
- 한국어 답변 품질 최우선

### 1.2 핵심 요구사항
| 요구사항 | 상세 |
|----------|------|
| **HWP 지원** | 한글 파일(.hwp) 다수 존재, LangChain 미지원 → LibreOffice 변환 필요 |
| **스캔/사진 문서** | 텍스트가 이미지인 문서 다수 → VLM(Vision LLM) 필수 |
| **OCR 품질** | Tesseract 한국어 성능 부족 → VLM 자체 OCR + PaddleOCR 보조 |
| **한국어** | 답변, 검색, 임베딩 모두 한국어 최적화 필요 |
| **메타데이터 필터링** | 문서량 증가 시 검색 품질 유지를 위한 필터링 필수 |
| **GPU 공존** | 동일 H100에서 PV 태양광 예측 파이프라인과 공존 |

### 1.3 설계 원칙
- **양자화 최소화**: 4-bit AWQ 대신 FP8(H100 네이티브) 사용하여 품질 손실 방지
- **GPU는 LLM 전용**: 임베딩/리랭커는 CPU로 분리하여 KV Cache 극대화
- **Vision RAG**: OCR 파이프라인 대신 문서 페이지 이미지를 VLM이 직접 이해

---

## 2. 서버 환경

### 2.1 하드웨어
```
호스트: jhko@dl380gen11
IP: 172.16.10.30
SSH Port: 7980
GPU: NVIDIA H100 PCIe 80GB × 1
```

### 2.2 기존 서비스 (PV 태양광 예측 파이프라인)
```
컨테이너: pv-jupyter (Docker)
API: uvicorn (port 5000) - /forecast 엔드포인트
모델: LightGBM (CPU only)
GPU 사용: 0GB (CUDA_VISIBLE_DEVICES=-1)
학습: cron 자동 + 수동 (CPU only)
```

> **PV 파이프라인은 GPU를 전혀 사용하지 않으므로 RAG 챗봇과 GPU 충돌 없음.**
> 다만 미래에 LightGBM GPU 학습 전환 시 +1~2GB, 딥러닝 전환 시 +4~8GB 추가 가능.
> 이 경우에도 vLLM의 `--gpu-memory-utilization` 조정으로 공존 가능.

### 2.3 PV 파이프라인 uvicorn 관리 명령어 (참고용)
```bash
# 종료
docker exec -u root pv-jupyter bash -lc 'pkill -f "uvicorn pv_api.main:app" || true'

# 시작
docker exec -d pv-jupyter bash -c 'cd /tf && env TZ=KST-9 CUDA_VISIBLE_DEVICES=-1 TF_CPP_MIN_LOG_LEVEL=3 \
  PV_API_DEBUG=1 PYTHONUNBUFFERED=1 \
  PYTHONPATH=/tf/pv_api \
  KMA_SERVICE_KEY="l0skHtQyz+jIza9ev/7u59ZRRMAJqX485lfn7EeNavtJlWOXwqnT3bTcCPTOxQpszWLJjKEfp7AujYY1Av6kVg==" \
  ASOS_SERVICE_KEY="l0skHtQyz+jIza9ev/7u59ZRRMAJqX485lfn7EeNavtJlWOXwqnT3bTcCPTOxQpszWLJjKEfp7AujYY1Av6kVg==" \
  DB_HOST="172.16.30.57" DB_PORT="14000" DB_USER="nuriai" \
  DB_PASSWORD="aeb29187ead19d008e17" DB_NAME="pulsar" \
  nohup uvicorn pv_api.main:app --host 0.0.0.0 --port 5000 \
  --log-level warning --no-access-log \
  >> /tf/pv_forecast/logs/uvicorn_cpu.log 2>&1'
```

---

## 3. 최종 기술 스택

### 3.1 GPU (vLLM 전용 - 64GB)

| 구성요소 | 모델 | HuggingFace ID | VRAM |
|----------|------|----------------|------|
| VLM (LLM+Vision) | Qwen3-VL-32B FP8 | `Qwen/Qwen3-VL-32B-Instruct-FP8` | 32.5GB |
| KV Cache | vLLM 자동 할당 | - | 31.5GB |
| **GPU 합계** | | | **64GB** |

- **Thinking 변형**: `Qwen/Qwen3-VL-32B-Thinking-FP8` (복잡한 문서 분석 시 깊은 추론)
- **설정**: `--gpu-memory-utilization 0.80`
- **동시 처리**: 8~10건 (30~50명 서비스 가능)

### 3.2 CPU / RAM

| 구성요소 | 모델/도구 | HuggingFace ID / 설치 | RAM |
|----------|----------|----------------------|-----|
| 문서 이미지 임베딩 | Nemotron ColEmbed V2 4B | `nvidia/nemotron-colembed-vl-4b-v2` | ~8GB |
| 텍스트 임베딩 | BGE-m3-ko (568M) | `upskyy/bge-m3-ko` | ~1.5GB |
| 리랭커 | bge-reranker-v2-m3 | `BAAI/bge-reranker-v2-m3` | ~1.5GB |
| OCR 보조 | PaddleOCR v5 한국어 | `pip install paddlepaddle paddleocr` | ~1GB |
| 벡터 DB | Qdrant | Docker: `qdrant/qdrant` | ~2GB+ |
| HWP 변환 | LibreOffice headless | `apt install libreoffice` | 미미 |
| 웹 UI | Open WebUI 또는 Gradio | - | 미미 |
| API 서버 | FastAPI | `pip install fastapi uvicorn` | 미미 |

### 3.3 VRAM 예산 요약
```
H100 80GB
├── vLLM (0.80)                64.0GB
│   ├── Qwen3-VL-32B FP8      32.5GB
│   └── KV Cache               31.5GB  → 동시 8~10건
├── OS / Driver / 버퍼         16.0GB
└── 임베딩/리랭커/OCR           CPU/RAM에서 실행 (GPU 미사용)
```

---

## 4. 모델 선정 근거

### 4.1 VLM: Qwen3-VL-32B FP8을 선택한 이유

**탈락 후보들:**

| 모델 | 탈락 사유 |
|------|----------|
| Qwen2.5-VL-72B AWQ 4-bit | 4-bit 양자화 → 한국어 품질 열화, 수치/표 오류 증가 |
| InternVL3-78B AWQ 4-bit | 동일한 4-bit 문제 + 공식 FP8 미제공 |
| InternVL3.5-38B | 공식 FP8 없음, vLLM 버전별 호환 버그 |
| Qwen2.5-VL-32B FP8 | 구세대, OCR/DocVQA 점수 열세 |
| Qwen3-VL-30B-A3B (MoE) | 활성 파라미터 3B뿐 → 32B Dense 대비 품질 열세 |
| GLM-4.5V | 106B MoE → 단일 H100에서 실용적 양자화 불가 |

**Qwen3-VL-32B 선택 근거:**

| 기준 | 점수/결과 |
|------|----------|
| OCRBench | 875 (32B급 1위) |
| DocVQA | 96.5% (32B급 1위) |
| 한국어 | Qwen 시리즈 고유 강점 |
| 양자화 | FP8 = H100 네이티브, 품질 손실 거의 없음 |
| HuggingFace | Qwen 공식 FP8 제공 |
| vLLM | 공식 지원, 안정적 |
| Thinking 모드 | 복잡 추론 시 전환 가능 |

### 4.2 Dense vs MoE 선택 기준

```
Dense (밀집형): 모든 파라미터가 매번 연산에 참여
  → 품질 안정적, VRAM = 연산 파라미터

MoE (전문가 혼합): 전문가 N명 중 소수만 활성화
  → 빠르지만, 30B 모델이라도 실제 일하는 건 3B뿐
  → 메모리는 30B 차지하면서 품질은 3B급

결론: 답변 품질 1순위 → Dense 선택
```

### 4.3 FP8 vs 4-bit AWQ

```
FP8:     소수점 8자리 → H100 하드웨어 네이티브 → 품질 손실 거의 없음
AWQ 4-bit: 소수점 4자리 → 한국어 조사/수치 오류 증가 → 사내 문서 RAG에 부적합

단일 H100 80GB에서 나사 안 빠진 최대 Dense VLM = 32B FP8 (스윗스팟)
72B는 4-bit 필수 → 크기 이점이 양자화 손실로 상쇄
```

### 4.4 문서 이미지 임베딩: Nemotron ColEmbed V2 4B

- ViDoRe V3 리더보드 1위 (2026.01 기준, nDCG@10 = 63.54)
- NVIDIA 공식 모델, Qwen3-VL-8B 기반
- ColQwen2.5 대비 검색 품질 우위
- CPU에서도 실용적 속도 (~300ms/페이지)

### 4.5 벡터 DB: Qdrant

- Rust 기반 고성능
- Named Vectors: 이미지 벡터 + 텍스트 벡터 동시 저장
- 메타데이터 필터링: 부서, 문서 유형, 날짜 등으로 검색 범위 축소
- 하이브리드 검색: Dense + Sparse 결합
- Docker 배포 간편

---

## 5. 아키텍처

### 5.1 전체 서빙 구조
```
[사용자 브라우저]
       │
[Web UI]  ← Open WebUI 또는 Gradio
       │
[FastAPI 서버]  ← CPU: 임베딩, 리랭커, OCR, 문서 변환
       │
       ├── 검색 ──→ [Qdrant] (Docker, CPU/RAM)
       │
       └── 답변 ──→ [vLLM] (GPU, port 8000) ──→ Qwen3-VL-32B FP8
```

### 5.2 문서 인제스트 파이프라인
```
문서 업로드 (HWP, PDF, 이미지, 스캔문서)
       │
       ├── HWP → LibreOffice headless → PDF
       │
       ▼
  PDF → 페이지별 이미지 추출 (pdf2image 등)
       │
       ├── Nemotron ColEmbed V2 → 이미지 벡터 (CPU)
       │     └→ Qdrant 저장 (image_vector)
       │
       ├── PaddleOCR → 텍스트 추출 (보조) (CPU)
       │     └→ BGE-m3-ko → 텍스트 벡터 (CPU)
       │           └→ Qdrant 저장 (text_vector)
       │
       └── 메타데이터 저장 (파일명, 부서, 날짜, 페이지 번호 등)
```

### 5.3 질의응답 파이프라인
```
사용자 질문
       │
  BGE-m3-ko → 쿼리 벡터 (CPU)
       │
  Qdrant 하이브리드 검색 (이미지+텍스트 벡터)
  + 메타데이터 필터 (부서, 문서유형 등)
       │
  bge-reranker → 상위 K개 재정렬 (CPU)
       │
  상위 문서 페이지 이미지 + 추출 텍스트
       │
  Qwen3-VL-32B → 답변 생성 (GPU, vLLM)
       │
  사용자에게 응답 (+ 출처 페이지 표시)
```

---

## 6. 서빙 설정

### 6.1 vLLM 시작 명령어
```bash
python -m vllm.entrypoints.openai.api_server \
  --model Qwen/Qwen3-VL-32B-Instruct-FP8 \
  --gpu-memory-utilization 0.80 \
  --max-model-len 8192 \
  --port 8000
```

### 6.2 주요 파라미터
| 파라미터 | 값 | 설명 |
|----------|-----|------|
| `--gpu-memory-utilization` | 0.80 | 80GB × 0.8 = 64GB (모델 32.5GB + KV Cache 31.5GB) |
| `--max-model-len` | 8192 | 최대 컨텍스트 길이 (KV Cache 절약) |
| `--port` | 8000 | OpenAI 호환 API 엔드포인트 |

### 6.3 vLLM API 호출 예시
```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="dummy")

response = client.chat.completions.create(
    model="Qwen/Qwen3-VL-32B-Instruct-FP8",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_b64}"}},
                {"type": "text", "text": "이 문서의 핵심 내용을 요약해주세요."}
            ]
        }
    ],
    max_tokens=2048
)
```

### 6.4 Qdrant 시작 명령어
```bash
docker run -d --name qdrant \
  -p 6333:6333 -p 6334:6334 \
  -v /path/to/qdrant/storage:/qdrant/storage \
  qdrant/qdrant
```

---

## 7. HuggingFace 다운로드 목록

```bash
# VLM (GPU - vLLM 서빙)
huggingface-cli download Qwen/Qwen3-VL-32B-Instruct-FP8

# 깊은 추론 변형 (선택)
huggingface-cli download Qwen/Qwen3-VL-32B-Thinking-FP8

# 문서 이미지 임베딩 (CPU)
huggingface-cli download nvidia/nemotron-colembed-vl-4b-v2

# 텍스트 임베딩 (CPU)
huggingface-cli download upskyy/bge-m3-ko

# 리랭커 (CPU)
huggingface-cli download BAAI/bge-reranker-v2-m3
```

---

## 8. 미래 확장성

### 8.1 사용자 증가 시
| 규모 | 대응 |
|------|------|
| 50명 이하 | 현재 설정 유지 (0.80, max-model-len 8192) |
| 50~100명 | `--max-model-len 4096`으로 줄여 동시 처리량 2배 확보 |
| 100명+ | GPU 증설 또는 vLLM 멀티 GPU 분산 |

### 8.2 PV 파이프라인 GPU 전환 시
| 상황 | VRAM 추가 | 대응 |
|------|----------|------|
| LightGBM GPU 학습 | +1~2GB | `--gpu-memory-utilization 0.78`로 미세 조정 |
| 딥러닝(LSTM 등) 전환 | +4~8GB | 학습/서빙 시간 분리 또는 vLLM 0.70으로 조정 |

### 8.3 모델 업그레이드 경로
```
현재:  Qwen3-VL-32B FP8 (32.5GB)
미래:  차세대 모델이 같은 크기에서 성능 향상 시 교체
       (vLLM 모델 경로만 변경하면 되므로 아키텍처 변경 불필요)
```

---

## 9. 리스크 및 주의사항

| 리스크 | 대응 |
|--------|------|
| vLLM 기본값 0.90으로 시작하면 다른 프로세스 OOM | **반드시** `--gpu-memory-utilization 0.80` 설정 |
| PV 파이프라인과 포트 충돌 | PV=5000, vLLM=8000, Qdrant=6333 (겹치지 않음) |
| HWP 변환 실패 | LibreOffice가 지원 못하는 HWP 버전 존재 가능 → PaddleOCR로 fallback |
| 긴 문서(100페이지+) 인제스트 느림 | 페이지 단위 비동기 처리, 큐 시스템 도입 |
| Nemotron ColEmbed CPU 속도 | 인제스트는 배치 처리라 느려도 무방, 검색 시에는 쿼리 임베딩만 하므로 빠름 |

---

## 10. 구현 순서 (권장)

```
Phase 1: 기반 인프라
  □ vLLM + Qwen3-VL-32B FP8 서빙 확인
  □ Qdrant Docker 구동
  □ FastAPI 서버 뼈대

Phase 2: 문서 인제스트
  □ LibreOffice HWP→PDF 변환
  □ PDF→페이지 이미지 추출
  □ Nemotron ColEmbed 이미지 임베딩
  □ BGE-m3-ko 텍스트 임베딩
  □ Qdrant 저장 (Named Vectors)

Phase 3: 검색 + 답변
  □ 하이브리드 검색 구현
  □ bge-reranker 리랭킹
  □ VLM 답변 생성 (이미지 + 텍스트 컨텍스트)
  □ 출처 페이지 표시

Phase 4: UI + 운영
  □ Web UI 연동
  □ 메타데이터 필터 UI
  □ 문서 업로드 기능
  □ 로깅 / 모니터링
```

---

## 11. 참고 링크

- [Qwen3-VL-32B-Instruct-FP8](https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct-FP8)
- [Qwen3-VL-32B-Thinking-FP8](https://huggingface.co/Qwen/Qwen3-VL-32B-Thinking-FP8)
- [Qwen3-VL GitHub](https://github.com/QwenLM/Qwen3-VL)
- [Qwen3-VL Technical Report](https://arxiv.org/abs/2511.21631)
- [Nemotron ColEmbed V2 4B](https://huggingface.co/nvidia/nemotron-colembed-vl-4b-v2)
- [BGE-m3-ko](https://huggingface.co/upskyy/bge-m3-ko)
- [bge-reranker-v2-m3](https://huggingface.co/BAAI/bge-reranker-v2-m3)
- [Qdrant 공식 문서](https://qdrant.tech/documentation/)
- [vLLM Qwen3-VL 가이드](https://docs.vllm.ai/projects/recipes/en/latest/Qwen/Qwen3-VL.html)
- [Open VLM Leaderboard](https://huggingface.co/spaces/opencompass/open_vlm_leaderboard)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
