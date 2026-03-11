import base64
import io
import logging

from PIL import Image

from app.config import settings

logger = logging.getLogger(__name__)

# Qwen3-VL: 28×28 pixels = 1 token. 2048px 이미지 ≈ ~5,300 tokens
VLM_MAX_IMAGE_SIZE = 2048


def _resize_for_vlm(image: Image.Image) -> Image.Image:
    """VLM 전송용 이미지 리사이즈 (토큰 수 제한)."""
    w, h = image.size
    if max(w, h) <= VLM_MAX_IMAGE_SIZE:
        return image
    scale = VLM_MAX_IMAGE_SIZE / max(w, h)
    new_w, new_h = int(w * scale), int(h * scale)
    return image.resize((new_w, new_h), Image.LANCZOS)


def _image_to_base64(image: Image.Image, quality: int = 85) -> str:
    """PIL 이미지를 JPEG base64 문자열로 변환 (리사이즈 포함)."""
    image = _resize_for_vlm(image)
    buf = io.BytesIO()
    image.save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def describe_image_for_search(image: Image.Image) -> str:
    """이미지 파일 인제스트 시 VLM으로 검색용 설명 생성.

    OCR 텍스트에 "급식", "메뉴" 같은 컨텍스트 키워드가 빠지는 문제를 해결.
    생성된 설명을 OCR 텍스트 앞에 붙여서 검색/리랭킹 품질을 높인다.
    """
    from openai import OpenAI

    client = OpenAI(base_url=settings.vllm_base_url, api_key="dummy")

    img_b64 = _image_to_base64(image, quality=80)

    response = client.chat.completions.create(
        model=settings.vllm_model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                    },
                    {
                        "type": "text",
                        "text": (
                            "/no_think\n"
                            "이 이미지의 내용을 검색에 유용하도록 2~3문장으로 설명하세요.\n"
                            "- 문서 종류 (예: 급식 메뉴표, 조직도, 공지사항 등)\n"
                            "- 핵심 키워드 나열\n"
                            "- 날짜, 기간 등 시간 정보가 있으면 포함\n"
                            "설명만 출력하세요."
                        ),
                    },
                ],
            },
        ],
        max_tokens=200,
        temperature=0.1,
    )

    answer = response.choices[0].message.content.strip()
    if "</think>" in answer:
        answer = answer.split("</think>")[-1].strip()
    logger.info("이미지 설명 생성: %s", answer[:100])
    return answer


def generate_answer(
    question: str,
    page_images: list[Image.Image | None],
    ocr_texts: list[str],
    source_info: list[dict],
    max_tokens: int = 1536,
) -> dict:
    """페이지 이미지/텍스트를 Qwen3-VL에 전송하여 답변 생성.

    Args:
        question: 사용자 질문
        page_images: 관련 페이지 이미지 (None이면 텍스트 전용)
        ocr_texts: 각 페이지의 추출 텍스트
        source_info: 각 페이지의 출처 정보 [{"file_name", "page_number"}, ...]
        max_tokens: 최대 생성 토큰 수

    Returns:
        {"answer": str, "sources": list[dict]}
    """
    from openai import OpenAI

    client = OpenAI(base_url=settings.vllm_base_url, api_key="dummy")

    # 시스템 프롬프트
    system_prompt = (
        "/no_think\n"
        "당신은 사내 문서 기반 질의응답 AI 어시스턴트입니다.\n"
        "제공된 문서 페이지 이미지와 추출 텍스트를 분석하여 질문에 답변하세요.\n\n"
        "## 핵심 원칙\n"
        "- 문서에 명시된 내용만 근거로 답변하세요. 추론, 추측, 가정하지 마세요.\n"
        "- 문서에서 답을 찾을 수 없으면 '제공된 문서에서 해당 정보를 찾을 수 없습니다'라고만 답하세요.\n"
        "- 답변은 간결하고 직접적으로 하세요. 불필요한 서론, 반복, 헷징(~일 수 있습니다, ~가능할 수 있습니다) 없이 핵심만 전달하세요.\n"
        "- 질문과 직접 관련 없는 문서 내용은 언급하지 마세요.\n\n"
        "## 문서 분석 규칙\n"
        "- 이미지의 모든 텍스트를 빠짐없이 읽으세요. 제목, 헤더, 본문, 주석, 범례 등 위치와 크기에 관계없이 모든 텍스트가 중요합니다.\n"
        "- 표를 읽을 때 행(가로)과 열(세로) 헤더를 정확히 매핑하세요.\n"
        "- 특수 기호(●, ○, ■, △, ✓ 등)는 해당 문서의 범례를 참조하여 의미를 파악하세요.\n"
        "- 추출 텍스트가 제공된 경우 이미지와 교차 검증하여 누락 없이 답변하세요.\n\n"
        "## 답변 형식\n"
        "- 한국어로 답변하세요.\n"
        "- 답변 마지막에 출처를 표시하세요: [출처: 파일명, n페이지]\n"
    )

    # 사용자 메시지: 이미지/텍스트 + 출처 정보 + 질문
    content = []
    for i, (img, ocr_text, info) in enumerate(
        zip(page_images, ocr_texts, source_info)
    ):
        # 이미지가 있으면 포함 (텍스트 전용 문서는 None)
        if img is not None:
            img_b64 = _image_to_base64(img)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
            })
        # 페이지 식별 + 추출 텍스트 (이미지 유무 관계없이 항상 추가)
        # 텍스트 전용(이미지 없음)이면 더 많은 텍스트 허용 (이미지 토큰 절약분 활용)
        text_limit = 3000 if img is None else 1500
        page_label = f"[문서 {i + 1}: {info['file_name']} {info['page_number']}페이지]"
        if ocr_text and ocr_text.strip():
            page_label += f"\n--- 추출된 텍스트 ---\n{ocr_text[:text_limit]}\n---"
        content.append({
            "type": "text",
            "text": page_label,
        })

    content.append({
        "type": "text",
        "text": f"\n질문: {question}",
    })

    response = client.chat.completions.create(
        model=settings.vllm_model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": content},
        ],
        max_tokens=max_tokens,
        temperature=0.2,
    )

    answer = response.choices[0].message.content
    # thinking 토큰이 포함된 경우 제거
    if answer and "</think>" in answer:
        answer = answer.split("</think>")[-1].strip()
    logger.info("VLM 답변 생성 완료 (%d 토큰)", response.usage.completion_tokens)

    return {
        "answer": answer,
        "sources": source_info,
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
        },
    }
