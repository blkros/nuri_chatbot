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


def generate_answer(
    question: str,
    page_images: list[Image.Image],
    ocr_texts: list[str],
    source_info: list[dict],
    max_tokens: int = 2048,
) -> dict:
    """페이지 이미지들을 Qwen3-VL에 전송하여 답변 생성.

    Args:
        question: 사용자 질문
        page_images: 관련 페이지 이미지 (상위 K개)
        ocr_texts: 각 페이지의 OCR 텍스트
        source_info: 각 페이지의 출처 정보 [{"file_name", "page_number"}, ...]
        max_tokens: 최대 생성 토큰 수

    Returns:
        {"answer": str, "sources": list[dict]}
    """
    from openai import OpenAI

    client = OpenAI(base_url=settings.vllm_base_url, api_key="dummy")

    # 시스템 프롬프트
    system_prompt = (
        "당신은 사내 문서 기반 질의응답 AI 어시스턴트입니다.\n"
        "제공된 문서 페이지 이미지들을 분석하여 질문에 정확하게 답변하세요.\n\n"
        "규칙:\n"
        "1. 문서에 있는 내용만 근거로 답변하세요.\n"
        "2. 답변 마지막에 참조한 출처를 표시하세요 (예: [출처: 파일명, n페이지]).\n"
        "3. 문서에서 답을 찾을 수 없으면 솔직히 '제공된 문서에서 해당 정보를 찾을 수 없습니다'라고 답하세요.\n"
        "4. 한국어로 답변하세요.\n"
        "5. 표, 서식, 양식 문서를 분석할 때는 반드시 범례, 기호 설명, 머리글, "
        "바닥글, 주석을 먼저 확인하세요. 특수 기호(●, ○, ■, △, ✓ 등)가 "
        "사용된 경우 해당 문서 내의 범례나 설명을 참조하여 각 기호의 의미를 "
        "정확히 파악한 뒤 답변하세요.\n"
        "6. 표의 행과 열 헤더를 정확히 매핑하여 읽으세요. 특히 결재/승인 "
        "문서에서는 직급별 권한, 금액 구간별 결재자를 정확히 구분하세요.\n"
    )

    # 사용자 메시지: 이미지들 + 출처 정보 + 질문
    content = []
    for i, (img, ocr_text, info) in enumerate(
        zip(page_images, ocr_texts, source_info)
    ):
        img_b64 = _image_to_base64(img)
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
        })
        # 페이지 식별 + 추출 텍스트 (VLM에 텍스트 컨텍스트 제공)
        page_label = f"[문서 {i + 1}: {info['file_name']} {info['page_number']}페이지]"
        if ocr_text and ocr_text.strip():
            page_label += f"\n--- 추출된 텍스트 ---\n{ocr_text[:2000]}\n---"
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
    logger.info("VLM 답변 생성 완료 (%d 토큰)", response.usage.completion_tokens)

    return {
        "answer": answer,
        "sources": source_info,
        "usage": {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
        },
    }
