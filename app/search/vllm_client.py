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


def rewrite_query(question: str, history: list[dict] | None = None) -> str:
    """사용자 질문을 검색에 유리한 형태로 확장.

    짧거나 모호한 질문에 관련 키워드를 추가하여
    임베딩 검색과 리랭킹 품질을 높인다.
    멀티턴 대화 시 이전 대화 맥락을 반영하여 후속 질문을 자기 완결적으로 변환.
    텍스트 전용 호출이라 빠름 (~0.5초).
    """
    from openai import OpenAI

    client = OpenAI(base_url=settings.vllm_base_url, api_key="dummy")

    # 멀티턴 히스토리가 있으면 시스템 프롬프트에 맥락 해석 지시 추가
    if history:
        system_content = (
            "/no_think\n"
            "사내 문서 검색 쿼리 확장기.\n"
            "이전 대화 맥락이 주어집니다. 현재 질문이 후속 질문이면 "
            "이전 맥락을 반영하여 자기 완결적인 검색 쿼리로 변환하세요.\n"
            "예: 이전='위임전결 규정 알려줘', 현재='표로 정리해줘' → '위임전결 규정 표 정리'\n"
            "질문에 없는 내용을 추가하지 마세요. 1줄, 50자 이내."
        )
    else:
        system_content = (
            "/no_think\n"
            "사내 문서 검색 쿼리 확장기.\n"
            "질문을 검색에 유리하게 다듬되, 질문에 없는 내용을 절대 추가하지 마세요.\n"
            "동의어/유의어/공식 용어 추가만 허용. 1줄, 50자 이내."
        )

    messages = [{"role": "system", "content": system_content}]

    # few-shot 예시
    messages.extend([
        {"role": "user", "content": "결재 어떻게해?"},
        {"role": "assistant", "content": "결재 승인 절차 전결 위임전결 규정"},
        {"role": "user", "content": "급식 메뉴 알려줘"},
        {"role": "assistant", "content": "급식 메뉴 식단표 중식 석식"},
    ])

    # 멀티턴 히스토리가 있으면 이전 대화 추가 (최근 2턴)
    if history:
        recent = history[-4:]  # 최근 2턴 (user+ai 각 1쌍 = 4메시지)
        for msg in recent:
            role = "user" if msg["role"] == "user" else "assistant"
            # 검색 쿼리 확장용이므로 답변은 요약만
            content = msg["content"]
            if role == "assistant":
                content = content[:100]
            messages.append({"role": role, "content": content})

    messages.append({"role": "user", "content": question})

    response = client.chat.completions.create(
        model=settings.vllm_model,
        messages=messages,
        max_tokens=80,
        temperature=0.1,
    )

    rewritten = response.choices[0].message.content.strip()
    if "</think>" in rewritten:
        rewritten = rewritten.split("</think>")[-1].strip()
    # 빈 응답이나 비정상적으로 긴 경우 원본 사용
    if not rewritten or len(rewritten) > 200:
        return question
    logger.info("쿼리 리라이팅: '%s' → '%s'", question, rewritten)
    return rewritten


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


_SYSTEM_PROMPT = (
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
    "- 출처나 참조 문서를 답변에 포함하지 마세요. 시스템이 자동으로 처리합니다.\n"
)


def _build_messages(
    question: str,
    page_images: list[Image.Image | None],
    ocr_texts: list[str],
    source_info: list[dict],
    history: list[dict] | None = None,
) -> list[dict]:
    """VLM API 호출용 메시지 리스트 구성 (generate_answer / generate_answer_stream 공용)."""
    messages = [{"role": "system", "content": _SYSTEM_PROMPT}]

    # 멀티턴: 이전 대화를 user/assistant 메시지로 추가 (최근 3턴, 토큰 절약)
    if history:
        recent = history[-6:]  # 최근 3턴 (user+ai 쌍)
        for msg in recent:
            role = "user" if msg["role"] == "user" else "assistant"
            # 이전 답변은 500자로 제한 (토큰 절약)
            content = msg["content"][:500] if role == "assistant" else msg["content"]
            messages.append({"role": role, "content": content})

    # 현재 질문 + 문서 컨텍스트
    content = []
    for i, (img, ocr_text, info) in enumerate(
        zip(page_images, ocr_texts, source_info)
    ):
        if img is not None:
            img_b64 = _image_to_base64(img)
            content.append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
            })
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

    messages.append({"role": "user", "content": content})
    return messages


def generate_answer(
    question: str,
    page_images: list[Image.Image | None],
    ocr_texts: list[str],
    source_info: list[dict],
    max_tokens: int = 1536,
    history: list[dict] | None = None,
) -> dict:
    """페이지 이미지/텍스트를 Qwen3-VL에 전송하여 답변 생성.

    Args:
        question: 사용자 질문
        page_images: 관련 페이지 이미지 (None이면 텍스트 전용)
        ocr_texts: 각 페이지의 추출 텍스트
        source_info: 각 페이지의 출처 정보 [{"file_name", "page_number"}, ...]
        max_tokens: 최대 생성 토큰 수
        history: 이전 대화 히스토리 [{"role": "user"|"ai", "content": str}, ...]

    Returns:
        {"answer": str, "sources": list[dict]}
    """
    from openai import OpenAI

    client = OpenAI(base_url=settings.vllm_base_url, api_key="dummy")
    messages = _build_messages(question, page_images, ocr_texts, source_info, history)

    response = client.chat.completions.create(
        model=settings.vllm_model,
        messages=messages,
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


def generate_answer_stream(
    question: str,
    page_images: list[Image.Image | None],
    ocr_texts: list[str],
    source_info: list[dict],
    max_tokens: int = 1536,
    history: list[dict] | None = None,
):
    """SSE 스트리밍용 답변 생성 제너레이터.

    Yields:
        str: 토큰 텍스트 청크 (thinking 토큰 제거 후)
    """
    from openai import OpenAI

    client = OpenAI(base_url=settings.vllm_base_url, api_key="dummy")
    messages = _build_messages(question, page_images, ocr_texts, source_info, history)

    stream = client.chat.completions.create(
        model=settings.vllm_model,
        messages=messages,
        max_tokens=max_tokens,
        temperature=0.2,
        stream=True,
    )

    # /no_think 사용 시에도 간혹 <think>...</think>가 나올 수 있음
    # thinking 구간을 버퍼링하고 </think> 이후부터 yield
    in_think = False
    think_buffer = ""

    for chunk in stream:
        delta = chunk.choices[0].delta if chunk.choices else None
        if not delta or not delta.content:
            continue

        text = delta.content

        if in_think:
            think_buffer += text
            if "</think>" in think_buffer:
                # thinking 끝 → 이후 텍스트만 yield
                after = think_buffer.split("</think>", 1)[1]
                in_think = False
                think_buffer = ""
                if after:
                    yield after
            continue

        if "<think>" in text:
            # thinking 시작
            before, _, remainder = text.partition("<think>")
            if before:
                yield before
            in_think = True
            think_buffer = remainder
            if "</think>" in think_buffer:
                after = think_buffer.split("</think>", 1)[1]
                in_think = False
                think_buffer = ""
                if after:
                    yield after
            continue

        yield text

    logger.info("VLM 스트리밍 답변 생성 완료")
