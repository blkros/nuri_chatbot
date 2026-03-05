import base64
import io
import json
import logging
import re

from PIL import Image

from app.config import settings

logger = logging.getLogger(__name__)


def classify_document_vlm(
    page_image: Image.Image | None,
    text_content: str,
    file_name: str,
) -> dict:
    """VLM을 사용하여 문서 메타데이터 자동 생성.

    첫 페이지 이미지 + 추출 텍스트를 VLM에 전송하여
    department, doc_type, summary를 자동 분류.
    page_image=None이면 키워드 기반 폴백 (Excel 등 텍스트 전용 문서).

    Returns:
        {"department": str, "doc_type": str, "summary": str}
    """
    if page_image is None:
        dept = _classify_by_keywords(file_name, text_content)
        return {"department": dept, "doc_type": "기타", "summary": ""}

    from openai import OpenAI

    client = OpenAI(base_url=settings.vllm_base_url, api_key="dummy")

    # 이미지를 base64로 변환 (분류용이므로 작게 리사이즈)
    img = page_image.copy()
    w, h = img.size
    if max(w, h) > 1024:
        scale = 1024 / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=80)
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    prompt = (
        f"다음 문서의 첫 페이지를 분석하여 메타데이터를 JSON으로 생성하세요.\n\n"
        f"파일명: {file_name}\n\n"
    )
    if text_content and text_content.strip():
        prompt += f"추출된 텍스트 (앞부분):\n{text_content[:1500]}\n\n"

    prompt += (
        "아래 JSON 형식으로만 답변하세요. 다른 텍스트 없이 JSON만 출력하세요.\n"
        "```json\n"
        "{\n"
        '  "department": "인사|재무|개발|영업|경영|기타 중 하나",\n'
        '  "doc_type": "규정|보고서|양식|계약서|회의록|매뉴얼|기타 중 하나",\n'
        '  "summary": "문서 내용 1줄 요약 (30자 이내)"\n'
        "}\n"
        "```"
    )

    try:
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
                        {"type": "text", "text": prompt},
                    ],
                },
            ],
            max_tokens=200,
            temperature=0.1,
        )

        answer = response.choices[0].message.content.strip()
        logger.info("VLM 분류 응답: %s", answer)

        # JSON 파싱 (코드블록 안에 있을 수 있음)
        json_match = re.search(r"\{[^}]+\}", answer, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
        else:
            result = json.loads(answer)

        # 필드 검증
        valid_departments = {"인사", "재무", "개발", "영업", "경영", "기타"}
        valid_doc_types = {"규정", "보고서", "양식", "계약서", "회의록", "매뉴얼", "기타"}

        if result.get("department") not in valid_departments:
            result["department"] = "기타"
        if result.get("doc_type") not in valid_doc_types:
            result["doc_type"] = "기타"
        if "summary" not in result:
            result["summary"] = ""

        logger.info(
            "VLM 문서 분류: %s → %s / %s",
            file_name, result["department"], result["doc_type"],
        )
        return result

    except Exception as e:
        logger.warning("VLM 분류 실패, 키워드 폴백 사용: %s", e)
        # 폴백: 키워드 기반 분류
        dept = _classify_by_keywords(file_name, text_content)
        return {"department": dept, "doc_type": "기타", "summary": ""}


def _classify_by_keywords(file_name: str, text_content: str = "") -> str:
    """VLM 실패 시 키워드 기반 폴백 분류."""
    CATEGORY_KEYWORDS = {
        "인사": ["인사", "급여", "채용", "퇴직", "복리후생", "근태", "연봉", "승진", "휴가", "취업규칙"],
        "재무": ["재무", "회계", "결산", "예산", "세무", "자금", "원가", "결제", "전결", "위임전결"],
        "개발": ["개발", "R&D", "연구", "기술", "소프트웨어", "시스템", "프로젝트", "설계", "특허"],
        "영업": ["영업", "매출", "고객", "판매", "계약", "견적", "수주", "거래처", "납품", "마케팅"],
        "경영": ["경영", "전략", "기획", "이사회", "정관", "조직", "규정", "내규", "지침", "사업계획"],
    }

    scores: dict[str, float] = {cat: 0.0 for cat in CATEGORY_KEYWORDS}
    name_clean = re.sub(r"\.[^.]+$", "", file_name).lower()

    for category, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in name_clean:
                scores[category] += 3.0

    text_lower = text_content[:3000].lower() if text_content else ""
    if text_lower:
        for category, keywords in CATEGORY_KEYWORDS.items():
            for kw in keywords:
                count = text_lower.count(kw.lower())
                if count > 0:
                    scores[category] += min(count, 5)

    best_cat = max(scores, key=scores.get)
    return best_cat if scores[best_cat] >= 1.0 else "기타"
