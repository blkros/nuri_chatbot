import logging
import re

logger = logging.getLogger(__name__)

# 카테고리별 키워드 (우선순위 높은 것부터)
CATEGORY_KEYWORDS: dict[str, list[str]] = {
    "인사": [
        "인사", "급여", "채용", "퇴직", "복리후생", "근태", "인력",
        "연봉", "승진", "평가", "교육훈련", "직원", "인재", "휴가",
        "취업규칙", "복무", "상벌", "징계",
    ],
    "재무": [
        "재무", "회계", "결산", "예산", "세무", "자금", "원가",
        "매입", "매출전표", "손익", "재무제표", "감사", "세금",
        "결제", "전결", "위임전결",
    ],
    "개발": [
        "개발", "R&D", "연구", "기술", "소프트웨어", "시스템",
        "프로젝트", "설계", "테스트", "특허", "기술개발", "연구소",
    ],
    "영업": [
        "영업", "매출", "고객", "판매", "계약", "견적", "수주",
        "거래처", "납품", "마케팅", "제안", "입찰",
    ],
    "경영": [
        "경영", "전략", "기획", "이사회", "정관", "조직",
        "경영계획", "사업계획", "규정", "내규", "지침",
    ],
}


def classify_document(file_name: str, text_content: str = "") -> str:
    """파일명과 텍스트 내용을 기반으로 문서 카테고리를 자동 분류.

    Args:
        file_name: 업로드된 파일명
        text_content: 추출된 텍스트 (첫 페이지 또는 전체, 최대 3000자 권장)

    Returns:
        카테고리 문자열 ("인사", "재무", "개발", "영업", "경영", "기타")
    """
    scores: dict[str, float] = {cat: 0.0 for cat in CATEGORY_KEYWORDS}

    # 파일명에서 확장자 제거
    name_clean = re.sub(r"\.[^.]+$", "", file_name).lower()

    # 파일명 키워드 매칭 (가중치 3배 — 파일명이 더 의도적)
    for category, keywords in CATEGORY_KEYWORDS.items():
        for kw in keywords:
            if kw.lower() in name_clean:
                scores[category] += 3.0

    # 텍스트 내용 키워드 매칭
    text_lower = text_content[:3000].lower() if text_content else ""
    if text_lower:
        for category, keywords in CATEGORY_KEYWORDS.items():
            for kw in keywords:
                count = text_lower.count(kw.lower())
                if count > 0:
                    scores[category] += min(count, 5)  # 최대 5점 캡

    best_cat = max(scores, key=scores.get)
    best_score = scores[best_cat]

    if best_score < 1.0:
        result = "기타"
    else:
        result = best_cat

    logger.info(
        "문서 자동 분류: %s → %s (점수: %s)",
        file_name, result,
        {k: v for k, v in scores.items() if v > 0},
    )
    return result
