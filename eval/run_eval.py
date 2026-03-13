"""RAG 평가 스크립트 — /ask API 호출 → 정답 포함 여부 자동 체크."""

import json
import sys
import time
from pathlib import Path

import requests

API_BASE = "http://172.16.10.30:3857"
TEST_CASES_PATH = Path(__file__).parent / "test_cases.json"


def load_test_cases() -> list[dict]:
    with open(TEST_CASES_PATH, encoding="utf-8") as f:
        return json.load(f)


def ask_question(question: str) -> str:
    """POST /ask 호출 → answer 텍스트 반환."""
    resp = requests.post(
        f"{API_BASE}/ask",
        data={"question": question},
        timeout=120,
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("answer", "")


def check_answer(answer: str, expected: list[str]) -> bool:
    """expected 키워드가 모두 answer에 포함되면 PASS."""
    answer_lower = answer.lower().replace(",", "").replace(" ", "")
    for keyword in expected:
        keyword_clean = keyword.lower().replace(",", "").replace(" ", "")
        if keyword_clean not in answer_lower:
            return False
    return True


def run_eval():
    cases = load_test_cases()
    results = []
    pass_count = 0
    fail_count = 0

    print(f"\n{'='*70}")
    print(f"  RAG 평가 시작 — {len(cases)}문항")
    print(f"{'='*70}\n")

    for case in cases:
        qid = case["id"]
        doc = case["doc"]
        question = case["question"]
        expected = case["expected"]

        print(f"[{qid:02d}/{len(cases):02d}] {doc} | {question[:50]}...")

        try:
            start = time.time()
            answer = ask_question(question)
            elapsed = time.time() - start
            passed = check_answer(answer, expected)
        except Exception as e:
            answer = f"ERROR: {e}"
            elapsed = 0
            passed = False

        status = "✅ PASS" if passed else "❌ FAIL"
        if passed:
            pass_count += 1
        else:
            fail_count += 1

        print(f"       {status}  ({elapsed:.1f}s)")
        if not passed:
            print(f"       기대: {expected}")
            print(f"       답변: {answer[:200]}")
        print()

        results.append({
            "id": qid,
            "doc": doc,
            "question": question,
            "expected": expected,
            "answer": answer[:500],
            "passed": passed,
            "elapsed_sec": round(elapsed, 1),
        })

    # 결과 요약
    total = len(cases)
    print(f"{'='*70}")
    print(f"  결과: {pass_count}/{total} PASS ({pass_count/total*100:.0f}%)")
    print(f"  PASS: {pass_count}  |  FAIL: {fail_count}")
    print(f"{'='*70}")

    # 문서별 요약
    docs = sorted(set(c["doc"] for c in cases))
    for doc in docs:
        doc_results = [r for r in results if r["doc"] == doc]
        doc_pass = sum(1 for r in doc_results if r["passed"])
        print(f"  {doc}: {doc_pass}/{len(doc_results)}")

    # 실패 문항 목록
    failed = [r for r in results if not r["passed"]]
    if failed:
        print(f"\n  실패 문항:")
        for r in failed:
            print(f"    [{r['id']:02d}] {r['question'][:60]}")

    # 결과 JSON 저장
    output_path = Path(__file__).parent / "eval_results.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  상세 결과 저장: {output_path}")

    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(run_eval())
