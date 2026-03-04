const API_BASE = window.location.origin;

// DOM 요소
const landing = document.getElementById("landing");
const resultArea = document.getElementById("result-area");
const queryInput = document.getElementById("query-input");
const topQueryInput = document.getElementById("top-query-input");
const loading = document.getElementById("loading");
const answerBox = document.getElementById("answer-box");
const answerText = document.getElementById("answer-text");
const sources = document.getElementById("sources");

// 검색 실행
async function doSearch(question) {
  if (!question.trim()) return;

  // UI 전환: 랜딩 → 결과
  landing.classList.add("hidden");
  resultArea.classList.remove("hidden");
  topQueryInput.value = question;

  // 로딩 표시
  loading.classList.remove("hidden");
  answerBox.classList.add("hidden");

  try {
    const formData = new FormData();
    formData.append("question", question);

    const res = await fetch(`${API_BASE}/ask`, {
      method: "POST",
      body: formData,
    });

    const data = await res.json();

    if (data.answer) {
      answerText.innerHTML = formatAnswer(data.answer);
      sources.innerHTML = formatSources(data.sources || []);
      answerBox.classList.remove("hidden");
    } else {
      answerText.textContent = data.detail || "답변을 생성할 수 없습니다.";
      sources.innerHTML = "";
      answerBox.classList.remove("hidden");
    }
  } catch (err) {
    answerText.textContent = "서버 연결에 실패했습니다. 잠시 후 다시 시도해주세요.";
    sources.innerHTML = "";
    answerBox.classList.remove("hidden");
  } finally {
    loading.classList.add("hidden");
  }
}

// 마크다운 기본 포맷 (bold, 출처 태그)
function formatAnswer(text) {
  return text
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
    .replace(/\n/g, "<br>");
}

// 출처 표시
function formatSources(srcList) {
  if (!srcList.length) return "";
  const tags = srcList
    .map((s) => `<span class="source-tag">${s.file_name} ${s.page_number}p</span>`)
    .join("");
  return `<div style="color:#5a6a7a;font-size:0.8rem;margin-bottom:6px;">참조 문서</div>${tags}`;
}

// 초기 화면으로 복귀
function resetUI() {
  resultArea.classList.add("hidden");
  landing.classList.remove("hidden");
  queryInput.value = "";
  queryInput.focus();
}

// 이벤트 바인딩
queryInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") doSearch(queryInput.value);
});

document.getElementById("search-btn").addEventListener("click", () => {
  doSearch(queryInput.value);
});

topQueryInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") doSearch(topQueryInput.value);
});

document.getElementById("top-search-btn").addEventListener("click", () => {
  doSearch(topQueryInput.value);
});
