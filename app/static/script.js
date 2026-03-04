const API_BASE = window.location.origin;

// DOM 요소
const landing = document.getElementById("landing");
const chatArea = document.getElementById("chat-area");
const chatMessages = document.getElementById("chat-messages");
const queryInput = document.getElementById("query-input");
const bottomInput = document.getElementById("bottom-query-input");
const fileInput = document.getElementById("file-input");
const fileBadge = document.getElementById("file-badge");
const fileNameSpan = document.getElementById("file-name");
const bottomFileBadge = document.getElementById("bottom-file-badge");
const bottomFileNameSpan = document.getElementById("bottom-file-name");

// 상태
let pendingFile = null;
let selectedDept = "";

// ── 카테고리 칩 ──

document.querySelectorAll(".chip-row").forEach((row) => {
  row.addEventListener("click", (e) => {
    const chip = e.target.closest(".chip");
    if (!chip) return;

    const dept = chip.dataset.dept;
    selectedDept = dept;

    // 모든 칩 행 동기화
    document.querySelectorAll(".chip").forEach((c) => {
      c.classList.toggle("active", c.dataset.dept === dept);
    });
  });
});

// ── 파일 첨부 ──

function openFilePicker() {
  fileInput.click();
}

fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  if (!file) return;
  pendingFile = file;

  // 현재 보이는 화면에 따라 뱃지 표시
  if (!chatArea.classList.contains("visible")) {
    fileNameSpan.textContent = file.name;
    fileBadge.classList.remove("hidden");
  } else {
    bottomFileNameSpan.textContent = file.name;
    bottomFileBadge.classList.remove("hidden");
  }
});

function clearFile() {
  pendingFile = null;
  fileInput.value = "";
  fileBadge.classList.add("hidden");
  bottomFileBadge.classList.add("hidden");
}

document.getElementById("file-remove").addEventListener("click", clearFile);
document.getElementById("bottom-file-remove").addEventListener("click", clearFile);

document.getElementById("attach-btn").addEventListener("click", openFilePicker);
document.getElementById("bottom-attach-btn").addEventListener("click", openFilePicker);

// ── 메시지 추가 헬퍼 ──

function addUserMessage(text) {
  const div = document.createElement("div");
  div.className = "msg msg-user";
  div.innerHTML = `<div class="bubble">${escapeHtml(text)}</div>`;
  chatMessages.appendChild(div);
  scrollToBottom();
}

function addAIMessage(html, sourcesHtml) {
  const div = document.createElement("div");
  div.className = "msg msg-ai";
  let inner = `<div class="bubble-wrap"><div class="bubble">${html}</div>`;
  if (sourcesHtml) {
    inner += `<div class="msg-sources">${sourcesHtml}</div>`;
  }
  inner += `</div>`;
  div.innerHTML = inner;
  chatMessages.appendChild(div);
  scrollToBottom();
}

function addLoadingIndicator() {
  const div = document.createElement("div");
  div.className = "msg msg-loading";
  div.id = "loading-msg";
  div.innerHTML = `<div class="bubble"><div class="typing-dots"><span></span><span></span><span></span></div><span>답변을 생성하고 있습니다...</span></div>`;
  chatMessages.appendChild(div);
  scrollToBottom();
}

function removeLoadingIndicator() {
  const el = document.getElementById("loading-msg");
  if (el) el.remove();
}

function scrollToBottom() {
  chatMessages.scrollTop = chatMessages.scrollHeight;
}

function escapeHtml(text) {
  const div = document.createElement("div");
  div.textContent = text;
  return div.innerHTML;
}

// ── 파일 업로드 (ingest) ──

async function uploadFile(file) {
  const formData = new FormData();
  formData.append("file", file);
  if (selectedDept) {
    formData.append("department", selectedDept);
  }

  const res = await fetch(`${API_BASE}/ingest`, {
    method: "POST",
    body: formData,
  });

  const data = await res.json();
  if (!res.ok || data.status === "error") {
    throw new Error(data.detail || "업로드 실패");
  }
  return data;
}

// ── 화면 전환 애니메이션 ──

let isTransitioning = false;

function transitionToChat() {
  return new Promise((resolve) => {
    if (chatArea.classList.contains("visible")) {
      resolve();
      return;
    }

    isTransitioning = true;

    // 1. 랜딩 위로 슬라이드 아웃
    landing.classList.add("slide-out");

    // 2. 채팅 영역 준비 (보이지만 투명)
    setTimeout(() => {
      chatArea.classList.add("visible");

      // 3. 다음 프레임에서 채팅 페이드인
      requestAnimationFrame(() => {
        chatArea.classList.add("show");
      });
    }, 300);

    // 4. 트랜지션 완료 후 랜딩 완전 제거
    setTimeout(() => {
      landing.classList.add("gone");
      isTransitioning = false;
      resolve();
    }, 700);
  });
}

// ── 검색/대화 실행 ──

async function doSearch(question) {
  const hasQuestion = question.trim().length > 0;
  const hasFile = pendingFile !== null;

  if (!hasQuestion && !hasFile) return;
  if (isTransitioning) return;

  // 랜딩 → 채팅 전환 (애니메이션)
  await transitionToChat();

  // 사용자 메시지 표시
  if (hasQuestion) {
    addUserMessage(question);
  }

  // 입력 초기화
  queryInput.value = "";
  bottomInput.value = "";

  try {
    // 1. 첨부 파일이 있으면 먼저 인제스트
    if (hasFile) {
      const file = pendingFile;
      clearFile();

      if (hasQuestion) {
        addLoadingIndicator();
      }

      showToast(`${file.name} 업로드 중...`, "info");
      const ingestResult = await uploadFile(file);
      showToast(`${ingestResult.file_name} 인덱싱 완료 (${ingestResult.pages}페이지)`, "success");

      // 질문 없이 파일만 업로드한 경우
      if (!hasQuestion) {
        addAIMessage(
          `<strong>${escapeHtml(ingestResult.file_name)}</strong> 문서가 등록되었습니다 (${ingestResult.pages}페이지).<br>이제 이 문서에 대해 질문할 수 있습니다.`,
          ""
        );
        bottomInput.focus();
        return;
      }
    } else {
      addLoadingIndicator();
    }

    // 2. 질문 검색
    const formData = new FormData();
    formData.append("question", question);
    if (selectedDept) {
      formData.append("department", selectedDept);
    }

    const res = await fetch(`${API_BASE}/ask`, {
      method: "POST",
      body: formData,
    });

    const data = await res.json();
    removeLoadingIndicator();

    if (data.answer) {
      addAIMessage(
        formatAnswer(data.answer),
        formatSources(data.sources || [])
      );
    } else {
      addAIMessage(escapeHtml(data.detail || "답변을 생성할 수 없습니다."), "");
    }
  } catch (err) {
    removeLoadingIndicator();
    if (err.message.includes("업로드")) {
      showToast(err.message, "error");
      if (!hasQuestion) return;
    }
    addAIMessage("서버 연결에 실패했습니다. 잠시 후 다시 시도해주세요.", "");
  }

  bottomInput.focus();
}

// ── 마크다운 기본 포맷 ──

function formatAnswer(text) {
  return text
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
    .replace(/\n/g, "<br>");
}

// ── 출처 표시 ──

function formatSources(srcList) {
  if (!srcList.length) return "";
  const tags = srcList
    .map((s) => `<span class="source-tag">${escapeHtml(s.file_name)} ${s.page_number}p</span>`)
    .join("");
  return `<div class="msg-sources-label">참조 문서</div>${tags}`;
}

// ── 토스트 알림 ──

function showToast(message, type = "info") {
  const container = document.getElementById("toast-container");
  const toast = document.createElement("div");
  toast.className = `toast ${type}`;
  toast.textContent = message;
  container.appendChild(toast);
  setTimeout(() => {
    toast.style.opacity = "0";
    toast.style.transition = "opacity 0.3s";
    setTimeout(() => toast.remove(), 300);
  }, 4000);
}

// ── 초기 화면으로 복귀 ──

function resetUI() {
  chatArea.classList.remove("show");
  setTimeout(() => {
    chatArea.classList.remove("visible");
    chatMessages.innerHTML = "";

    landing.classList.remove("gone", "slide-out");
    queryInput.value = "";
    clearFile();
    queryInput.focus();
  }, 300);
}

// ── 이벤트 바인딩 ──

// 랜딩 화면
queryInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") doSearch(queryInput.value);
});

document.getElementById("search-btn").addEventListener("click", () => {
  doSearch(queryInput.value);
});

// 하단 입력창
bottomInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") doSearch(bottomInput.value);
});

document.getElementById("bottom-send-btn").addEventListener("click", () => {
  doSearch(bottomInput.value);
});
