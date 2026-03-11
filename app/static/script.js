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
const sidebar = document.getElementById("sidebar");
const historyList = document.getElementById("history-list");

// 상태
let pendingFile = null;
let currentConversationId = null;

// ── 대화 이력 관리 (localStorage) ──

function loadConversations() {
  try {
    return JSON.parse(localStorage.getItem("nuri_conversations") || "[]");
  } catch { return []; }
}

function saveConversations(convs) {
  localStorage.setItem("nuri_conversations", JSON.stringify(convs));
}

function createConversation(title) {
  const conv = {
    id: Date.now().toString(),
    title: title.slice(0, 60),
    messages: [],  // [{role: "user"|"ai", content: string}]
    createdAt: Date.now(),
  };
  const convs = loadConversations();
  convs.unshift(conv);
  // 최대 50개 보관
  if (convs.length > 50) convs.length = 50;
  saveConversations(convs);
  currentConversationId = conv.id;
  return conv;
}

function appendMessage(role, content) {
  if (!currentConversationId) return;
  const convs = loadConversations();
  const conv = convs.find(c => c.id === currentConversationId);
  if (conv) {
    conv.messages.push({ role, content });
    saveConversations(convs);
  }
}

function deleteConversation(id) {
  let convs = loadConversations();
  convs = convs.filter(c => c.id !== id);
  saveConversations(convs);
  if (currentConversationId === id) {
    currentConversationId = null;
  }
  renderHistoryList();
}

function renderHistoryList() {
  const convs = loadConversations();
  if (convs.length === 0) {
    historyList.innerHTML = '<div class="history-empty">대화 이력이 없습니다</div>';
    return;
  }
  historyList.innerHTML = "";
  convs.forEach(conv => {
    const item = document.createElement("div");
    item.className = "history-item" + (conv.id === currentConversationId ? " active" : "");
    item.textContent = conv.title;
    item.addEventListener("click", () => loadConversation(conv.id));

    const delBtn = document.createElement("button");
    delBtn.className = "history-delete";
    delBtn.innerHTML = "&times;";
    delBtn.title = "삭제";
    delBtn.addEventListener("click", (e) => {
      e.stopPropagation();
      deleteConversation(conv.id);
    });
    item.appendChild(delBtn);

    historyList.appendChild(item);
  });
}

function loadConversation(id) {
  const convs = loadConversations();
  const conv = convs.find(c => c.id === id);
  if (!conv) return;

  currentConversationId = id;
  chatMessages.innerHTML = "";

  // 메시지 복원
  conv.messages.forEach(msg => {
    if (msg.role === "user") {
      addUserMessage(msg.content, false);
    } else {
      addAIMessage(formatAnswer(msg.content), false);
    }
  });

  renderHistoryList();

  // 이미 채팅 모드라면 스크롤만
  if (chatArea.classList.contains("visible")) {
    scrollToBottom();
    bottomInput.focus();
  } else {
    // 랜딩에서 전환
    transitionToChat().then(() => bottomInput.focus());
  }
}

// ── 사이드바 토글 ──

let sidebarOpen = false;

function toggleSidebar() {
  sidebarOpen = !sidebarOpen;
  sidebar.classList.toggle("open", sidebarOpen);
}

function openSidebar() {
  if (!sidebarOpen) {
    sidebarOpen = true;
    sidebar.classList.add("open");
  }
}

document.getElementById("sidebar-toggle").addEventListener("click", toggleSidebar);
document.getElementById("new-chat-btn").addEventListener("click", () => {
  currentConversationId = null;
  chatMessages.innerHTML = "";
  renderHistoryList();
  bottomInput.focus();
});

// ── 파일 첨부 ──

function openFilePicker() {
  fileInput.click();
}

fileInput.addEventListener("change", () => {
  const file = fileInput.files[0];
  if (!file) return;
  pendingFile = file;

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

function addUserMessage(text, save = true) {
  const div = document.createElement("div");
  div.className = "msg msg-user";
  div.innerHTML = `<div class="bubble">${escapeHtml(text)}</div>`;
  chatMessages.appendChild(div);
  scrollToBottom();
  if (save) appendMessage("user", text);
}

const AI_AVATAR = '<div class="ai-avatar">N</div>';

function addAIMessage(html, save = true) {
  const div = document.createElement("div");
  div.className = "msg msg-ai";
  div.innerHTML = `${AI_AVATAR}<div class="bubble-wrap"><div class="bubble">${html}</div></div>`;
  chatMessages.appendChild(div);
  scrollToBottom();
}

function addStreamingAIMessage() {
  const div = document.createElement("div");
  div.className = "msg msg-ai";
  div.id = "streaming-msg";
  div.innerHTML = `<div class="ai-avatar thinking" id="streaming-avatar">N</div><div class="bubble-wrap"><div class="bubble" id="streaming-bubble"></div></div>`;
  chatMessages.appendChild(div);
  scrollToBottom();
  return document.getElementById("streaming-bubble");
}

function finalizeStreamingMessage(fullText) {
  const bubble = document.getElementById("streaming-bubble");
  if (!bubble) return;
  bubble.innerHTML = formatAnswer(fullText);
  bubble.removeAttribute("id");
  // 아바타 펄스 중지
  const avatar = document.getElementById("streaming-avatar");
  if (avatar) {
    avatar.classList.remove("thinking");
    avatar.removeAttribute("id");
  }
  const msg = document.getElementById("streaming-msg");
  if (msg) msg.removeAttribute("id");
  scrollToBottom();

  // 대화 이력에 저장
  appendMessage("ai", fullText);
  renderHistoryList();
}

function addLoadingIndicator() {
  const div = document.createElement("div");
  div.className = "msg msg-loading";
  div.id = "loading-msg";
  div.innerHTML = `<div class="ai-avatar thinking">N</div><div class="bubble"><div class="typing-dots"><span></span><span></span><span></span></div><span>답변을 생성하고 있습니다...</span></div>`;
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

    landing.classList.add("slide-out");

    setTimeout(() => {
      chatArea.classList.add("visible");

      requestAnimationFrame(() => {
        chatArea.classList.add("show");
        // 사이드바 슬라이드 인 (약간 딜레이)
        setTimeout(() => openSidebar(), 200);
      });
    }, 300);

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

  // 새 대화 시작 (현재 대화가 없으면)
  if (!currentConversationId && hasQuestion) {
    createConversation(question.trim());
    renderHistoryList();
  }

  await transitionToChat();

  if (hasQuestion) {
    addUserMessage(question);
  }

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
      const deptLabel = ingestResult.department || "";
      const docTypeLabel = ingestResult.doc_type || "";
      const summaryText = ingestResult.summary || "";
      showToast(`${ingestResult.file_name} 인덱싱 완료 (${ingestResult.pages}페이지)`, "success");

      if (!hasQuestion) {
        // 파일만 업로드 — 대화 시작
        if (!currentConversationId) {
          createConversation(`📎 ${ingestResult.file_name}`);
          renderHistoryList();
        }

        let msgHtml = `<strong>${escapeHtml(ingestResult.file_name)}</strong> 문서가 등록되었습니다 (${ingestResult.pages}페이지).`;
        if (deptLabel || docTypeLabel) {
          msgHtml += `<br>분류: ${deptLabel ? `[${escapeHtml(deptLabel)}]` : ""} ${docTypeLabel ? escapeHtml(docTypeLabel) : ""}`;
        }
        if (summaryText) {
          msgHtml += `<br>요약: ${escapeHtml(summaryText)}`;
        }
        msgHtml += `<br>이제 이 문서에 대해 질문할 수 있습니다.`;
        addAIMessage(msgHtml);
        appendMessage("ai", `${ingestResult.file_name} 문서가 등록되었습니다 (${ingestResult.pages}페이지).`);
        renderHistoryList();
        bottomInput.focus();
        return;
      }
    } else {
      addLoadingIndicator();
    }

    // 2. 질문 → SSE 스트리밍
    const formData = new FormData();
    formData.append("question", question);

    removeLoadingIndicator();
    const bubble = addStreamingAIMessage();
    let fullText = "";

    try {
      const res = await fetch(`${API_BASE}/ask/stream`, {
        method: "POST",
        body: formData,
      });

      if (!res.ok) {
        const errData = await res.json();
        throw new Error(errData.detail || "서버 오류");
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop();

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const payload = line.slice(6);
          if (payload === "[DONE]") break;

          try {
            const msg = JSON.parse(payload);
            if (msg.token) {
              fullText += msg.token;
              bubble.textContent = fullText;
              let cursor = bubble.querySelector(".streaming-cursor");
              if (!cursor) {
                cursor = document.createElement("span");
                cursor.className = "streaming-cursor";
              }
              bubble.appendChild(cursor);
              scrollToBottom();
            } else if (msg.answer) {
              // 비스트리밍 폴백 (결과 없음 등)
              fullText = msg.answer;
              bubble.textContent = fullText;
            } else if (msg.error) {
              fullText = "답변 생성 중 오류가 발생했습니다.";
              bubble.textContent = fullText;
            }
            // msg.sources는 무시 (출처 표시 안 함)
          } catch (e) { /* JSON 파싱 실패 무시 */ }
        }
      }

      finalizeStreamingMessage(fullText);
    } catch (streamErr) {
      fullText = fullText || "서버 연결에 실패했습니다.";
      bubble.textContent = fullText;
      finalizeStreamingMessage(fullText);
    }
  } catch (err) {
    removeLoadingIndicator();
    if (err.message.includes("업로드")) {
      showToast(err.message, "error");
      if (!hasQuestion) return;
    }
    addAIMessage("서버 연결에 실패했습니다. 잠시 후 다시 시도해주세요.");
  }

  bottomInput.focus();
}

// ── 마크다운 기본 포맷 ──

function formatAnswer(text) {
  // VLM이 출처를 생성하더라도 제거
  text = text.replace(/\[출처:.*?\]/g, "").replace(/\n*참조\s*문서[：:].*$/gm, "").trim();
  return escapeHtml(text)
    .replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>")
    .replace(/\n/g, "<br>");
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
  sidebarOpen = false;
  sidebar.classList.remove("open");
  currentConversationId = null;

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

queryInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") doSearch(queryInput.value);
});

document.getElementById("search-btn").addEventListener("click", () => {
  doSearch(queryInput.value);
});

bottomInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter") doSearch(bottomInput.value);
});

document.getElementById("bottom-send-btn").addEventListener("click", () => {
  doSearch(bottomInput.value);
});

// ── 초기화 ──
renderHistoryList();
