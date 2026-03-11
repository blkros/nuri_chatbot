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
let isProcessing = false;  // 중복 전송 방지

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

// ── 입력 잠금/해제 (중복 전송 방지) ──

function lockInput() {
  isProcessing = true;
  queryInput.disabled = true;
  bottomInput.disabled = true;
  document.querySelectorAll(".search-btn").forEach(btn => btn.disabled = true);
}

function unlockInput() {
  isProcessing = false;
  queryInput.disabled = false;
  bottomInput.disabled = false;
  document.querySelectorAll(".search-btn").forEach(btn => btn.disabled = false);
}

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
  let inner = `${AI_AVATAR}<div class="bubble-wrap"><div class="bubble">${html}</div>`;
  inner += `<button class="copy-btn" title="복사">복사</button>`;
  inner += `</div>`;
  div.innerHTML = inner;
  // 복사 버튼 이벤트
  div.querySelector(".copy-btn").addEventListener("click", function() {
    const bubble = div.querySelector(".bubble");
    navigator.clipboard.writeText(bubble.innerText).then(() => {
      this.textContent = "복사됨";
      setTimeout(() => { this.textContent = "복사"; }, 1500);
    });
  });
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

  // 복사 버튼 추가
  const wrap = bubble.closest(".bubble-wrap");
  if (wrap) {
    const copyBtn = document.createElement("button");
    copyBtn.className = "copy-btn";
    copyBtn.title = "복사";
    copyBtn.textContent = "복사";
    copyBtn.addEventListener("click", function() {
      navigator.clipboard.writeText(bubble.innerText).then(() => {
        copyBtn.textContent = "복사됨";
        setTimeout(() => { copyBtn.textContent = "복사"; }, 1500);
      });
    });
    wrap.appendChild(copyBtn);
  }

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

function addLoadingIndicator(text = "문서를 검색하고 있습니다...") {
  removeLoadingIndicator();
  const div = document.createElement("div");
  div.className = "msg msg-loading";
  div.id = "loading-msg";
  div.innerHTML = `<div class="ai-avatar thinking">N</div><div class="bubble"><div class="typing-dots"><span></span><span></span><span></span></div><span id="loading-text">${escapeHtml(text)}</span></div>`;
  chatMessages.appendChild(div);
  scrollToBottom();
}

function updateLoadingText(text) {
  const el = document.getElementById("loading-text");
  if (el) el.textContent = text;
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

  if (!hasQuestion && !hasFile) {
    showToast("질문을 입력하거나 파일을 첨부해주세요.", "info");
    return;
  }
  if (isTransitioning || isProcessing) return;

  lockInput();

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
        addLoadingIndicator("문서를 등록하고 있습니다...");
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
          createConversation(`${ingestResult.file_name}`);
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
        unlockInput();
        bottomInput.focus();
        return;
      }

      // 파일 + 질문: 인제스트 완료 → 검색으로 이어감
      updateLoadingText("문서 검색 중...");
    } else {
      addLoadingIndicator("문서를 검색하고 있습니다...");
    }

    // 2. 질문 → SSE 스트리밍
    const formData = new FormData();
    formData.append("question", question);

    // 멀티턴: 현재 대화의 이전 메시지를 히스토리로 전송 (최근 6개 = 3턴)
    if (currentConversationId) {
      const convs = loadConversations();
      const conv = convs.find(c => c.id === currentConversationId);
      if (conv && conv.messages.length > 0) {
        // 방금 추가한 현재 user 메시지 제외 (마지막 1개)
        const prevMessages = conv.messages.slice(0, -1).slice(-6);
        if (prevMessages.length > 0) {
          formData.append("history", JSON.stringify(prevMessages));
        }
      }
    }

    let fullText = "";
    let streamStarted = false;

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
      let bubble = null;
      let renderTimer = null;

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

            if (msg.sources) {
              // 출처 정보 수신 → 로딩 텍스트 업데이트
              updateLoadingText("답변을 생성하고 있습니다...");
            } else if (msg.token) {
              // 첫 토큰 → 로딩 제거 + 스트리밍 버블 생성
              if (!streamStarted) {
                removeLoadingIndicator();
                bubble = addStreamingAIMessage();
                streamStarted = true;
              }
              fullText += msg.token;
              // 스트리밍 중 실시간 마크다운 렌더링 (throttle: 100ms)
              if (!renderTimer) {
                renderTimer = setTimeout(() => {
                  if (bubble) {
                    bubble.innerHTML = formatAnswer(fullText);
                    scrollToBottom();
                  }
                  renderTimer = null;
                }, 100);
              }
            } else if (msg.answer) {
              // 비스트리밍 폴백 (결과 없음 등)
              if (!streamStarted) {
                removeLoadingIndicator();
                bubble = addStreamingAIMessage();
                streamStarted = true;
              }
              fullText = msg.answer;
              if (bubble) bubble.innerHTML = formatAnswer(fullText);
            } else if (msg.error) {
              if (!streamStarted) {
                removeLoadingIndicator();
                bubble = addStreamingAIMessage();
                streamStarted = true;
              }
              fullText = _friendlyError(msg.error);
              if (bubble) bubble.innerHTML = formatAnswer(fullText);
            }
          } catch (e) { /* JSON 파싱 실패 무시 */ }
        }
      }

      // 마지막 렌더 보장
      if (renderTimer) {
        clearTimeout(renderTimer);
      }

      if (!streamStarted) {
        // 스트리밍이 한 번도 시작 안 된 경우
        removeLoadingIndicator();
        bubble = addStreamingAIMessage();
        fullText = fullText || "답변을 생성할 수 없습니다.";
      }

      finalizeStreamingMessage(fullText);
    } catch (streamErr) {
      removeLoadingIndicator();
      if (!streamStarted) {
        addAIMessage(formatAnswer(fullText || "서버 연결에 실패했습니다."));
      } else {
        finalizeStreamingMessage(fullText || "서버 연결에 실패했습니다.");
      }
    }
  } catch (err) {
    removeLoadingIndicator();
    if (err.message.includes("업로드")) {
      showToast(err.message, "error");
      if (!question.trim()) { unlockInput(); return; }
    }
    addAIMessage(_friendlyError(err.message));
  }

  unlockInput();
  bottomInput.focus();
}

// ── 마크다운 포맷 ──

function formatAnswer(text) {
  // VLM이 출처를 생성하더라도 제거
  text = text.replace(/\[출처:.*?\]/g, "").replace(/\n*참조\s*문서[：:].*$/gm, "").trim();

  const lines = text.split("\n");
  const result = [];
  let i = 0;

  while (i < lines.length) {
    // 코드 블록: ```lang ... ```
    if (/^\s*```/.test(lines[i])) {
      const lang = lines[i].replace(/^\s*```/, "").trim();
      const codeLines = [];
      i++;
      while (i < lines.length && !/^\s*```\s*$/.test(lines[i])) {
        codeLines.push(lines[i]);
        i++;
      }
      if (i < lines.length) i++; // skip closing ```
      const codeId = "code-" + Date.now() + "-" + Math.random().toString(36).slice(2, 6);
      result.push(
        `<div class="code-block">` +
        `<div class="code-header"><span>${escapeHtml(lang || "code")}</span><button class="code-copy-btn" data-target="${codeId}">복사</button></div>` +
        `<pre id="${codeId}"><code>${escapeHtml(codeLines.join("\n"))}</code></pre></div>`
      );
      continue;
    }

    // 마크다운 테이블 감지: | ... | 패턴이 연속되는 구간
    if (/^\s*\|.+\|/.test(lines[i])) {
      const tableLines = [];
      while (i < lines.length && /^\s*\|.+\|/.test(lines[i])) {
        tableLines.push(lines[i]);
        i++;
      }
      result.push(_renderTable(tableLines));
      continue;
    }

    const line = lines[i];
    result.push(_renderLine(line));
    i++;
  }

  return result.join("");
}

function _renderTable(lines) {
  // 구분선(---|---) 행 필터링 + 헤더/바디 분리
  const rows = [];
  let hasHeader = false;

  for (let i = 0; i < lines.length; i++) {
    const cells = lines[i].split("|").slice(1, -1).map(c => c.trim());
    if (cells.every(c => /^[-:]+$/.test(c))) {
      hasHeader = true;
      continue;
    }
    rows.push(cells);
  }

  if (rows.length === 0) return "";

  let html = '<table class="md-table">';

  if (hasHeader && rows.length > 0) {
    html += "<thead><tr>";
    rows[0].forEach(cell => {
      html += `<th>${_inlineFormat(cell)}</th>`;
    });
    html += "</tr></thead><tbody>";
    for (let i = 1; i < rows.length; i++) {
      html += "<tr>";
      rows[i].forEach(cell => {
        html += `<td>${_inlineFormat(cell)}</td>`;
      });
      html += "</tr>";
    }
    html += "</tbody>";
  } else {
    html += "<tbody>";
    rows.forEach(row => {
      html += "<tr>";
      row.forEach(cell => {
        html += `<td>${_inlineFormat(cell)}</td>`;
      });
      html += "</tr>";
    });
    html += "</tbody>";
  }

  html += "</table>";
  return html;
}

function _inlineFormat(text) {
  return escapeHtml(text).replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");
}

function _renderLine(line) {
  if (!line.trim()) return "<br>";

  // 헤더
  const h3 = line.match(/^###\s+(.+)/);
  if (h3) return `<h4>${_inlineFormat(h3[1])}</h4>`;
  const h2 = line.match(/^##\s+(.+)/);
  if (h2) return `<h3>${_inlineFormat(h2[1])}</h3>`;
  const h1 = line.match(/^#\s+(.+)/);
  if (h1) return `<h3>${_inlineFormat(h1[1])}</h3>`;

  // 리스트
  const ul = line.match(/^(\s*)[*-]\s+(.+)/);
  if (ul) {
    return `<div style="padding-left:${Math.min(ul[1].length, 4) + 1}em">\u2022 ${_inlineFormat(ul[2])}</div>`;
  }

  const ol = line.match(/^(\s*)\d+[.)]\s+(.+)/);
  if (ol) {
    const num = line.match(/\d+[.)]/)[0];
    return `<div style="padding-left:${Math.min(ol[1].length, 4) + 1}em">${escapeHtml(num)} ${_inlineFormat(ol[2])}</div>`;
  }

  return _inlineFormat(line) + "<br>";
}

// ── 에러 메시지 사용자 친화적 변환 ──

function _friendlyError(msg) {
  if (!msg) return "알 수 없는 오류가 발생했습니다. 잠시 후 다시 시도해주세요.";
  const m = msg.toLowerCase();
  if (m.includes("timeout") || m.includes("timed out"))
    return "서버 응답 시간이 초과되었습니다. 잠시 후 다시 시도해주세요.";
  if (m.includes("connect") || m.includes("network") || m.includes("fetch"))
    return "서버에 연결할 수 없습니다. 네트워크 상태를 확인해주세요.";
  if (m.includes("500") || m.includes("internal"))
    return "서버 내부 오류가 발생했습니다. 관리자에게 문의해주세요.";
  if (m.includes("413") || m.includes("too large"))
    return "파일 크기가 너무 큽니다. 더 작은 파일로 시도해주세요.";
  if (m.includes("404") || m.includes("not found"))
    return "요청한 리소스를 찾을 수 없습니다.";
  return "오류가 발생했습니다. 잠시 후 다시 시도해주세요.";
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
  if (e.key === "Enter" && !e.shiftKey) doSearch(queryInput.value);
});

document.getElementById("search-btn").addEventListener("click", () => {
  doSearch(queryInput.value);
});

bottomInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) doSearch(bottomInput.value);
});

document.getElementById("bottom-send-btn").addEventListener("click", () => {
  doSearch(bottomInput.value);
});

// ── 코드 블록 복사 (이벤트 위임) ──
chatMessages.addEventListener("click", (e) => {
  if (!e.target.classList.contains("code-copy-btn")) return;
  const targetId = e.target.dataset.target;
  const pre = document.getElementById(targetId);
  if (!pre) return;
  navigator.clipboard.writeText(pre.textContent).then(() => {
    e.target.textContent = "복사됨";
    setTimeout(() => { e.target.textContent = "복사"; }, 1500);
  });
});

// ── 키보드 단축키 ──
document.addEventListener("keydown", (e) => {
  // Escape: 사이드바 닫기
  if (e.key === "Escape" && sidebarOpen) {
    toggleSidebar();
    return;
  }
  // Ctrl+Shift+N: 새 대화
  if (e.ctrlKey && e.shiftKey && e.key === "N") {
    e.preventDefault();
    currentConversationId = null;
    chatMessages.innerHTML = "";
    renderHistoryList();
    if (chatArea.classList.contains("visible")) {
      bottomInput.focus();
    } else {
      queryInput.focus();
    }
    return;
  }
});

// ── 대화 이력 검색 ──
function filterHistory(keyword) {
  const convs = loadConversations();
  const items = historyList.querySelectorAll(".history-item");
  const kw = keyword.trim().toLowerCase();
  if (!kw) {
    items.forEach(item => item.style.display = "");
    return;
  }
  items.forEach((item, idx) => {
    const conv = convs[idx];
    if (!conv) { item.style.display = "none"; return; }
    const match = conv.title.toLowerCase().includes(kw) ||
                  conv.messages.some(m => m.content.toLowerCase().includes(kw));
    item.style.display = match ? "" : "none";
  });
}

// ── 모바일 가상 키보드 대응 ──
if (window.visualViewport) {
  window.visualViewport.addEventListener("resize", () => {
    const bottomBar = document.querySelector(".bottom-bar");
    if (bottomBar) {
      // 가상 키보드가 올라오면 하단 바 위치 조정
      const offset = window.innerHeight - window.visualViewport.height;
      bottomBar.style.paddingBottom = offset > 0 ? `${offset}px` : "";
    }
  });
}

// ── 대화 이력 검색 이벤트 ──
const historySearchInput = document.getElementById("history-search");
if (historySearchInput) {
  historySearchInput.addEventListener("input", (e) => {
    filterHistory(e.target.value);
  });
}

// ── 초기화 ──
renderHistoryList();
