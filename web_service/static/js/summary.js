/* Summary tab - tóm tắt cuộc họp với citation click-to-seek */

let _summarizerAvailable = false;
let _summaryData = null;
let _summaryFileId = null;
let _summaryLoading = false;

// === Init: check summarizer availability ===

async function initSummaryTab() {
    try {
        const resp = await fetch('/api/summarizer/status', {
            headers: window.authToken ? { 'Authorization': 'Bearer ' + window.authToken } : {},
        });
        if (resp.ok) {
            const data = await resp.json();
            _summarizerAvailable = data.available;
        }
    } catch (e) {
        _summarizerAvailable = false;
    }
    updateSummaryTabVisibility();
}

function updateSummaryTabVisibility() {
    const tabBtn = document.getElementById('tab-summary');
    if (tabBtn) {
        tabBtn.style.display = _summarizerAvailable ? '' : 'none';
    }
}

// === Load existing summary ===

async function loadSummaryForFile(fileId) {
    _summaryFileId = fileId;
    _summaryData = null;

    const container = document.getElementById('result-summary');
    if (!container) return;

    if (!_summarizerAvailable) {
        container.innerHTML = '';
        return;
    }

    try {
        const resp = await fetch(`/api/files/${fileId}/summary`, {
            headers: window.authToken ? { 'Authorization': 'Bearer ' + window.authToken } : {},
        });
        if (resp.ok) {
            _summaryData = await resp.json();
            renderSummary(_summaryData);
        } else {
            renderSummaryEmpty();
        }
    } catch (e) {
        renderSummaryEmpty();
    }
}

// === Trigger summarization ===

async function triggerSummarize() {
    if (!_summaryFileId || _summaryLoading) return;

    _summaryLoading = true;
    const container = document.getElementById('result-summary');
    container.innerHTML = `
        <div class="summary-progress">
            <div class="summary-progress-spinner"></div>
            <div class="summary-progress-text">Đang đưa vào hàng đợi...</div>
        </div>`;

    try {
        const resp = await fetch(`/api/files/${_summaryFileId}/summarize`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                ...(window.authToken ? { 'Authorization': 'Bearer ' + window.authToken } : {}),
            },
        });
        const data = await resp.json();
        if (!resp.ok) {
            container.innerHTML = `<div class="summary-error">❌ ${esc(data.detail || 'Lỗi không xác định')}</div>`;
            _summaryLoading = false;
            return;
        }
        if (data.position > 0) {
            container.querySelector('.summary-progress-text').textContent =
                `Đang chờ trong hàng đợi (vị trí #${data.position})...`;
        }
    } catch (e) {
        container.innerHTML = `<div class="summary-error">❌ Không thể kết nối server</div>`;
        _summaryLoading = false;
    }
}

// === Render ===

function renderSummaryEmpty() {
    const container = document.getElementById('result-summary');
    container.innerHTML = `
        <div class="summary-empty">
            <p>Chưa có tóm tắt cho file này.</p>
            <button class="btn-summarize" onclick="triggerSummarize()">📝 Tạo tóm tắt</button>
            <p class="summary-hint">Quá trình tóm tắt mất khoảng 1-2 phút. Bạn có thể nghe lại audio trong lúc đợi.</p>
        </div>`;
}

function renderSummary(summary) {
    const container = document.getElementById('result-summary');
    if (!summary || !summary.title) {
        renderSummaryEmpty();
        return;
    }

    const segments = summary._segments || {};
    let html = '<div class="summary-container">';

    // Title
    html += `<div class="summary-title">${esc(summary.title)}</div>`;

    // Summary overview
    if (summary.summary) {
        html += `<div class="summary-overview">${esc(summary.summary)}</div>`;
    }

    // Key points
    if (summary.key_points && summary.key_points.length) {
        html += '<div class="summary-section">';
        html += '<div class="summary-section-title">Điểm chính</div>';
        html += '<ul class="summary-list">';
        for (const pt of summary.key_points) {
            html += '<li class="summary-item">';
            html += `<span class="summary-item-text">${esc(pt.text)}</span>`;
            if (pt.speaker) html += ` <span class="summary-speaker">— ${esc(pt.speaker)}</span>`;
            html += renderCitations(pt.refs, segments);
            html += '</li>';
        }
        html += '</ul></div>';
    }

    // Decisions
    if (summary.decisions && summary.decisions.length) {
        html += '<div class="summary-section">';
        html += '<div class="summary-section-title">Quyết định</div>';
        html += '<ul class="summary-list">';
        for (const d of summary.decisions) {
            html += `<li class="summary-item"><span class="summary-item-text">${esc(d.text)}</span>`;
            html += renderCitations(d.refs, segments);
            html += '</li>';
        }
        html += '</ul></div>';
    }

    // Action items
    if (summary.action_items && summary.action_items.length) {
        html += '<div class="summary-section">';
        html += '<div class="summary-section-title">Công việc</div>';
        html += '<ul class="summary-list">';
        for (const a of summary.action_items) {
            html += '<li class="summary-item">';
            html += `<span class="summary-item-text">${esc(a.text)}</span>`;
            if (a.assignee) html += ` <span class="summary-assignee">→ ${esc(a.assignee)}</span>`;
            if (a.deadline && a.deadline !== 'null') html += ` <span class="summary-deadline">📅 ${esc(a.deadline)}</span>`;
            html += renderCitations(a.refs, segments);
            html += '</li>';
        }
        html += '</ul></div>';
    }

    // Conclusion
    if (summary.conclusion && summary.conclusion !== 'null') {
        html += '<div class="summary-section">';
        html += '<div class="summary-section-title">Kết luận</div>';
        html += `<div class="summary-conclusion">${esc(summary.conclusion)}</div>`;
        html += '</div>';
    }

    // Re-summarize button
    html += `<div class="summary-actions">
        <button class="btn-resummarize" onclick="triggerSummarize()">🔄 Tóm tắt lại</button>
    </div>`;

    html += '</div>';
    container.innerHTML = html;
}

function renderCitations(refs, segments) {
    if (!refs || !refs.length) return '';
    let html = ' <span class="summary-citations">';
    for (const ref of refs) {
        const seg = segments[String(ref)];
        let timeStr = `#${ref}`;
        if (seg) {
            const t = seg.start_time || 0;
            const mm = String(Math.floor(t / 60)).padStart(2, '0');
            const ss = String(Math.floor(t % 60)).padStart(2, '0');
            timeStr = `${mm}:${ss}`;
        }
        html += `<a class="citation-link" onclick="citationSeek(${ref})" title="Nghe đoạn gốc">▶${timeStr}</a> `;
    }
    html += '</span>';
    return html;
}

function citationSeek(segIdx) {
    // Switch to content tab first so user sees highlighted text
    switchTab('content');
    // seekToSegment is defined in player.js
    if (typeof seekToSegment === 'function') {
        seekToSegment(segIdx);
    }
}

function esc(s) {
    if (!s) return '';
    const div = document.createElement('div');
    div.textContent = s;
    return div.innerHTML;
}

// === WebSocket handlers ===

function onSummaryStarted(data) {
    if (data.file_id !== _summaryFileId) return;
    _summaryLoading = true;
    const container = document.getElementById('result-summary');
    if (container) {
        container.innerHTML = `
            <div class="summary-progress">
                <div class="summary-progress-spinner"></div>
                <div class="summary-progress-text">Đang bắt đầu tóm tắt...</div>
            </div>`;
    }
}

function onSummaryComplete(data) {
    if (data.file_id !== _summaryFileId) return;
    _summaryLoading = false;
    _summaryData = data.summary;
    renderSummary(data.summary);
}

function onSummaryError(data) {
    if (data.file_id !== _summaryFileId) return;
    _summaryLoading = false;
    const container = document.getElementById('result-summary');
    if (container) {
        container.innerHTML = `
            <div class="summary-error">
                ❌ Lỗi tóm tắt: ${esc(data.error)}
                <br><br>
                <button class="btn-summarize" onclick="triggerSummarize()">🔄 Thử lại</button>
            </div>`;
    }
}

// Summary progress updates come through the normal progress WS handler
// We hook into it by checking the phase name
const _origOnProgress = typeof onProgress === 'function' ? onProgress : null;
function _summaryProgressHook(data) {
    if (data.phase === 'Summary' && data.file_id === _summaryFileId && _summaryLoading) {
        const container = document.getElementById('result-summary');
        const textEl = container ? container.querySelector('.summary-progress-text') : null;
        if (textEl) {
            textEl.textContent = data.message || 'Đang xử lý...';
        }
    }
    // Call original handler
    if (_origOnProgress) _origOnProgress(data);
}
