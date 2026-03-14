/* Meetings management for logged-in users */

let _allMeetings = [];
let _meetingsPage = 1;
const MEETINGS_PER_PAGE = 20;

// === Meeting Name Dialog ===

function showMeetingNameModal() {
    const fileName = uploadedFile ? uploadedFile.name.replace(/\.[^.]+$/, '') : '';
    document.getElementById('meeting-name-input').value = '';
    document.getElementById('meeting-name-input').placeholder = fileName || 'Nhập tên cuộc họp...';
    document.getElementById('meeting-name-modal').style.display = 'flex';
    document.getElementById('meeting-name-input').focus();
}

function hideMeetingNameModal() {
    document.getElementById('meeting-name-modal').style.display = 'none';
}

async function confirmMeetingName() {
    hideMeetingNameModal();
    const meetingName = document.getElementById('meeting-name-input').value.trim();
    await doProcessFile(meetingName);
}

// === Meetings Panel ===

function toggleMeetingsPanel() {
    const panel = document.getElementById('meetings-panel');
    if (panel.style.display === 'none' || !panel.style.display) {
        panel.style.display = 'flex';
        _meetingsPage = 1;
        loadMeetings();
    } else {
        closeMeetingsPanel();
    }
}

function closeMeetingsPanel() {
    document.getElementById('meetings-panel').style.display = 'none';
}

async function loadMeetings(search) {
    const listEl = document.getElementById('meetings-list');
    listEl.innerHTML = '<div class="meetings-loading">Đang tải...</div>';

    try {
        let url = '/api/meetings';
        if (search) url += '?search=' + encodeURIComponent(search);
        _allMeetings = await apiFetch(url);
        renderMeetingsPage();
    } catch (e) {
        listEl.innerHTML = '<div class="meetings-loading" style="color:var(--danger)">Lỗi: ' + e.message + '</div>';
    }
}

let _searchTimeout = null;
function searchMeetings() {
    clearTimeout(_searchTimeout);
    _searchTimeout = setTimeout(() => {
        const q = document.getElementById('meetings-search').value.trim();
        _meetingsPage = 1;
        loadMeetings(q || undefined);
    }, 300);
}

function renderMeetingsPage() {
    const listEl = document.getElementById('meetings-list');
    const total = _allMeetings.length;

    if (!total) {
        listEl.innerHTML = '<div class="meetings-loading">Chưa có cuộc họp nào</div>';
        updateMeetingsToolbar();
        return;
    }

    const totalPages = Math.ceil(total / MEETINGS_PER_PAGE);
    if (_meetingsPage > totalPages) _meetingsPage = totalPages;
    const start = (_meetingsPage - 1) * MEETINGS_PER_PAGE;
    const page = _allMeetings.slice(start, start + MEETINGS_PER_PAGE);

    let html = '<div class="meetings-grid">';
    for (const m of page) {
        const statusClass = 'status-' + m.status;
        const statusLabel = { waiting: 'Chờ', processing: 'Đang xử lý', completed: 'Hoàn thành', error: 'Lỗi' }[m.status] || m.status;
        const clickable = m.status === 'completed';
        const size = formatFileSize(m.file_size);
        const date = formatDate(m.created_at);

        html += `
        <div class="mg-item${clickable ? ' mg-clickable' : ''}" data-id="${m.id}"
             ${clickable ? `onclick="loadMeeting(${m.id})"` : ''}>
            <div class="mg-check" onclick="event.stopPropagation()">
                ${m.status !== 'processing' ? `<input type="checkbox" class="mg-cb" value="${m.id}">` : ''}
            </div>
            <div class="mg-info">
                <div class="mg-name" ondblclick="event.stopPropagation(); startRenameMeeting(${m.id}, this)">${escapeHtml(m.meeting_name)}</div>
                <div class="mg-meta">
                    <span class="mg-file">${escapeHtml(m.original_filename)}</span>
                    <span>${size}</span>
                    <span>${date}</span>
                    ${m.error_message && m.status === 'error' ? `<span class="mg-err">${escapeHtml(m.error_message)}</span>` : ''}
                </div>
            </div>
            <div class="mg-status ${statusClass}">${statusLabel}</div>
        </div>`;
    }
    html += '</div>';

    // Pagination
    if (totalPages > 1) {
        html += '<div class="mg-pager">';
        html += `<button class="btn btn-sm" onclick="meetingsPagePrev()" ${_meetingsPage <= 1 ? 'disabled' : ''}>&laquo;</button>`;
        html += `<span class="mg-page-info">Trang ${_meetingsPage}/${totalPages} (${total} cuộc họp)</span>`;
        html += `<button class="btn btn-sm" onclick="meetingsPageNext()" ${_meetingsPage >= totalPages ? 'disabled' : ''}>&raquo;</button>`;
        html += '</div>';
    }

    listEl.innerHTML = html;
    updateMeetingsToolbar();
}

function meetingsPagePrev() {
    if (_meetingsPage > 1) { _meetingsPage--; renderMeetingsPage(); }
}

function meetingsPageNext() {
    const totalPages = Math.ceil(_allMeetings.length / MEETINGS_PER_PAGE);
    if (_meetingsPage < totalPages) { _meetingsPage++; renderMeetingsPage(); }
}

// === Toolbar: select all, delete selected ===

function updateMeetingsToolbar() {
    const bar = document.getElementById('meetings-toolbar');
    const cbs = document.querySelectorAll('.mg-cb');
    bar.style.display = cbs.length > 0 ? 'flex' : 'none';
}

function toggleSelectAll() {
    const master = document.getElementById('mg-select-all');
    document.querySelectorAll('.mg-cb').forEach(cb => cb.checked = master.checked);
}

async function deleteSelectedMeetings() {
    const ids = [];
    document.querySelectorAll('.mg-cb:checked').forEach(cb => ids.push(parseInt(cb.value)));
    if (!ids.length) { showToast('Chưa chọn cuộc họp nào', 'error'); return; }
    if (!confirm(`Xóa ${ids.length} cuộc họp đã chọn? Không thể hoàn tác.`)) return;

    let ok = 0, fail = 0;
    for (const id of ids) {
        try {
            await apiFetch(`/api/meetings/${id}`, { method: 'DELETE' });
            ok++;
        } catch { fail++; }
    }
    showToast(`Đã xóa ${ok} cuộc họp` + (fail ? `, ${fail} lỗi` : ''), ok ? 'success' : 'error');
    document.getElementById('mg-select-all').checked = false;
    loadMeetings(document.getElementById('meetings-search').value.trim() || undefined);
}

// === Load meeting result ===

async function loadMeeting(meetingId) {
    try {
        const data = await apiFetch(`/api/meetings/${meetingId}`);
        if (data.asr_result) {
            currentASRData = data.asr_result;
            renderASRResult(data.asr_result);
            clearDirty();
            currentFileId = data.file_id;
            loadAudioFromUrl(`/api/meetings/${meetingId}/audio`);

            document.querySelector('.drop-zone-text').style.display = 'none';
            document.getElementById('file-selected').style.display = 'flex';
            document.getElementById('file-name').textContent = data.original_filename;
            document.getElementById('file-size').textContent = '';

            closeMeetingsPanel();
            showToast('Đã tải: ' + data.meeting_name, 'success');
        } else {
            showToast('Cuộc họp chưa có kết quả ASR', 'error');
        }
    } catch (e) {
        showToast('Lỗi: ' + e.message, 'error');
    }
}

// === Rename ===

function startRenameMeeting(meetingId, el) {
    const currentName = el.textContent;
    const input = document.createElement('input');
    input.type = 'text';
    input.value = currentName;
    input.className = 'meeting-rename-input';
    input.onkeydown = async (e) => {
        if (e.key === 'Enter') { e.preventDefault(); await saveRenameMeeting(meetingId, input.value.trim(), el, currentName); }
        else if (e.key === 'Escape') { el.textContent = currentName; }
    };
    input.onblur = async () => {
        if (input.value.trim() && input.value.trim() !== currentName) {
            await saveRenameMeeting(meetingId, input.value.trim(), el, currentName);
        } else {
            el.textContent = currentName;
        }
    };
    el.textContent = '';
    el.appendChild(input);
    input.focus();
    input.select();
}

async function saveRenameMeeting(meetingId, newName, el, oldName) {
    if (!newName) { el.textContent = oldName; return; }
    try {
        await apiFetch(`/api/meetings/${meetingId}`, {
            method: 'PUT',
            body: JSON.stringify({ meeting_name: newName }),
        });
        el.textContent = newName;
    } catch (e) {
        el.textContent = oldName;
        showToast('Lỗi đổi tên: ' + e.message, 'error');
    }
}

// === Helpers ===

function formatFileSize(bytes) {
    if (!bytes) return '0 B';
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1048576) return (bytes / 1024).toFixed(1) + ' KB';
    if (bytes < 1073741824) return (bytes / 1048576).toFixed(1) + ' MB';
    return (bytes / 1073741824).toFixed(2) + ' GB';
}

function formatDate(dateStr) {
    if (!dateStr) return '';
    try {
        const d = new Date(dateStr + 'Z');
        return d.toLocaleDateString('vi-VN') + ' ' + d.toLocaleTimeString('vi-VN', { hour: '2-digit', minute: '2-digit' });
    } catch {
        return dateStr;
    }
}
