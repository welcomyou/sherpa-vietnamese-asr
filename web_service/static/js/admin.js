/* Admin Panel - Quan tri he thong qua web
   Tuong duong server_gui.py nhung chay tren browser */

let _adminTab = 'stats';
let _adminRefreshTimer = null;
let _adminData = {};

// === Panel toggle ===
function toggleAdminPanel() {
    const panel = document.getElementById('admin-panel');
    if (panel.style.display === 'none') {
        panel.style.display = 'flex';
        const mp = document.getElementById('meetings-panel');
        if (mp) mp.style.display = 'none';
        loadAdminTab();
        _adminRefreshTimer = setInterval(loadAdminTab, 10000);
    } else {
        closeAdminPanel();
    }
}

function closeAdminPanel() {
    document.getElementById('admin-panel').style.display = 'none';
    if (_adminRefreshTimer) { clearInterval(_adminRefreshTimer); _adminRefreshTimer = null; }
}

function switchAdminTab(tab, el) {
    _adminTab = tab;
    document.querySelectorAll('.admin-tabs .tab-btn').forEach(b => b.classList.remove('active'));
    if (el) el.classList.add('active');
    loadAdminTab();
}

async function loadAdminTab() {
    const el = document.getElementById('admin-content');
    try {
        if (_adminTab === 'stats') await renderAdminStats(el);
        else if (_adminTab === 'sessions') await renderAdminSessions(el);
        else if (_adminTab === 'queue') await renderAdminQueue(el);
        else if (_adminTab === 'users') await renderAdminUsers(el);
        else if (_adminTab === 'config') await renderAdminConfig(el);
    } catch (e) {
        el.innerHTML = '<div class="admin-empty"><p style="color:var(--danger)">' + escapeHtml(e.message) + '</p></div>';
    }
}

// === Helpers ===
function _fmtTime(iso) {
    if (!iso) return '-';
    return iso.substring(11, 19);
}

function _fmtDate(iso) {
    if (!iso) return '-';
    return iso.substring(0, 10);
}

function _fmtSize(bytes) {
    if (!bytes || bytes === 0) return '0';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(0) + ' KB';
    if (bytes < 1024 * 1024 * 1024) return (bytes / 1024 / 1024).toFixed(1) + ' MB';
    return (bytes / 1024 / 1024 / 1024).toFixed(2) + ' GB';
}

function _badge(text, type) {
    const colors = {
        success: 'var(--success)', danger: 'var(--danger)',
        warning: 'var(--warning)', info: 'var(--accent)', muted: 'var(--text-secondary)',
    };
    const c = colors[type] || colors.muted;
    return '<span class="admin-badge" style="color:' + c + '">' + escapeHtml(text) + '</span>';
}

// ============================================================
//  TAB 1: TONG QUAN (Stats)
// ============================================================
async function renderAdminStats(el) {
    const [stats, locks] = await Promise.all([
        apiFetch('/api/admin/stats'),
        apiFetch('/api/admin/rate-limits'),
    ]);
    _adminData.stats = stats;

    let html = '<div class="admin-grid">';
    const cards = [
        { val: stats.active_sessions || 0, lbl: 'Phien hoat dong', icon: 'session' },
        { val: stats.total_users || 0, lbl: 'Nguoi dung', icon: 'user' },
        { val: stats.total_files || 0, lbl: 'Tong file', icon: 'file' },
        { val: stats.queue_waiting || 0, lbl: 'Dang cho', icon: 'queue' },
        { val: stats.queue_processing || 0, lbl: 'Dang xu ly', icon: 'process' },
        { val: stats.files_today || 0, lbl: 'File hom nay', icon: 'today' },
    ];
    for (const c of cards) {
        html += '<div class="admin-stat">';
        html += '<span class="admin-stat-val">' + c.val + '</span>';
        html += '<span class="admin-stat-lbl">' + c.lbl + '</span>';
        html += '</div>';
    }
    html += '</div>';

    // Rate limit section
    if (locks.length > 0) {
        html += '<div class="admin-section admin-section-warn">';
        html += '<div class="admin-section-header">';
        html += '<strong>IP bi khoa dang nhap (' + locks.length + ')</strong>';
        html += '<button class="btn btn-sm" onclick="adminClearRateLimits()">Mo khoa tat ca</button>';
        html += '</div>';
        html += '<div class="admin-table-wrap"><table class="admin-table">';
        html += '<tr><th>IP</th><th>So lan sai</th><th>Mo khoa sau</th></tr>';
        for (const l of locks) {
            html += '<tr><td>' + escapeHtml(l.ip) + '</td>';
            html += '<td>' + l.attempts + ' lan</td>';
            html += '<td>' + l.unlock_in_seconds + 's</td></tr>';
        }
        html += '</table></div></div>';
    }

    // Server info
    html += '<div class="admin-section">';
    html += '<div class="admin-section-header"><strong>Thong tin server</strong></div>';
    html += '<div class="admin-info-grid">';
    if (stats.db_size_mb) html += '<div class="admin-info-item"><span class="admin-info-lbl">Database</span><span>' + stats.db_size_mb + ' MB</span></div>';
    if (stats.uploads_size_mb) html += '<div class="admin-info-item"><span class="admin-info-lbl">Uploads</span><span>' + stats.uploads_size_mb + ' MB</span></div>';
    if (stats.total_completed !== undefined) html += '<div class="admin-info-item"><span class="admin-info-lbl">Da xu ly</span><span>' + stats.total_completed + ' file</span></div>';
    html += '</div></div>';

    el.innerHTML = html;
}

async function adminClearRateLimits() {
    if (!confirm('Mo khoa tat ca IP bi khoa dang nhap?')) return;
    await apiFetch('/api/admin/rate-limits/clear', { method: 'POST' });
    showToast('Da mo khoa tat ca IP');
    loadAdminTab();
}

// ============================================================
//  TAB 2: PHIEN (Sessions)
// ============================================================
async function renderAdminSessions(el) {
    const sessions = await apiFetch('/api/admin/sessions');

    let html = '<div class="admin-toolbar">';
    html += '<div class="admin-toolbar-left">';
    html += '<button class="btn btn-sm" onclick="adminCleanupSessions()">Don dep het han</button>';
    html += '</div>';
    html += '<span class="admin-count">' + sessions.length + ' phien</span>';
    html += '</div>';

    if (sessions.length === 0) {
        html += '<div class="admin-empty">Khong co phien nao</div>';
        el.innerHTML = html;
        return;
    }

    html += '<div class="admin-table-wrap"><table class="admin-table">';
    html += '<tr><th>ID</th><th>IP</th><th>Loai</th><th>Heartbeat</th><th style="width:60px"></th></tr>';
    for (const s of sessions) {
        const isAnon = s.is_anonymous;
        const type = isAnon ? _badge('Anonymous', 'muted') : _badge(s.username || 'User #' + s.user_id, 'info');
        const sid = escapeHtml(String(s.id).substring(0, 8));
        const fullSid = escapeHtml(String(s.id));
        html += '<tr>';
        html += '<td class="admin-mono" title="' + fullSid + '">' + sid + '...</td>';
        html += '<td>' + escapeHtml(s.ip_address || '-') + '</td>';
        html += '<td>' + type + '</td>';
        html += '<td class="admin-mono">' + _fmtTime(s.last_heartbeat) + '</td>';
        html += '<td><button class="btn btn-sm btn-danger" onclick="adminKillSession(\'' + fullSid + '\')">Kick</button></td>';
        html += '</tr>';
    }
    html += '</table></div>';
    el.innerHTML = html;
}

async function adminKillSession(sid) {
    if (!confirm('Kick phien ' + sid.substring(0, 8) + '...?')) return;
    await apiFetch('/api/admin/sessions/' + sid, { method: 'DELETE' });
    showToast('Da kick phien');
    loadAdminTab();
}

async function adminCleanupSessions() {
    const r = await apiFetch('/api/admin/sessions/cleanup', { method: 'POST' });
    showToast('Da don dep ' + (r.cleaned_count || 0) + ' phien');
    loadAdminTab();
}

// ============================================================
//  TAB 3: HANG DOI (Queue)
// ============================================================
async function renderAdminQueue(el) {
    const items = await apiFetch('/api/admin/queue');

    let html = '<div class="admin-toolbar">';
    html += '<div class="admin-toolbar-left">';
    html += '<button class="btn btn-sm" onclick="adminPauseQueue()">Tam dung</button>';
    html += '<button class="btn btn-sm btn-primary" onclick="adminResumeQueue()">Tiep tuc</button>';
    html += '</div>';
    html += '<span class="admin-count">' + items.length + ' muc</span>';
    html += '</div>';

    if (items.length === 0) {
        html += '<div class="admin-empty">Hang doi trong</div>';
        el.innerHTML = html;
        return;
    }

    html += '<div class="admin-table-wrap"><table class="admin-table">';
    html += '<tr><th>#</th><th>Ten file</th><th>Trang thai</th><th>Tien do</th><th style="width:60px"></th></tr>';
    for (const q of items) {
        let statusBadge;
        if (q.status === 'processing') statusBadge = _badge('Dang xu ly', 'success');
        else if (q.status === 'waiting') statusBadge = _badge('Dang cho', 'warning');
        else if (q.status === 'error') statusBadge = _badge('Loi', 'danger');
        else statusBadge = _badge(q.status, 'muted');

        html += '<tr>';
        html += '<td>' + q.file_id + '</td>';
        html += '<td class="admin-ellipsis" title="' + escapeHtml(q.original_filename || '') + '">' + escapeHtml(q.original_filename || '-') + '</td>';
        html += '<td>' + statusBadge + '</td>';
        html += '<td class="admin-ellipsis">' + escapeHtml(q.progress_message || '-') + '</td>';
        html += '<td><button class="btn btn-sm btn-danger" onclick="adminCancelQueue(' + q.file_id + ')">Huy</button></td>';
        html += '</tr>';
    }
    html += '</table></div>';
    el.innerHTML = html;
}

async function adminPauseQueue() {
    await apiFetch('/api/admin/queue/pause', { method: 'POST' });
    showToast('Da tam dung hang doi');
}

async function adminResumeQueue() {
    await apiFetch('/api/admin/queue/resume', { method: 'POST' });
    showToast('Da tiep tuc hang doi');
}

async function adminCancelQueue(fileId) {
    if (!confirm('Huy xu ly file #' + fileId + '?')) return;
    await apiFetch('/api/admin/queue/cancel/' + fileId, { method: 'POST' });
    showToast('Da huy');
    loadAdminTab();
}

// ============================================================
//  TAB 4: NGUOI DUNG (Users)
// ============================================================
async function renderAdminUsers(el) {
    const users = await apiFetch('/api/admin/users');

    let html = '<div class="admin-toolbar">';
    html += '<div class="admin-toolbar-left">';
    html += '<button class="btn btn-sm btn-primary" onclick="adminShowCreateUser()">Tao nguoi dung</button>';
    html += '</div>';
    html += '<span class="admin-count">' + users.length + ' nguoi dung</span>';
    html += '</div>';

    html += '<div class="admin-table-wrap"><table class="admin-table">';
    html += '<tr><th>ID</th><th>Ten</th><th>Quyen</th><th>File</th><th>Dung luong</th><th>Trang thai</th><th style="width:120px">Thao tac</th></tr>';
    for (const u of users) {
        const used = _fmtSize(u.storage_used_bytes || 0);
        const limit = u.storage_limit_gb > 0 ? u.storage_limit_gb + ' GB' : 'KGH';
        const storage = used + ' / ' + limit;
        const active = u.is_active ? _badge('Hoat dong', 'success') : _badge('Bi khoa', 'danger');
        const isAdmin = u.role === 'admin';

        html += '<tr>';
        html += '<td>' + u.id + '</td>';
        html += '<td><strong>' + escapeHtml(u.username) + '</strong></td>';
        html += '<td>' + _badge(u.role, isAdmin ? 'info' : 'muted') + '</td>';
        html += '<td>' + (u.file_count || 0) + '</td>';
        html += '<td class="admin-mono" style="font-size:11px">' + storage + '</td>';
        html += '<td>' + active + '</td>';
        html += '<td class="admin-actions">';
        if (!isAdmin) {
            html += '<button class="btn btn-sm" onclick="adminResetPassword(' + u.id + ',\'' + escapeHtml(u.username) + '\')" title="Doi mat khau">MK</button>';
            html += '<button class="btn btn-sm" onclick="adminToggleActive(' + u.id + ',' + (u.is_active ? 0 : 1) + ')" title="' + (u.is_active ? 'Khoa' : 'Mo khoa') + '">' + (u.is_active ? 'Khoa' : 'Mo') + '</button>';
            html += '<button class="btn btn-sm btn-danger" onclick="adminDeleteUser(' + u.id + ',\'' + escapeHtml(u.username) + '\')" title="Xoa">Xoa</button>';
        }
        html += '</td>';
        html += '</tr>';
    }
    html += '</table></div>';
    el.innerHTML = html;
}

// --- User CRUD ---
function adminShowCreateUser() {
    const el = document.getElementById('admin-content');
    el.innerHTML =
        '<div class="admin-form">' +
        '<h4>Tao nguoi dung moi</h4>' +
        '<div class="form-group"><label>Ten dang nhap</label><input id="adm-new-user" type="text" autocomplete="off"></div>' +
        '<div class="form-group"><label>Mat khau</label><input id="adm-new-pass" type="password" autocomplete="new-password"></div>' +
        '<div class="form-group"><label>Gioi han luu tru (GB, 0 = khong gioi han)</label><input id="adm-new-storage" type="number" value="5" min="0" step="0.5"></div>' +
        '<div id="adm-create-err" class="error-msg"></div>' +
        '<div class="admin-form-actions">' +
        '<button class="btn btn-primary" onclick="adminDoCreateUser()">Tao nguoi dung</button>' +
        '<button class="btn" onclick="switchAdminTab(\'users\')">Huy</button>' +
        '</div>' +
        '</div>';
    document.getElementById('adm-new-user').focus();
}

async function adminDoCreateUser() {
    const username = document.getElementById('adm-new-user').value.trim();
    const password = document.getElementById('adm-new-pass').value;
    const storage = parseFloat(document.getElementById('adm-new-storage').value) || 5;
    const errEl = document.getElementById('adm-create-err');
    errEl.textContent = '';
    if (!username || username.length < 2) { errEl.textContent = 'Ten dang nhap it nhat 2 ky tu'; return; }
    if (password.length < 6) { errEl.textContent = 'Mat khau it nhat 6 ky tu'; return; }
    try {
        await apiFetch('/api/admin/users', {
            method: 'POST',
            body: JSON.stringify({ username, password, storage_limit_gb: storage }),
        });
        showToast('Da tao nguoi dung: ' + username);
        _adminTab = 'users';
        loadAdminTab();
    } catch (e) {
        errEl.textContent = e.message;
    }
}

async function adminResetPassword(uid, username) {
    const pw = prompt('Mat khau moi cho "' + username + '" (it nhat 6 ky tu):');
    if (pw === null) return;
    if (pw.length < 6) { alert('Mat khau it nhat 6 ky tu'); return; }
    try {
        await apiFetch('/api/admin/users/' + uid + '/reset-password', {
            method: 'POST', body: JSON.stringify({ password: pw }),
        });
        showToast('Da doi mat khau cho ' + username);
    } catch (e) {
        alert('Loi: ' + e.message);
    }
}

async function adminToggleActive(uid, active) {
    try {
        await apiFetch('/api/admin/users/' + uid, {
            method: 'PUT', body: JSON.stringify({ is_active: !!active }),
        });
        showToast(active ? 'Da mo khoa tai khoan' : 'Da khoa tai khoan');
        loadAdminTab();
    } catch (e) {
        alert('Loi: ' + e.message);
    }
}

async function adminDeleteUser(uid, username) {
    if (!confirm('Xoa nguoi dung "' + username + '"? Hanh dong nay khong the hoan tac.')) return;
    try {
        await apiFetch('/api/admin/users/' + uid, { method: 'DELETE' });
        showToast('Da xoa: ' + username);
        loadAdminTab();
    } catch (e) {
        alert('Loi: ' + e.message);
    }
}

// ============================================================
//  TAB 5: CAU HINH (Config)
// ============================================================
async function renderAdminConfig(el) {
    const config = await apiFetch('/api/admin/config');

    // Check summarizer status
    let summStatus = { available: false };
    try { summStatus = await apiFetch('/api/summarizer/status'); } catch(e) {}

    const summEnabled = config.summarizer_enabled === '1';
    const summPath = config.summarizer_model_path || '';
    const summModel = config.summarizer_ollama_model || 'qwen3.5:4b';

    let html = '';

    // === Summarizer Section ===
    html += '<div class="admin-section">';
    html += '<div class="admin-section-header"><strong>Tom tat cuoc hop (LLM)</strong></div>';
    html += '<div class="admin-form">';

    // Toggle
    html += '<div class="form-group form-row">';
    html += '<label class="form-check"><input type="checkbox" id="cfg-summ-enabled" ' + (summEnabled ? 'checked' : '') + '> Bat chuc nang tom tat</label>';
    html += '</div>';

    // Status
    let statusHtml = '';
    if (summEnabled) {
        if (summStatus.available) {
            statusHtml = '<span style="color:var(--success)">San sang</span>';
        } else if (summPath && summPath.startsWith('http')) {
            statusHtml = '<span style="color:var(--warning)">Ollama: kiem tra ket noi...</span>';
        } else if (summPath) {
            statusHtml = '<span style="color:var(--danger)">Model khong tim thay: ' + escapeHtml(summPath) + '</span>';
        } else {
            statusHtml = '<span style="color:var(--warning)">Chua cau hinh model. Nhan "Tai model" hoac nhap duong dan.</span>';
        }
    } else {
        statusHtml = '<span style="color:var(--text-secondary)">Da tat</span>';
    }
    html += '<div class="form-group form-row"><label>Trang thai:</label> ' + statusHtml + '</div>';

    // Model path
    html += '<div class="form-group">';
    html += '<label>Model path / Ollama URL</label>';
    html += '<input type="text" id="cfg-summ-path" value="' + escapeHtml(summPath) + '" placeholder="models/Qwen3.5-4B-Q4_K_M.gguf hoac http://localhost:11434" style="width:100%">';
    html += '<div class="form-hint">File GGUF local (uu tien, dung llama.cpp) hoac URL Ollama server</div>';
    html += '</div>';

    // Ollama model name
    html += '<div class="form-group">';
    html += '<label>Ten model Ollama</label>';
    html += '<input type="text" id="cfg-summ-model" value="' + escapeHtml(summModel) + '" placeholder="qwen3.5:4b" style="width:200px">';
    html += '<div class="form-hint">Chi dung khi tro toi Ollama URL</div>';
    html += '</div>';

    // Buttons
    html += '<div class="admin-form-actions">';
    html += '<button class="btn btn-primary" onclick="adminSaveConfig()">Luu cau hinh</button>';
    html += '<button class="btn" id="btn-dl-model" onclick="adminDownloadModel()">Tai model Qwen3.5-4B (~2.7 GB)</button>';
    html += '</div>';

    html += '<div id="cfg-summ-msg" class="form-hint" style="margin-top:8px"></div>';
    html += '</div></div>';

    // === General Config Section ===
    html += '<div class="admin-section" style="margin-top:16px">';
    html += '<div class="admin-section-header"><strong>Cau hinh chung</strong></div>';
    html += '<div class="admin-form">';

    const generalFields = [
        { key: 'cpu_threads', label: 'So luong CPU', type: 'number' },
        { key: 'max_upload_mb', label: 'Upload toi da (MB)', type: 'number' },
        { key: 'anonymous_timeout_minutes', label: 'Timeout anonymous (phut)', type: 'number' },
        { key: 'storage_per_user_gb', label: 'Luu tru / user (GB)', type: 'number' },
        { key: 'max_sessions', label: 'Phien toi da', type: 'number' },
        { key: 'jwt_expire_minutes', label: 'JWT het han (phut)', type: 'number' },
    ];

    for (const f of generalFields) {
        const val = config[f.key] || '';
        html += '<div class="form-group form-row">';
        html += '<label>' + f.label + '</label>';
        html += '<input type="' + f.type + '" id="cfg-' + f.key + '" value="' + escapeHtml(val) + '" style="width:120px">';
        html += '</div>';
    }

    html += '<div class="admin-form-actions">';
    html += '<button class="btn btn-primary" onclick="adminSaveGeneralConfig()">Luu cau hinh chung</button>';
    html += '</div>';
    html += '</div></div>';

    el.innerHTML = html;
}

async function adminSaveConfig() {
    const enabled = document.getElementById('cfg-summ-enabled').checked;
    const path = document.getElementById('cfg-summ-path').value.trim();
    const model = document.getElementById('cfg-summ-model').value.trim();

    try {
        await apiFetch('/api/admin/config', {
            method: 'PUT',
            body: JSON.stringify({
                summarizer_enabled: enabled ? '1' : '0',
                summarizer_model_path: path,
                summarizer_ollama_model: model || 'qwen3.5:4b',
            }),
        });
        showToast('Da luu cau hinh tom tat');
        // Refresh summary tab visibility
        if (typeof initSummaryTab === 'function') initSummaryTab();
        loadAdminTab();
    } catch (e) {
        alert('Loi: ' + e.message);
    }
}

async function adminSaveGeneralConfig() {
    const fields = ['cpu_threads', 'max_upload_mb', 'anonymous_timeout_minutes',
                    'storage_per_user_gb', 'max_sessions', 'jwt_expire_minutes'];
    const body = {};
    for (const key of fields) {
        const el = document.getElementById('cfg-' + key);
        if (el) body[key] = el.value;
    }
    try {
        await apiFetch('/api/admin/config', { method: 'PUT', body: JSON.stringify(body) });
        showToast('Da luu cau hinh chung');
    } catch (e) {
        alert('Loi: ' + e.message);
    }
}

async function adminDownloadModel() {
    const btn = document.getElementById('btn-dl-model');
    const msg = document.getElementById('cfg-summ-msg');
    btn.disabled = true;
    btn.textContent = 'Dang tai...';
    msg.textContent = 'Dang tai model tu HuggingFace (~2.7 GB), vui long doi...';
    msg.style.color = 'var(--warning)';

    try {
        const resp = await apiFetch('/api/admin/download-summarizer-model', { method: 'POST' });
        msg.textContent = 'Tai thanh cong: ' + (resp.path || '');
        msg.style.color = 'var(--success)';
        btn.textContent = 'Da tai xong';
        // Auto-fill path
        const pathEl = document.getElementById('cfg-summ-path');
        if (pathEl && resp.path) pathEl.value = resp.path;
    } catch (e) {
        msg.textContent = 'Loi: ' + e.message;
        msg.style.color = 'var(--danger)';
        btn.textContent = 'Thu lai';
        btn.disabled = false;
    }
}
