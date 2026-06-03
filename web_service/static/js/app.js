/* Main app logic - init, render ASR, auth, UI state */

// === PWA Setup ===
let deferredInstallPrompt = null;

if ('serviceWorker' in navigator) {
    navigator.serviceWorker.register('/static/sw.js', { scope: '/static/' }).catch((err) => {
        console.warn('SW registration failed:', err);
    });
    navigator.serviceWorker.addEventListener('controllerchange', () => {
        window.location.reload();
    });
}

function isIOS() {
    return /iPad|iPhone|iPod/.test(navigator.userAgent) ||
        (navigator.platform === 'MacIntel' && navigator.maxTouchPoints > 1);
}

function isInStandaloneMode() {
    return window.matchMedia('(display-mode: standalone)').matches ||
        window.navigator.standalone === true;
}

function detectDevice() {
    const ua = navigator.userAgent;
    if (/Samsung/i.test(ua)) return 'samsung';
    if (/Android/i.test(ua)) return 'android';
    if (isIOS()) return 'ios';
    if (/Macintosh|Mac OS/i.test(ua)) return 'macos';
    if (/Windows/i.test(ua)) return 'windows';
    return 'other';
}

function isCertTrusted() {
    // Neu truy cap localhost thi khong can cert
    const host = window.location.hostname;
    if (host === 'localhost' || host === '127.0.0.1') return true;
    // Neu SW da dang ky thanh cong thi cert duoc trust
    if (navigator.serviceWorker && navigator.serviceWorker.controller) return true;
    return false;
}

function isAppInstalled() {
    // Chi an panel khi THUC SU dang chay trong standalone mode
    // Khong dua vao localStorage vi no van ton tai sau khi go app
    return isInStandaloneMode();
}

function setupCspSafeEventDelegation() {
    const closestEventTarget = (event, selector) => {
        const target = event.target;
        if (target && typeof target.closest === 'function') return target.closest(selector);
        return target?.parentElement?.closest(selector) || null;
    };

    const clickActions = {
        'show-about': () => showAboutDialog(),
        'show-login': () => showLoginModal(),
        'toggle-admin-panel': () => toggleAdminPanel(),
        'toggle-meetings-panel': () => toggleMeetingsPanel(),
        'toggle-user-menu': (event) => toggleUserMenu(event),
        'show-change-password': () => showChangePasswordModal(),
        logout: () => logout(),
        'toggle-panel': (_event, target) => togglePanel(target.dataset.panel),
        'show-cert-guide': () => showCertGuide(),
        'install-pwa': () => installPWA(),
        'run-server-calibration': () => runServerCalibration(),
        'clear-file': () => clearFile(),
        'process-file': () => processFile(),
        'cancel-process': () => cancelProcess(),
        'load-json': () => loadJSON(),
        'save-json': () => saveJSON(),
        'download-audio': () => downloadAudio(),
        'copy-text': () => copyText(),
        'switch-tab': (_event, target) => switchTab(target.dataset.tab),
        'search-nav': (_event, target) => searchNav(parseInt(target.dataset.dir, 10)),
        'clear-search': () => clearSearch(),
        'toggle-play': () => togglePlay(),
        'save-edits': () => saveEdits(),
        'scroll-top': () => scrollToTop(),
        'resolve-calibration-mode': (_event, target) => resolveCalibrationModeChoice(target.dataset.choice),
        'hide-login': () => hideLoginModal(),
        'login-submit': () => doLogin(),
        'hide-split-speaker': () => hideSplitSpeakerModal(),
        'split-speaker-submit': () => doSplitSpeaker(),
        'hide-change-password': () => hideChangePasswordModal(),
        'change-password-submit': () => doChangePassword(),
        'ctx-split-speaker': () => ctxSplitSpeaker(),
        'ctx-merge-up': () => ctxMergeUp(),
        'ctx-merge-down': () => ctxMergeDown(),
        'ctx-rename-speaker': () => ctxRenameSpeaker(),
        'ctx-copy': () => ctxCopy(),
        'hide-rename': () => hideRenameModal(),
        'rename-speaker': (_event, target) => doRenameSpeaker(target.dataset.all === 'true'),
        'hide-meeting-name': () => hideMeetingNameModal(),
        'confirm-meeting-name': () => confirmMeetingName(),
        'close-meetings-panel': () => closeMeetingsPanel(),
        'delete-selected-meetings': () => deleteSelectedMeetings(),
        'close-admin-panel': () => closeAdminPanel(),
        'switch-admin-tab': (_event, target) => switchAdminTab(
            target.dataset.tab,
            target.closest('.admin-tabs') ? target : null
        ),
        'load-meeting': (_event, target) => loadMeeting(parseInt(target.dataset.meetingId, 10)),
        'meetings-page-prev': () => meetingsPagePrev(),
        'meetings-page-next': () => meetingsPageNext(),
        'trigger-summarize': () => triggerSummarize(),
        'citation-seek': (_event, target) => citationSeek(parseInt(target.dataset.ref, 10)),
        'admin-clear-rate-limits': () => adminClearRateLimits(),
        'admin-cleanup-sessions': () => adminCleanupSessions(),
        'admin-kill-session': (_event, target) => adminKillSession(target.dataset.sid),
        'admin-pause-queue': () => adminPauseQueue(),
        'admin-resume-queue': () => adminResumeQueue(),
        'admin-cancel-queue': (_event, target) => adminCancelQueue(parseInt(target.dataset.fileId, 10)),
        'admin-show-create-user': () => adminShowCreateUser(),
        'admin-reset-password': (_event, target) => adminResetPassword(parseInt(target.dataset.uid, 10), target.dataset.uname),
        'admin-toggle-active': (_event, target) => adminToggleActive(parseInt(target.dataset.uid, 10), parseInt(target.dataset.active, 10)),
        'admin-delete-user': (_event, target) => adminDeleteUser(parseInt(target.dataset.uid, 10), target.dataset.uname),
        'admin-create-user': () => adminDoCreateUser(),
        'admin-download-model': () => adminDownloadModel(),
        'admin-save-config': () => adminSaveConfig(),
        'admin-save-general-config': () => adminSaveGeneralConfig(),
        'switch-cert-tab': (_event, target) => switchCertTab(target.dataset.deviceId),
        'remove-closest': (_event, target) => target.closest(target.dataset.closest)?.remove(),
    };

    const inputActions = {
        'search-input': () => onSearchInput(),
        'seek-audio': (_event, target) => seekAudio(target.value),
        'search-meetings': () => searchMeetings(),
    };

    const changeActions = {
        'sync-split-speaker': (_event, target) => {
            document.getElementById('split-speaker-input').value = target.value;
        },
        'sync-rename-speaker': (_event, target) => {
            document.getElementById('rename-input').value = target.value;
        },
        'toggle-select-all': () => toggleSelectAll(),
        'json-selected': (_event, target) => onJSONSelected(target),
        'admin-toggle-summ-backend': () => adminToggleSummBackend(),
    };

    const enterKeyActions = {
        'login-submit': () => doLogin(),
        'split-speaker-submit': () => doSplitSpeaker(),
        'change-password-submit': () => doChangePassword(),
        'confirm-meeting-name': () => confirmMeetingName(),
    };

    document.addEventListener('click', (event) => {
        if (closestEventTarget(event, '[data-stop-click]')) {
            event.stopPropagation();
            return;
        }
        const target = closestEventTarget(event, '[data-action]');
        if (!target) return;
        const handler = clickActions[target.dataset.action];
        if (!handler) return;
        event.preventDefault();
        handler(event, target);
    });

    document.addEventListener('input', (event) => {
        const target = closestEventTarget(event, '[data-input-action]');
        if (!target) return;
        const handler = inputActions[target.dataset.inputAction];
        if (handler) handler(event, target);
    });

    document.addEventListener('change', (event) => {
        const target = closestEventTarget(event, '[data-change-action]');
        if (!target) return;
        const handler = changeActions[target.dataset.changeAction];
        if (handler) handler(event, target);
    });

    document.addEventListener('keydown', (event) => {
        const target = closestEventTarget(event, '[data-key-action]');
        if (!target) return;
        if (target.dataset.keyAction === 'search-keydown') {
            onSearchKeydown(event);
            return;
        }
        if (event.key !== 'Enter') return;
        const handler = enterKeyActions[target.dataset.keyAction];
        if (!handler) return;
        event.preventDefault();
        handler(event, target);
    });

    document.addEventListener('dblclick', (event) => {
        const target = closestEventTarget(event, '[data-dbl-action]');
        if (!target || target.dataset.dblAction !== 'rename-meeting') return;
        event.stopPropagation();
        startRenameMeeting(parseInt(target.dataset.meetingId, 10), target);
    });
}

setupCspSafeEventDelegation();

// Hien/an install panel
window.addEventListener('DOMContentLoaded', () => {
    const panel = document.getElementById('install-panel');
    if (!panel) return;

    // Dang chay trong standalone mode -> da cai roi, an panel
    if (isInStandaloneMode()) return;

    // Mo trong browser -> hien panel de user co the cai
    panel.style.display = '';

    // Kiem tra cert trusted -> enable nut B2
    checkAndEnableInstallBtn();
});

// Bat su kien beforeinstallprompt (Android Chrome)
window.addEventListener('beforeinstallprompt', (e) => {
    e.preventDefault();
    deferredInstallPrompt = e;
    const panel = document.getElementById('install-panel');
    if (panel && !isInStandaloneMode()) panel.style.display = '';
    checkAndEnableInstallBtn();
});

window.addEventListener('appinstalled', () => {
    deferredInstallPrompt = null;
    const panel = document.getElementById('install-panel');
    if (panel) panel.style.display = 'none';
});

function checkAndEnableInstallBtn() {
    const btn = document.getElementById('btn-install-app');
    const msg = document.getElementById('install-status-msg');
    if (!btn) return;
    // Enable khi: co beforeinstallprompt HOAC la iOS
    const canInstall = deferredInstallPrompt || isIOS();
    btn.disabled = !canInstall;
    if (canInstall) {
        btn.classList.add('btn-install-ready');
        if (msg) msg.style.display = 'none';
    } else {
        btn.classList.remove('btn-install-ready');
        if (msg) {
            msg.style.display = '';
            msg.textContent = 'Ứng dụng có tên "Chuyển văn bản", vui lòng kiểm tra xem đã cài chưa.';
        }
    }
}

function installPWA() {
    if (deferredInstallPrompt) {
        deferredInstallPrompt.prompt();
        deferredInstallPrompt.userChoice.then((result) => {
            if (result.outcome === 'accepted') {
                const panel = document.getElementById('install-panel');
                if (panel) panel.style.display = 'none';
            }
            deferredInstallPrompt = null;
        });
        return;
    }
    if (isIOS()) {
        showIOSInstallGuide();
        return;
    }
}

function showIOSInstallGuide() {
    const overlay = document.createElement('div');
    overlay.className = 'ios-install-overlay';
    overlay.onclick = (e) => { if (e.target === overlay) overlay.remove(); };
    overlay.innerHTML =
        '<div class="ios-install-dialog">' +
            '<h3>Cài đặt ứng dụng</h3>' +
            '<p>Trên thiết bị iOS:</p>' +
            '<ol>' +
                '<li>Nhấn nút <strong>Chia sẻ</strong> ' +
                    '<svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" style="vertical-align:middle">' +
                        '<path d="M4 12v8a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2v-8"/>' +
                        '<polyline points="16 6 12 2 8 6"/>' +
                        '<line x1="12" y1="2" x2="12" y2="15"/>' +
                    '</svg> ở thanh dưới' +
                '</li>' +
                '<li><strong>Vuốt lên</strong> trong menu chia sẻ để tìm mục <strong>"Thêm vào Màn hình chính"</strong></li>' +
                '<li>Nhấn <strong>"Thêm"</strong> (góc phải)</li>' +
            '</ol>' +
            '<button class="btn btn-sm" style="width:100%" data-action="remove-closest" data-closest=".ios-install-overlay">Đã hiểu</button>' +
        '</div>';
    document.body.appendChild(overlay);
}

// === SSL Cert Guide ===
function showCertGuide() {
    const guide = document.getElementById('cert-guide');
    if (!guide) return;
    if (guide.style.display !== 'none') {
        guide.style.display = 'none';
        return;
    }

    const device = detectDevice();
    guide.innerHTML = buildCertGuideHTML(device);
    guide.style.display = '';
}

function buildCertGuideHTML(activeDevice) {
    const devices = [
        { id: 'samsung', label: 'Samsung', icon: '📱' },
        { id: 'android', label: 'Android khác', icon: '📱' },
        { id: 'ios', label: 'iPhone / iPad', icon: '🍎' },
        { id: 'windows', label: 'Windows', icon: '💻' },
        { id: 'macos', label: 'MacOS', icon: '🖥' },
    ];

    const guides = {
        samsung:
            '<ol>' +
                '<li>Nhấn <strong>"Tải cert"</strong> bên dưới</li>' +
                '<li>Mở <strong>Cài đặt</strong> → <strong>Bảo mật và riêng tư</strong> → <strong>Cài đặt bảo mật khác</strong></li>' +
                '<li>Chọn <strong>"Cài đặt từ bộ nhớ thiết bị"</strong></li>' +
                '<li>Chọn <strong>"Chứng chỉ CA"</strong> → nhấn <strong>"Tiếp tục cài đặt"</strong></li>' +
                '<li>Chọn file <strong>sherpa-asr-vn.crt</strong> vừa tải</li>' +
                '<li><strong>Đóng hoàn toàn trình duyệt</strong> (vuốt tắt app) rồi mở lại</li>' +
            '</ol>',
        android:
            '<ol>' +
                '<li>Nhấn <strong>"Tải cert"</strong> bên dưới</li>' +
                '<li>Mở <strong>Cài đặt → Bảo mật → Mã hóa và thông tin xác thực</strong></li>' +
                '<li>Chọn <strong>"Cài chứng chỉ"</strong></li>' +
                '<li>⚠️ Quan trọng: chọn <strong>"Chứng chỉ CA"</strong> (KHÔNG chọn "VPN và ứng dụng")</li>' +
                '<li>Chọn file <strong>sherpa-asr-vn.crt</strong> vừa tải</li>' +
                '<li><strong>Đóng hoàn toàn trình duyệt</strong> (vuốt tắt app) rồi mở lại</li>' +
            '</ol>',
        ios:
            '<ol>' +
                '<li>Nhấn <strong>"Tải cert"</strong> bên dưới → chọn <strong>"Cho phép"</strong></li>' +
                '<li>Mở <strong>Cài đặt</strong> → <strong>Cài đặt chung</strong> → <strong>Quản lý VPN & thiết bị</strong></li>' +
                '<li>Trong mục "Hồ sơ đã tải về", nhấn vào <strong>Sherpa Vietnamese ASR</strong></li>' +
                '<li>Nhấn <strong>"Cài đặt"</strong> (góc phải) → nhấn <strong>"Cài đặt"</strong> lần nữa ở màn Cảnh báo</li>' +
                '<li>Vào <strong>Cài đặt → Cài đặt chung → Giới thiệu</strong> → kéo xuống cuối chọn <strong>"Cài đặt tin cậy chứng nhận"</strong></li>' +
                '<li>Bật công tắc cho <strong>Sherpa Vietnamese ASR</strong></li>' +
                '<li><strong>Đóng hoàn toàn Safari</strong> rồi mở lại</li>' +
            '</ol>',
        windows:
            '<ol>' +
                '<li>Nhấn <strong>"Tải cert"</strong> bên dưới</li>' +
                '<li>Mở file <strong>sherpa-asr-vn.crt</strong> vừa tải</li>' +
                '<li>Nhấn <strong>"Install Certificate..."</strong></li>' +
                '<li>Chọn <strong>Current User</strong> → Next</li>' +
                '<li>Chọn <strong>"Place all certificates in the following store"</strong> → Browse → <strong>Trusted Root Certification Authorities</strong></li>' +
                '<li>Nhấn Next → Finish → Yes (xác nhận)</li>' +
                '<li><strong>Khởi động lại trình duyệt</strong></li>' +
            '</ol>',
        macos:
            '<ol>' +
                '<li>Nhấn <strong>"Tải cert"</strong> bên dưới</li>' +
                '<li>Mở file <strong>sherpa-asr-vn.crt</strong> → tự mở Keychain Access</li>' +
                '<li>Chọn Keychain: <strong>System</strong> hoặc <strong>login</strong></li>' +
                '<li>Tìm cert <strong>"Sherpa Vietnamese ASR"</strong>, double-click</li>' +
                '<li>Mở <strong>Trust → When using this certificate</strong> → chọn <strong>Always Trust</strong></li>' +
                '<li>Đóng cửa sổ, nhập mật khẩu xác nhận</li>' +
                '<li><strong>Khởi động lại trình duyệt</strong></li>' +
            '</ol>',
    };

    let html = '<div class="cert-guide-content">';
    html += '<div class="cert-device-tabs">';
    devices.forEach((d) => {
        const active = d.id === activeDevice ? ' active' : '';
        html += '<button class="cert-tab' + active + '" data-action="switch-cert-tab" data-device-id="' + d.id + '">' + d.icon + ' ' + d.label + '</button>';
    });
    html += '</div>';

    devices.forEach((d) => {
        const show = d.id === activeDevice ? '' : ' style="display:none"';
        html += '<div class="cert-tab-content" id="cert-tab-' + d.id + '"' + show + '>' + (guides[d.id] || '') + '</div>';
    });

    html += '<a href="/install-cert" class="btn btn-sm btn-install" style="display:inline-flex;text-decoration:none;margin-top:8px;width:100%;justify-content:center">' +
        '<svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>' +
        ' Tải cert</a>';
    html += '</div>';
    return html;
}

function switchCertTab(deviceId) {
    document.querySelectorAll('.cert-tab-content').forEach((el) => el.style.display = 'none');
    document.querySelectorAll('.cert-tab').forEach((el) => el.classList.remove('active'));
    const tab = document.getElementById('cert-tab-' + deviceId);
    if (tab) tab.style.display = '';
    document.querySelectorAll('.cert-tab').forEach((btn) => {
        if (btn.dataset.deviceId === deviceId) btn.classList.add('active');
    });
}

// === App State ===

let currentASRData = null;
window.authToken = null;
window.appConfig = null;
let _dirty = false;
let currentProcessStartedAt = null;
let activeFileStatusPollTimer = null;
let activeFileStatusPollFileId = null;
let activeFileStatusPollInFlight = false;
const SPEAKER_EDIT_HISTORY_LIMIT = 100;
let speakerUndoStack = [];
let speakerRedoStack = [];

// === Init ===

document.addEventListener('DOMContentLoaded', async () => {
    initUpload();
    initPlayer();
    initContextMenu();
    setupSpeakerEditShortcuts();

    // Disable scroll wheel trên config inputs (tránh thay đổi nhầm)
    document.querySelectorAll('input[type="range"], select').forEach(el => {
        el.addEventListener('wheel', e => { e.preventDefault(); }, { passive: false });
    });

    // Tao session truoc (1 request duy nhat), cookie se duoc set tu response
    try {
        await fetch('/api/session', { method: 'POST', credentials: 'same-origin' });
    } catch (e) {
        console.error('Failed to create session:', e);
    }

    // Sau khi co session cookie, ket noi WebSocket va load config
    connectWebSocket();

    // Init summary tab (check availability)
    if (typeof initSummaryTab === 'function') initSummaryTab();

    // Load config
    try {
        const [models, defaults] = await Promise.all([
            apiFetch('/api/config/models'),
            apiFetch('/api/config/defaults'),
        ]);

        window.appConfig = defaults;
        populateModels(models, defaults);

        // Offline download link (chỉ cho phép http/https)
        const offlineUrl = defaults.offline_download_url;
        const btnOffline = document.getElementById('btn-download-offline');
        const lblUnavailable = document.getElementById('offline-unavailable');
        if (offlineUrl && /^https?:\/\//i.test(offlineUrl)) {
            if (btnOffline) { btnOffline.href = offlineUrl; btnOffline.style.display = ''; }
            if (lblUnavailable) lblUnavailable.style.display = 'none';
        }
    } catch (e) {
        console.error('Failed to load config:', e);
    }
    // Ẩn loading overlay
    const loadingEl = document.getElementById('config-loading');
    if (loadingEl) loadingEl.classList.add('hidden');

    // Init sliders
    initSliders();

    // Num speakers dropdown (2-20)
    const numSpeakers = document.getElementById('cfg-num-speakers');
    for (let i = 2; i <= 20; i++) {
        numSpeakers.innerHTML += `<option value="${i}">${i}</option>`;
    }

    // Diarization checkbox toggle
    document.getElementById('cfg-diarization').addEventListener('change', (e) => {
        document.getElementById('diarization-options').style.display = e.target.checked ? '' : 'none';
    });

    // Check auth via HttpOnly cookie; do not persist JWT in localStorage.
    try {
        const me = await apiFetch('/api/auth/me');
        window.authToken = '__cookie__';
        showLoggedIn(me);
    } catch {
        window.authToken = null;
        updateHeaderTitle(false);
    }

    // Restore session state cho moi user (anonymous + login)
    // Login da goi trong showLoggedIn, chi goi them cho anonymous
    if (!window.authToken) {
        restoreSessionState();
    }
});

function updateHeaderTitle(isLoggedIn) {
    // Warning in empty drop zone
    const warnEmpty = document.getElementById('anonymous-warning-empty');
    if (warnEmpty) {
        warnEmpty.style.display = isLoggedIn ? 'none' : '';
    }
    // Warning shown when file is selected (only show if file is selected and anonymous)
    const warnFile = document.getElementById('anonymous-warning');
    if (warnFile) {
        // Only show if anonymous AND a file is selected
        const hasFile = document.getElementById('file-selected')?.style.display !== 'none';
        warnFile.style.display = (!isLoggedIn && hasFile) ? 'block' : 'none';
    }
    syncJsonUploadVisibility();
}

function syncJsonUploadVisibility() {
    const isAnonymous = !window.authToken;
    const btnLoadJson = document.getElementById('btn-load-json');
    if (btnLoadJson) {
        const hasUploadedFile = typeof uploadedFile !== 'undefined' && !!uploadedFile;
        btnLoadJson.style.display = isAnonymous ? '' : 'none';
        btnLoadJson.disabled = !isAnonymous || !hasUploadedFile;
    }
    const fileInput = document.getElementById('file-input');
    if (fileInput) {
        const baseAccept = '.mp3,.wav,.m4a,.flac,.aac,.wma,.ogg,.opus,.mp4,.mkv,.avi,.mov,.webm,.flv,.wmv';
        fileInput.setAttribute('accept', isAnonymous ? baseAccept + ',.json' : baseAccept);
    }
}


function populateModels(models, defaults) {
    const asrSelect = document.getElementById('cfg-model');
    asrSelect.innerHTML = '';
    for (const m of models.asr_models) {
        const opt = document.createElement('option');
        opt.value = m.id;
        opt.textContent = m.name;
        if (m.id === defaults.asr_model) opt.selected = true;
        asrSelect.appendChild(opt);
    }

    const spkSelect = document.getElementById('cfg-speaker-model');
    spkSelect.innerHTML = '';
    for (const m of models.speaker_models) {
        const opt = document.createElement('option');
        opt.value = m.id;
        opt.textContent = m.name;
        if (m.id === defaults.speaker_model) opt.selected = true;
        spkSelect.appendChild(opt);
    }

    // Luu default thresholds tu API de tu dong dieu chinh khi doi model
    window._speakerModelThresholds = {};
    for (const m of models.speaker_models) {
        if (m.default_threshold) {
            window._speakerModelThresholds[m.id] = m.default_threshold;
        }
    }

    // Update threshold default when model changes (internal, hidden from user)
    spkSelect.addEventListener('change', () => {
        const modelId = spkSelect.value;
        const newThreshold = window._speakerModelThresholds[modelId] || 70;
        document.getElementById('cfg-threshold').value = newThreshold;
    });

    // Apply defaults
    document.getElementById('cfg-punct').value = defaults.punctuation_confidence;
    document.getElementById('cfg-case').value = defaults.case_confidence;
    document.getElementById('cfg-threshold').value = defaults.diarization_threshold;
    updateCalibrationStatus(defaults.execution_provider || 'cpu');
    updateSliderLabels();
}

function initSliders() {
    const sliderLabels = {
        'cfg-punct': { el: 'cfg-punct-label', fmt: (v) => getConfLabel(v) + ` (${v})` },
        'cfg-case': { el: 'cfg-case-label', fmt: (v) => getConfLabel(v) + ` (${v})` },
    };

    for (const [sliderId, info] of Object.entries(sliderLabels)) {
        document.getElementById(sliderId).addEventListener('input', () => updateSliderLabels());
    }
    updateSliderLabels();
}

function updateSliderLabels() {
    const p = document.getElementById('cfg-punct').value;
    document.getElementById('cfg-punct-label').textContent = getConfLabel(p) + ` (${p})`;

    const c = document.getElementById('cfg-case').value;
    document.getElementById('cfg-case-label').textContent = getConfLabel(c) + ` (${c})`;

    const thresholdEl = document.getElementById('cfg-threshold');
    const thresholdLabel = document.getElementById('cfg-threshold-label');
    if (thresholdEl && thresholdLabel) {
        thresholdLabel.textContent = (thresholdEl.value / 100).toFixed(2);
    }
}

function getConfLabel(v) {
    v = parseInt(v);
    if (v <= 2) return 'Rất ít';
    if (v <= 4) return 'Ít';
    if (v <= 6) return 'Vừa';
    if (v <= 8) return 'Nhiều';
    return 'Rất nhiều';
}

function updateCalibrationStatus(provider) {
    const el = document.getElementById('calibration-status');
    if (!el) return;
    const value = (provider || 'cpu').toLowerCase();
    el.textContent = value === 'auto' ? 'Tối ưu: GPU auto' : 'Tối ưu: CPU-only';
}

function hasGpuAutoConfig() {
    if ((window.appConfig?.execution_provider || 'cpu').toLowerCase() === 'auto') return true;
    const stageProfile = window.appConfig?.stage_execution_providers || {};
    return Object.values(stageProfile).some((value) => String(value).toLowerCase() === 'auto');
}

function isGpuAutoMode() {
    return (window.appConfig?.execution_provider || 'cpu').toLowerCase() === 'auto';
}

let calibrationModeChoiceResolver = null;

function chooseCalibrationMode() {
    const modal = document.getElementById('calibration-mode-modal');
    if (!modal) return Promise.resolve('rerun');
    const gpuAuto = isGpuAutoMode();
    const textEl = document.getElementById('calibration-mode-text');
    const subtextEl = document.getElementById('calibration-mode-subtext');
    const cpuBtn = document.getElementById('btn-calibration-mode-cpu');
    const gpuBtn = document.getElementById('btn-calibration-mode-gpu');
    if (textEl) {
        textEl.textContent = gpuAuto
            ? 'Thiết bị hiện đang dùng GPU Auto theo kết quả tối ưu đã lưu.'
            : 'Thiết bị hiện đang dùng CPU-only, nhưng vẫn còn kết quả tối ưu GPU đã lưu.';
    }
    if (subtextEl) {
        subtextEl.textContent = gpuAuto
            ? 'Bạn có thể tối ưu lại hoặc chuyển về CPU-only.'
            : 'Bạn có thể bật lại GPU auto hoặc tối ưu lại.';
    }
    if (cpuBtn) cpuBtn.style.display = gpuAuto ? '' : 'none';
    if (gpuBtn) gpuBtn.style.display = gpuAuto ? 'none' : '';
    modal.style.display = 'flex';
    return new Promise((resolve) => {
        calibrationModeChoiceResolver = resolve;
    });
}

function resolveCalibrationModeChoice(choice) {
    const modal = document.getElementById('calibration-mode-modal');
    if (modal) modal.style.display = 'none';
    const resolver = calibrationModeChoiceResolver;
    calibrationModeChoiceResolver = null;
    if (resolver) resolver(choice || 'cancel');
}

async function setServerCpuOnly() {
    const result = await apiFetch('/api/calibration/cpu-only', {
        method: 'POST',
        body: JSON.stringify({}),
    });
    window.appConfig.execution_provider = 'cpu';
    window.appConfig.stage_execution_providers = result.current_stage_execution_providers || window.appConfig.stage_execution_providers || {};
    updateCalibrationStatus('cpu');
    showToast('Đã chuyển sang CPU-only. Kết quả tối ưu GPU vẫn được giữ lại.', 'success');
}

async function setServerGpuAuto() {
    const result = await apiFetch('/api/calibration/gpu-auto', {
        method: 'POST',
        body: JSON.stringify({}),
    });
    window.appConfig.execution_provider = 'auto';
    window.appConfig.stage_execution_providers = result.current_stage_execution_providers || window.appConfig.stage_execution_providers || {};
    updateCalibrationStatus('auto');
    showToast('Đã chuyển sang GPU auto', 'success');
}

function calibrationHardwareText(status) {
    const ram = status?.ram || {};
    let text = status?.hardware_summary || 'Không đọc được thông tin phần cứng';
    if (ram.total_mb) {
        text += `\nRAM: ${ram.available_mb || '?'} / ${ram.total_mb} MB khả dụng/tổng`;
    }
    return text;
}

function gpuModelsDownloadHint(gpuModels) {
    if (!gpuModels || gpuModels.installed) return '';
    const missing = gpuModels.missing_paths || gpuModels.expected_paths || [];
    const expected = missing
        .map((item) => item.display_path || item.relative_path || '')
        .filter(Boolean)
        .map((path) => `- ${path}`)
        .join('\n');
    return `\n\nHãy tải thêm: ${gpuModels.zip_name || 'gpu-models-win64-<version>.zip'}` +
        '\nGiải nén vào thư mục gốc của ứng dụng.' +
        (expected ? `\nSau khi giải nén phải có:\n${expected}` : '');
}

function formatCalibrationNumber(value, suffix = '') {
    if (value === null || value === undefined || value === '') return 'N/A';
    const number = Number(value);
    if (!Number.isFinite(number)) return String(value);
    if (Math.abs(number) < 0.001 && number !== 0) return number.toExponential(2) + suffix;
    return number.toFixed(3) + suffix;
}

function shortCalibrationStageLabel(item) {
    const key = item.key || '';
    const label = item.label || key || 'Stage';
    const labels = {
        speaker_campp_embedding: 'CAM++ embedding',
        speaker_pyannote_embedding: 'Pyannote embedding',
        dnsmos: 'DNSMOS',
        punctuation: 'Punctuation',
    };
    if (labels[key]) return labels[key];
    return label
        .replace('Speaker Diarization: ', '')
        .replace('Pyannote embedding encoder', 'Pyannote embedding')
        .replace('Punctuation: ViBERT punctuation fp32', 'Punctuation')
        .replace('DNSMOS quality', 'DNSMOS');
}

function shortCalibrationReason(item) {
    if (item.missing) return 'thiếu model';
    const reasons = {
        accepted: 'đạt',
        diff_exceeds_tolerance: 'diff vượt ngưỡng',
        gpu_not_faster_enough: 'GPU chưa nhanh hơn 20%',
        gpu_batch_tune_failed: 'batch tune lỗi',
        model_missing: 'thiếu model',
        provider_unavailable: 'provider GPU chưa sẵn sàng',
        gpu_provider_fell_back_to_cpu: 'ORT fallback về CPU',
        provider_runtime_unsupported: 'ORT/DirectML lỗi runtime',
    };
    if (reasons[item.reason]) return reasons[item.reason];
    if (item.error) return 'lỗi benchmark';
    if (item.skipped || item.speedup === null || item.speedup === undefined) {
        return item.reason || 'không đo GPU';
    }
    return item.reason || 'GPU không đạt ngưỡng';
}

function shortCalibrationError(item, maxLen = 220) {
    const retries = item.provider_retries || [];
    let text = item.error || item.error_detail || (retries.length ? retries[retries.length - 1].error : '') || '';
    if (!text) return '';
    text = String(text).replace(/\s+/g, ' ').trim();
    const replacements = {
        'requested DmlExecutionProvider but ORT created CPUExecutionProvider':
            'ORT fallback: requested DmlExecutionProvider, actual CPUExecutionProvider',
        'requested CUDAExecutionProvider but ORT created CPUExecutionProvider':
            'ORT fallback: requested CUDAExecutionProvider, actual CPUExecutionProvider',
        'requested OpenVINOExecutionProvider but ORT created CPUExecutionProvider':
            'ORT fallback: requested OpenVINOExecutionProvider, actual CPUExecutionProvider',
    };
    for (const [needle, replacement] of Object.entries(replacements)) {
        if (text.includes(needle)) {
            text = text === needle ? replacement : text.replace(needle, replacement);
            break;
        }
    }
    return text.length > maxLen ? text.slice(0, maxLen).trimEnd() + '...' : text;
}

function formatCalibrationCpuLine(label, item, speedupText = '') {
    const parts = [shortCalibrationReason(item)];
    const error = shortCalibrationError(item);
    if (error) parts.push(`lỗi: ${error}`);
    const text = parts.filter(Boolean).join('; ');
    return speedupText ? `- ${label}: ${speedupText}, ${text}` : `- ${label}: ${text}`;
}

function calibrationStageActiveForCurrentProfile(item, stageProfile = {}) {
    const key = item.key || '';
    const pipelineKey = item.pipeline_key || key;
    if (key === 'speaker_campp_embedding') {
        return stageProfile.diarization === 'auto' && stageProfile.diarization_campp === 'auto';
    }
    if (key === 'speaker_pyannote_embedding') {
        return stageProfile.diarization === 'auto' && stageProfile.diarization_pyannote === 'auto';
    }
    return stageProfile[pipelineKey] === 'auto';
}

function inactiveCalibrationStageNote(item) {
    const key = item.key || '';
    if (key === 'speaker_campp_embedding') return 'khi chọn CAM++';
    if (key === 'speaker_pyannote_embedding') return 'khi chọn Pyannote';
    return 'khi bật stage này';
}

function formatCalibrationStageLines(comparison, stageProfile = {}) {
    const lines = [];
    const gpuLines = [];
    const inactiveGpuLines = [];
    const cpuLines = [];
    const details = comparison?.stage_details || [];
    for (const item of details) {
        const label = shortCalibrationStageLabel(item);
        const speedup = formatCalibrationNumber(item.speedup, 'x');
        const sampleText = item.sample_items && item.sample_items_total
            ? `, mẫu ${item.sample_items}/${item.sample_items_total}`
            : '';
        if (item.skipped || item.speedup === null || item.speedup === undefined) {
            cpuLines.push(formatCalibrationCpuLine(label, item));
            continue;
        }
        const batchText = item.batch ? `, batch ${item.batch}` : '';
        const provider = (item.actual_provider || '').replace('ExecutionProvider', '');
        const providerText = provider ? `, ${provider}` : '';
        if (item.selected_provider === 'auto') {
            const line = `- ${label}: ${speedup}${batchText}${providerText}${sampleText}`;
            if (calibrationStageActiveForCurrentProfile(item, stageProfile)) {
                gpuLines.push(line);
            } else {
                inactiveGpuLines.push(`${line} (${inactiveCalibrationStageNote(item)})`);
            }
        } else {
            cpuLines.push(formatCalibrationCpuLine(label, item, speedup));
        }
    }

    lines.push('GPU dùng với cấu hình hiện tại:');
    lines.push(...(gpuLines.length ? gpuLines : ['- Không có stage nào đạt ngưỡng']));
    if (inactiveGpuLines.length) {
        lines.push('');
        lines.push('GPU khả dụng khi đổi cấu hình:');
        lines.push(...inactiveGpuLines);
    }
    lines.push('');
    lines.push('CPU giữ:');
    lines.push(...(cpuLines.length ? cpuLines : ['- Không có stage đo nào bị giữ CPU']));

    if (details.some((item) => item.error || item.error_detail)) {
        lines.push('');
        lines.push('Chi tiết lỗi đầy đủ: temp/device_calibration_last.json');
    }

    if (comparison?.gpu_advice) {
        lines.push('');
        lines.push('Gợi ý: ' + comparison.gpu_advice);
    }

    const fixed = comparison?.fixed_cpu_stages || [];
    if (fixed.length) {
        const fixedMap = {
            'ASR encoder/decoder/joiner': 'ASR',
            'Audio decode / resample': 'decode/resample',
            'Silero VAD': 'VAD',
            'Speaker segmentation / VBx / clustering': 'speaker postprocess',
        };
        const fixedLabels = fixed.map((item) => fixedMap[item.label] || item.label || 'stage CPU');
        lines.push('');
        lines.push(`Luôn CPU theo PWA: ${fixedLabels.join(', ')}.`);
    }
    return lines.join('\n');
}

async function runServerCalibration() {
    const btn = document.getElementById('btn-calibration');
    const statusEl = document.getElementById('calibration-status');
    if (hasGpuAutoConfig()) {
        const choice = await chooseCalibrationMode();
        if (choice === 'cpu') {
            try {
                if (btn) btn.disabled = true;
                if (statusEl) statusEl.textContent = 'Tối ưu: đang chuyển CPU-only...';
                await setServerCpuOnly();
            } catch (e) {
                updateCalibrationStatus(window.appConfig?.execution_provider || 'cpu');
                showToast('Không chuyển được CPU-only: ' + e.message, 'error');
            } finally {
                if (btn) btn.disabled = false;
            }
            return;
        }
        if (choice === 'gpu') {
            try {
                if (btn) btn.disabled = true;
                if (statusEl) statusEl.textContent = 'Tối ưu: đang bật GPU auto...';
                await setServerGpuAuto();
            } catch (e) {
                updateCalibrationStatus(window.appConfig?.execution_provider || 'cpu');
                showToast('Không bật được GPU auto: ' + e.message, 'error');
            } finally {
                if (btn) btn.disabled = false;
            }
            return;
        }
        if (choice !== 'rerun') {
            return;
        }
    }
    if (btn) btn.disabled = true;
    if (statusEl) statusEl.textContent = 'Đang kiểm tra phần cứng...';

    try {
        const status = await apiFetch('/api/calibration/status');
        const addon = status.recommended_addon || null;
        const gpuModels = status.recommended_gpu_models || null;
        if (addon && !addon.installed && !status.provider_ready) {
            window.appConfig.execution_provider = 'cpu';
            updateCalibrationStatus('cpu');
            let expected = addon.expected_display_path || `/gpu_addons/${addon.id || '<id>'}/Lib/site-packages/onnxruntime/`;
            expected += gpuModelsDownloadHint(gpuModels);
            alert(
                'Phát hiện GPU nhưng chưa có gói tối ưu GPU phù hợp.\n\n' +
                calibrationHardwareText(status) +
                `\n\nHãy tải: ${addon.zip_name || (addon.artifact + '-<version>.zip')}` +
                '\nGiải nén vào thư mục gốc của ứng dụng.' +
                `\nSau khi giải nén phải có đường dẫn: ${expected}` +
                '\nSau đó mở lại ứng dụng và bấm Tối ưu thiết bị.'
            );
            return;
        }
        if (addon && addon.installed && !status.provider_ready) {
            window.appConfig.execution_provider = 'cpu';
            updateCalibrationStatus('cpu');
            const expected = addon.expected_display_path || `/gpu_addons/${addon.id || '<id>'}/Lib/site-packages/onnxruntime/`;
            alert(
                'Đã thấy gói tối ưu GPU nhưng ONNX Runtime chưa nạp được provider GPU.\n\n' +
                calibrationHardwareText(status) +
                `\n\nĐường dẫn cần có trong thư mục gốc của ứng dụng: ${expected}` +
                '\nHãy đóng hẳn server/app, mở lại rồi bấm Tối ưu thiết bị.' +
                '\nNếu vẫn lỗi, hãy xóa thư mục /gpu_addons/ rồi giải nén lại gói phù hợp vào đúng thư mục gốc của ứng dụng.'
            );
            return;
        }
        if (gpuModels && !gpuModels.installed) {
            window.appConfig.execution_provider = 'cpu';
            updateCalibrationStatus('cpu');
            alert(
                'Phát hiện GPU và gói tối ưu GPU đã sẵn sàng, nhưng còn thiếu model GPU cho ViBERT.\n\n' +
                calibrationHardwareText(status) +
                gpuModelsDownloadHint(gpuModels) +
                '\nSau đó mở lại server/app và bấm Tối ưu thiết bị.'
            );
            return;
        }
        if (!status.can_optimize) {
            window.appConfig.execution_provider = 'cpu';
            updateCalibrationStatus('cpu');
            alert('Không tìm thấy GPU/provider phù hợp. Cấu hình hiện tại đã tối ưu ở chế độ CPU-only.\n\n' + calibrationHardwareText(status));
            return;
        }

        const ok = confirm(
            'Phát hiện GPU có thể tối ưu xử lý.\n\n' +
            calibrationHardwareText(status) +
            `\nProvider đề xuất: ${status.preferred_provider || 'GPU'}\n\n` +
            'Chạy tối ưu bằng file mẫu 10 phút? Quá trình này có thể mất vài phút.'
        );
        if (!ok) {
            updateCalibrationStatus(window.appConfig?.execution_provider || 'cpu');
            return;
        }

        if (statusEl) statusEl.textContent = 'Đang tối ưu bằng file mẫu 10 phút...';
        const report = await apiFetch('/api/calibration/run', {
            method: 'POST',
            body: JSON.stringify({
                model: document.getElementById('cfg-model')?.value,
                speaker_model: document.getElementById('cfg-speaker-model')?.value,
            }),
        });

        const selected = report.current_execution_provider || report.selected_execution_provider || 'cpu';
        window.appConfig.execution_provider = selected;
        window.appConfig.stage_execution_providers = report.current_stage_execution_providers || report.stage_execution_providers || {};
        updateCalibrationStatus(selected);

        const cmp = report.comparison || {};
        const accepted = cmp.accepted_stage_count || 0;
        const measured = cmp.measured_stage_count || 0;
        let providerText = selected === 'auto' ? 'GPU theo từng stage' : 'CPU-only';
        const stageProfile = window.appConfig.stage_execution_providers || {};
        const activeGpu = (cmp.stage_details || []).filter(
            (item) => item.selected_provider === 'auto' && calibrationStageActiveForCurrentProfile(item, stageProfile)
        ).length;
        if (selected !== 'auto' && activeGpu === 0 && accepted > 0) {
            providerText = 'CPU-only hiện tại; GPU khả dụng khi đổi cấu hình';
        }
        const stageLines = formatCalibrationStageLines(cmp, stageProfile);
        const speedupMin = Number(cmp.speedup_min || 1.20);
        const savedText = selected === 'auto' && activeGpu > 0
            ? 'Đã lưu cấu hình. File chạy sau calibration sẽ dùng ngay cấu hình này.'
            : accepted > 0
                ? 'Đã lưu kết quả tối ưu. Cấu hình hiện tại vẫn chạy CPU; GPU sẽ dùng khi đổi sang stage/model đạt benchmark.'
                : 'Đã lưu cấu hình CPU-only cho máy này.';
        alert(
            `Tối ưu hoàn tất. Kết luận: ${providerText}\n` +
            `GPU active hiện tại: ${activeGpu} stage. Đạt benchmark: ${accepted}/${measured} stage.\n` +
            `${savedText}\n\n` +
            `Ngưỡng: inference GPU >= ${speedupMin.toFixed(2)}x và diff đạt.\n` +
            'Số bên dưới là tốc độ inference stage, không phải thời gian xử lý cả file.\n\n' +
            stageLines
        );
    } catch (e) {
        updateCalibrationStatus(window.appConfig?.execution_provider || 'cpu');
        showToast('Tối ưu thiết bị thất bại: ' + e.message, 'error');
    } finally {
        if (btn) btn.disabled = false;
    }
}

// === Panel toggle ===

function togglePanel(panelId) {
    document.getElementById(panelId).classList.toggle('collapsed');
}

// === Process file ===

async function processFile() {
    currentProcessStartedAt = performance.now();
    if (!uploadedFile) return;

    // User đã login → hỏi tên cuộc họp trước
    if (window.authToken) {
        // Upload file trước nếu chưa upload
        if (!currentFileId) {
            try {
                await uploadFile();
            } catch (e) {
                // Thử tạo lại session rồi retry 1 lần
                try {
                    await fetch('/api/session', { method: 'POST', credentials: 'same-origin' });
                    await uploadFile();
                } catch (e2) {
                    showToast('Lỗi upload: ' + (e2.message || e.message), 'error');
                    resetProcessUI();
                    return;
                }
            }
        }
        if (!currentFileId) return;
        showMeetingNameModal();
        return;
    }

    await doProcessFile();
}

async function doProcessFile(meetingName) {
    try {
        // Upload file neu chua upload
        if (!currentFileId) {
            try {
                await uploadFile();
            } catch (e) {
                // Thử tạo lại session rồi retry 1 lần
                await fetch('/api/session', { method: 'POST', credentials: 'same-origin' });
                await uploadFile();
            }
        }

        if (!currentFileId) return;

        // Get config
        const config = getASRConfig();
        if (meetingName !== undefined) {
            config.meeting_name = meetingName;
        }

        // GUI state
        document.getElementById('btn-process').disabled = true;
        document.getElementById('btn-cancel').style.display = '';
        document.getElementById('process-progress').style.display = 'flex';
        document.getElementById('queue-info').style.display = 'none';

        showProcessProgress('Đang gửi yêu cầu xử lý...', 0, '');

        // Subscribe to WS updates
        subscribeQueue(currentFileId);

        // Send process request
        const result = await apiFetch(`/api/process/${currentFileId}`, {
            method: 'POST',
            body: JSON.stringify(config),
        });

        startActiveFileStatusPolling(currentFileId);

        if (result.position > 0) {
            showQueuePosition(result.position, result.total);
            showProcessProgress('File đã được đưa vào hàng đợi. Vui lòng đợi tới lượt xử lý.', 0, '');
            showToast('File đã vào hàng đợi. Vui lòng đợi tới lượt xử lý.', 'success');
        }

    } catch (e) {
        showToast('Lỗi: ' + e.message, 'error');
        resetProcessUI();
    }
}

function getASRConfig() {
    return {
        model: document.getElementById('cfg-model').value,
        speaker_diarization: document.getElementById('cfg-diarization').checked,
        speaker_model: document.getElementById('cfg-speaker-model').value,
        num_speakers: parseInt(document.getElementById('cfg-num-speakers').value),
        punctuation_confidence: parseInt(document.getElementById('cfg-punct').value),
        case_confidence: parseInt(document.getElementById('cfg-case').value),
        diarization_threshold: parseInt(document.getElementById('cfg-threshold').value),
        execution_provider: window.appConfig?.execution_provider || 'cpu',
        stage_execution_providers: window.appConfig?.stage_execution_providers || {},
        gap_recover: false,
        rms_normalize: document.getElementById('cfg-rms-normalize').checked,
        bypass_vad: document.getElementById('cfg-bypass-vad').checked,
    };
}

async function cancelProcess() {
    if (!currentFileId) return;
    try {
        await apiFetch(`/api/cancel/${currentFileId}`, { method: 'POST' });
        showToast('Đã hủy xử lý', 'success');
    } catch (e) {
        showToast('Lỗi: ' + e.message, 'error');
    }
}

function startActiveFileStatusPolling(fileId) {
    if (!fileId) return;
    if (activeFileStatusPollTimer && activeFileStatusPollFileId === fileId) return;
    stopActiveFileStatusPolling();
    activeFileStatusPollFileId = fileId;
    activeFileStatusPollTimer = setInterval(() => {
        pollActiveFileStatus(fileId);
    }, 2000);
    pollActiveFileStatus(fileId);
}

function stopActiveFileStatusPolling(fileId) {
    if (fileId && activeFileStatusPollFileId && activeFileStatusPollFileId !== fileId) return;
    if (activeFileStatusPollTimer) {
        clearInterval(activeFileStatusPollTimer);
    }
    activeFileStatusPollTimer = null;
    activeFileStatusPollFileId = null;
    activeFileStatusPollInFlight = false;
}

async function pollActiveFileStatus(fileId) {
    if (!fileId || fileId !== currentFileId || activeFileStatusPollInFlight) return;
    activeFileStatusPollInFlight = true;
    try {
        const status = await apiFetch(`/api/files/${fileId}/status`);
        await applyActiveFileStatus(status);
    } catch (e) {
        console.debug('Status poll failed:', e);
    } finally {
        activeFileStatusPollInFlight = false;
    }
}

async function applyActiveFileStatus(status) {
    if (!status || status.file_id !== currentFileId) return;

    if (status.status === 'queued') {
        if (status.queue_position > 0) {
            showQueuePosition(status.queue_position, status.queue_total);
        }
        showProcessProgress('File đã được đưa vào hàng đợi. Vui lòng đợi tới lượt xử lý.', 0, '');
        return;
    }

    if (status.status === 'processing') {
        const queue = document.getElementById('queue-info');
        const progress = document.getElementById('process-progress');
        const msgEl = document.getElementById('process-message');
        const msg = msgEl ? msgEl.textContent : '';
        if (queue) queue.style.display = 'none';
        if (!progress || progress.style.display === 'none' || msg.includes('hàng đợi') || msg.includes('gửi yêu cầu')) {
            showProcessProgress('Đang xử lý...', 0, '');
        }
        return;
    }

    if (status.status === 'completed') {
        stopActiveFileStatusPolling(status.file_id);
        const result = await apiFetch(`/api/files/${status.file_id}/result`);
        onASRComplete({ file_id: status.file_id, result });
        return;
    }

    if (status.status === 'error') {
        stopActiveFileStatusPolling(status.file_id);
        onASRError({
            file_id: status.file_id,
            error: 'Lỗi xử lý file. Vui lòng thử lại.',
        });
        return;
    }

    if (status.status === 'cancelled') {
        stopActiveFileStatusPolling(status.file_id);
        onASRCancelled({ file_id: status.file_id });
        return;
    }

    if (status.status === 'uploaded') {
        stopActiveFileStatusPolling(status.file_id);
        resetProcessUI();
    }
}

// === WS event handlers ===

function onQueuePosition(data) {
    if (data.file_id !== currentFileId) return;
    if (data.position > 0) {
        showQueuePosition(data.position, data.total);
        startActiveFileStatusPolling(currentFileId);
    } else if (data.position === 0) {
        document.getElementById('queue-info').style.display = 'none';
    }
}

function onProcessingStarted(data) {
    if (data.file_id !== currentFileId) return;
    startActiveFileStatusPolling(currentFileId);
    document.getElementById('queue-info').style.display = 'none';
    showProcessProgress('Bắt đầu xử lý...', 0, '');
}

function onProgress(data) {
    if (data.file_id !== currentFileId) return;
    // Hook summary progress
    if (data.phase === 'Summary' && typeof _summaryProgressHook === 'function') {
        _summaryProgressHook(data);
        return;
    }
    showProcessProgress(data.message, data.percent, data.phase || '');
}

function onASRComplete(data) {
    if (data.file_id !== currentFileId) return;
    stopActiveFileStatusPolling(data.file_id);
    if (currentProcessStartedAt) {
        data.result.timing = data.result.timing || {};
        data.result.timing.total = (performance.now() - currentProcessStartedAt) / 1000;
        currentProcessStartedAt = null;
    }
    currentASRData = data.result;
    renderASRResult(data.result);
    clearDirty();
    resetSpeakerEditHistory();
    resetProcessUI();
    loadAudio(currentFileId);
    showToast('Xử lý hoàn tất!', 'success');

    // Refresh meetings panel nếu đang mở
    const mp = document.getElementById('meetings-panel');
    if (mp && mp.style.display !== 'none') {
        loadMeetings();
    }
}

function onASRError(data) {
    if (data.file_id !== currentFileId) return;
    stopActiveFileStatusPolling(data.file_id);
    currentProcessStartedAt = null;
    showToast('Lỗi xử lý: ' + data.error, 'error');
    resetProcessUI();
}

function onASRCancelled(data) {
    if (data.file_id !== currentFileId) return;
    stopActiveFileStatusPolling(data.file_id);
    currentProcessStartedAt = null;
    showToast('Đã hủy xử lý', 'success');
    resetProcessUI();
}

function onQueueUpdated(data) {
    // Queue da thay doi, server se gui queue_position rieng neu can
}

function onSessionExpired(data) {
    showToast('Phiên làm việc đã hết hạn. Vui lòng tải lại trang.', 'error');
    // Disable UI
    document.getElementById('btn-process').disabled = true;
    document.getElementById('btn-cancel').disabled = true;
    const dropZone = document.getElementById('drop-zone');
    if (dropZone) dropZone.style.pointerEvents = 'none';
}

// === UI helpers ===

function showProcessProgress(message, percent, phase) {
    const container = document.getElementById('process-progress');
    const bar = document.getElementById('process-bar');
    const pctEl = document.getElementById('process-percent');
    const msgEl = document.getElementById('process-message');
    const phaseEl = document.getElementById('process-phase');
    if (container) container.style.display = 'flex';
    if (bar) bar.style.width = percent + '%';
    if (pctEl) pctEl.textContent = percent > 0 ? percent + '%' : '';
    if (msgEl) msgEl.textContent = message;
    if (phaseEl) phaseEl.textContent = phase || '';
}

function showQueuePosition(position, total) {
    document.getElementById('queue-info').style.display = 'flex';
    document.getElementById('queue-position').textContent = position;
    document.getElementById('queue-total').textContent = total;
}

function resetProcessUI() {
    const btnProcess = document.getElementById('btn-process');
    const btnCancel = document.getElementById('btn-cancel');
    const progress = document.getElementById('process-progress');
    const queue = document.getElementById('queue-info');
    if (btnProcess) btnProcess.disabled = false;
    if (btnCancel) btnCancel.style.display = 'none';
    if (progress) progress.style.display = 'none';
    if (queue) queue.style.display = 'none';
}

function hideResults() {
    document.getElementById('result-panel').style.display = 'none';
    currentASRData = null;
    resetSpeakerEditHistory();
    clearDirty();
    document.getElementById('btn-save-json').disabled = true;
    document.getElementById('btn-copy').disabled = true;
}

function cloneASRData(data) {
    if (!data) return null;
    if (typeof structuredClone === 'function') {
        return structuredClone(data);
    }
    return JSON.parse(JSON.stringify(data));
}

function resetSpeakerEditHistory() {
    speakerUndoStack = [];
    speakerRedoStack = [];
}

function pushSpeakerEditUndoState() {
    if (!currentASRData) return;
    speakerUndoStack.push(cloneASRData(currentASRData));
    if (speakerUndoStack.length > SPEAKER_EDIT_HISTORY_LIMIT) {
        speakerUndoStack.shift();
    }
    speakerRedoStack = [];
}

function restoreSpeakerEditState(state) {
    if (!state) return;
    currentASRData = cloneASRData(state);
    renderASRResult(currentASRData);
    markDirty();
}

function undoSpeakerEdit() {
    if (!speakerUndoStack.length || !currentASRData) return;
    speakerRedoStack.push(cloneASRData(currentASRData));
    restoreSpeakerEditState(speakerUndoStack.pop());
}

function redoSpeakerEdit() {
    if (!speakerRedoStack.length || !currentASRData) return;
    speakerUndoStack.push(cloneASRData(currentASRData));
    restoreSpeakerEditState(speakerRedoStack.pop());
}

function setupSpeakerEditShortcuts() {
    document.addEventListener('keydown', (event) => {
        const key = (event.key || '').toLowerCase();
        if (!event.ctrlKey || event.altKey || event.metaKey) return;
        if (event.target?.isContentEditable || ['INPUT', 'TEXTAREA', 'SELECT'].includes(event.target?.tagName)) return;
        if (key === 'z') {
            event.preventDefault();
            undoSpeakerEdit();
        } else if (key === 'y') {
            event.preventDefault();
            redoSpeakerEdit();
        }
    });
}

// === Render ASR result ===

function renderTextWithConfidence(text, rawWords) {
    /* Đổi màu chữ vùng nghi ngờ ASR lỗi sang cam kem nhẹ (#DFC8B0).
       Chỉ đổi color — không highlight, không border, không phân tâm. */
    if (!rawWords || rawWords.length === 0) return escapeHtml(text);

    const escaped = escapeHtml(text);
    const lower = escaped.toLowerCase();

    // Tìm vị trí từ nghi ngờ
    const positions = [];
    let searchPos = 0;
    for (const rw of rawWords) {
        const word = (rw.text || '').trim();
        if (!word) continue;
        const idx = lower.indexOf(word.toLowerCase(), searchPos);
        if (idx < 0) continue;

        const isSuspect = rw.suspect || rw.gap_after_ms || rw.gap_before_ms;
        if (isSuspect) {
            positions.push({start: idx, end: idx + word.length});
        }
        searchPos = idx + word.length;
    }

    if (positions.length === 0) return escaped;

    // Gom từ liền kề thành vùng
    const regions = [];
    let i = 0;
    while (i < positions.length) {
        let rStart = positions[i].start;
        let rEnd = positions[i].end;
        let j = i + 1;
        while (j < positions.length) {
            if (escaped.substring(rEnd, positions[j].start).trim().length === 0) {
                rEnd = positions[j].end;
                j++;
            } else {
                break;
            }
        }
        regions.push({start: rStart, end: rEnd});
        i = j;
    }

    // Build HTML — chỉ đổi color
    let result = '';
    let lastEnd = 0;
    for (const r of regions) {
        result += escaped.substring(lastEnd, r.start);
        const regionText = escaped.substring(r.start, r.end);
        result += `<span class="word-suspect">${regionText}</span>`;
        lastEnd = r.end;
    }
    result += escaped.substring(lastEnd);
    return result;
}

function renderASRResult(data) {
    currentASRData = data;
    const segments = data.segments || [];
    const speakerNames = data.speaker_names || {};
    const hasSpeakers = segments.some(s => s.type === 'speaker');

    // Set segments for player
    setPlayerSegments(segments);

    const contentEl = document.getElementById('result-content');

    if (hasSpeakers) {
        contentEl.innerHTML = renderSpeakerView(segments, speakerNames);
    } else {
        contentEl.innerHTML = renderPlainView(segments);
    }

    document.getElementById('result-panel').style.display = 'flex';
    document.getElementById('btn-save-json').disabled = false;
    document.getElementById('btn-copy').disabled = false;
    if (typeof syncDownloadAudioButton === 'function') syncDownloadAudioButton();

    // Render quality strip + timing info
    renderQualityStrip(data.quality_info);
    renderTimingInfo(data.timing);

    // Attach click handlers
    contentEl.querySelectorAll('.seg-span').forEach(span => {
        span.addEventListener('click', () => {
            const idx = parseInt(span.dataset.seg);
            seekToSegment(idx);
        });
    });

    // P1 XSS: Gắn event listener cho speaker-label qua addEventListener (không inline onclick)
    contentEl.querySelectorAll('.speaker-label[data-spk]').forEach(label => {
        label.addEventListener('click', () => {
            const spkId = parseInt(label.dataset.spk);
            const blockIdx = parseInt(label.dataset.blockIdx) || 0;
            ctxRenameSpeakerDirect(spkId, blockIdx);
        });
    });

    // Load summary if available
    if (typeof loadSummaryForFile === 'function' && currentFileId) {
        loadSummaryForFile(currentFileId);
    }
}

function _qColor(score, thresholds) {
    // thresholds: [[value, color], ...] descending
    for (const [v, c] of thresholds) { if (score >= v) return c; }
    return thresholds[thresholds.length - 1][1];
}

function renderQualityStrip(quality) {
    const strip = document.getElementById('quality-strip');
    if (!strip) return;

    if (!quality) {
        strip.style.display = 'none';
        return;
    }

    const dnsThresh = [[4.0, '#28a745'], [3.0, '#5cb85c'], [2.0, '#ffc107'], [0, '#dc3545']];
    const confThresh = [[0.80, '#28a745'], [0.60, '#ffc107'], [0, '#dc3545']];

    let items = [];

    if (quality.dnsmos_sig !== undefined) {
        const c = _qColor(quality.dnsmos_sig, dnsThresh);
        items.push(`<span class="qs-item" title="DNSMOS - Ch\u1ea5t l\u01b0\u1ee3ng gi\u1ecdng n\u00f3i">Gi\u1ecdng n\u00f3i <span class="qs-val" style="color:${c}">${quality.dnsmos_sig.toFixed(1)}/5</span></span>`);
    }
    if (quality.dnsmos_bak !== undefined) {
        const c = _qColor(quality.dnsmos_bak, dnsThresh);
        items.push(`<span class="qs-item" title="DNSMOS - Nhi\u1ec5u, vang n\u1ec1n">Nhi\u1ec5u n\u1ec1n <span class="qs-val" style="color:${c}">${quality.dnsmos_bak.toFixed(1)}/5</span></span>`);
    }
    if (quality.dnsmos_ovrl !== undefined) {
        const c = _qColor(quality.dnsmos_ovrl, dnsThresh);
        items.push(`<span class="qs-item" title="DNSMOS - Ch\u1ea5t l\u01b0\u1ee3ng t\u1ed5ng th\u1ec3">T\u1ed5ng th\u1ec3 <span class="qs-val" style="color:${c}">${quality.dnsmos_ovrl.toFixed(1)}/5</span></span>`);
    }
    if (quality.asr_confidence !== undefined) {
        const pct = (quality.asr_confidence * 100).toFixed(1);
        const c = _qColor(quality.asr_confidence, confThresh);
        items.push(`<span class="qs-item" title="M\u1ee9c \u0111\u1ed9 t\u1ef1 tin d\u1ecbch ch\u00ednh x\u00e1c c\u1ee7a m\u00f4 h\u00ecnh ASR">M\u1ee9c \u0111\u1ed9 t\u1ef1 tin d\u1ecbch ch\u00ednh x\u00e1c <span class="qs-val" style="color:${c}">${pct}%</span></span>`);
    }

    if (items.length === 0) {
        strip.style.display = 'none';
        return;
    }

    strip.style.display = 'flex';
    strip.innerHTML = `<span class="qs-label">Ch\u1ea5t l\u01b0\u1ee3ng:</span>${items.join('<span class="qs-sep">\u00b7</span>')}`;
}
function renderTimingInfo(timing) {
    const el = document.getElementById('result-timing');
    if (!el) return;

    if (!timing || Object.keys(timing).length === 0) {
        el.style.display = 'none';
        return;
    }

    const normalized = {
        preprocessing: Number(timing.preprocessing ?? timing.upload_convert ?? 0),
        transcription_detail: Number(timing.transcription_detail ?? timing.asr ?? timing.transcription ?? 0),
        diarization: Number(timing.diarization ?? 0),
        punctuation: Number(timing.punctuation ?? 0),
        total: Number(timing.total ?? 0),
    };
    const labels = {
        preprocessing: 'PreProcessing',
        transcription_detail: 'ASR',
        diarization: 'Ph\u00e2n t\u00e1ch ng\u01b0\u1eddi n\u00f3i',
        punctuation: 'D\u1ea5u c\u00e2u',
        total: 'T\u1ed5ng th\u1eddi gian',
    };

    let html = '';
    for (const [key, label] of Object.entries(labels)) {
        const value = normalized[key];
        if (Number.isFinite(value) && value > 0) {
            html += `
                <div class="timing-item">
                    <span class="timing-label">${label}:</span>
                    <span class="timing-value">${value.toFixed(1)}s</span>
                </div>
            `;
        }
    }

    el.innerHTML = html;
    el.style.display = html ? 'flex' : 'none';
}
function _safeColor(c) {
    if (!c) return '';
    // A03: Chỉ cho phép #hex, rgb(), rgba(), named colors — chặn CSS injection
    return /^(#[0-9a-fA-F]{3,8}|rgba?\(\s*\d{1,3}\s*,\s*\d{1,3}\s*,\s*\d{1,3}\s*(,\s*[\d.]+\s*)?\)|[a-zA-Z]{1,20})$/.test(c) ? c : '';
}

function renderSpeakerView(segments, speakerNames) {
    let html = '';
    let currentSpeaker = null;
    let currentBlock = [];
    let blockIdx = 0;
    let textIdx = 0;
    const speakerColors = (currentASRData && currentASRData.speaker_colors) || {};

    const flushBlock = () => {
        if (currentBlock.length === 0) return;
        const name = currentSpeaker ? (speakerNames[currentSpeaker] || `Người nói ${Number(currentSpeaker) + 1}`) : 'Unknown';
        const color = _safeColor(currentSpeaker ? (speakerColors[currentSpeaker] || '') : '');
        const borderStyle = color ? `border-left-color:${color}` : '';
        const labelStyle = color ? `color:${color}` : '';
        // P1 XSS: Không nội suy speaker_id vào inline onclick — dùng data-attribute
        // Event listener được gắn qua addEventListener sau khi render (xem initSpeakerLabelListeners)
        const safeSpkNum = parseInt(currentSpeaker) || 0; // speaker_id là số nguyên từ backend
        const safeBlockIdx = parseInt(blockIdx) || 0;
        html += `<div class="speaker-block" data-block="${safeBlockIdx}" data-speaker-id="${safeSpkNum}" style="${borderStyle}">`;
        html += `<div class="speaker-label" style="${labelStyle}" data-spk="${safeSpkNum}" data-block-idx="${safeBlockIdx}">${escapeHtml(name)}:</div>`;
        html += `<div class="speaker-text">${currentBlock.join(' ')}</div>`;
        html += `</div>`;
        blockIdx++;
        currentBlock = [];
    };

    for (const seg of segments) {
        if (seg.type === 'speaker') {
            flushBlock();
            currentSpeaker = seg.speaker_id !== undefined ? String(seg.speaker_id) : null;
        } else if (seg.type === 'text') {
            const text = seg.text || '';
            const rendered = renderTextWithConfidence(text, seg.raw_words);
            currentBlock.push(`<span class="seg-span" data-seg="${textIdx}">${rendered}</span>`);
            textIdx++;
        }
    }
    flushBlock();

    return html;
}

function renderPlainView(segments) {
    let html = '<div class="plain-text">';
    let textIdx = 0;
    for (const seg of segments) {
        if (seg.type === 'text') {
            const text = seg.text || '';
            const rendered = renderTextWithConfidence(text, seg.raw_words);
            html += `<span class="seg-span" data-seg="${textIdx}">${rendered}</span> `;
            textIdx++;
        }
    }
    html += '</div>';
    return html;
}

function renderRawSpeakers(segments, speakerNames) {
    const speakerColors = (currentASRData && currentASRData.speaker_colors) || {};
    let html = '<table style="width:100%;font-size:13px;"><tr><th>Time</th><th>Speaker</th><th>Text</th></tr>';
    let textIdx = 0;
    let currentSpeaker = '';
    let currentSpkId = '';

    for (const seg of segments) {
        if (seg.type === 'speaker') {
            currentSpkId = String(seg.speaker_id);
            currentSpeaker = speakerNames[currentSpkId] || `Người nói ${seg.speaker_id + 1}`;
        } else if (seg.type === 'text') {
            const time = formatTime(seg.start_time || 0);
            const color = _safeColor(speakerColors[currentSpkId] || '') || 'var(--accent)';
            html += `<tr><td style="color:var(--text-secondary)">${time}</td>`;
            html += `<td style="color:${color}">${escapeHtml(currentSpeaker)}</td>`;
            html += `<td>${escapeHtml(seg.text || '')}</td></tr>`;
            textIdx++;
        }
    }
    html += '</table>';
    return html;
}

// === Tab switching ===

function switchTab(tab) {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.getElementById('result-content').style.display = 'none';
    document.getElementById('result-summary').style.display = 'none';

    const tabMap = {
        content: { el: 'result-content', idx: 0 },
        summary: { el: 'result-summary', btn: 'tab-summary' },
    };
    const info = tabMap[tab];
    if (info) {
        document.getElementById(info.el).style.display = '';
        if (info.btn) {
            const btn = document.getElementById(info.btn);
            if (btn) btn.classList.add('active');
        } else {
            document.querySelectorAll('.tab-btn')[info.idx].classList.add('active');
        }
    }
}

// === JSON load/save ===

function loadJSON() {
    document.getElementById('json-input').click();
}

function formatCurrentASRTranscriptText() {
    if (!currentASRData) return '';
    const segments = currentASRData.segments || [];
    const speakerNames = currentASRData.speaker_names || {};
    const hasSpeakers = segments.some(s => s.type === 'speaker');

    if (!hasSpeakers) {
        return segments.filter(s => s.type === 'text').map(s => s.text || '').join(' ').trim();
    }

    let text = '';
    let currentSpeaker = '';
    let blockTexts = [];
    for (const seg of segments) {
        if (seg.type === 'speaker') {
            if (blockTexts.length) {
                text += currentSpeaker + ':\n' + blockTexts.join(' ') + '\n\n';
            }
            currentSpeaker = speakerNames[seg.speaker_id] || `Ng\u01b0\u1eddi n\u00f3i ${seg.speaker_id + 1}`;
            blockTexts = [];
        } else if (seg.type === 'text') {
            blockTexts.push(seg.text || '');
        }
    }
    if (blockTexts.length) {
        text += currentSpeaker + ':\n' + blockTexts.join(' ') + '\n';
    }
    return text.trim();
}

async function onJSONSelected(input) {
    if (!input.files.length || !currentFileId) {
        // Neu chua upload audio, upload truoc
        if (!currentFileId && uploadedFile) {
            try {
                await uploadFile();
            } catch (e) {
                // Thử tạo lại session rồi retry 1 lần
                try {
                    await fetch('/api/session', { method: 'POST', credentials: 'same-origin' });
                    await uploadFile();
                } catch (e2) {
                    showToast('Lỗi upload: ' + (e2.message || e.message), 'error');
                    return;
                }
            }
        }
        if (!currentFileId) {
            showToast('Vui lòng chọn file âm thanh trước', 'error');
            return;
        }
    }

    const jsonFile = input.files[0];
    const formData = new FormData();
    formData.append('file', jsonFile);

    try {
        await apiFetch(`/api/upload-json/${currentFileId}`, {
            method: 'POST',
            body: formData,
            rawBody: true,
        });

        // Load result
        const result = await apiFetch(`/api/files/${currentFileId}/result`);
        renderASRResult(result);
        clearDirty();
        resetSpeakerEditHistory();
        loadAudio(currentFileId);
        showToast('Đã load JSON', 'success');
    } catch (e) {
        showToast('Lỗi: ' + e.message, 'error');
    }

    if (input && input.value !== undefined) {
        input.value = '';
    }
}

async function saveJSON() {
    if (currentASRData) {
        const data = cloneASRData(currentASRData);
        data.transcript_text = formatCurrentASRTranscriptText();
        const blob = new Blob([JSON.stringify(data, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = (uploadedFile ? uploadedFile.name.replace(/\.[^.]+$/, '') : 'result') + '.asr.json';
        a.click();
        URL.revokeObjectURL(url);
        showToast('\u0110\u00e3 t\u1ea3i JSON', 'success');
        return;
    }
    if (!currentFileId) return;
    try {
        const resp = await fetch(`/api/files/${currentFileId}/download-json`, {
            headers: window.authToken ? { 'Authorization': 'Bearer ' + window.authToken } : {},
        });
        if (!resp.ok) throw new Error('Download failed');

        const blob = await resp.blob();
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = (uploadedFile ? uploadedFile.name.replace(/\.[^.]+$/, '') : 'result') + '.asr.json';
        a.click();
        URL.revokeObjectURL(url);
        showToast('Đã tải JSON', 'success');
    } catch (e) {
        showToast('Lỗi: ' + e.message, 'error');
    }
}

function triggerFileDownload(blob, filename) {
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename || 'audio';
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function filenameFromContentDisposition(header) {
    if (!header) return '';
    const utf8Match = header.match(/filename\*=UTF-8''([^;]+)/i);
    if (utf8Match) {
        try {
            return decodeURIComponent(utf8Match[1].trim());
        } catch (e) {
            return utf8Match[1].trim();
        }
    }
    const asciiMatch = header.match(/filename="?([^";]+)"?/i);
    return asciiMatch ? asciiMatch[1].trim() : '';
}

function currentAudioDownloadName() {
    if (uploadedFile?.name) return uploadedFile.name;
    const displayedName = document.getElementById('file-name')?.textContent?.trim();
    return displayedName || 'audio';
}

async function downloadAudio() {
    try {
        if (uploadedFile) {
            triggerFileDownload(uploadedFile, uploadedFile.name || 'audio');
            showToast('Đã tải audio', 'success');
            return;
        }

        if (!currentFileId) {
            showToast('Chưa có file audio để tải', 'error');
            return;
        }

        const resp = await fetch(`/api/files/${currentFileId}/download-audio`, {
            headers: window.authToken ? { 'Authorization': 'Bearer ' + window.authToken } : {},
            credentials: 'same-origin',
        });
        if (!resp.ok) {
            let msg = 'Download failed';
            try {
                msg = (await resp.text()) || msg;
            } catch (e) { }
            throw new Error(msg);
        }

        const blob = await resp.blob();
        const filename = filenameFromContentDisposition(resp.headers.get('Content-Disposition')) || currentAudioDownloadName();
        triggerFileDownload(blob, filename);
        showToast('Đã tải audio', 'success');
    } catch (e) {
        showToast('Lỗi: ' + e.message, 'error');
    }
}

function copyText() {
    if (!currentASRData) return;

    const segments = currentASRData.segments || [];
    const speakerNames = currentASRData.speaker_names || {};
    const hasSpeakers = segments.some(s => s.type === 'speaker');

    let text = '';
    if (hasSpeakers) {
        let currentSpeaker = '';
        let blockTexts = [];
        for (const seg of segments) {
            if (seg.type === 'speaker') {
                if (blockTexts.length) {
                    text += currentSpeaker + ':\n' + blockTexts.join(' ') + '\n\n';
                }
                currentSpeaker = speakerNames[seg.speaker_id] || `Người nói ${seg.speaker_id + 1}`;
                blockTexts = [];
            } else if (seg.type === 'text') {
                blockTexts.push(seg.text || '');
            }
        }
        if (blockTexts.length) {
            text += currentSpeaker + ':\n' + blockTexts.join(' ') + '\n';
        }
    } else {
        text = segments.filter(s => s.type === 'text').map(s => s.text || '').join(' ');
    }

    navigator.clipboard.writeText(text.trim()).then(() => {
        showToast(`Đã copy ${text.length} ký tự`, 'success');
    });
}

// === Auth ===

function showLoginModal() {
    document.getElementById('login-modal').style.display = 'flex';
    document.getElementById('login-username').focus();
    document.getElementById('login-error').style.display = 'none';
}

function hideLoginModal() {
    document.getElementById('login-modal').style.display = 'none';
}

async function doLogin() {
    const username = document.getElementById('login-username').value.trim();
    const password = document.getElementById('login-password').value;

    if (!username || !password) {
        document.getElementById('login-error').textContent = 'Vui lòng nhập đầy đủ';
        document.getElementById('login-error').style.display = 'block';
        return;
    }

    try {
        const resp = await apiFetch('/api/auth/login', {
            method: 'POST',
            body: JSON.stringify({ username, password }),
        });

        window.authToken = '__cookie__';
        currentFileId = null;
        if (uploadedFile) {
            hideResults();
            if (typeof loadLocalAudioPreview === 'function') loadLocalAudioPreview(uploadedFile);
            if (typeof syncDownloadAudioButton === 'function') syncDownloadAudioButton();
        } else {
            clearFile();
        }
        if (typeof reconnectWebSocket === 'function') reconnectWebSocket();
        showLoggedIn(resp.user);
        hideLoginModal();
        showToast(`Xin chào, ${resp.user.username}!`, 'success');
    } catch (e) {
        document.getElementById('login-error').textContent = e.message;
        document.getElementById('login-error').style.display = 'block';
    }
}

function showLoggedIn(user) {
    document.getElementById('auth-anonymous').style.display = 'none';
    document.getElementById('auth-loggedin').style.display = 'flex';
    document.getElementById('username-display').textContent = user.username;
    updateHeaderTitle(true);

    // Hien nut admin neu co quyen
    const btnAdmin = document.getElementById('btn-admin');
    if (btnAdmin) btnAdmin.style.display = (user.role === 'admin') ? '' : 'none';

    // Khôi phục trạng thái nếu đang xử lý
    restoreSessionState();
}

async function logout() {
    // P2: Gửi Authorization header để backend revoke JWT token ngay lập tức
    try {
        const headers = { 'Content-Type': 'application/json' };
        if (window.authToken) headers['Authorization'] = 'Bearer ' + window.authToken;
        await fetch('/api/auth/logout', { method: 'POST', credentials: 'same-origin', headers });
    } catch (e) {
        console.error('Logout API error:', e);
    }

    window.authToken = null;

    // Reset toàn bộ UI về trạng thái anonymous ban đầu
    document.getElementById('auth-anonymous').style.display = 'flex';
    document.getElementById('auth-loggedin').style.display = 'none';
    const btnAdmin = document.getElementById('btn-admin');
    if (btnAdmin) btnAdmin.style.display = 'none';
    closeAdminPanel();
    updateHeaderTitle(false);

    // Ẩn kết quả, progress, queue, player
    resetProcessUI();
    hideResults();
    hidePlayer();
    clearFile();

    // Đóng meetings panel nếu đang mở
    closeMeetingsPanel();

    // Reconnect WebSocket với session mới
    reconnectWebSocket();

    showToast('Đã đăng xuất', 'success');
}

// === User Menu ===

function toggleUserMenu(event) {
    event.stopPropagation();
    const dropdown = document.getElementById('user-dropdown');
    dropdown.style.display = dropdown.style.display === 'none' ? 'block' : 'none';
}

// Close dropdown when clicking outside
document.addEventListener('click', (e) => {
    const dropdown = document.getElementById('user-dropdown');
    if (dropdown && dropdown.style.display !== 'none') {
        const wrapper = e.target.closest('.user-menu-wrapper');
        if (!wrapper) {
            dropdown.style.display = 'none';
        }
    }
});

// === Change Password ===

function showChangePasswordModal() {
    document.getElementById('user-dropdown').style.display = 'none';
    document.getElementById('change-password-modal').style.display = 'flex';
    document.getElementById('chpw-old').value = '';
    document.getElementById('chpw-new').value = '';
    document.getElementById('chpw-confirm').value = '';
    document.getElementById('chpw-error').style.display = 'none';
    document.getElementById('chpw-old').focus();
}

function hideChangePasswordModal() {
    document.getElementById('change-password-modal').style.display = 'none';
}

async function doChangePassword() {
    const oldPw = document.getElementById('chpw-old').value;
    const newPw = document.getElementById('chpw-new').value;
    const confirmPw = document.getElementById('chpw-confirm').value;
    const errorEl = document.getElementById('chpw-error');

    if (!oldPw || !newPw) {
        errorEl.textContent = 'Vui lòng nhập đầy đủ';
        errorEl.style.display = 'block';
        return;
    }

    if (newPw !== confirmPw) {
        errorEl.textContent = 'Mật khẩu xác nhận không khớp';
        errorEl.style.display = 'block';
        return;
    }

    if (newPw.length < 8) {
        errorEl.textContent = 'Mật khẩu mới phải có ít nhất 8 ký tự';
        errorEl.style.display = 'block';
        return;
    }

    try {
        await apiFetch('/api/auth/change-password', {
            method: 'POST',
            body: JSON.stringify({ old_password: oldPw, new_password: newPw }),
        });
        hideChangePasswordModal();
        showToast('Đổi mật khẩu thành công!', 'success');
    } catch (e) {
        errorEl.textContent = e.message;
        errorEl.style.display = 'block';
    }
}

// === Speaker rename direct (click on label) ===

function ctxRenameSpeakerDirect(speakerId, blockIndex) {
    ctxSpeakerId = speakerId;
    ctxBlockIndex = blockIndex;
    ctxRenameSpeaker();
}

// === Session state restoration (anonymous + logged-in) ===

async function restoreSessionState() {
    try {
        const status = await apiFetch('/api/session/status');

        // Uu tien queue_item (dang waiting/processing)
        if (status.queue_item) {
            currentFileId = status.queue_item.file_id;
            if (typeof syncDownloadAudioButton === 'function') syncDownloadAudioButton();

            // Hien thi thong tin file
            document.querySelector('.drop-zone-text').style.display = 'none';
            document.getElementById('file-selected').style.display = 'flex';
            document.getElementById('file-name').textContent = status.queue_item.original_filename || 'File đang xử lý';
            document.getElementById('file-size').textContent = '';

            if (status.queue_item.status === 'processing') {
                document.getElementById('btn-process').disabled = true;
                document.getElementById('btn-cancel').style.display = '';
                showProcessProgress(
                    status.queue_item.progress_message || 'Đang xử lý...',
                    status.queue_item.progress_percent || 0, '');
                subscribeQueue(currentFileId);
                startActiveFileStatusPolling(currentFileId);
            } else if (status.queue_item.status === 'waiting') {
                document.getElementById('btn-process').disabled = true;
                document.getElementById('btn-cancel').style.display = '';
                subscribeQueue(currentFileId);
                startActiveFileStatusPolling(currentFileId);
            }
            return;
        }

        // Khong co queue active -> check file gan nhat da completed
        if (status.latest_file && status.latest_file.has_result) {
            currentFileId = status.latest_file.file_id;
            if (typeof syncDownloadAudioButton === 'function') syncDownloadAudioButton();

            // Hien thi ten file
            document.querySelector('.drop-zone-text').style.display = 'none';
            document.getElementById('file-selected').style.display = 'flex';
            document.getElementById('file-name').textContent = status.latest_file.original_filename;
            document.getElementById('file-size').textContent = '';

            // Load result JSON + audio
            try {
                const result = await apiFetch(`/api/files/${currentFileId}/result`);
                renderASRResult(result);
                clearDirty();
                resetSpeakerEditHistory();
                loadAudio(currentFileId);
            } catch (e) {
                console.error('Failed to load completed result:', e);
            }
        }
    } catch (e) {
        console.error('Failed to restore session state:', e);
    }
}

function loadAudioFromUrl(url) {
    // Load audio vào player từ URL (cho meetings)
    if (!audio) initPlayer();
    if (typeof revokeLocalPreviewAudioUrl === 'function') {
        revokeLocalPreviewAudioUrl();
    }
    const headers = {};
    if (window.authToken) {
        headers['Authorization'] = 'Bearer ' + window.authToken;
    }
    // Luu lai URL goc de co the reload khi mobile browser giai phong audio
    window._audioOriginalUrl = url;
    // Fetch as blob to include auth header
    fetch(url, { headers, credentials: 'same-origin' }).then(r => r.blob()).then(blob => {
        const blobUrl = URL.createObjectURL(blob);
        audio.src = blobUrl;
        lastAudioSrc = blobUrl;
        audio.load();
        document.getElementById('player-panel').style.display = 'flex';
    }).catch(e => console.error('Failed to load audio:', e));
}

// === API helper ===

async function apiFetch(url, options = {}) {
    const headers = { 'Content-Type': 'application/json' };
    if (window.authToken) {
        headers['Authorization'] = 'Bearer ' + window.authToken;
    }

    const fetchOptions = { method: options.method || 'GET', headers, credentials: 'same-origin' };

    if (options.body) {
        if (options.rawBody) {
            // FormData - don't set Content-Type
            delete fetchOptions.headers['Content-Type'];
            fetchOptions.body = options.body;
        } else {
            fetchOptions.body = options.body;
        }
    }

    const resp = await fetch(url, fetchOptions);

    if (!resp.ok) {
        // Token hết hạn → tự động logout
        if (resp.status === 401 && window.authToken) {
            window.authToken = null;
            document.getElementById('auth-anonymous').style.display = 'flex';
            document.getElementById('auth-loggedin').style.display = 'none';
            updateHeaderTitle(false);
        }
        let msg = `HTTP ${resp.status}`;
        try {
            const data = await resp.json();
            msg = data.detail || data.message || msg;
        } catch { }
        throw new Error(msg);
    }

    return resp.json();
}

// === Toast ===

function showToast(message, type = 'success') {
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 4000);
}

// === Dirty flag & Save ===

function markDirty() {
    _dirty = true;
    if (!window.authToken) return; // anonymous: no save needed
    const btn = document.getElementById('btn-save-edits');
    if (btn) btn.style.display = '';
}

function clearDirty() {
    _dirty = false;
    const btn = document.getElementById('btn-save-edits');
    if (btn) btn.style.display = 'none';
}

async function saveEdits() {
    if (!_dirty || !currentFileId || !currentASRData) return;

    const btn = document.getElementById('btn-save-edits');
    btn.disabled = true;
    btn.textContent = 'Đang lưu...';

    try {
        await apiFetch(`/api/files/${currentFileId}/save-result`, {
            method: 'POST',
            body: JSON.stringify({ asr_result: currentASRData }),
        });
        clearDirty();
        showToast('Đã lưu thay đổi', 'success');
    } catch (e) {
        showToast('Lỗi lưu: ' + e.message, 'error');
    } finally {
        btn.disabled = false;
        btn.textContent = 'Lưu thay đổi';
    }
}

// === Scroll to top ===

function scrollToTop() {
    window.scrollTo({ top: 0, behavior: 'instant' });
}

window.addEventListener('scroll', () => {
    const btn = document.getElementById('btn-scroll-top');
    if (btn) {
        btn.classList.toggle('visible', window.scrollY > 300);
    }
});

// === Unload Warning ===

window.addEventListener('beforeunload', function (e) {
    if (_dirty || !window.authToken) {
        e.preventDefault();
        e.returnValue = '';
    }
});
