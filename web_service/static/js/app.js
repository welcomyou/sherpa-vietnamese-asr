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
            '<button class="btn btn-sm" style="width:100%" onclick="this.closest(\'.ios-install-overlay\').remove()">Đã hiểu</button>' +
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
        html += '<button class="cert-tab' + active + '" onclick="switchCertTab(\'' + d.id + '\')">' + d.icon + ' ' + d.label + '</button>';
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
    document.querySelectorAll('.cert-tab').forEach((el) => {
        if (el.onclick && el.textContent.toLowerCase().includes(deviceId)) el.classList.add('active');
    });
    // Dung data attribute de match chinh xac
    document.querySelectorAll('.cert-tab').forEach((btn) => {
        if (btn.getAttribute('onclick') === "switchCertTab('" + deviceId + "')") btn.classList.add('active');
    });
}

// === App State ===

let currentASRData = null;
window.authToken = null;
window.appConfig = null;
let _dirty = false;

// === Init ===

document.addEventListener('DOMContentLoaded', async () => {
    initUpload();
    initPlayer();
    initContextMenu();

    // Tao session truoc (1 request duy nhat), cookie se duoc set tu response
    try {
        await fetch('/api/session', { method: 'POST', credentials: 'same-origin' });
    } catch (e) {
        console.error('Failed to create session:', e);
    }

    // Sau khi co session cookie, ket noi WebSocket va load config
    connectWebSocket();

    // Load config
    try {
        const [models, defaults] = await Promise.all([
            apiFetch('/api/config/models'),
            apiFetch('/api/config/defaults'),
        ]);

        window.appConfig = defaults;
        populateModels(models, defaults);
    } catch (e) {
        console.error('Failed to load config:', e);
    }

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

    // Check stored auth
    const storedToken = localStorage.getItem('authToken');
    if (storedToken) {
        window.authToken = storedToken;
        try {
            const me = await apiFetch('/api/auth/me');
            showLoggedIn(me);
        } catch {
            localStorage.removeItem('authToken');
            window.authToken = null;
            updateHeaderTitle(false);
        }
    } else {
        updateHeaderTitle(false);
    }

    // Restore session state cho moi user (anonymous + login)
    // Login da goi trong showLoggedIn, chi goi them cho anonymous
    if (!window.authToken) {
        restoreSessionState();
    }
});

function showAboutDialog() {
    document.getElementById('about-modal').style.display = 'flex';
}

function hideAboutDialog() {
    document.getElementById('about-modal').style.display = 'none';
}

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

    // Tu dong dieu chinh threshold khi doi speaker model (giong desktop)
    spkSelect.addEventListener('change', () => {
        const modelId = spkSelect.value;
        const newThreshold = window._speakerModelThresholds[modelId] || 70;
        document.getElementById('cfg-threshold').value = newThreshold;
        updateSliderLabels();
    });

    // Apply defaults
    document.getElementById('cfg-punct').value = defaults.punctuation_confidence;
    document.getElementById('cfg-case').value = defaults.case_confidence;
    document.getElementById('cfg-threshold').value = defaults.diarization_threshold;
    updateSliderLabels();
}

function initSliders() {
    const sliderLabels = {
        'cfg-punct': { el: 'cfg-punct-label', fmt: (v) => getConfLabel(v) + ` (${v})` },
        'cfg-case': { el: 'cfg-case-label', fmt: (v) => getConfLabel(v) + ` (${v})` },
        'cfg-threshold': { el: 'cfg-threshold-label', fmt: (v) => (v / 100).toFixed(2) },
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

    const t = document.getElementById('cfg-threshold').value;
    document.getElementById('cfg-threshold-label').textContent = (t / 100).toFixed(2);
}

function getConfLabel(v) {
    v = parseInt(v);
    if (v <= 2) return 'Rất ít';
    if (v <= 4) return 'Ít';
    if (v <= 6) return 'Vừa';
    if (v <= 8) return 'Nhiều';
    return 'Rất nhiều';
}

// === Panel toggle ===

function togglePanel(panelId) {
    document.getElementById(panelId).classList.toggle('collapsed');
}

// === Process file ===

async function processFile() {
    if (!uploadedFile) return;

    // User đã login → hỏi tên cuộc họp trước
    if (window.authToken) {
        // Upload file trước nếu chưa upload
        if (!currentFileId) {
            showProcessProgress('Đang tải file lên...', 0, '');
            try {
                await uploadFile();
            } catch (e) {
                showToast('Lỗi upload: ' + e.message, 'error');
                resetProcessUI();
                return;
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
            showProcessProgress('Đang tải file lên...', 0, '');
            await uploadFile();
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

        if (result.position > 0) {
            showQueuePosition(result.position, result.total);
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

// === WS event handlers ===

function onQueuePosition(data) {
    if (data.file_id !== currentFileId) return;
    if (data.position > 0) {
        showQueuePosition(data.position, data.total);
    } else if (data.position === 0) {
        document.getElementById('queue-info').style.display = 'none';
    }
}

function onProcessingStarted(data) {
    if (data.file_id !== currentFileId) return;
    document.getElementById('queue-info').style.display = 'none';
    showProcessProgress('Bắt đầu xử lý...', 0, '');
}

function onProgress(data) {
    if (data.file_id !== currentFileId) return;
    showProcessProgress(data.message, data.percent, data.phase || '');
}

function onASRComplete(data) {
    if (data.file_id !== currentFileId) return;
    currentASRData = data.result;
    renderASRResult(data.result);
    clearDirty();
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
    showToast('Lỗi xử lý: ' + data.error, 'error');
    resetProcessUI();
}

function onASRCancelled(data) {
    if (data.file_id !== currentFileId) return;
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
    document.getElementById('process-progress').style.display = 'flex';
    document.getElementById('process-bar').style.width = percent + '%';
    document.getElementById('process-percent').textContent = percent > 0 ? percent + '%' : '';
    document.getElementById('process-message').textContent = message;
    const phaseEl = document.getElementById('process-phase');
    if (phaseEl) phaseEl.textContent = phase || '';
}

function showQueuePosition(position, total) {
    document.getElementById('queue-info').style.display = 'flex';
    document.getElementById('queue-position').textContent = position;
    document.getElementById('queue-total').textContent = total;
}

function resetProcessUI() {
    document.getElementById('btn-process').disabled = false;
    document.getElementById('btn-cancel').style.display = 'none';
    document.getElementById('process-progress').style.display = 'none';
    document.getElementById('queue-info').style.display = 'none';
}

function hideResults() {
    document.getElementById('result-panel').style.display = 'none';
    currentASRData = null;
    clearDirty();
    document.getElementById('btn-save-json').disabled = true;
    document.getElementById('btn-copy').disabled = true;
}

// === Render ASR result ===

function renderASRResult(data) {
    currentASRData = data;
    const segments = data.segments || [];
    const speakerNames = data.speaker_names || {};
    const hasSpeakers = segments.some(s => s.type === 'speaker');

    // Set segments for player
    setPlayerSegments(segments);

    const contentEl = document.getElementById('result-content');
    const speakersEl = document.getElementById('result-speakers');

    if (hasSpeakers) {
        contentEl.innerHTML = renderSpeakerView(segments, speakerNames);
        speakersEl.innerHTML = renderRawSpeakers(segments, speakerNames);
    } else {
        contentEl.innerHTML = renderPlainView(segments);
        speakersEl.innerHTML = '<p style="color:var(--text-secondary)">Không có dữ liệu người nói</p>';
    }

    document.getElementById('result-panel').style.display = 'flex';
    document.getElementById('btn-save-json').disabled = false;
    document.getElementById('btn-copy').disabled = false;

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
        items.push(`<span class="qs-item" title="DNSMOS - Chất lượng giọng nói">Giọng nói <span class="qs-val" style="color:${c}">${quality.dnsmos_sig.toFixed(1)}/5</span></span>`);
    }
    if (quality.dnsmos_bak !== undefined) {
        const c = _qColor(quality.dnsmos_bak, dnsThresh);
        items.push(`<span class="qs-item" title="DNSMOS - Nhiễu, vang nền">Nhiễu nền <span class="qs-val" style="color:${c}">${quality.dnsmos_bak.toFixed(1)}/5</span></span>`);
    }
    if (quality.dnsmos_ovrl !== undefined) {
        const c = _qColor(quality.dnsmos_ovrl, dnsThresh);
        items.push(`<span class="qs-item" title="DNSMOS - Chất lượng tổng thể">Tổng thể <span class="qs-val" style="color:${c}">${quality.dnsmos_ovrl.toFixed(1)}/5</span></span>`);
    }
    if (quality.asr_confidence !== undefined) {
        const pct = (quality.asr_confidence * 100).toFixed(1);
        const c = _qColor(quality.asr_confidence, confThresh);
        items.push(`<span class="qs-item" title="Mức độ tự tin dịch chính xác của mô hình ASR">Mức độ tự tin dịch chính xác <span class="qs-val" style="color:${c}">${pct}%</span></span>`);
    }

    if (items.length === 0) {
        strip.style.display = 'none';
        return;
    }

    strip.style.display = 'flex';
    strip.innerHTML = `<span class="qs-label">Chất lượng:</span>${items.join('<span class="qs-sep">·</span>')}`;
}

function renderTimingInfo(timing) {
    const el = document.getElementById('result-timing');
    if (!el) return;

    if (!timing || Object.keys(timing).length === 0) {
        el.style.display = 'none';
        return;
    }

    el.style.display = 'flex';

    // Mapping keys to labels (from asr_engine.py timing_details)
    const labels = {
        'upload_convert': 'Tải audio',
        'transcription_detail': 'Chuyển văn bản',
        'punctuation': 'Dấu câu',
        'alignment': 'Căn chỉnh',
        'diarization': 'Phân tách người nói',
        'total': 'Tổng cộng'
    };

    let html = '';
    for (const [key, label] of Object.entries(labels)) {
        const value = timing[key];
        if (value !== undefined && value !== null && value > 0) {
            html += `
                <div class="timing-item">
                    <span class="timing-label">${label}:</span>
                    <span class="timing-value">${value.toFixed(1)}s</span>
                </div>
            `;
        }
    }

    // Neu chi co "total" thi hien thi total
    if (html === '' && timing.total) {
        html = `
            <div class="timing-item">
                <span class="timing-label">Tổng thời gian:</span>
                <span class="timing-value">${timing.total.toFixed(1)}s</span>
            </div>
        `;
    }

    el.innerHTML = html;
}

function renderSpeakerView(segments, speakerNames) {
    let html = '';
    let currentSpeaker = null;
    let currentBlock = [];
    let blockIdx = 0;
    let textIdx = 0;

    const flushBlock = () => {
        if (currentBlock.length === 0) return;
        const name = currentSpeaker ? (speakerNames[currentSpeaker] || `Speaker ${currentSpeaker}`) : 'Unknown';
        html += `<div class="speaker-block" data-block="${blockIdx}" data-speaker-id="${currentSpeaker || ''}">`;
        html += `<div class="speaker-label" onclick="ctxRenameSpeakerDirect('${currentSpeaker}', ${blockIdx})">${escapeHtml(name)}:</div>`;
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
            currentBlock.push(`<span class="seg-span" data-seg="${textIdx}">${escapeHtml(text)}</span>`);
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
            html += `<span class="seg-span" data-seg="${textIdx}">${escapeHtml(text)}</span> `;
            textIdx++;
        }
    }
    html += '</div>';
    return html;
}

function renderRawSpeakers(segments, speakerNames) {
    let html = '<table style="width:100%;font-size:13px;"><tr><th>Time</th><th>Speaker</th><th>Text</th></tr>';
    let textIdx = 0;
    let currentSpeaker = '';

    for (const seg of segments) {
        if (seg.type === 'speaker') {
            currentSpeaker = speakerNames[seg.speaker_id] || `Speaker ${seg.speaker_id}`;
        } else if (seg.type === 'text') {
            const time = formatTime(seg.start_time || 0);
            html += `<tr><td style="color:var(--text-secondary)">${time}</td>`;
            html += `<td style="color:var(--accent)">${escapeHtml(currentSpeaker)}</td>`;
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
    if (tab === 'content') {
        document.getElementById('result-content').style.display = '';
        document.getElementById('result-speakers').style.display = 'none';
        document.querySelectorAll('.tab-btn')[0].classList.add('active');
    } else {
        document.getElementById('result-content').style.display = 'none';
        document.getElementById('result-speakers').style.display = '';
        document.querySelectorAll('.tab-btn')[1].classList.add('active');
    }
}

// === JSON load/save ===

function loadJSON() {
    document.getElementById('json-input').click();
}

async function onJSONSelected(input) {
    if (!input.files.length || !currentFileId) {
        // Neu chua upload audio, upload truoc
        if (!currentFileId && uploadedFile) {
            try {
                await uploadFile();
            } catch (e) {
                showToast('Lỗi upload: ' + e.message, 'error');
                return;
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
                currentSpeaker = speakerNames[seg.speaker_id] || `Speaker ${seg.speaker_id}`;
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

        window.authToken = resp.token;
        localStorage.setItem('authToken', resp.token);
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

    // Khôi phục trạng thái nếu đang xử lý
    restoreSessionState();
}

async function logout() {
    // Gọi backend tạo session anonymous mới (tách khỏi session cũ đang processing)
    try {
        await fetch('/api/auth/logout', { method: 'POST', credentials: 'same-origin' });
    } catch (e) {
        console.error('Logout API error:', e);
    }

    window.authToken = null;
    localStorage.removeItem('authToken');

    // Reset toàn bộ UI về trạng thái anonymous ban đầu
    document.getElementById('auth-anonymous').style.display = 'flex';
    document.getElementById('auth-loggedin').style.display = 'none';
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

    if (newPw.length < 6) {
        errorEl.textContent = 'Mật khẩu mới phải có ít nhất 6 ký tự';
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
            } else if (status.queue_item.status === 'waiting') {
                document.getElementById('btn-process').disabled = true;
                document.getElementById('btn-cancel').style.display = '';
                subscribeQueue(currentFileId);
            }
            return;
        }

        // Khong co queue active -> check file gan nhat da completed
        if (status.latest_file && status.latest_file.has_result) {
            currentFileId = status.latest_file.file_id;

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
    const headers = {};
    if (window.authToken) {
        headers['Authorization'] = 'Bearer ' + window.authToken;
    }
    // Luu lai URL goc de co the reload khi mobile browser giai phong audio
    window._audioOriginalUrl = url;
    // Fetch as blob to include auth header
    fetch(url, { headers }).then(r => r.blob()).then(blob => {
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

    const fetchOptions = { method: options.method || 'GET', headers };

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
            localStorage.removeItem('authToken');
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
