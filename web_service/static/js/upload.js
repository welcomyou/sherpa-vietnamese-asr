/* Upload file + progress */

let currentFileId = null;
let uploadedFile = null;

function initUpload() {
    const dropZone = document.getElementById('drop-zone');
    const fileInput = document.getElementById('file-input');

    // Click to select
    dropZone.addEventListener('click', (e) => {
        if (e.target.closest('.btn-clear')) return;
        fileInput.click();
    });

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
            selectFile(fileInput.files[0]);
        }
    });

    // Drag & drop
    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.classList.add('drag-over');
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.classList.remove('drag-over');
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        if (e.dataTransfer.files.length > 0) {
            selectFile(e.dataTransfer.files[0]);
        }
    });
}

function selectFile(file) {
    const allowedExts = ['mp3', 'wav', 'm4a', 'flac', 'aac', 'wma', 'ogg', 'opus',
        'mp4', 'mkv', 'avi', 'mov', 'webm', 'flv', 'wmv'];
    const ext = file.name.split('.').pop().toLowerCase();

    // Xử lý kéo thả file JSON
    if (ext === 'json') {
        if (!uploadedFile) {
            showToast('Vui lòng kéo thả file âm thanh trước', 'error');
            return;
        }

        // Kiểm tra tên file (bỏ qua extension .asr.json hoặc .json)
        const audioBaseName = uploadedFile.name.replace(/\.[^.]+$/, '');
        let jsonBaseName = file.name;
        if (jsonBaseName.endsWith('.asr.json')) {
            jsonBaseName = jsonBaseName.substring(0, jsonBaseName.length - 9); // Remove .asr.json
        } else {
            jsonBaseName = jsonBaseName.replace(/\.[^.]+$/, ''); // Remove .json
        }

        if (audioBaseName !== jsonBaseName) {
            showToast('Tên file JSON không khớp với file âm thanh đã chọn', 'error');
            return;
        }

        // Gọi hàm xử lý file JSON như khi click nút Tải JSON
        onJSONSelected({ files: [file], value: '' });
        return;
    }

    if (!allowedExts.includes(ext)) {
        showToast(`Định dạng .${ext} không được hỗ trợ`, 'error');
        return;
    }

    // Check size (client-side)
    const maxMB = window.appConfig?.max_upload_mb || 500;
    if (file.size > maxMB * 1024 * 1024) {
        showToast(`File quá lớn. Tối đa ${maxMB}MB`, 'error');
        return;
    }

    uploadedFile = file;

    // Hien thi ten file
    document.querySelector('.drop-zone-text').style.display = 'none';
    document.getElementById('file-selected').style.display = 'flex';
    document.getElementById('file-name').textContent = file.name;
    document.getElementById('file-size').textContent = formatSize(file.size);

    // Show anonymous warning if in anonymous mode
    const anonWarning = document.getElementById('anonymous-warning');
    if (anonWarning && !window.authToken) {
        anonWarning.style.display = 'block';
    }

    // Enable buttons
    document.getElementById('btn-process').disabled = false;
    document.getElementById('btn-load-json').disabled = false;

    // Reset state
    currentFileId = null;
    hideResults();
}

function clearFile() {
    uploadedFile = null;
    currentFileId = null;

    document.querySelector('.drop-zone-text').style.display = '';
    document.getElementById('file-selected').style.display = 'none';
    document.getElementById('file-input').value = '';

    // Hide anonymous warning
    const anonWarning = document.getElementById('anonymous-warning');
    if (anonWarning) {
        anonWarning.style.display = 'none';
    }

    document.getElementById('btn-process').disabled = true;
    document.getElementById('btn-cancel').style.display = 'none';
    document.getElementById('btn-load-json').disabled = true;
    document.getElementById('btn-save-json').disabled = true;
    document.getElementById('btn-copy').disabled = true;

    hideResults();
    hidePlayer();
}

async function uploadFile() {
    if (!uploadedFile) return null;

    const formData = new FormData();
    formData.append('file', uploadedFile);

    const progressContainer = document.getElementById('upload-progress');
    const progressBar = document.getElementById('upload-bar');
    const progressPercent = document.getElementById('upload-percent');

    progressContainer.style.display = 'flex';
    progressBar.style.width = '0%';

    return new Promise((resolve, reject) => {
        const xhr = new XMLHttpRequest();

        xhr.upload.onprogress = (e) => {
            if (e.lengthComputable) {
                const pct = Math.round((e.loaded / e.total) * 100);
                progressBar.style.width = pct + '%';
                progressPercent.textContent = pct + '%';
            }
        };

        xhr.onload = () => {
            progressContainer.style.display = 'none';
            if (xhr.status === 200) {
                const data = JSON.parse(xhr.responseText);
                currentFileId = data.file_id;
                resolve(data);
            } else {
                let msg = 'Upload thất bại';
                try { msg = JSON.parse(xhr.responseText).detail || msg; } catch (e) { }
                reject(new Error(msg));
            }
        };

        xhr.onerror = () => {
            progressContainer.style.display = 'none';
            reject(new Error('Lỗi kết nối'));
        };

        xhr.open('POST', '/api/upload');
        if (window.authToken) {
            xhr.setRequestHeader('Authorization', 'Bearer ' + window.authToken);
        }
        xhr.send(formData);
    });
}

function formatSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    if (bytes < 1024 * 1024) return (bytes / 1024).toFixed(1) + ' KB';
    return (bytes / (1024 * 1024)).toFixed(1) + ' MB';
}
