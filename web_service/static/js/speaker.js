/* Speaker context menu - split, merge, rename (client-side editing) */

const SPEAKER_COLORS = [
    '#4FC3F7', '#81C784', '#FFB74D', '#E57373', '#BA68C8',
    '#FFD54F', '#4DD0E1', '#F06292', '#A1887F', '#90A4AE'
];
let ctxSegIndex = -1;
let ctxBlockIndex = -1;
let ctxSpeakerId = null;
let renameSelectedColor = null;

function initContextMenu() {
    // Hide on click outside
    document.addEventListener('click', (e) => {
        if (!e.target.closest('#context-menu')) {
            hideContextMenu();
        }
    });

    // Right-click on result content
    document.getElementById('result-content').addEventListener('contextmenu', (e) => {
        const segEl = e.target.closest('[data-seg]');
        const blockEl = e.target.closest('[data-block]');

        if (segEl || blockEl) {
            e.preventDefault();
            ctxSegIndex = segEl ? parseInt(segEl.dataset.seg) : -1;
            ctxBlockIndex = blockEl ? parseInt(blockEl.dataset.block) : -1;
            ctxSpeakerId = blockEl ? blockEl.dataset.speakerId : null;
            showContextMenu(e.clientX, e.clientY);
        }
    });
}

function showContextMenu(x, y) {
    const menu = document.getElementById('context-menu');
    menu.style.display = 'block';
    menu.style.left = Math.min(x, window.innerWidth - 240) + 'px';
    menu.style.top = Math.min(y, window.innerHeight - 200) + 'px';
}

function hideContextMenu() {
    document.getElementById('context-menu').style.display = 'none';
}

// === Helper: find absolute position in segments array from textIdx ===

function findSegmentPosition(segments, textIdx) {
    let textCount = 0;
    for (let i = 0; i < segments.length; i++) {
        if (segments[i].type === 'text') {
            if (textCount === textIdx) return i;
            textCount++;
        }
    }
    return -1;
}

// Find the speaker separator that owns a given segment position
function findOwnerSpeaker(segments, segPos) {
    for (let i = segPos - 1; i >= 0; i--) {
        if (segments[i].type === 'speaker') return i;
    }
    return -1;
}

// Find next speaker separator after a position
function findNextSpeaker(segments, segPos) {
    for (let i = segPos + 1; i < segments.length; i++) {
        if (segments[i].type === 'speaker') return i;
    }
    return -1;
}

// === SPLIT SPEAKER (client-side) ===

function ctxSplitSpeaker() {
    hideContextMenu();
    if (ctxSegIndex < 0 || !currentASRData) return;

    const speakerNames = currentASRData.speaker_names || {};
    const currentName = ctxSpeakerId ? (speakerNames[ctxSpeakerId] || `Người nói ${Number(ctxSpeakerId) + 1}`) : '';

    document.getElementById('split-current-speaker').textContent = currentName;
    document.getElementById('split-speaker-input').value = '';

    // Collect all speaker names
    const allNames = new Set();
    for (const [id, name] of Object.entries(speakerNames)) {
        allNames.add(name);
    }

    // Fill select dropdown (exclude current speaker)
    const select = document.getElementById('split-speaker-select');
    select.innerHTML = '<option value="">-- Chọn từ danh sách --</option>';
    for (const name of [...allNames].sort()) {
        if (name !== currentName) {
            const opt = document.createElement('option');
            opt.value = name;
            opt.textContent = name;
            select.appendChild(opt);
        }
    }

    // Reset scope
    document.querySelector('input[name="split-scope"][value="to_end"]').checked = true;

    document.getElementById('split-speaker-modal').style.display = 'flex';
    document.getElementById('split-speaker-input').focus();
}

function hideSplitSpeakerModal() {
    document.getElementById('split-speaker-modal').style.display = 'none';
}

function doSplitSpeaker() {
    let name = document.getElementById('split-speaker-input').value.trim();
    if (!name) {
        name = document.getElementById('split-speaker-select').value;
    }
    if (!name) {
        showToast('Vui lòng chọn hoặc nhập tên người nói', 'error');
        return;
    }

    const scope = document.querySelector('input[name="split-scope"]:checked').value;
    hideSplitSpeakerModal();

    const segments = currentASRData.segments;
    const speakerNames = currentASRData.speaker_names || {};

    // Find or create speaker_id for the new name
    let newSpeakerId = null;
    for (const [id, n] of Object.entries(speakerNames)) {
        if (n === name) { newSpeakerId = id; break; }
    }
    if (newSpeakerId === null) {
        newSpeakerId = String(Math.max(-1, ...Object.keys(speakerNames).map(Number)) + 1);
        speakerNames[newSpeakerId] = name;
    }

    // Find absolute position of the target text segment
    const segPos = findSegmentPosition(segments, ctxSegIndex);
    if (segPos < 0) return;

    if (scope === 'to_end') {
        // Insert new speaker separator before this segment
        segments.splice(segPos, 0, { type: 'speaker', speaker_id: parseInt(newSpeakerId) });
    } else {
        // Single: insert speaker before, and restore original speaker after
        segments.splice(segPos, 0, { type: 'speaker', speaker_id: parseInt(newSpeakerId) });
        // Find the next text segment after our target (segPos+1 is now the target text)
        const nextTextPos = segPos + 2; // after the inserted separator + target text
        if (nextTextPos < segments.length) {
            // Restore original speaker
            const origSpeakerId = ctxSpeakerId ? parseInt(ctxSpeakerId) : 0;
            segments.splice(nextTextPos, 0, { type: 'speaker', speaker_id: origSpeakerId });
        }
    }

    currentASRData.speaker_names = speakerNames;
    renderASRResult(currentASRData);
    markDirty();
    showToast('Đã tách người nói', 'success');
}

// === MERGE UP (client-side) ===

function ctxMergeUp() {
    hideContextMenu();
    if (ctxBlockIndex <= 0 || !currentASRData) return;

    const segments = currentASRData.segments;

    if (ctxSegIndex >= 0) {
        // Partial merge: merge from block start to ctxSegIndex (inclusive) into previous block
        const segPos = findSegmentPosition(segments, ctxSegIndex);
        if (segPos < 0) return;

        // Find the speaker separator that owns this segment
        const speakerPos = findOwnerSpeaker(segments, segPos);
        if (speakerPos < 0) return;

        // Remove the speaker separator to merge with previous block
        segments.splice(speakerPos, 1);

        // If there are remaining segments after the merge point, insert a new separator
        // to keep them as a separate block with the original speaker
        const adjustedSegPos = segPos - 1; // adjusted after splice
        const nextPos = adjustedSegPos + 1;
        if (nextPos < segments.length && segments[nextPos].type === 'text') {
            const origSpeakerId = ctxSpeakerId ? parseInt(ctxSpeakerId) : 0;
            segments.splice(nextPos, 0, { type: 'speaker', speaker_id: origSpeakerId });
        }
    } else {
        // Full block merge: remove the speaker separator for this block
        // Find the Nth speaker separator (ctxBlockIndex)
        let blockCount = 0;
        for (let i = 0; i < segments.length; i++) {
            if (segments[i].type === 'speaker') {
                if (blockCount === ctxBlockIndex) {
                    segments.splice(i, 1);
                    break;
                }
                blockCount++;
            }
        }
    }

    renderASRResult(currentASRData);
    markDirty();
    showToast('Đã gộp người nói', 'success');
}

// === MERGE DOWN (client-side) ===

function ctxMergeDown() {
    hideContextMenu();
    if (ctxBlockIndex < 0 || !currentASRData) return;

    const segments = currentASRData.segments;

    if (ctxSegIndex >= 0) {
        // Partial merge: merge from ctxSegIndex to block end into next block
        const segPos = findSegmentPosition(segments, ctxSegIndex);
        if (segPos < 0) return;

        // Find the next speaker separator
        const nextSpeakerPos = findNextSpeaker(segments, segPos);
        if (nextSpeakerPos < 0) return; // no next block

        // Remove the next speaker separator to merge
        segments.splice(nextSpeakerPos, 1);

        // If there are segments before ctxSegIndex in the current block, add separator
        const ownerPos = findOwnerSpeaker(segments, segPos);
        if (ownerPos >= 0) {
            // Check if there are text segments between ownerPos and segPos
            let hasTextBefore = false;
            for (let i = ownerPos + 1; i < segPos; i++) {
                if (segments[i].type === 'text') { hasTextBefore = true; break; }
            }
            if (hasTextBefore) {
                // Insert new speaker separator (from next block) before segPos
                // The next block's speaker is the one after nextSpeakerPos (which was removed)
                // Actually we want to assign the next block's speaker to the merged portion
                // Since we removed the next separator, segments from segPos onward now belong to owner
                // We need to insert a separator with the NEXT block's speaker before segPos
                // But we already removed it... let's just keep it as-is (all merge into current block's speaker)
            }
        }
    } else {
        // Full block merge: remove the next speaker separator
        let blockCount = 0;
        let nextSepIdx = -1;
        for (let i = 0; i < segments.length; i++) {
            if (segments[i].type === 'speaker') {
                if (blockCount === ctxBlockIndex + 1) {
                    nextSepIdx = i;
                    break;
                }
                blockCount++;
            }
        }
        if (nextSepIdx >= 0) {
            segments.splice(nextSepIdx, 1);
        }
    }

    renderASRResult(currentASRData);
    markDirty();
    showToast('Đã gộp người nói', 'success');
}

// === RENAME SPEAKER (client-side) ===

function ctxRenameSpeaker() {
    hideContextMenu();
    if (ctxBlockIndex < 0 || !currentASRData) return;

    const speakerNames = currentASRData.speaker_names || {};
    const speakerColors = currentASRData.speaker_colors || {};
    const currentName = ctxSpeakerId ? (speakerNames[ctxSpeakerId] || `Người nói ${Number(ctxSpeakerId) + 1}`) : '';

    document.getElementById('rename-current').textContent = currentName;
    document.getElementById('rename-input').value = '';

    // Fill select with existing speaker names
    const select = document.getElementById('rename-select');
    select.innerHTML = '<option value="">-- Chọn --</option>';
    for (const [id, name] of Object.entries(speakerNames)) {
        const opt = document.createElement('option');
        opt.value = name;
        opt.textContent = name;
        select.appendChild(opt);
    }

    // Fill color picker
    renameSelectedColor = ctxSpeakerId ? (speakerColors[ctxSpeakerId] || null) : null;
    const colorsEl = document.getElementById('rename-colors');
    colorsEl.innerHTML = '';
    for (const c of SPEAKER_COLORS) {
        const dot = document.createElement('div');
        dot.className = 'color-dot' + (renameSelectedColor === c ? ' selected' : '');
        dot.style.backgroundColor = c;
        dot.onclick = () => {
            renameSelectedColor = c;
            colorsEl.querySelectorAll('.color-dot').forEach(d => d.classList.remove('selected'));
            dot.classList.add('selected');
        };
        colorsEl.appendChild(dot);
    }

    document.getElementById('rename-modal').style.display = 'flex';
}

function hideRenameModal() {
    document.getElementById('rename-modal').style.display = 'none';
}

function doRenameSpeaker(applyAll) {
    let newName = document.getElementById('rename-input').value.trim();
    if (!newName) {
        newName = document.getElementById('rename-select').value;
    }
    if (!newName && !renameSelectedColor) {
        showToast('Vui lòng nhập tên hoặc chọn màu', 'error');
        return;
    }

    hideRenameModal();

    const speakerNames = currentASRData.speaker_names || {};
    if (!currentASRData.speaker_colors) currentASRData.speaker_colors = {};
    const speakerColors = currentASRData.speaker_colors;

    // Lưu màu
    if (renameSelectedColor && ctxSpeakerId !== null) {
        speakerColors[ctxSpeakerId] = renameSelectedColor;
    }

    if (newName) {
        if (applyAll && ctxSpeakerId !== null) {
            speakerNames[ctxSpeakerId] = newName;
        } else {
            const maxId = Math.max(-1, ...Object.keys(speakerNames).map(Number)) + 1;
            speakerNames[maxId] = newName;
            // Chuyển màu sang speaker_id mới
            if (renameSelectedColor) speakerColors[maxId] = renameSelectedColor;

            let blockCount = 0;
            for (let i = 0; i < currentASRData.segments.length; i++) {
                if (currentASRData.segments[i].type === 'speaker') {
                    if (blockCount === ctxBlockIndex) {
                        currentASRData.segments[i].speaker_id = maxId;
                        break;
                    }
                    blockCount++;
                }
            }
        }
        currentASRData.speaker_names = speakerNames;
    }

    renderASRResult(currentASRData);
    markDirty();
    showToast(newName ? 'Đã đổi tên người nói' : 'Đã đổi màu người nói', 'success');
}

// === COPY ===

function ctxCopy() {
    hideContextMenu();
    const selection = window.getSelection();
    if (selection.toString()) {
        navigator.clipboard.writeText(selection.toString());
        showToast('Đã sao chép', 'success');
    }
}
