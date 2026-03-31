/* Search text trong ket qua ASR - cross-segment + fuzzy (Vietnamese accent-insensitive) */

let searchMatches = [];
let searchCurrentIdx = -1;   // Logical index (cross-segment = 1 logical match)
let searchTotalLogical = 0;
let searchDebounceTimer = null;

function onSearchInput() {
    clearTimeout(searchDebounceTimer);
    searchDebounceTimer = setTimeout(() => {
        performSearch();
    }, 300);
}

function onSearchKeydown(e) {
    if (e.key === 'Enter') {
        e.preventDefault();
        if (searchMatches.length === 0) {
            // Chưa có kết quả → search ngay (bỏ debounce)
            clearTimeout(searchDebounceTimer);
            performSearch();
        } else {
            searchNav(e.shiftKey ? -1 : 1);
        }
    }
}

// === Vietnamese normalization (port from core/utils.py) ===

function normalizeVietnamese(text) {
    if (!text) return '';
    text = text.toLowerCase();
    text = text.replace(/đ/g, 'd');
    // NFD decompose then remove combining marks (diacritics)
    text = text.normalize('NFD').replace(/[\u0300-\u036f]/g, '');
    return text;
}

function mapNormToOrig(original, normIdx) {
    /**
     * Map vị trí trong chuỗi normalized về vị trí trong chuỗi gốc.
     * Cần vì chuỗi gốc có dấu (ế = e + combining) dài hơn chuỗi normalized.
     */
    if (normIdx <= 0) return 0;

    let baseCount = 0;
    for (let i = 0; i < original.length; i++) {
        const c = original[i];
        if (c.toLowerCase() === 'đ') {
            baseCount++;
            if (baseCount > normIdx) return i;
            continue;
        }
        const decomposed = c.normalize('NFD');
        // Ký tự đầu tiên của decomposed không phải combining mark → là base char
        const firstCode = decomposed.charCodeAt(0);
        const isBase = decomposed.length === 0 || !(firstCode >= 0x0300 && firstCode <= 0x036f);
        if (decomposed.length === 0 || isBase) {
            baseCount++;
        }
        if (baseCount > normIdx) return i;
    }
    return original.length;
}

// === Main search ===

function performSearch() {
    const query = document.getElementById('search-input').value.trim();

    searchMatches = [];
    searchCurrentIdx = -1;

    // Xóa highlight cũ
    clearHighlights();

    if (!query || query.length < 1) {
        document.getElementById('search-count').textContent = '';
        return;
    }

    const queryLower = query.toLowerCase();
    const queryNorm = normalizeVietnamese(query);

    // Lấy tất cả seg-span và build concatenated text
    const spans = document.querySelectorAll('#result-content .seg-span');
    if (spans.length === 0) {
        updateSearchCount();
        return;
    }

    const segBoundaries = [];
    const parts = [];
    let globalPos = 0;

    spans.forEach((span, idx) => {
        const text = span.textContent;
        if (!text) return;

        segBoundaries.push({
            globalStart: globalPos,
            globalEnd: globalPos + text.length,
            spanIdx: idx,
            text: text,
        });

        parts.push(text);
        globalPos += text.length + 1; // +1 cho space separator
        parts.push(' ');
    });

    const concatenated = parts.join('');
    const concatenatedLower = concatenated.toLowerCase();
    const concatenatedNorm = normalizeVietnamese(concatenated);

    // Helper: map global position → (spanIdx, localPos)
    function mapGlobalToSeg(gPos) {
        for (const b of segBoundaries) {
            if (b.globalStart <= gPos && gPos < b.globalEnd) {
                return { spanIdx: b.spanIdx, local: gPos - b.globalStart };
            }
        }
        // Vị trí cuối segment
        for (const b of segBoundaries) {
            if (gPos === b.globalEnd) {
                return { spanIdx: b.spanIdx, local: gPos - b.globalStart };
            }
        }
        return null;
    }

    // Helper: thêm match (single hoặc cross-segment)
    function addMatch(startGlobal, matchLen, score) {
        const endGlobal = startGlobal + matchLen;
        const startSeg = mapGlobalToSeg(startGlobal);
        const endSeg = mapGlobalToSeg(endGlobal - 1);

        if (!startSeg) return;

        if (!endSeg || startSeg.spanIdx === endSeg.spanIdx) {
            // Match trong 1 segment
            const segText = segBoundaries.find(b => b.spanIdx === startSeg.spanIdx)?.text || '';
            const endPos = Math.min(startSeg.local + matchLen, segText.length);
            searchMatches.push({
                spanIdx: startSeg.spanIdx,
                charStart: startSeg.local,
                charEnd: endPos,
                score: score,
            });
        } else {
            // Match span qua nhiều segment
            const firstBoundary = segBoundaries.find(b => b.spanIdx === startSeg.spanIdx);
            const firstText = firstBoundary?.text || '';
            const firstMatchLen = firstText.length - startSeg.local;

            searchMatches.push({
                spanIdx: startSeg.spanIdx,
                charStart: startSeg.local,
                charEnd: firstText.length,
                score: score,
                spansToNext: true,
            });

            // Phần còn lại ở segment thứ 2
            const remaining = matchLen - firstMatchLen - 1; // -1 cho space separator
            const secondText = segBoundaries.find(b => b.spanIdx === endSeg.spanIdx)?.text || '';
            const secondEnd = Math.min(Math.max(0, remaining), secondText.length);
            if (secondEnd > 0) {
                searchMatches.push({
                    spanIdx: endSeg.spanIdx,
                    charStart: 0,
                    charEnd: secondEnd,
                    score: score,
                    continuedFromPrev: true,
                });
            }
        }
    }

    // Helper: thêm normalized match (cần map vị trí norm → orig)
    function addNormMatch(normStartGlobal, normMatchLen, score) {
        const normEndGlobal = normStartGlobal + normMatchLen;
        const startSeg = mapGlobalToSeg(normStartGlobal);
        const endSeg = mapGlobalToSeg(normEndGlobal - 1);

        if (!startSeg) return;

        const firstBoundary = segBoundaries.find(b => b.spanIdx === startSeg.spanIdx);
        if (!firstBoundary) return;
        const segText = firstBoundary.text;
        const origStart = mapNormToOrig(segText, startSeg.local);

        // Kiểm tra trùng với exact match
        for (const existing of searchMatches) {
            if (existing.spanIdx === startSeg.spanIdx && Math.abs(existing.charStart - origStart) < 2) {
                return; // Đã tìm thấy bằng exact search
            }
        }

        if (origStart >= segText.length) return;

        if (!endSeg || startSeg.spanIdx === endSeg.spanIdx) {
            // Match trong 1 segment
            const origEnd = Math.min(mapNormToOrig(segText, startSeg.local + normMatchLen), segText.length);
            searchMatches.push({
                spanIdx: startSeg.spanIdx,
                charStart: origStart,
                charEnd: origEnd,
                score: score,
            });
        } else {
            // Cross-segment normalized match
            searchMatches.push({
                spanIdx: startSeg.spanIdx,
                charStart: origStart,
                charEnd: segText.length,
                score: score,
                spansToNext: true,
            });

            const secondBoundary = segBoundaries.find(b => b.spanIdx === endSeg.spanIdx);
            if (secondBoundary) {
                const segText2 = secondBoundary.text;
                const firstNormLen = firstBoundary.globalEnd - (firstBoundary.globalStart + startSeg.local);
                const remainingNorm = normMatchLen - firstNormLen - 1; // -1 cho space
                const origEnd2 = Math.min(mapNormToOrig(segText2, Math.max(0, remainingNorm)), segText2.length);
                if (origEnd2 > 0) {
                    searchMatches.push({
                        spanIdx: endSeg.spanIdx,
                        charStart: 0,
                        charEnd: origEnd2,
                        score: score,
                        continuedFromPrev: true,
                    });
                }
            }
        }
    }

    // Pass 1: Exact case-insensitive search trên concatenated text
    let idx = 0;
    while ((idx = concatenatedLower.indexOf(queryLower, idx)) !== -1) {
        addMatch(idx, query.length, 1.0);
        idx += 1;
    }

    // Pass 2: Normalized (accent-insensitive) search
    idx = 0;
    while ((idx = concatenatedNorm.indexOf(queryNorm, idx)) !== -1) {
        addNormMatch(idx, queryNorm.length, 0.9);
        idx += 1;
    }

    // Sort theo thứ tự xuất hiện
    searchMatches.sort((a, b) => a.spanIdx - b.spanIdx || a.charStart - b.charStart);

    // Gán logicalIdx: cross-segment match (continuedFromPrev) chung logicalIdx với phần trước
    let logIdx = 0;
    searchMatches.forEach((m, i) => {
        if (m.continuedFromPrev) {
            m.logicalIdx = searchMatches[i - 1].logicalIdx;
        } else {
            m.logicalIdx = logIdx++;
        }
    });
    searchTotalLogical = logIdx;

    if (searchTotalLogical > 0) {
        searchCurrentIdx = 0;
        applySearchHighlight();
    }

    updateSearchCount();
}

// === Highlight ===

function clearHighlights() {
    document.querySelectorAll('.search-match, .search-current-match').forEach(el => {
        el.replaceWith(document.createTextNode(el.textContent));
    });
    // Normalize text nodes sau khi xóa highlight
    document.querySelectorAll('#result-content .seg-span').forEach(span => span.normalize());
}

function applySearchHighlight() {
    const spans = document.querySelectorAll('#result-content .seg-span');

    // Xóa highlight cũ
    spans.forEach(span => {
        const marks = span.querySelectorAll('.search-match, .search-current-match');
        marks.forEach(mark => mark.replaceWith(document.createTextNode(mark.textContent)));
        span.normalize();
    });

    // Group matches theo spanIdx
    const matchesBySpan = {};
    searchMatches.forEach((m, mIdx) => {
        if (!matchesBySpan[m.spanIdx]) matchesBySpan[m.spanIdx] = [];
        matchesBySpan[m.spanIdx].push({ ...m, matchIdx: mIdx });
    });

    for (const [spanIdx, matches] of Object.entries(matchesBySpan)) {
        const span = spans[parseInt(spanIdx)];
        if (!span) continue;

        const text = span.textContent;
        let html = '';
        let lastEnd = 0;

        // Sort by charStart, merge overlapping
        matches.sort((a, b) => a.charStart - b.charStart);

        for (const m of matches) {
            if (m.charStart < lastEnd) continue; // Bỏ qua overlap
            html += escapeHtml(text.slice(lastEnd, m.charStart));
            const cls = m.logicalIdx === searchCurrentIdx ? 'search-current-match' : 'search-match';
            html += `<span class="${cls}">${escapeHtml(text.slice(m.charStart, m.charEnd))}</span>`;
            lastEnd = m.charEnd;
        }
        html += escapeHtml(text.slice(lastEnd));
        span.innerHTML = html;
    }

    // Scroll to current match
    const current = document.querySelector('.search-current-match');
    if (current) {
        current.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
}

// === Navigation ===

function searchNav(direction) {
    if (searchTotalLogical === 0) return;
    searchCurrentIdx = (searchCurrentIdx + direction + searchTotalLogical) % searchTotalLogical;
    applySearchHighlight();
    updateSearchCount();
}

function clearSearch() {
    document.getElementById('search-input').value = '';
    searchMatches = [];
    searchCurrentIdx = -1;
    searchTotalLogical = 0;
    document.getElementById('search-count').textContent = '';
    clearHighlights();
}

function updateSearchCount() {
    const el = document.getElementById('search-count');
    if (searchTotalLogical === 0) {
        el.textContent = document.getElementById('search-input').value ? '0/0' : '';
    } else {
        el.textContent = `${searchCurrentIdx + 1}/${searchTotalLogical}`;
    }
}

function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}
