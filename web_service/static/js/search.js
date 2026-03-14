/* Search text trong ket qua ASR */

let searchMatches = [];
let searchCurrentIdx = -1;
let searchDebounceTimer = null;

function onSearchInput() {
    clearTimeout(searchDebounceTimer);
    searchDebounceTimer = setTimeout(() => {
        performSearch();
    }, 300);
}

function performSearch() {
    const query = document.getElementById('search-input').value.trim().toLowerCase();

    searchMatches = [];
    searchCurrentIdx = -1;

    // Xoa highlight cu
    document.querySelectorAll('.search-match, .search-current-match').forEach(el => {
        const text = el.textContent;
        el.replaceWith(document.createTextNode(text));
    });

    if (!query || query.length < 1) {
        document.getElementById('search-count').textContent = '';
        return;
    }

    // Tim trong tat ca seg-span
    const spans = document.querySelectorAll('#result-content .seg-span');
    spans.forEach((span, spanIdx) => {
        const text = span.textContent;
        const lower = text.toLowerCase();
        let idx = 0;
        while ((idx = lower.indexOf(query, idx)) !== -1) {
            searchMatches.push({ spanIdx, charStart: idx, charEnd: idx + query.length });
            idx += query.length;
        }
    });

    if (searchMatches.length > 0) {
        searchCurrentIdx = 0;
        applySearchHighlight();
    }

    updateSearchCount();
}

function applySearchHighlight() {
    // Reset all highlights first by re-rendering
    // Thay vi re-render toan bo, ta dung mark approach
    const spans = document.querySelectorAll('#result-content .seg-span');

    // Xoa highlight cu
    spans.forEach(span => {
        const marks = span.querySelectorAll('.search-match, .search-current-match');
        marks.forEach(mark => {
            mark.replaceWith(document.createTextNode(mark.textContent));
        });
        // Normalize text nodes
        span.normalize();
    });

    // Ap dung highlight moi (tu cuoi ve dau de khong lech index)
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

        // Sort by charStart
        matches.sort((a, b) => a.charStart - b.charStart);

        for (const m of matches) {
            html += escapeHtml(text.slice(lastEnd, m.charStart));
            const cls = m.matchIdx === searchCurrentIdx ? 'search-current-match' : 'search-match';
            html += `<span class="${cls}">${escapeHtml(text.slice(m.charStart, m.charEnd))}</span>`;
            lastEnd = m.charEnd;
        }
        html += escapeHtml(text.slice(lastEnd));
        span.innerHTML = html;
    }

    // Scroll to current
    const current = document.querySelector('.search-current-match');
    if (current) {
        current.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }
}

function searchNav(direction) {
    if (searchMatches.length === 0) return;
    searchCurrentIdx = (searchCurrentIdx + direction + searchMatches.length) % searchMatches.length;
    applySearchHighlight();
    updateSearchCount();
}

function clearSearch() {
    document.getElementById('search-input').value = '';
    searchMatches = [];
    searchCurrentIdx = -1;
    document.getElementById('search-count').textContent = '';

    // Xoa highlight
    document.querySelectorAll('.search-match, .search-current-match').forEach(el => {
        el.replaceWith(document.createTextNode(el.textContent));
    });
}

function updateSearchCount() {
    const el = document.getElementById('search-count');
    if (searchMatches.length === 0) {
        el.textContent = document.getElementById('search-input').value ? '0/0' : '';
    } else {
        el.textContent = `${searchCurrentIdx + 1}/${searchMatches.length}`;
    }
}

function escapeHtml(str) {
    const div = document.createElement('div');
    div.textContent = str;
    return div.innerHTML;
}
