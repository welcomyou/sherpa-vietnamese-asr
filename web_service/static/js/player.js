/* Audio player - play, seek, highlight */

let audio = null;
let playerSegments = [];  // segments voi timestamp
let currentHighlightIdx = -1;
let isUserSeeking = false;
let lastAudioSrc = '';  // luu lai src de reload khi mobile browser giai phong audio

function initPlayer() {
    audio = new Audio();
    audio.addEventListener('timeupdate', onTimeUpdate);
    audio.addEventListener('ended', () => {
        showPlayIcon(true);
    });
    audio.addEventListener('loadedmetadata', () => {
        document.getElementById('player-seek').max = audio.duration;
        updateTimeDisplay();
    });

    // Mobile browser giai phong audio khi an app - reload khi quay lai
    document.addEventListener('visibilitychange', () => {
        if (!document.hidden && audio && lastAudioSrc && audio.readyState === 0) {
            reloadAudio();
        }
    });
}

function reloadAudio(onReady) {
    // Neu la blob URL bi mat, fetch lai tu URL goc
    if (window._audioOriginalUrl && (!lastAudioSrc || lastAudioSrc.startsWith('blob:'))) {
        const headers = {};
        if (window.authToken) {
            headers['Authorization'] = 'Bearer ' + window.authToken;
        }
        fetch(window._audioOriginalUrl, { headers })
            .then(r => r.blob())
            .then(blob => {
                lastAudioSrc = URL.createObjectURL(blob);
                _doReloadAudio(onReady);
            })
            .catch(e => console.error('Failed to reload audio:', e));
        return;
    }
    if (!lastAudioSrc) return;
    _doReloadAudio(onReady);
}

function _doReloadAudio(onReady) {
    const savedTime = audio.currentTime || 0;
    audio.src = lastAudioSrc;
    audio.load();
    audio.addEventListener('loadedmetadata', function onReloadMeta() {
        audio.removeEventListener('loadedmetadata', onReloadMeta);
        if (savedTime > 0 && savedTime < audio.duration) {
            audio.currentTime = savedTime;
        }
        updateTimeDisplay();
        if (onReady) onReady();
    });
}

function loadAudio(fileId) {
    if (!audio) initPlayer();
    lastAudioSrc = `/api/files/${fileId}/audio`;
    audio.src = lastAudioSrc;
    audio.load();
    document.getElementById('player-panel').style.display = 'flex';
}

function hidePlayer() {
    document.getElementById('player-panel').style.display = 'none';
    if (audio) {
        audio.pause();
        audio.src = '';
    }
    lastAudioSrc = '';
    currentHighlightIdx = -1;
}

function togglePlay() {
    if (!audio || !audio.src) return;
    if (audio.paused) {
        // Mobile browser da giai phong audio - reload roi play
        if (audio.readyState === 0 && (lastAudioSrc || window._audioOriginalUrl)) {
            reloadAudio(() => {
                audio.play();
                showPlayIcon(false);
            });
            return;
        }
        audio.play();
        showPlayIcon(false);
    } else {
        audio.pause();
        showPlayIcon(true);
    }
}

function seekAudio(value) {
    if (audio) {
        isUserSeeking = true;
        audio.currentTime = parseFloat(value);
        setTimeout(() => { isUserSeeking = false; }, 200);
    }
}

function seekToSegment(segIndex) {
    if (!audio || segIndex < 0 || segIndex >= playerSegments.length) return;
    const seg = playerSegments[segIndex];
    if (seg && seg.start_time !== undefined) {
        // Mobile browser da giai phong audio - can reload truoc
        if (audio.readyState === 0 && (lastAudioSrc || window._audioOriginalUrl)) {
            reloadAudio(() => {
                audio.currentTime = seg.start_time;
                updateTimeDisplay();
                highlightSegment(segIndex);
            });
            return;
        }
        audio.currentTime = seg.start_time;
        updateTimeDisplay();
        highlightSegment(segIndex);
    }
}

function onTimeUpdate() {
    if (!audio || isUserSeeking) return;

    // Update seek slider
    const slider = document.getElementById('player-seek');
    slider.value = audio.currentTime;
    updateTimeDisplay();

    // Auto-highlight
    const segIdx = findSegmentAtTime(audio.currentTime);
    if (segIdx >= 0 && segIdx !== currentHighlightIdx) {
        highlightSegment(segIdx);
    }
}

function findSegmentAtTime(time) {
    let best = -1;
    for (let i = 0; i < playerSegments.length; i++) {
        const seg = playerSegments[i];
        if (seg.start_time !== undefined && seg.start_time <= time) {
            best = i;
        }
    }
    return best;
}

function highlightSegment(segIndex) {
    // Xoa highlight cu
    document.querySelectorAll('.seg-highlight').forEach(el => {
        el.classList.remove('seg-highlight');
    });

    currentHighlightIdx = segIndex;

    // Them highlight moi
    const el = document.querySelector(`[data-seg="${segIndex}"]`);
    if (el) {
        el.classList.add('seg-highlight');
        // Scroll to view
        el.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
}

function updateTimeDisplay() {
    if (!audio) return;
    const current = formatTime(audio.currentTime);
    const total = formatTime(audio.duration || 0);
    document.getElementById('player-time').textContent = `${current} / ${total}`;
}

function formatTime(sec) {
    if (isNaN(sec)) return '00:00';
    const m = Math.floor(sec / 60);
    const s = Math.floor(sec % 60);
    return `${m.toString().padStart(2, '0')}:${s.toString().padStart(2, '0')}`;
}

function showPlayIcon(isPlay) {
    const btn = document.getElementById('btn-play');
    const playIcon = btn.querySelector('.play-icon');
    const pauseIcon = btn.querySelector('.pause-icon');
    if (playIcon) playIcon.style.display = isPlay ? '' : 'none';
    if (pauseIcon) pauseIcon.style.display = isPlay ? 'none' : '';
}

function setPlayerSegments(segments) {
    playerSegments = [];
    let textIdx = 0;
    for (const seg of segments) {
        if (seg.type === 'text') {
            playerSegments.push({
                index: textIdx,
                start_time: seg.start_time || 0,
                text: seg.text || '',
            });
            textIdx++;
        }
    }
}
