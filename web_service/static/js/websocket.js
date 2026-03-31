/* WebSocket - heartbeat, progress, queue updates */

let ws = null;
let wsReconnectTimer = null;
let heartbeatTimer = null;

function connectWebSocket() {
    const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const url = `${protocol}//${location.host}/ws`;

    ws = new WebSocket(url);

    ws.onopen = () => {
        console.log('[WS] Connected');
        if (wsReconnectTimer) {
            clearInterval(wsReconnectTimer);
            wsReconnectTimer = null;
        }
        // Heartbeat moi 30 giay
        heartbeatTimer = setInterval(() => {
            if (ws && ws.readyState === WebSocket.OPEN) {
                ws.send(JSON.stringify({ type: 'heartbeat' }));
            }
        }, 30000);
    };

    // Mobile browser: reconnect ngay khi user quay lai app
    if (!window._wsVisibilityListenerAdded) {
        document.addEventListener('visibilitychange', () => {
            if (!document.hidden) {
                // Gui HTTP heartbeat ngay lap tuc de update DB
                // (WS co the chua reconnect xong, cleanup loop co the chay bat cu luc nao)
                fetch('/api/session/heartbeat', {
                    method: 'POST', credentials: 'same-origin',
                    headers: window.authToken ? { 'Authorization': 'Bearer ' + window.authToken } : {},
                }).catch(() => {});

                // Khi quay lai, kiem tra WS con song khong
                if (!ws || ws.readyState === WebSocket.CLOSED || ws.readyState === WebSocket.CLOSING) {
                    console.log('[WS] Page visible, reconnecting...');
                    connectWebSocket();
                } else if (ws.readyState === WebSocket.OPEN) {
                    // Gui heartbeat qua WS luon
                    ws.send(JSON.stringify({ type: 'heartbeat' }));
                }
            }
        });
        window._wsVisibilityListenerAdded = true;
    }

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleWSMessage(data);
    };

    ws.onclose = () => {
        console.log('[WS] Disconnected');
        if (heartbeatTimer) {
            clearInterval(heartbeatTimer);
            heartbeatTimer = null;
        }
        // Auto reconnect sau 3 giay
        if (!wsReconnectTimer) {
            wsReconnectTimer = setInterval(() => {
                console.log('[WS] Reconnecting...');
                connectWebSocket();
            }, 3000);
        }
    };

    ws.onerror = () => {
        ws.close();
    };
}

function handleWSMessage(data) {
    switch (data.type) {
        case 'heartbeat_ack':
            break;

        case 'queue_position':
            onQueuePosition(data);
            break;

        case 'processing_started':
            onProcessingStarted(data);
            break;

        case 'progress':
            onProgress(data);
            break;

        case 'asr_complete':
            onASRComplete(data);
            break;

        case 'asr_error':
            onASRError(data);
            break;

        case 'asr_cancelled':
            onASRCancelled(data);
            break;

        case 'queue_updated':
            onQueueUpdated(data);
            break;

        case 'session_expired':
            onSessionExpired(data);
            break;

        case 'summary_started':
            if (typeof onSummaryStarted === 'function') onSummaryStarted(data);
            break;

        case 'summary_complete':
            if (typeof onSummaryComplete === 'function') onSummaryComplete(data);
            break;

        case 'summary_error':
            if (typeof onSummaryError === 'function') onSummaryError(data);
            break;

        default:
            console.log('[WS] Unknown message:', data);
    }
}

function wsSend(data) {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify(data));
    }
}

function subscribeQueue(fileId) {
    wsSend({ type: 'subscribe_queue', file_id: fileId });
}

function reconnectWebSocket() {
    // Đóng kết nối cũ, tạo kết nối mới (với session cookie mới)
    if (heartbeatTimer) {
        clearInterval(heartbeatTimer);
        heartbeatTimer = null;
    }
    if (wsReconnectTimer) {
        clearInterval(wsReconnectTimer);
        wsReconnectTimer = null;
    }
    if (ws) {
        ws.onclose = null; // Tránh auto-reconnect với session cũ
        ws.close();
        ws = null;
    }
    connectWebSocket();
}
