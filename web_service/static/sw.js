// Service Worker cho Sherpa Vietnamese ASR PWA
const CACHE_VERSION = 'v2';
const CACHE_NAME = 'asr-vn-' + CACHE_VERSION;

const STATIC_ASSETS = [
  '/static/index.html',
  '/static/offline.html',
  '/static/css/style.css',
  '/static/js/app.js',
  '/static/js/websocket.js',
  '/static/js/upload.js',
  '/static/js/player.js',
  '/static/js/speaker.js',
  '/static/js/search.js',
  '/static/js/meetings.js',
  '/static/icons/icon-192.png',
  '/static/icons/icon-512.png',
  '/static/manifest.json'
];

// Install: pre-cache tung file rieng le, bo qua file loi de khong fail toan bo
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME).then((cache) => {
      return Promise.all(
        STATIC_ASSETS.map((url) =>
          cache.add(url).catch((err) => {
            console.warn('SW: failed to cache', url, err);
          })
        )
      );
    }).then(() => self.skipWaiting())
  );
});

// Activate: xoa toan bo cache cu, chiem quyen ngay
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((cacheNames) => {
      return Promise.all(
        cacheNames
          .filter((name) => name !== CACHE_NAME)
          .map((name) => caches.delete(name))
      );
    }).then(() => self.clients.claim())
  );
});

// Fetch: cache-first cho static, network-first cho API
self.addEventListener('fetch', (event) => {
  const url = new URL(event.request.url);

  // Bo qua WebSocket va non-GET requests
  if (event.request.method !== 'GET' || url.pathname.startsWith('/ws')) {
    return;
  }

  // Static assets: network-first, fallback cache (dam bao luon load code moi nhat)
  if (url.pathname.startsWith('/static/')) {
    event.respondWith(
      fetch(event.request).then((response) => {
        const clone = response.clone();
        caches.open(CACHE_NAME).then((cache) => cache.put(event.request, clone));
        return response;
      }).catch(() => {
        return caches.match(event.request).then((cached) => {
          if (cached) return cached;
          if (event.request.mode === 'navigate') {
            return caches.match('/static/offline.html');
          }
        });
      })
    );
    return;
  }

  // API & other requests: network-first, offline fallback
  event.respondWith(
    fetch(event.request).catch(() => {
      if (event.request.mode === 'navigate') {
        return caches.match('/static/offline.html');
      }
    })
  );
});
