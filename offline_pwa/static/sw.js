const CACHE_PREFIX = "offline-pwa-";
const CACHE_VERSION = "offline-pwa-v141";
const NETWORK_TIMEOUT_MS = 1500;
const INSTALL_CORE_TIMEOUT_MS = 15000;
const CORE_SHELL = [
  "/",
  "/index.html",
  "/manifest.json",
  "/css/app.css",
  "/js/app.js",
  "/icons/icon-192.png",
  "/icons/icon-512.png"
];
const APP_SHELL = [
  ...CORE_SHELL,
  "/hotword.txt",
  "/api/model-manifest",
  "/shared/css/style.css",
  "/shared/js/about.js",
  "/shared/js/status.js",
  "/js/asr-worker.js",
  "/js/ffmpeg-decode-worker.js",
  "/js/pure-ort-asr-worker.js",
  "/vendor/ffmpeg/ffmpeg/classes.js",
  "/vendor/ffmpeg/ffmpeg/const.js",
  "/vendor/ffmpeg/ffmpeg/errors.js",
  "/vendor/ffmpeg/ffmpeg/index.js",
  "/vendor/ffmpeg/ffmpeg/types.js",
  "/vendor/ffmpeg/ffmpeg/utils.js",
  "/vendor/ffmpeg/ffmpeg/worker.js",
  "/vendor/ffmpeg/core/ffmpeg-core.js",
  "/vendor/ffmpeg/core/ffmpeg-core.wasm",
  "/vendor/onnxruntime-web/ort.wasm.min.js",
  "/vendor/onnxruntime-web/ort.webgpu.min.js",
  "/vendor/onnxruntime-web/ort-wasm-simd-threaded.wasm",
  "/vendor/onnxruntime-web/ort-wasm-simd-threaded.mjs",
  "/vendor/onnxruntime-web/ort-wasm-simd-threaded.jsep.wasm",
  "/vendor/onnxruntime-web/ort-wasm-simd-threaded.jsep.mjs",
  "/vendor/onnxruntime-web/ort-wasm-simd-threaded.asyncify.wasm",
  "/vendor/onnxruntime-web/ort-wasm-simd-threaded.asyncify.mjs",
  "/vendor/longform-clustering/longform-clustering.js",
  "/vendor/mpg123-decoder/mpg123-decoder.min.js",
  "/vendor/zstd-wasm/zstd-wrapper.js",
  "/vendor/zstd-wasm/zstd.js",
  "/vendor/zstd-wasm/zstd.wasm",
  "/vendor/sherpa-onnx-wasm/sherpa-onnx-asr.js",
  "/vendor/sherpa-onnx-wasm/sherpa-onnx-wasm-main-vad-asr.js",
  "/vendor/sherpa-onnx-wasm/sherpa-onnx-wasm-main-vad-asr.wasm",
  "/calibration/1hour_qh_10min.mp3"
];

const OFFLINE_FALLBACK_HTML = `<!doctype html>
<html lang="vi">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Sherpa Vietnamese ASR</title>
  <style>
    body{margin:0;background:#2b2b2b;color:#f5f5f5;font:16px system-ui,-apple-system,Segoe UI,sans-serif}
    main{max-width:720px;margin:64px auto;padding:24px}
    .panel{border:1px solid #555;border-radius:6px;background:#363636;padding:18px}
    h1{font-size:22px;margin:0 0 12px;color:#1683ff}
    p{line-height:1.5}
  </style>
</head>
<body>
  <main>
    <div class="panel">
      <h1>Sherpa Vietnamese ASR</h1>
      <p>Ứng dụng chưa có đủ cache giao diện để mở hoàn toàn khi offline.</p>
      <p>Hãy bật lại kết nối và mở app một lần cho đến khi giao diện xử lý hiện ra, sau đó app sẽ mở offline được.</p>
    </div>
  </main>
</body>
</html>`;

function fetchWithTimeout(request, timeoutMs = NETWORK_TIMEOUT_MS) {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  return fetch(request, { signal: controller.signal })
    .finally(() => clearTimeout(timer));
}

async function localFirst(event, cacheKey) {
  const cache = await caches.open(CACHE_VERSION);
  const cached = await cache.match(cacheKey);
  if (cached) {
    return cached;
  }
  try {
    const response = await fetchWithTimeout(event.request);
    if (response.ok) {
      await cache.put(cacheKey, response.clone());
    }
    return response;
  } catch (_) {
    // Fall through to the offline response below.
  }
  if (event.request.mode === "navigate") {
    const shell = await cache.match("/");
    if (shell) return shell;
    return new Response(OFFLINE_FALLBACK_HTML, {
      status: 200,
      headers: { "Content-Type": "text/html; charset=utf-8" },
    });
  }
  return new Response("Offline asset is not cached yet.", { status: 503 });
}

self.addEventListener("install", (event) => {
  event.waitUntil(
    caches.open(CACHE_VERSION)
      .then(async (cache) => {
        await Promise.all(CORE_SHELL.map(async (url) => {
          const request = new Request(url, { cache: "reload" });
          const response = await fetchWithTimeout(request, INSTALL_CORE_TIMEOUT_MS);
          if (!response.ok) throw new Error(`core shell cache failed: ${url} ${response.status}`);
          await cache.put(url, response);
        }));
        await Promise.all(APP_SHELL.map(async (url) => {
          if (CORE_SHELL.includes(url)) return;
          try {
            const request = new Request(url, { cache: "reload" });
            const response = await fetchWithTimeout(request);
            if (response.ok) await cache.put(url, response);
          } catch (_) {
            // Optional shell/model helpers are refreshed opportunistically at runtime.
          }
        }));
      })
      .then(() => self.skipWaiting())
  );
});

self.addEventListener("activate", (event) => {
  event.waitUntil(
    caches.keys()
      .then((keys) => Promise.all(
        keys
          .filter((key) => key.startsWith(CACHE_PREFIX) && key !== CACHE_VERSION)
          .map((key) => caches.delete(key))
      ))
      .then(() => self.clients.claim())
  );
});

self.addEventListener("fetch", (event) => {
  const url = new URL(event.request.url);
  if (event.request.method !== "GET") return;

  if (url.pathname === "/api/version") {
    return;
  }

  if (event.request.headers.get("X-ASR-Offline-Cache-Fill") === "1") {
    return;
  }

  if (url.pathname.startsWith("/api/model-files/")) {
    return;
  }

  const shouldCache =
    APP_SHELL.includes(url.pathname) ||
    url.pathname.startsWith("/shared/") ||
    url.pathname.startsWith("/css/") ||
    url.pathname.startsWith("/js/") ||
    url.pathname.startsWith("/vendor/") ||
    url.pathname.startsWith("/calibration/") ||
    url.pathname.startsWith("/icons/");

  if (shouldCache) {
    event.respondWith(localFirst(event, url.pathname));
    return;
  }

  if (event.request.mode === "navigate") {
    event.respondWith(localFirst(event, "/"));
  }
});

self.addEventListener("notificationclick", (event) => {
  event.notification.close();
  const targetUrl = event.notification?.data?.url || "/";
  event.waitUntil((async () => {
    const allClients = await self.clients.matchAll({
      type: "window",
      includeUncontrolled: true,
    });
    const sameOriginUrl = new URL(targetUrl, self.location.origin);
    for (const client of allClients) {
      const clientUrl = new URL(client.url);
      if (clientUrl.origin === sameOriginUrl.origin && "focus" in client) {
        await client.focus();
        return;
      }
    }
    if (self.clients.openWindow) {
      await self.clients.openWindow(sameOriginUrl.pathname + sameOriginUrl.search + sameOriginUrl.hash);
    }
  })());
});
