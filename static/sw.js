const CACHE = 'partisolator-v1';
const SHELL = ['/', '/static/manifest.json'];

// API paths that must never be served from cache
const API_PATHS = ['/isolate', '/status', '/download', '/parts'];

self.addEventListener('install', e =>
  e.waitUntil(
    caches.open(CACHE).then(c => c.addAll(SHELL))
  )
);

self.addEventListener('activate', e =>
  e.waitUntil(
    caches.keys().then(keys =>
      Promise.all(keys.filter(k => k !== CACHE).map(k => caches.delete(k)))
    )
  )
);

self.addEventListener('fetch', e => {
  const url = new URL(e.request.url);
  if (e.request.method !== 'GET') return;
  if (API_PATHS.some(p => url.pathname.startsWith(p))) return;

  e.respondWith(
    caches.match(e.request).then(cached => cached || fetch(e.request))
  );
});
