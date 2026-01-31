// FootyPredict Pro - Service Worker
// Version: 2.1.0 - Fixed stability issues

const CACHE_NAME = 'footypredict-v2.1';
const OFFLINE_URL = '/offline.html';

// Only cache truly static assets (CSS, JS, icons)
const STATIC_ASSETS = [
  '/styles.css',
  '/app.js',
  '/money-zone.js',
  '/money-zone.css',
  '/manifest.json',
  '/offline.html',
  '/icons/icon.svg'
];

// Install: Cache static assets only
self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE_NAME)
      .then((cache) => {
        console.log('[SW] Caching static assets');
        return cache.addAll(STATIC_ASSETS);
      })
      .then(() => self.skipWaiting())
  );
});

// Activate: Clean old caches immediately
self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys()
      .then((cacheNames) => {
        return Promise.all(
          cacheNames
            .filter((name) => name !== CACHE_NAME)
            .map((name) => {
              console.log('[SW] Deleting old cache:', name);
              return caches.delete(name);
            })
        );
      })
      .then(() => self.clients.claim())
  );
});

// Fetch handler with proper strategies
self.addEventListener('fetch', (event) => {
  const { request } = event;
  const url = new URL(request.url);
  
  // Skip non-GET requests
  if (request.method !== 'GET') return;
  
  // Skip cross-origin requests
  if (url.origin !== location.origin) {
    return;
  }
  
  // HTML pages: Network-first (always get fresh content)
  if (request.mode === 'navigate' || url.pathname.endsWith('.html') || url.pathname === '/') {
    event.respondWith(
      fetch(request)
        .then((response) => {
          return response;
        })
        .catch(() => {
          // Only show offline page if network completely fails
          return caches.match(OFFLINE_URL);
        })
    );
    return;
  }
  
  // Static assets (CSS, JS): Cache-first with network fallback
  if (url.pathname.endsWith('.css') || url.pathname.endsWith('.js') || 
      url.pathname.endsWith('.svg') || url.pathname.endsWith('.png') ||
      url.pathname.endsWith('.ico') || url.pathname.endsWith('.webp')) {
    event.respondWith(
      caches.match(request)
        .then((cached) => {
          if (cached) {
            return cached;
          }
          return fetch(request).then((response) => {
            if (response.ok) {
              const clone = response.clone();
              caches.open(CACHE_NAME).then((cache) => {
                cache.put(request, clone);
              });
            }
            return response;
          });
        })
    );
    return;
  }
  
  // Everything else: Network-first
  event.respondWith(
    fetch(request).catch(() => caches.match(request))
  );
});

// Push notifications
self.addEventListener('push', (event) => {
  const data = event.data?.json() || {};
  
  const options = {
    body: data.body || 'New prediction available!',
    icon: '/icons/icon.svg',
    vibrate: [100, 50, 100],
    data: { url: data.url || '/' }
  };
  
  event.waitUntil(
    self.registration.showNotification(
      data.title || '⚽ FootyPredict Pro',
      options
    )
  );
});

// Notification click
self.addEventListener('notificationclick', (event) => {
  event.notification.close();
  event.waitUntil(
    clients.openWindow(event.notification.data?.url || '/')
  );
});
