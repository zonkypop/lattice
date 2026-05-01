// 00-globals.js - Bootstrap globals FIRST, no imports


// Minimal event target (inline to avoid import order issues)
class SimpleEventTarget {
  constructor() {
    this._listeners = {};
  }
  addEventListener(type, fn) {
    (this._listeners[type] ??= []).push(fn);
  }
  removeEventListener(type, fn) {
    if (this._listeners[type]) {
      this._listeners[type] = this._listeners[type].filter((f) => f !== fn);
    }
  }
  dispatchEvent(evt) {
    for (const fn of this._listeners[evt.type] || []) {
      try {
        fn(evt);
      } catch (e) {
        console.error(e);
      }
    }
    return !evt.defaultPrevented;
  }
}

globalThis.SimpleEventTarget = SimpleEventTarget;

// Window proxy - assignments to window.x also set globalThis.x
const windowTarget = {
  devicePixelRatio: 1,
  innerWidth: 800,
  innerHeight: 600,
  nativeXR: false,
};

const windowEventTarget = new SimpleEventTarget();
Object.assign(windowTarget, {
  addEventListener: windowEventTarget.addEventListener.bind(windowEventTarget),
  removeEventListener:
    windowEventTarget.removeEventListener.bind(windowEventTarget),
  dispatchEvent: windowEventTarget.dispatchEvent.bind(windowEventTarget),
  _eventTarget: windowEventTarget,
});

globalThis.window = new Proxy(windowTarget, {
  set(t, p, v) {
    t[p] = v;
    if (typeof p === "string" && p !== "window" && p !== "self") {
      globalThis[p] = v;
    }
    return true;
  },
  get(t, p) {
    return p in t ? t[p] : globalThis[p];
  },
  has(t, p) {
    return p in t || p in globalThis;
  },
});

globalThis.self ??= globalThis;
globalThis.devicePixelRatio = 1;

// ImageData polyfill (used by texture workers for RGBA pixel buffers)
if (typeof ImageData === "undefined") {
  globalThis.ImageData = class ImageData {
    constructor(dataOrWidth, widthOrHeight, heightOrUndefined) {
      if (dataOrWidth instanceof Uint8ClampedArray) {
        this.data = dataOrWidth;
        this.width = widthOrHeight;
        this.height =
          heightOrUndefined ?? (dataOrWidth.length / 4 / widthOrHeight) | 0;
      } else {
        this.width = dataOrWidth;
        this.height = widthOrHeight;
        this.data = new Uint8ClampedArray(this.width * this.height * 4);
      }
    }
  };
}
globalThis.performance ??= { now: () => Date.now() };

// Override fetch to use synchronous native HTTP when available.
// Deno's built-in async fetch deadlocks on the single-threaded tokio runtime.
if (globalThis.__httpFetch) {
  const _nativeFetch = globalThis.__httpFetch;
  const _denoFetch = globalThis.fetch; // keep for non-HTTP (e.g. data: URIs)
  globalThis.fetch = function fetch(input, init) {
    let url = typeof input === 'string' ? input : input?.url;
    // Only intercept GET/HEAD on http(s) URLs
    const method = init?.method?.toUpperCase() || 'GET';
    if (url && (url.startsWith('http://') || url.startsWith('https://')) && (method === 'GET' || method === 'HEAD')) {
      try {
        const bytes = _nativeFetch(url);
        const headers = new Headers();
        if (url.endsWith('.json')) headers.set('content-type', 'application/json');
        else if (url.endsWith('.wasm')) headers.set('content-type', 'application/wasm');
        else if (url.endsWith('.js') || url.endsWith('.mjs')) headers.set('content-type', 'application/javascript');
        return Promise.resolve(new Response(bytes.buffer, { status: 200, headers }));
      } catch (e) {
        return Promise.reject(new TypeError(`fetch failed: ${e.message}`));
      }
    }
    // Fall through to deno's fetch for non-HTTP or non-GET
    if (_denoFetch) return _denoFetch(input, init);
    return Promise.reject(new TypeError('fetch not available'));
  };
}

// WebGL enum constants (used by glTF loaders for component types, topology, samplers)
if (typeof WebGLRenderingContext === 'undefined') {
  globalThis.WebGLRenderingContext = {
    BYTE: 0x1400, UNSIGNED_BYTE: 0x1401,
    SHORT: 0x1402, UNSIGNED_SHORT: 0x1403,
    UNSIGNED_INT: 0x1405, FLOAT: 0x1406,
    TRIANGLES: 0x0004, TRIANGLE_STRIP: 0x0005,
    LINES: 0x0001, LINE_STRIP: 0x0003, POINTS: 0x0000,
    NEAREST: 0x2600, LINEAR: 0x2601,
    NEAREST_MIPMAP_NEAREST: 0x2700, LINEAR_MIPMAP_NEAREST: 0x2701,
    NEAREST_MIPMAP_LINEAR: 0x2702, LINEAR_MIPMAP_LINEAR: 0x2703,
    REPEAT: 0x2901, CLAMP_TO_EDGE: 0x812F, MIRRORED_REPEAT: 0x8370,
  };
}

// RAF system
let __rafCallbacks = [];
let __rafId = 0;

globalThis.requestAnimationFrame =
  globalThis.window.requestAnimationFrame =
  globalThis.self.requestAnimationFrame =
    (cb) => {
      const id = ++__rafId;
      __rafCallbacks.push({ id, callback: cb });
      return id;
    };

globalThis.cancelAnimationFrame =
  globalThis.window.cancelAnimationFrame =
  globalThis.self.cancelAnimationFrame =
    (id) => {
      __rafCallbacks = __rafCallbacks.filter((c) => c.id !== id);
    };

globalThis.__runAnimationFrames = () => {
  // Dispatch queued input events BEFORE RAF callbacks,
  // matching browser ordering where input fires before rAF.
  globalThis.__dispatchInputEvents?.();

  const cbs = __rafCallbacks;
  __rafCallbacks = [];
  const now = performance.now();
  for (const { callback } of cbs) {
    try {
      callback(now);
    } catch (e) {
      console.error("RAF error:", e);
    }
  }
};
