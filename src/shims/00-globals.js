// 00-globals.js - Bootstrap globals FIRST, no imports

// Truncate large console output to avoid flooding the terminal
{
  const MAX_ARG_LENGTH = 100;
  const TRUNCATE_SUFFIX = "\n  ... [truncated]";
  const originalConsole = { log: console.log, warn: console.warn, error: console.error, info: console.info, debug: console.debug };

  function truncateArg(arg) {
    if (typeof arg === "string") {
      return arg.length > MAX_ARG_LENGTH ? arg.slice(0, MAX_ARG_LENGTH) + TRUNCATE_SUFFIX : arg;
    }
    if (arg && typeof arg === "object") {
      try {
        const s = JSON.stringify(arg, null, 2);
        if (s && s.length > MAX_ARG_LENGTH) {
          return s.slice(0, MAX_ARG_LENGTH) + TRUNCATE_SUFFIX;
        }
      } catch { /* circular or non-serializable, let Deno handle it */ }
    }
    return arg;
  }

  for (const method of ["log", "warn", "error", "info", "debug"]) {
    const orig = originalConsole[method];
    if (!orig) continue;
    console[method] = (...args) => orig.call(console, ...args.map(truncateArg));
  }
}

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
globalThis.performance ??= { now: () => Date.now() };

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
