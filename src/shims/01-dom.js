// 01-dom.js - Document and DOM shims

const { SimpleEventTarget } = globalThis;

let __pointerLockElement = null;
const documentEventTarget = new SimpleEventTarget();
const bodyEventTarget = new SimpleEventTarget();

// Body setup
Object.assign(bodyEventTarget, {
  appendChild() {},
  removeChild() {},
  style: {},
  ownerDocument: null,
});

let __currentCursor = "auto";
Object.defineProperty(bodyEventTarget.style, "cursor", {
  get: () => __currentCursor,
  set(v) {
    __currentCursor = v;
    globalThis.__input?.op_input_set_cursor_style?.(v);
  },
});

// Document
globalThis.document = {
  body: bodyEventTarget,
  createElement(name) {
    if (name === "canvas") return globalThis.__makeCanvas();
    // Stub element for libraries that create DOM elements for URL parsing etc.
    const el = { tagName: name.toUpperCase(), style: {}, dataset: {} };
    if (name === "a") {
      // URL resolution: setting .href on an <a> parses the URL
      let _url = null;
      Object.defineProperty(el, "href", {
        get() { return _url ? _url.href : ""; },
        set(v) { try { _url = new URL(v, globalThis.location?.href || "file:///"); } catch { _url = null; } },
      });
      Object.defineProperty(el, "pathname", {
        get() { return _url ? _url.pathname : ""; },
      });
      Object.defineProperty(el, "hostname", {
        get() { return _url ? _url.hostname : ""; },
      });
      Object.defineProperty(el, "protocol", {
        get() { return _url ? _url.protocol : ""; },
      });
      Object.defineProperty(el, "origin", {
        get() { return _url ? _url.origin : ""; },
      });
    }
    return el;
  },
  addEventListener:
    documentEventTarget.addEventListener.bind(documentEventTarget),
  removeEventListener:
    documentEventTarget.removeEventListener.bind(documentEventTarget),
  dispatchEvent: documentEventTarget.dispatchEvent.bind(documentEventTarget),
  _eventTarget: documentEventTarget,
  get pointerLockElement() {
    return globalThis.__input?.op_input_is_pointer_locked?.()
      ? __pointerLockElement
      : null;
  },
  exitPointerLock() {
    if (__pointerLockElement) {
      __pointerLockElement = null;
      globalThis.__input?.op_input_exit_pointer_lock?.();
      this.dispatchEvent({
        type: "pointerlockchange",
        bubbles: true,
        cancelable: false,
        timeStamp: performance.now(),
        isTrusted: true,
        preventDefault() {},
        stopPropagation() {},
        stopImmediatePropagation() {},
      });
    }
  },
};

bodyEventTarget.ownerDocument = globalThis.document;

// Canvas factory
globalThis.__inputCanvases = new Set();
globalThis.__pointerLockElement = null;

globalThis.__makeCanvas = function () {
  const eventTarget = new SimpleEventTarget();

  const canvas = {
    width: globalThis.__windowInnerWidth ?? 800,
    height: globalThis.__windowInnerHeight ?? 600,
    style: {},
    get clientWidth() {
      return this.width;
    },
    get clientHeight() {
      return this.height;
    },
    getBoundingClientRect() {
      return {
        x: 0,
        y: 0,
        top: 0,
        left: 0,
        right: this.width,
        bottom: this.height,
        width: this.width,
        height: this.height,
        toJSON() {
          return { ...this };
        },
      };
    },
    getContext(kind) {
      if (kind !== "webgpu") throw new Error("Only webgpu context supported");
      globalThis.__gpuCanvasContext ??= new globalThis.GPUCanvasContext();
      return globalThis.__gpuCanvasContext;
    },
    addEventListener: eventTarget.addEventListener.bind(eventTarget),
    removeEventListener: eventTarget.removeEventListener.bind(eventTarget),
    dispatchEvent: eventTarget.dispatchEvent.bind(eventTarget),
    _eventTarget: eventTarget,
    getRootNode: () => globalThis.document,
    setPointerCapture() {},
    releasePointerCapture() {},
    requestPointerLock() {
      __pointerLockElement = this;
      globalThis.__pointerLockElement = this;
      globalThis.__input?.op_input_request_pointer_lock?.();
    },
  };

  globalThis.__inputCanvases.add(canvas);
  globalThis.__windowInnerWidth = canvas.width;
  globalThis.__windowInnerHeight = canvas.height;
  return canvas;
};

// Export for pointer lock access
globalThis.__setPointerLockElement = (el) => {
  __pointerLockElement = el;
};
globalThis.__getPointerLockElement = () => __pointerLockElement;

console.log("[shims] DOM initialized");
