// setup.js - imports in deterministic order

// Order matters with Deno (import order can work differently from vite etc)
// each file depends on previous ones being initialized.
import "./00-globals.js"; // window, self, RAF, performance
import "./01-dom.js"; // document, body, canvas factory
import "./02-input.js"; // input event dispatching
import "./03-audio.js"; // AudioContext stubs
import "./04-indexeddb.js"; // IndexedDB (if native ops present)
import "./04a-localstorage.js"; // localStorage (native SQLite, separate DB)
import "./05-webgpu.js"; // WebGPU (if native ops present)
import "./06-webxr.js"; // WebXR (if native ops present)
import "./07-webrtc.js"; // WebRTC data channels (if native ops present)

// Navigator - must be set before any user code that checks browser compat
globalThis.navigator ??= {};
Object.defineProperty(globalThis.navigator, "onLine", {
  value: true,
  writable: true,
  configurable: true,
});
Object.defineProperty(globalThis.navigator, "platform", {
  value: "Linux aarch64",
  writable: true,
  configurable: true,
});
Object.defineProperty(globalThis.navigator, "userAgent", {
  value: "Mozilla/5.0 (Linux; aarch64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0",
  writable: true,
  configurable: true,
});
// PeerJS detectBrowser() uses userAgentData for Chrome detection
Object.defineProperty(globalThis.navigator, "userAgentData", {
  value: {
    brands: [
      { brand: "Chromium", version: "120" },
      { brand: "Not:A-Brand", version: "8" },
    ],
    mobile: false,
    platform: "Linux",
  },
  writable: true,
  configurable: true,
});
const __lang = globalThis.__locale__ || "en-US";
Object.defineProperty(globalThis.navigator, "language", {
  value: __lang,
  writable: true,
  configurable: true,
});
Object.defineProperty(globalThis.navigator, "languages", {
  value: Object.freeze([__lang]),
  writable: true,
  configurable: true,
});
globalThis.navigator.sendBeacon ??= () => false;
globalThis.navigator.mediaDevices ??= {
  getUserMedia: async (constraints) => {
    if (!constraints?.audio) throw new DOMException("Only audio capture is supported", "NotSupportedError");
    const __rtc = globalThis.__webrtc;
    if (!__rtc || !__rtc.op_rtc_get_user_media) {
      throw new DOMException("getUserMedia not available", "NotSupportedError");
    }
    const result = await __rtc.op_rtc_get_user_media();
    const track = new MediaStreamTrack(result.stream_id, "audio");
    return new MediaStream(result.stream_id, [track]);
  },
  enumerateDevices: () => Promise.resolve([]),
};

console.log("[shims] setup complete");
