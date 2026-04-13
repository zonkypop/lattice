// setup.js - imports in deterministic order

// Order matters with Deno (import order can work differently from vite etc)
// each file depends on previous ones being initialized.
import "./00-globals.js"; // window, self, RAF, performance
import "./01-dom.js"; // document, body, canvas factory
import "./02-input.js"; // input event dispatching
import "./03-audio.js"; // AudioContext stubs
import "./04-indexeddb.js"; // IndexedDB (if native ops present)
import "./05-webgpu.js"; // WebGPU (if native ops present)
import "./06-webxr.js"; // WebXR (if native ops present)

// Navigator
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

console.log("[shims] setup complete");
