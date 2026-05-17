// 04a-localstorage.js - localStorage backed by native SQLite (separate DB)

const ops = globalThis.__localStorage;
if (!ops) {
  console.log("[shims] localStorage skipped (no native ops)");
} else {
  const storage = {
    getItem(key) {
      return ops.op_localstorage_get(String(key)) ?? null;
    },
    setItem(key, value) {
      ops.op_localstorage_set(String(key), String(value));
    },
    removeItem(key) {
      ops.op_localstorage_remove(String(key));
    },
    clear() {
      ops.op_localstorage_clear();
    },
    key(index) {
      const keys = ops.op_localstorage_keys();
      return index >= 0 && index < keys.length ? keys[index] : null;
    },
    get length() {
      return ops.op_localstorage_length();
    },
  };

  // Proxy so storage["myKey"] works like storage.getItem("myKey")
  const handler = {
    get(target, prop) {
      if (prop in target) return target[prop];
      if (typeof prop === "symbol") return undefined;
      return target.getItem(prop);
    },
    set(target, prop, value) {
      if (prop in target) {
        target[prop] = value;
      } else {
        target.setItem(prop, value);
      }
      return true;
    },
    deleteProperty(target, prop) {
      target.removeItem(prop);
      return true;
    },
    has(target, prop) {
      if (prop in target) return true;
      return target.getItem(prop) !== null;
    },
    ownKeys() {
      return ops.op_localstorage_keys();
    },
    getOwnPropertyDescriptor(target, prop) {
      if (prop in target) return undefined;
      const val = target.getItem(prop);
      if (val === null) return undefined;
      return { value: val, writable: true, enumerable: true, configurable: true };
    },
  };

  const proxy = new Proxy(storage, handler);

  // Must use defineProperty — Deno defines localStorage as a getter,
  // so plain assignment silently fails.
  Object.defineProperty(globalThis, "localStorage", {
    value: proxy,
    writable: true,
    configurable: true,
    enumerable: true,
  });
  Object.defineProperty(globalThis, "sessionStorage", {
    value: proxy,
    writable: true,
    configurable: true,
    enumerable: true,
  });

  console.log("[shims] localStorage ready (native SQLite)");
}
