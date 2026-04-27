// 04-indexeddb.js - IndexedDB backed by native SQLite

const ops = globalThis.__indexeddb;
if (!ops) {
  console.log("[shims] IndexedDB skipped (no native ops)");
} else {
  const defer = (fn) => Promise.resolve().then(fn);

  // Binary-safe recursive serialization.
  // Walks the value tree, extracts binary data into a separate blob list,
  // and replaces them with placeholder references in a JSON-safe structure.
  // Final layout: [4-byte LE blob count] [4-byte LE JSON length] [JSON] [blob0] [blob1] ...
  // Each blob entry in the binary section: [4-byte LE mime length] [mime] [4-byte LE data length] [data]
  // JSON placeholders: { "__blob__": index } or { "__binary__": index }

  async function serializeValue(value, blobs) {
    if (value instanceof Blob) {
      const idx = blobs.length;
      const buf = await value.arrayBuffer();
      blobs.push({ mime: value.type || "", data: new Uint8Array(buf) });
      return { __blob__: idx };
    }
    if (value instanceof ArrayBuffer) {
      const idx = blobs.length;
      blobs.push({ mime: "", data: new Uint8Array(value) });
      return { __binary__: idx };
    }
    if (value instanceof Uint8Array) {
      const idx = blobs.length;
      blobs.push({ mime: "", data: value });
      return { __binary__: idx };
    }
    if (ArrayBuffer.isView(value)) {
      const idx = blobs.length;
      blobs.push({ mime: "", data: new Uint8Array(value.buffer, value.byteOffset, value.byteLength) });
      return { __binary__: idx };
    }
    if (Array.isArray(value)) {
      const out = [];
      for (let i = 0; i < value.length; i++) out.push(await serializeValue(value[i], blobs));
      return out;
    }
    if (value && typeof value === "object") {
      const out = {};
      for (const k of Object.keys(value)) out[k] = await serializeValue(value[k], blobs);
      return out;
    }
    return value; // primitives pass through to JSON
  }

  async function toBytes(value) {
    const blobs = [];
    const jsonVal = await serializeValue(value, blobs);
    const jsonBytes = new TextEncoder().encode(JSON.stringify(jsonVal));

    // Calculate total size
    let blobsSize = 0;
    for (const b of blobs) {
      const mimeBytes = new TextEncoder().encode(b.mime);
      blobsSize += 4 + mimeBytes.byteLength + 4 + b.data.byteLength;
      b._mimeBytes = mimeBytes; // cache for writing
    }

    const out = new Uint8Array(4 + 4 + jsonBytes.byteLength + blobsSize);
    const view = new DataView(out.buffer);
    let offset = 0;

    view.setUint32(offset, blobs.length, true); offset += 4;
    view.setUint32(offset, jsonBytes.byteLength, true); offset += 4;
    out.set(jsonBytes, offset); offset += jsonBytes.byteLength;

    for (const b of blobs) {
      view.setUint32(offset, b._mimeBytes.byteLength, true); offset += 4;
      out.set(b._mimeBytes, offset); offset += b._mimeBytes.byteLength;
      view.setUint32(offset, b.data.byteLength, true); offset += 4;
      out.set(b.data, offset); offset += b.data.byteLength;
    }

    return out;
  }

  function fromBytes(bytes) {
    if (!bytes || bytes.length < 8) return undefined;

    // Check for legacy format: old format started with JSON (first byte is '{' or '[' or a digit etc.)
    // New format starts with a 4-byte LE uint32 blob count. If byte[4..8] is a plausible JSON length
    // and byte[0] looks like a small number, it's new format. Otherwise try legacy.
    const firstByte = bytes[0];
    if (firstByte === 0x7b || firstByte === 0x5b || firstByte === 0x22 ||
        (firstByte >= 0x30 && firstByte <= 0x39) || firstByte === 0x74 ||
        firstByte === 0x66 || firstByte === 0x6e) {
      // Starts with { [ " 0-9 t f n — likely raw JSON (legacy)
      return fromBytesLegacy(bytes);
    }

    try {
      const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
      let offset = 0;

      const blobCount = view.getUint32(offset, true); offset += 4;
      const jsonLen = view.getUint32(offset, true); offset += 4;

      // Sanity check
      if (offset + jsonLen > bytes.byteLength) return fromBytesLegacy(bytes);

      const jsonStr = new TextDecoder().decode(bytes.subarray(offset, offset + jsonLen));
      offset += jsonLen;

      // Read blobs
      const blobs = [];
      for (let i = 0; i < blobCount; i++) {
        const mimeLen = view.getUint32(offset, true); offset += 4;
        const mime = new TextDecoder().decode(bytes.subarray(offset, offset + mimeLen));
        offset += mimeLen;
        const dataLen = view.getUint32(offset, true); offset += 4;
        const data = bytes.subarray(offset, offset + dataLen);
        offset += dataLen;
        blobs.push({ mime, data });
      }

      const jsonVal = JSON.parse(jsonStr);
      return deserializeValue(jsonVal, blobs);
    } catch {
      return fromBytesLegacy(bytes);
    }
  }

  function deserializeValue(val, blobs) {
    if (val === null || val === undefined) return val;
    if (typeof val !== "object") return val;
    if (val.__blob__ !== undefined) {
      const b = blobs[val.__blob__];
      return new Blob([b.data], { type: b.mime });
    }
    if (val.__binary__ !== undefined) {
      const b = blobs[val.__binary__];
      return b.data.buffer.slice(b.data.byteOffset, b.data.byteOffset + b.data.byteLength);
    }
    if (Array.isArray(val)) {
      for (let i = 0; i < val.length; i++) val[i] = deserializeValue(val[i], blobs);
      return val;
    }
    for (const k of Object.keys(val)) val[k] = deserializeValue(val[k], blobs);
    return val;
  }

  function fromBytesLegacy(bytes) {
    try {
      const parsed = JSON.parse(new TextDecoder().decode(bytes));
      if (parsed && parsed.type && typeof parsed.data !== "undefined") {
        return deserializeLegacy(parsed);
      }
      return parsed;
    } catch {
      return undefined;
    }
  }

  function deserializeLegacy(w) {
    if (!w) return undefined;
    switch (w.type) {
      case "arraybuffer":
        return new Uint8Array(w.data).buffer;
      case "uint8array":
        return new Uint8Array(w.data);
      case "blob":
        return new Blob([new Uint8Array(w.data)], { type: w.blobType || "" });
      case "array":
        return w.data.map(deserializeLegacy);
      case "object": {
        const r = {};
        for (const k of Object.keys(w.data)) r[k] = deserializeLegacy(w.data[k]);
        return r;
      }
      default:
        return w.data;
    }
  }

  // Request classes
  class IDBRequest {
    constructor() {
      this.result = undefined;
      this.error = null;
      this.readyState = "pending";
      this.onsuccess = null;
      this.onerror = null;
    }
    _succeed(r) {
      this.result = r;
      this.readyState = "done";
      this.onsuccess?.({ target: this });
    }
    _fail(e) {
      this.error = e;
      this.readyState = "done";
      this.onerror?.({ target: this, error: e });
    }
  }

  class IDBOpenDBRequest extends IDBRequest {
    constructor() {
      super();
      this.onupgradeneeded = null;
      this.onblocked = null;
    }
  }

  // Store
  class IDBObjectStore {
    constructor(db, name, tx) {
      this._db = db;
      this.name = name;
      this.transaction = tx;
    }

    get(key) {
      const req = new IDBRequest();
      try {
        const bytes = ops.op_indexeddb_get(
          this._db._name,
          this.name,
          String(key)
        );
        const value = bytes ? fromBytes(bytes) : undefined;
        req.result = value;
        req.readyState = "done";
        defer(() => req._succeed(value));
      } catch (e) {
        req.readyState = "done";
        defer(() => req._fail(e));
      }
      return req;
    }

    put(value, key) {
      const req = new IDBRequest();
      toBytes(value)
        .then((bytes) => {
          try {
            ops.op_indexeddb_put(this._db._name, this.name, String(key), bytes);
            req.result = key;
            req.readyState = "done";
            defer(() => req._succeed(key));
          } catch (e) {
            defer(() => req._fail(e));
          }
        })
        .catch((e) => defer(() => req._fail(e)));
      return req;
    }

    delete(key) {
      const req = new IDBRequest();
      try {
        ops.op_indexeddb_delete(this._db._name, this.name, String(key));
        defer(() => req._succeed(undefined));
      } catch (e) {
        defer(() => req._fail(e));
      }
      return req;
    }

    getAllKeys() {
      const req = new IDBRequest();
      try {
        const keys = ops.op_indexeddb_get_all_keys(this._db._name, this.name);
        defer(() => req._succeed(keys));
      } catch (e) {
        defer(() => req._fail(e));
      }
      return req;
    }

    clear() {
      const req = new IDBRequest();
      try {
        ops.op_indexeddb_clear(this._db._name, this.name);
        defer(() => req._succeed(undefined));
      } catch (e) {
        defer(() => req._fail(e));
      }
      return req;
    }
  }

  // Transaction
  class IDBTransaction {
    constructor(db, stores, mode) {
      this._db = db;
      this._stores = stores;
      this.mode = mode;
      this.db = db;
      this.oncomplete = null;
      this.onerror = null;
      this.onabort = null;
      defer(() => defer(() => this.oncomplete?.({ target: this })));
    }

    objectStore(name) {
      if (!this._stores.includes(name)) {
        throw new DOMException(`Store '${name}' not in scope`, "NotFoundError");
      }
      return new IDBObjectStore(this._db, name, this);
    }

    abort() {
      this.onabort?.({ target: this });
    }
  }

  // Database
  class IDBDatabase {
    constructor(name, version, stores) {
      this._name = name;
      this.name = name;
      this.version = version;
      this._stores = new Set(stores);
      this._updateStoreNames();
    }

    _updateStoreNames() {
      const arr = [...this._stores];
      this.objectStoreNames = {
        contains: (n) => this._stores.has(n),
        length: arr.length,
        item: (i) => arr[i],
        [Symbol.iterator]: () => this._stores.values(),
      };
    }

    createObjectStore(name) {
      this._stores.add(name);
      this._updateStoreNames();
      return new IDBObjectStore(this, name, null);
    }

    deleteObjectStore(name) {
      this._stores.delete(name);
      this._updateStoreNames();
    }

    transaction(stores, mode = "readonly") {
      return new IDBTransaction(
        this,
        Array.isArray(stores) ? stores : [stores],
        mode
      );
    }

    close() {}
  }

  // Factory
  class IDBFactory {
    open(name, version = 1) {
      const req = new IDBOpenDBRequest();
      defer(() => {
        try {
          const db = new IDBDatabase(name, version, []);
          req.result = db;
          req.onupgradeneeded?.({
            target: req,
            oldVersion: 0,
            newVersion: version,
          });
          const stores = [...db._stores];
          if (stores.length) ops.op_indexeddb_open(name, stores);
          req.readyState = "done";
          req.onsuccess?.({ target: req });
        } catch (e) {
          req._fail(e);
        }
      });
      return req;
    }

    deleteDatabase(name) {
      const req = new IDBRequest();
      defer(() => req._succeed(undefined));
      return req;
    }
  }

  globalThis.indexedDB = new IDBFactory();
  globalThis.IDBDatabase = IDBDatabase;
  globalThis.IDBTransaction = IDBTransaction;
  globalThis.IDBObjectStore = IDBObjectStore;
  globalThis.IDBRequest = IDBRequest;
  globalThis.IDBOpenDBRequest = IDBOpenDBRequest;

  console.log("[shims] IndexedDB initialized");
}
