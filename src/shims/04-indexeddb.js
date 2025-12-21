// 04-indexeddb.js - IndexedDB backed by native SQLite

const ops = globalThis.__indexeddb;
if (!ops) {
  console.log("[shims] IndexedDB skipped (no native ops)");
} else {
  const defer = (fn) => Promise.resolve().then(fn);

  // Serialization
  async function serializeValue(value) {
    if (value instanceof ArrayBuffer) {
      return { type: "arraybuffer", data: [...new Uint8Array(value)] };
    }
    if (value instanceof Uint8Array) {
      return { type: "uint8array", data: [...value] };
    }
    if (value instanceof Blob) {
      const buf = await value.arrayBuffer();
      return {
        type: "blob",
        data: [...new Uint8Array(buf)],
        blobType: value.type,
      };
    }
    if (Array.isArray(value)) {
      return {
        type: "array",
        data: await Promise.all(value.map(serializeValue)),
      };
    }
    if (value && typeof value === "object") {
      const data = {};
      for (const k of Object.keys(value))
        data[k] = await serializeValue(value[k]);
      return { type: "object", data };
    }
    return { type: "json", data: value };
  }

  function deserializeValue(w) {
    if (!w) return undefined;
    switch (w.type) {
      case "arraybuffer":
        return new Uint8Array(w.data).buffer;
      case "uint8array":
        return new Uint8Array(w.data);
      case "blob":
        return new Blob([new Uint8Array(w.data)], { type: w.blobType || "" });
      case "array":
        return w.data.map(deserializeValue);
      case "object": {
        const r = {};
        for (const k of Object.keys(w.data)) r[k] = deserializeValue(w.data[k]);
        return r;
      }
      default:
        return w.data;
    }
  }

  async function toBytes(value) {
    return new TextEncoder().encode(
      JSON.stringify(await serializeValue(value))
    );
  }

  function fromBytes(bytes) {
    if (!bytes?.length) return undefined;
    return deserializeValue(JSON.parse(new TextDecoder().decode(bytes)));
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
