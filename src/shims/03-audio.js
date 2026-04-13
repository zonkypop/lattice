// 03-audio.js - Audio (native-backed when ops available, stubs otherwise)

const audio = globalThis.__audio;

if (!audio) {
  // ======================= Stub fallback (no native ops) =======================
  console.log("[shims] audio: no native ops, using stubs");

  function createAudioParam(value) {
    return {
      value,
      setTargetAtTime(v) { this.value = v; },
      linearRampToValueAtTime(v) { this.value = v; },
      exponentialRampToValueAtTime(v) { this.value = v; },
      setValueAtTime(v) { this.value = v; },
      cancelScheduledValues() {},
    };
  }

  class AudioNode {
    constructor(ctx) { this.context = ctx; }
    connect() { return arguments[0]; }
    disconnect() {}
  }

  class GainNode extends AudioNode {
    constructor(ctx) { super(ctx); this.gain = createAudioParam(1); }
  }

  class AudioBufferSourceNode extends AudioNode {
    constructor(ctx) {
      super(ctx);
      this.buffer = null;
      this.loop = false;
      this.loopStart = 0;
      this.loopEnd = 0;
      this.onended = null;
      this.playbackRate = createAudioParam(1);
      this.detune = createAudioParam(0);
    }
    start() { this.onended?.(); }
    stop() {}
  }

  class AudioListener {
    constructor() {
      this.positionX = createAudioParam(0);
      this.positionY = createAudioParam(0);
      this.positionZ = createAudioParam(0);
      this.forwardX = createAudioParam(0);
      this.forwardY = createAudioParam(0);
      this.forwardZ = createAudioParam(-1);
      this.upX = createAudioParam(0);
      this.upY = createAudioParam(1);
      this.upZ = createAudioParam(0);
    }
    setPosition(x, y, z) {
      this.positionX.value = x; this.positionY.value = y; this.positionZ.value = z;
    }
    setOrientation(fx, fy, fz, ux, uy, uz) {
      this.forwardX.value = fx; this.forwardY.value = fy; this.forwardZ.value = fz;
      this.upX.value = ux; this.upY.value = uy; this.upZ.value = uz;
    }
  }

  class AudioContext {
    constructor() {
      this.currentTime = 0;
      this.sampleRate = 44100;
      this.listener = new AudioListener();
      this.destination = new AudioNode(this);
    }
    createGain() { return new GainNode(this); }
    createBufferSource() { return new AudioBufferSourceNode(this); }
    createOscillator() { return new AudioNode(this); }
    createBiquadFilter() { return new AudioNode(this); }
    createStereoPanner() { return new AudioNode(this); }
    createDelay() { return new AudioNode(this); }
    createDynamicsCompressor() { return new AudioNode(this); }
    createAnalyser() { return new AudioNode(this); }
    createPanner() { return new AudioNode(this); }
    createMediaElementSource() { return new AudioNode(this); }
    createMediaStreamSource() { return new AudioNode(this); }
    decodeAudioData(arrayBuffer, successCallback, errorCallback) {
      const p = Promise.resolve(null);
      if (typeof successCallback === 'function') p.then(successCallback);
      return p;
    }
    close() {}
  }

  globalThis.AudioContext = globalThis.window.AudioContext = AudioContext;
  globalThis.window.webkitAudioContext = AudioContext;
} else {
  // ======================= Native implementation =======================

  // --- AudioParam wrapper ---
  class NativeAudioParam {
    constructor(nodeId, paramName, defaultValue = 0) {
      this._nid = nodeId;
      this._p = paramName;
      this._value = defaultValue;
    }
    get value() { return this._value; }
    set value(v) {
      this._value = v;
      try { audio.op_audio_param_set_value(this._nid, this._p, v); } catch (_) {}
    }
    setValueAtTime(value, startTime) {
      this._value = value;
      try { audio.op_audio_param_set_value_at_time(this._nid, this._p, value, startTime); } catch (_) {}
      return this;
    }
    setTargetAtTime(value, startTime, timeConstant) {
      this._value = value;
      try { audio.op_audio_param_set_target_at_time(this._nid, this._p, value, startTime, timeConstant); } catch (_) {}
      return this;
    }
    linearRampToValueAtTime(value, endTime) {
      this._value = value;
      try { audio.op_audio_param_linear_ramp(this._nid, this._p, value, endTime); } catch (_) {}
      return this;
    }
    exponentialRampToValueAtTime(value, endTime) {
      this._value = value;
      try { audio.op_audio_param_exponential_ramp(this._nid, this._p, value, endTime); } catch (_) {}
      return this;
    }
    cancelScheduledValues() { return this; }
  }

  // --- Base AudioNode ---
  class NativeAudioNode {
    constructor(ctx, nodeId) {
      this.context = ctx;
      this.__id = nodeId;
    }
    connect(dest) {
      if (dest?.__id != null && this.__id != null) {
        try { audio.op_audio_connect(this.__id, dest.__id); } catch (_) {}
      }
      return dest;
    }
    disconnect() {
      if (this.__id != null) {
        try { audio.op_audio_disconnect(this.__id); } catch (_) {}
      }
    }
  }

  // --- GainNode ---
  class NativeGainNode extends NativeAudioNode {
    constructor(ctx, nodeId) {
      super(ctx, nodeId);
      this.gain = new NativeAudioParam(nodeId, "gain", 1);
    }
  }

  // --- AudioBufferSourceNode ---
  class NativeAudioBufferSourceNode extends NativeAudioNode {
    constructor(ctx, nodeId) {
      super(ctx, nodeId);
      this._buffer = null;
      this._loop = false;
      this._loopStart = 0;
      this._loopEnd = 0;
      this.onended = null;
      this.playbackRate = new NativeAudioParam(nodeId, "playbackRate", 1);
      this.detune = new NativeAudioParam(nodeId, "detune", 0);
    }
    get buffer() { return this._buffer; }
    set buffer(buf) {
      this._buffer = buf;
      if (buf?.__id != null && this.__id != null) {
        try { audio.op_audio_buffer_source_set_buffer(this.__id, buf.__id); } catch (_) {}
      }
    }
    get loop() { return this._loop; }
    set loop(v) {
      this._loop = v;
      try { audio.op_audio_buffer_source_set_loop(this.__id, !!v); } catch (_) {}
    }
    get loopStart() { return this._loopStart; }
    set loopStart(v) {
      this._loopStart = v;
      try { audio.op_audio_buffer_source_set_loop_start(this.__id, v); } catch (_) {}
    }
    get loopEnd() { return this._loopEnd; }
    set loopEnd(v) {
      this._loopEnd = v;
      try { audio.op_audio_buffer_source_set_loop_end(this.__id, v); } catch (_) {}
    }
    start(when = 0) {
      try { audio.op_audio_buffer_source_start(this.__id, when); } catch (_) {}
    }
    stop(when = 0) {
      try { audio.op_audio_buffer_source_stop(this.__id, when); } catch (_) {}
    }
  }

  // --- OscillatorNode ---
  class NativeOscillatorNode extends NativeAudioNode {
    constructor(ctx, nodeId) {
      super(ctx, nodeId);
      this._type = "sine";
      this.frequency = new NativeAudioParam(nodeId, "frequency", 440);
      this.detune = new NativeAudioParam(nodeId, "detune", 0);
    }
    get type() { return this._type; }
    set type(v) {
      this._type = v;
      try { audio.op_audio_oscillator_set_type(this.__id, v); } catch (_) {}
    }
    start(when = 0) {
      try { audio.op_audio_oscillator_start(this.__id, when); } catch (_) {}
    }
    stop(when = 0) {
      try { audio.op_audio_oscillator_stop(this.__id, when); } catch (_) {}
    }
  }

  // --- PannerNode ---
  class NativePannerNode extends NativeAudioNode {
    constructor(ctx, nodeId) {
      super(ctx, nodeId);
      this.positionX = new NativeAudioParam(nodeId, "positionX", 0);
      this.positionY = new NativeAudioParam(nodeId, "positionY", 0);
      this.positionZ = new NativeAudioParam(nodeId, "positionZ", 0);
      this.orientationX = new NativeAudioParam(nodeId, "orientationX", 1);
      this.orientationY = new NativeAudioParam(nodeId, "orientationY", 0);
      this.orientationZ = new NativeAudioParam(nodeId, "orientationZ", 0);
      this._panningModel = 'equalpower';
      this._distanceModel = 'inverse';
      this._refDistance = 1;
      this._maxDistance = 10000;
      this._rolloffFactor = 1;
      this._coneInnerAngle = 360;
      this._coneOuterAngle = 360;
      this._coneOuterGain = 0;
    }
    _configure(props) {
      if (this.__id != null) {
        try { audio.op_audio_panner_configure(this.__id, props); } catch (_) {}
      }
    }
    get panningModel() { return this._panningModel; }
    set panningModel(v) { this._panningModel = v; this._configure({ panning_model: v }); }
    get distanceModel() { return this._distanceModel; }
    set distanceModel(v) { this._distanceModel = v; this._configure({ distance_model: v }); }
    get refDistance() { return this._refDistance; }
    set refDistance(v) { this._refDistance = v; this._configure({ ref_distance: v }); }
    get maxDistance() { return this._maxDistance; }
    set maxDistance(v) { this._maxDistance = v; this._configure({ max_distance: v }); }
    get rolloffFactor() { return this._rolloffFactor; }
    set rolloffFactor(v) { this._rolloffFactor = v; this._configure({ rolloff_factor: v }); }
    get coneInnerAngle() { return this._coneInnerAngle; }
    set coneInnerAngle(v) { this._coneInnerAngle = v; this._configure({ cone_inner_angle: v }); }
    get coneOuterAngle() { return this._coneOuterAngle; }
    set coneOuterAngle(v) { this._coneOuterAngle = v; this._configure({ cone_outer_angle: v }); }
    get coneOuterGain() { return this._coneOuterGain; }
    set coneOuterGain(v) { this._coneOuterGain = v; this._configure({ cone_outer_gain: v }); }
    setPosition(x, y, z) {
      this.positionX.value = x;
      this.positionY.value = y;
      this.positionZ.value = z;
    }
    setOrientation(x, y, z) {
      this.orientationX.value = x;
      this.orientationY.value = y;
      this.orientationZ.value = z;
    }
  }

  // --- BiquadFilterNode ---
  class NativeBiquadFilterNode extends NativeAudioNode {
    constructor(ctx, nodeId) {
      super(ctx, nodeId);
      this._type = "lowpass";
      this.frequency = new NativeAudioParam(nodeId, "frequency", 350);
      this.Q = new NativeAudioParam(nodeId, "Q", 1);
      this.gain = new NativeAudioParam(nodeId, "gain", 0);
      this.detune = new NativeAudioParam(nodeId, "detune", 0);
    }
    get type() { return this._type; }
    set type(v) {
      this._type = v;
      try { audio.op_audio_biquad_set_type(this.__id, v); } catch (_) {}
    }
  }

  // --- StereoPannerNode ---
  class NativeStereoPannerNode extends NativeAudioNode {
    constructor(ctx, nodeId) {
      super(ctx, nodeId);
      this.pan = new NativeAudioParam(nodeId, "pan", 0);
    }
  }

  // --- DelayNode ---
  class NativeDelayNode extends NativeAudioNode {
    constructor(ctx, nodeId) {
      super(ctx, nodeId);
      this.delayTime = new NativeAudioParam(nodeId, "delayTime", 0);
    }
  }

  // --- DynamicsCompressorNode ---
  class NativeDynamicsCompressorNode extends NativeAudioNode {
    constructor(ctx, nodeId) {
      super(ctx, nodeId);
      this.threshold = new NativeAudioParam(nodeId, "threshold", -24);
      this.knee = new NativeAudioParam(nodeId, "knee", 30);
      this.ratio = new NativeAudioParam(nodeId, "ratio", 12);
      this.attack = new NativeAudioParam(nodeId, "attack", 0.003);
      this.release = new NativeAudioParam(nodeId, "release", 0.25);
    }
    get reduction() { return 0; }
  }

  // --- AnalyserNode (stub — analysis methods are TODO) ---
  class NativeAnalyserNode extends NativeAudioNode {
    constructor(ctx, nodeId) {
      super(ctx, nodeId);
      this.fftSize = 2048;
      this.smoothingTimeConstant = 0.8;
    }
    get frequencyBinCount() { return this.fftSize / 2; }
    getByteFrequencyData(arr) { if (arr) arr.fill(0); }
    getFloatFrequencyData(arr) { if (arr) arr.fill(-Infinity); }
    getByteTimeDomainData(arr) { if (arr) arr.fill(128); }
    getFloatTimeDomainData(arr) { if (arr) arr.fill(0); }
  }

  // --- AudioListener ---
  // Params must be reactive: Three.js sets listener.positionX.value = x each frame.
  // We batch updates via microtask so all 9 params flush once per frame.
  class NativeAudioListener {
    constructor(ctxId) {
      this._ctxId = ctxId;
      this._dirty = false;
      const self = this;
      const makeParam = (defaultVal) => {
        const p = { _v: defaultVal };
        Object.defineProperty(p, 'value', {
          get() { return p._v; },
          set(v) { p._v = v; self._markDirty(); },
        });
        p.setValueAtTime = (v) => { p.value = v; return p; };
        return p;
      };
      this.positionX = makeParam(0);
      this.positionY = makeParam(0);
      this.positionZ = makeParam(0);
      this.forwardX = makeParam(0);
      this.forwardY = makeParam(0);
      this.forwardZ = makeParam(-1);
      this.upX = makeParam(0);
      this.upY = makeParam(1);
      this.upZ = makeParam(0);
    }
    _markDirty() {
      if (this._dirty) return;
      this._dirty = true;
      queueMicrotask(() => {
        this._dirty = false;
        try {
          audio.op_audio_listener_set_position(this._ctxId,
            this.positionX._v, this.positionY._v, this.positionZ._v);
          audio.op_audio_listener_set_orientation(this._ctxId,
            this.forwardX._v, this.forwardY._v, this.forwardZ._v,
            this.upX._v, this.upY._v, this.upZ._v);
        } catch (_) {}
      });
    }
    setPosition(x, y, z) {
      this.positionX._v = x; this.positionY._v = y; this.positionZ._v = z;
      this._markDirty();
    }
    setOrientation(fx, fy, fz, ux, uy, uz) {
      this.forwardX._v = fx; this.forwardY._v = fy; this.forwardZ._v = fz;
      this.upX._v = ux; this.upY._v = uy; this.upZ._v = uz;
      this._markDirty();
    }
  }

  // --- AudioContext ---
  class NativeAudioContext {
    constructor() {
      const info = audio.op_audio_create_context();
      this._id = info.id;
      this.sampleRate = info.sample_rate;
      this.destination = new NativeAudioNode(this, info.destination_id);
      this.listener = new NativeAudioListener(this._id);
    }
    get currentTime() {
      return audio.op_audio_context_current_time(this._id);
    }
    createGain() {
      const { id } = audio.op_audio_create_gain(this._id);
      return new NativeGainNode(this, id);
    }
    createBufferSource() {
      const { id } = audio.op_audio_create_buffer_source(this._id);
      return new NativeAudioBufferSourceNode(this, id);
    }
    createOscillator() {
      const { id } = audio.op_audio_create_oscillator(this._id);
      return new NativeOscillatorNode(this, id);
    }
    createPanner() {
      const { id } = audio.op_audio_create_panner(this._id);
      return new NativePannerNode(this, id);
    }
    createBiquadFilter() {
      const { id } = audio.op_audio_create_biquad_filter(this._id);
      return new NativeBiquadFilterNode(this, id);
    }
    createStereoPanner() {
      const { id } = audio.op_audio_create_stereo_panner(this._id);
      return new NativeStereoPannerNode(this, id);
    }
    createDelay(maxDelayTime = 1.0) {
      const { id } = audio.op_audio_create_delay(this._id, maxDelayTime);
      return new NativeDelayNode(this, id);
    }
    createDynamicsCompressor() {
      const { id } = audio.op_audio_create_dynamics_compressor(this._id);
      return new NativeDynamicsCompressorNode(this, id);
    }
    createAnalyser() {
      const { id } = audio.op_audio_create_analyser(this._id);
      return new NativeAnalyserNode(this, id);
    }
    createMediaElementSource() { return new NativeAudioNode(this, null); }
    createMediaStreamSource() { return new NativeAudioNode(this, null); }
    decodeAudioData(arrayBuffer, successCallback, errorCallback) {
      const promise = new Promise((resolve, reject) => {
        try {
          const bytes = arrayBuffer instanceof Uint8Array
            ? arrayBuffer
            : new Uint8Array(arrayBuffer);
          const info = audio.op_audio_decode_audio_data(this._id, bytes);
          resolve({
            __id: info.id,
            duration: info.duration,
            length: info.length,
            sampleRate: info.sample_rate,
            numberOfChannels: info.number_of_channels,
          });
        } catch (e) {
          reject(e);
        }
      });
      // Support callback-style API (W3C spec allows both)
      if (typeof successCallback === 'function') {
        promise.then(successCallback, errorCallback || (() => {}));
      }
      return promise;
    }
    close() {
      audio.op_audio_context_close(this._id);
    }
  }

  globalThis.AudioContext = globalThis.window.AudioContext = NativeAudioContext;
  globalThis.window.webkitAudioContext = NativeAudioContext;

  console.log("[shims] audio initialized (native)");
}

// ======================= Audio (HTMLAudioElement) =======================
// Real implementation using native AudioContext + Deno.readFile.
// Set globalThis.__audioBasePath__ before modules load to control path resolution.
// Paths like "./src/audio/foo.mp3" are stripped of "./" and prefixed with the base.

const _bufferCache = new Map();
let _sharedAudioCtx = null;
let _resolvedPrefix = undefined; // undefined = not yet resolved, null = no prefix works

function _getAudioCtx() {
  if (!_sharedAudioCtx) {
    try { _sharedAudioCtx = new AudioContext(); } catch (_) {}
  }
  return _sharedAudioCtx;
}

async function _resolveAndRead(rawPath) {
  let path = rawPath.startsWith('./') ? rawPath.slice(2) : rawPath;

  // If we already know the working prefix, use it directly
  if (_resolvedPrefix !== undefined) {
    return Deno.readFile((_resolvedPrefix || '') + path);
  }

  // Try candidate prefixes in order until one works, then cache the winner
  const explicit = globalThis.__audioBasePath__;
  const scriptDir = globalThis.__scriptDir__ || '';
  const candidates = [];
  if (explicit) candidates.push(explicit);
  candidates.push('');             // cwd-relative
  if (scriptDir) {
    candidates.push(scriptDir);    // entry script dir  (e.g. "js/")
    // Also try one level deeper for common "web root inside script dir" layouts
    // e.g. js/u/ when entry is js/entry.js
    try {
      const entries = Deno.readDirSync(scriptDir);
      for (const e of entries) {
        if (e.isDirectory && !e.name.startsWith('.')) {
          candidates.push(scriptDir + e.name + '/');
        }
      }
    } catch (_) {}
  }

  for (const prefix of candidates) {
    try {
      const data = await Deno.readFile(prefix + path);
      _resolvedPrefix = prefix;
      if (prefix) console.log(`[Audio] resolved base path: ${prefix}`);
      return data;
    } catch (_) {}
  }

  _resolvedPrefix = null;
  throw new Error(`File not found: ${rawPath}`);
}

async function _loadAudioBuffer(src) {
  if (_bufferCache.has(src)) return _bufferCache.get(src);

  const data = await _resolveAndRead(src);
  const ctx = _getAudioCtx();
  if (!ctx) return null;

  const buffer = await ctx.decodeAudioData(data.buffer);
  _bufferCache.set(src, buffer);
  return buffer;
}

class Audio {
  constructor(src) {
    this.src = src || "";
    this.currentTime = 0;
    this.duration = 0;
    this.paused = true;
    this.ended = false;
    this.loop = false;
    this.volume = 1;
    this.muted = false;
    this.playbackRate = 1;
    this.readyState = 0;
    this.autoplay = false;
    this.preload = "auto";
    this.crossOrigin = null;
    this._listeners = {};
    this._buffer = null;
    this._source = null;
    this._loadPromise = this.src ? this._load() : Promise.resolve();
  }

  async _load() {
    try {
      this._buffer = await _loadAudioBuffer(this.src);
      if (this._buffer) {
        this.duration = this._buffer.duration;
        this.readyState = 4;
      }
    } catch (e) {
      console.warn(`[Audio] Failed to load ${this.src}: ${e.message}`);
    }
  }

  play() {
    if (!this._buffer) {
      // Not loaded yet — play after load completes
      this._loadPromise?.then(() => { if (this._buffer) this._playNow(); });
      return Promise.resolve();
    }
    this._playNow();
    return Promise.resolve();
  }

  _playNow() {
    const ctx = _getAudioCtx();
    if (!ctx || !this._buffer) return;
    // Stop any current playback
    if (this._source) { try { this._source.stop(); } catch (_) {} }
    const source = ctx.createBufferSource();
    source.buffer = this._buffer;
    source.loop = this.loop;
    const gain = ctx.createGain();
    gain.gain.value = this.muted ? 0 : this.volume;
    source.connect(gain);
    gain.connect(ctx.destination);
    source.start();
    this._source = source;
    this.paused = false;
    this.ended = false;
  }

  pause() {
    if (this._source) { try { this._source.stop(); } catch (_) {} this._source = null; }
    this.paused = true;
  }

  load() { if (this.src) this._loadPromise = this._load(); }

  addEventListener(e, cb) { (this._listeners[e] ??= []).push(cb); }
  removeEventListener(e, cb) {
    if (this._listeners[e]) this._listeners[e] = this._listeners[e].filter((c) => c !== cb);
  }
  dispatchEvent(evt) { (this._listeners[evt.type] || []).forEach((cb) => cb(evt)); return true; }
  cloneNode() { return new Audio(this.src); }
}

globalThis.Audio = globalThis.window.Audio = Audio;
