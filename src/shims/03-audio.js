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
    cancelScheduledValues(cancelTime = 0) {
      try { audio.op_audio_param_cancel_scheduled_values(this._nid, this._p, cancelTime); } catch (_) {}
      return this;
    }
    cancelAndHoldAtTime(cancelTime = 0) {
      try { audio.op_audio_param_cancel_and_hold(this._nid, this._p, cancelTime); } catch (_) {}
      return this;
    }
    setValueCurveAtTime(values, startTime, duration) { return this; }
  }

  // --- Base AudioNode ---
  class NativeAudioNode {
    constructor(ctx, nodeId) {
      this.context = ctx;
      this.__id = nodeId;
      this.numberOfInputs = 1;
      this.numberOfOutputs = 1;
      this._channelCount = 2;
      this._channelCountMode = "max";
      this._channelInterpretation = "speakers";
    }
    get channelCount() { return this._channelCount; }
    set channelCount(v) {
      this._channelCount = v;
      if (this.__id != null) {
        try { audio.op_audio_set_channel_count(this.__id, v); } catch (_) {}
      }
    }
    get channelCountMode() { return this._channelCountMode; }
    set channelCountMode(v) {
      this._channelCountMode = v;
      if (this.__id != null) {
        try { audio.op_audio_set_channel_count_mode(this.__id, v); } catch (_) {}
      }
    }
    get channelInterpretation() { return this._channelInterpretation; }
    set channelInterpretation(v) {
      this._channelInterpretation = v;
      if (this.__id != null) {
        try { audio.op_audio_set_channel_interpretation(this.__id, v); } catch (_) {}
      }
    }
    connect(dest, outputIndex = 0, inputIndex = 0) {
      if (this.__id == null) return dest;
      if (dest instanceof NativeAudioParam) {
        // node.connect(audioParam, output) — parameter modulation
        if (dest._nid >= 0) {
          try { audio.op_audio_connect_param(this.__id, dest._nid, dest._p, outputIndex); } catch (_) {}
        }
        return undefined; // Web Audio API spec: returns void for param connections
      }
      if (dest?.__id != null) {
        try { audio.op_audio_connect(this.__id, dest.__id, outputIndex, inputIndex); } catch (_) {}
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
        if (buf._syncToNative) buf._syncToNative();
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
    start(when = 0, offset = 0, duration = 0) {
      try { audio.op_audio_buffer_source_start(this.__id, when, offset, duration); } catch (_) {}
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
    setPeriodicWave(wave) {
      // Store for reference; native PeriodicWave support is TODO
      this._periodicWave = wave;
      this._type = "custom";
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

  // --- ListenerParam (extends NativeAudioParam for instanceof compatibility) ---
  // Audio libraries check `listener.positionX instanceof AudioParam`.
  // We batch-flush all listener params via microtask so setting 9 params per frame
  // only triggers one native call.
  class ListenerParam extends NativeAudioParam {
    constructor(listener, defaultVal) {
      // Pass dummy node/param — we override value set/get to use the batch flush
      super(-1, "", defaultVal);
      this._listener = listener;
    }
    get value() { return this._value; }
    set value(v) {
      this._value = v;
      this._listener._markDirty();
    }
  }

  // --- AudioListener ---
  class NativeAudioListener {
    constructor(ctxId) {
      this._ctxId = ctxId;
      this._dirty = false;
      this.positionX = new ListenerParam(this, 0);
      this.positionY = new ListenerParam(this, 0);
      this.positionZ = new ListenerParam(this, 0);
      this.forwardX = new ListenerParam(this, 0);
      this.forwardY = new ListenerParam(this, 0);
      this.forwardZ = new ListenerParam(this, -1);
      this.upX = new ListenerParam(this, 0);
      this.upY = new ListenerParam(this, 1);
      this.upZ = new ListenerParam(this, 0);
    }
    _markDirty() {
      if (this._dirty) return;
      this._dirty = true;
      queueMicrotask(() => {
        this._dirty = false;
        try {
          audio.op_audio_listener_set_position(this._ctxId,
            this.positionX._value, this.positionY._value, this.positionZ._value);
          audio.op_audio_listener_set_orientation(this._ctxId,
            this.forwardX._value, this.forwardY._value, this.forwardZ._value,
            this.upX._value, this.upY._value, this.upZ._value);
        } catch (_) {}
      });
    }
    setPosition(x, y, z) {
      this.positionX._value = x; this.positionY._value = y; this.positionZ._value = z;
      this._markDirty();
    }
    setOrientation(fx, fy, fz, ux, uy, uz) {
      this.forwardX._value = fx; this.forwardY._value = fy; this.forwardZ._value = fz;
      this.upX._value = ux; this.upY._value = uy; this.upZ._value = uz;
      this._markDirty();
    }
  }

  // --- WaveShaperNode (native) ---
  class NativeWaveShaperNode extends NativeAudioNode {
    constructor(ctx, nodeId) {
      super(ctx, nodeId);
      this._curve = null;
      this.oversample = "none";
    }
    get curve() { return this._curve; }
    set curve(v) {
      this._curve = v;
      if (v && this.__id != null) {
        try {
          const f32 = v instanceof Float32Array ? v : new Float32Array(v);
          audio.op_audio_wave_shaper_set_curve(this.__id, f32);
        } catch (_) {}
      }
    }
  }

  // --- ConstantSourceNode (native) ---
  class NativeConstantSourceNode extends NativeAudioNode {
    constructor(ctx, nodeId) {
      super(ctx, nodeId);
      this.offset = new NativeAudioParam(nodeId, "offset", 1);
    }
    start(when = 0) {
      try { audio.op_audio_constant_source_start(this.__id, when); } catch (_) {}
    }
    stop(when = 0) {
      try { audio.op_audio_constant_source_stop(this.__id, when); } catch (_) {}
    }
  }

  // --- ChannelMergerNode / ChannelSplitterNode (native) ---
  class NativeChannelMergerNode extends NativeAudioNode {
    constructor(ctx, nodeId, numberOfInputs = 6) {
      super(ctx, nodeId);
      this.numberOfInputs = numberOfInputs;
      this.numberOfOutputs = 1;
    }
  }

  class NativeChannelSplitterNode extends NativeAudioNode {
    constructor(ctx, nodeId, numberOfOutputs = 6) {
      super(ctx, nodeId);
      this.numberOfInputs = 1;
      this.numberOfOutputs = numberOfOutputs;
    }
  }

  // --- ConvolverNode (native) ---
  class NativeConvolverNode extends NativeAudioNode {
    constructor(ctx, nodeId) {
      super(ctx, nodeId);
      this._buffer = null;
      this.normalize = true;
    }
    get buffer() { return this._buffer; }
    set buffer(v) {
      this._buffer = v;
      if (v?.__id != null && this.__id != null) {
        if (v._syncToNative) v._syncToNative();
        try { audio.op_audio_convolver_set_buffer(this.__id, v.__id); } catch (_) {}
      }
    }
  }

  // --- AudioContext ---
  class NativeAudioContext {
    constructor(skipInit) {
      if (skipInit === true) return; // subclass handles init
      const info = audio.op_audio_create_context();
      this._id = info.id;
      this.sampleRate = info.sample_rate;
      this.state = "running";
      this.destination = new NativeAudioNode(this, info.destination_id);
      this.listener = new NativeAudioListener(this._id);
      this.audioWorklet = { addModule: () => Promise.resolve() };
    }
    get currentTime() {
      return audio.op_audio_context_current_time(this._id);
    }
    resume() { this.state = "running"; return Promise.resolve(); }
    suspend() { this.state = "suspended"; return Promise.resolve(); }
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
    createWaveShaper() {
      const { id } = audio.op_audio_create_wave_shaper(this._id);
      return new NativeWaveShaperNode(this, id);
    }
    createConstantSource() {
      const { id } = audio.op_audio_create_constant_source(this._id);
      return new NativeConstantSourceNode(this, id);
    }
    createChannelMerger(numberOfInputs = 6) {
      const { id } = audio.op_audio_create_channel_merger(this._id, numberOfInputs);
      return new NativeChannelMergerNode(this, id, numberOfInputs);
    }
    createChannelSplitter(numberOfOutputs = 6) {
      const { id } = audio.op_audio_create_channel_splitter(this._id, numberOfOutputs);
      return new NativeChannelSplitterNode(this, id, numberOfOutputs);
    }
    createConvolver() {
      const { id } = audio.op_audio_create_convolver(this._id);
      return new NativeConvolverNode(this, id);
    }
    createPeriodicWave(real, imag) { return { real, imag }; }
    createBuffer(numberOfChannels, length, sampleRate) {
      const info = audio.op_audio_create_buffer(this._id, numberOfChannels, length, sampleRate);
      const channels = [];
      for (let i = 0; i < numberOfChannels; i++) {
        channels.push(new Float32Array(length));
      }
      return new NativeAudioBuffer({
        __id: info.id,
        numberOfChannels: info.number_of_channels,
        length: info.length,
        sampleRate: info.sample_rate,
        duration: info.duration,
        _channels: channels,
      });
    }
    createMediaElementSource() { return new NativeAudioNode(this, null); }
    createMediaStreamSource() { return new NativeAudioNode(this, null); }
    createMediaStreamDestination() {
      const node = new NativeAudioNode(this, null);
      node.stream = { getTracks: () => [] };
      return node;
    }
    decodeAudioData(arrayBuffer, successCallback, errorCallback) {
      const promise = new Promise((resolve, reject) => {
        try {
          const bytes = arrayBuffer instanceof Uint8Array
            ? arrayBuffer
            : new Uint8Array(arrayBuffer);
          const info = audio.op_audio_decode_audio_data(this._id, bytes);
          resolve(new NativeAudioBuffer({
            __id: info.id,
            duration: info.duration,
            length: info.length,
            sampleRate: info.sample_rate,
            numberOfChannels: info.number_of_channels,
          }));
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
  globalThis.AudioNode = NativeAudioNode;
  globalThis.AudioParam = NativeAudioParam;
  class NativeAudioBuffer {
    constructor(options = {}) {
      this.__id = options.__id ?? null;
      this.numberOfChannels = options.numberOfChannels ?? 1;
      this.length = options.length ?? 0;
      this.sampleRate = options.sampleRate ?? 44100;
      this.duration = options.duration ?? 0;
      this._channels = options._channels ?? [];
      // Track whether JS-side channel data needs syncing to native.
      // Buffers from decodeAudioData/startRendering already have native data.
      // Buffers from createBuffer have local Float32Arrays that may be written to.
      this._localOnly = !!options._channels && options._channels.length > 0;
    }
    getChannelData(ch) {
      if (!this._channels[ch]) {
        // Try to fetch from native buffer (e.g. decoded audio data)
        if (this.__id != null && this.length > 0) {
          try {
            const data = audio.op_audio_buffer_get_channel_data(this.__id, ch);
            this._channels[ch] = new Float32Array(data.buffer, data.byteOffset, data.byteLength / 4);
          } catch (_) {
            this._channels[ch] = new Float32Array(this.length);
          }
        } else {
          this._channels[ch] = new Float32Array(this.length);
        }
      }
      this._localOnly = true;
      return this._channels[ch];
    }
    copyToChannel(source, ch, startInChannel = 0) {
      if (!this._channels[ch]) this._channels[ch] = new Float32Array(this.length);
      this._channels[ch].set(source, startInChannel);
      if (this.__id != null) {
        try { audio.op_audio_buffer_copy_to_channel(this.__id, source, ch); } catch (_) {}
      }
    }
    copyFromChannel(dest, ch, startInChannel = 0) {
      const src = this._channels[ch];
      if (src) dest.set(src.subarray(startInChannel, startInChannel + dest.length));
    }
    // Flush all JS-side channel data to the native buffer
    _syncToNative() {
      if (!this._localOnly || this.__id == null) return;
      for (let ch = 0; ch < this._channels.length; ch++) {
        const data = this._channels[ch];
        if (data && data.length > 0) {
          try { audio.op_audio_buffer_copy_to_channel(this.__id, data, ch); } catch (_) {}
        }
      }
      this._localOnly = false;
    }
  }
  globalThis.AudioBuffer = NativeAudioBuffer;
  globalThis.BaseAudioContext = NativeAudioContext;
  // --- OfflineAudioContext (native) ---
  // Must extend NativeAudioContext so instanceof AudioContext is true.
  // Libraries that deep-merge options treat AudioContext instances as leaf
  // values; without this, the offline context gets deep-merged incorrectly.
  class NativeOfflineAudioContext extends NativeAudioContext {
    constructor(numberOfChannels, length, sampleRate) {
      super(true); // skip NativeAudioContext init
      const info = audio.op_offline_context_create(numberOfChannels, length, sampleRate);
      this._id = info.id;
      this.sampleRate = info.sample_rate;
      this.state = "running";
      this.destination = new NativeAudioNode(this, info.destination_id);
      this.listener = new NativeAudioListener(this._id);
      this.length = length;
      this.audioWorklet = { addModule: () => Promise.resolve() };
    }
    get currentTime() {
      return audio.op_audio_context_current_time(this._id);
    }
    resume() { this.state = "running"; return Promise.resolve(); }
    suspend() { this.state = "suspended"; return Promise.resolve(); }
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
    createConstantSource() {
      const { id } = audio.op_audio_create_constant_source(this._id);
      return new NativeConstantSourceNode(this, id);
    }
    createConvolver() {
      const { id } = audio.op_audio_create_convolver(this._id);
      return new NativeConvolverNode(this, id);
    }
    createBiquadFilter() {
      const { id } = audio.op_audio_create_biquad_filter(this._id);
      return new NativeBiquadFilterNode(this, id);
    }
    createDelay(maxDelayTime = 1.0) {
      const { id } = audio.op_audio_create_delay(this._id, maxDelayTime);
      return new NativeDelayNode(this, id);
    }
    createDynamicsCompressor() {
      const { id } = audio.op_audio_create_dynamics_compressor(this._id);
      return new NativeDynamicsCompressorNode(this, id);
    }
    createStereoPanner() {
      const { id } = audio.op_audio_create_stereo_panner(this._id);
      return new NativeStereoPannerNode(this, id);
    }
    createAnalyser() {
      const { id } = audio.op_audio_create_analyser(this._id);
      return new NativeAnalyserNode(this, id);
    }
    createPanner() {
      const { id } = audio.op_audio_create_panner(this._id);
      return new NativePannerNode(this, id);
    }
    createWaveShaper() {
      const { id } = audio.op_audio_create_wave_shaper(this._id);
      return new NativeWaveShaperNode(this, id);
    }
    createChannelMerger(numberOfInputs = 6) {
      const { id } = audio.op_audio_create_channel_merger(this._id, numberOfInputs);
      return new NativeChannelMergerNode(this, id, numberOfInputs);
    }
    createChannelSplitter(numberOfOutputs = 6) {
      const { id } = audio.op_audio_create_channel_splitter(this._id, numberOfOutputs);
      return new NativeChannelSplitterNode(this, id, numberOfOutputs);
    }
    createPeriodicWave(real, imag) { return { real, imag }; }
    createBuffer(numberOfChannels, length, sampleRate) {
      const info = audio.op_audio_create_buffer(this._id, numberOfChannels, length, sampleRate);
      const channels = [];
      for (let i = 0; i < numberOfChannels; i++) {
        channels.push(new Float32Array(length));
      }
      return new NativeAudioBuffer({
        __id: info.id,
        numberOfChannels: info.number_of_channels,
        length: info.length,
        sampleRate: info.sample_rate,
        duration: info.duration,
        _channels: channels,
      });
    }
    decodeAudioData(arrayBuffer, successCallback, errorCallback) {
      const promise = new Promise((resolve, reject) => {
        try {
          const bytes = arrayBuffer instanceof Uint8Array
            ? arrayBuffer
            : new Uint8Array(arrayBuffer);
          const info = audio.op_audio_decode_audio_data(this._id, bytes);
          resolve(new NativeAudioBuffer({
            __id: info.id,
            duration: info.duration,
            length: info.length,
            sampleRate: info.sample_rate,
            numberOfChannels: info.number_of_channels,
          }));
        } catch (e) {
          reject(e);
        }
      });
      if (typeof successCallback === 'function') {
        promise.then(successCallback, errorCallback || (() => {}));
      }
      return promise;
    }
    startRendering() {
      return new Promise((resolve, reject) => {
        try {
          const info = audio.op_offline_context_start_rendering(this._id);
          this.state = "closed";
          resolve(new NativeAudioBuffer({
            __id: info.id,
            duration: info.duration,
            length: info.length,
            sampleRate: info.sample_rate,
            numberOfChannels: info.number_of_channels,
          }));
        } catch (e) {
          reject(e);
        }
      });
    }
    close() { this.state = "closed"; }
  }
  globalThis.OfflineAudioContext = NativeOfflineAudioContext;
  globalThis.AudioWorkletNode = globalThis.AudioWorkletNode ?? class AudioWorkletNode {};

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

async function _readFrom(url) {
  if (url.startsWith('http://') || url.startsWith('https://')) {
    const resp = await fetch(url);
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    return new Uint8Array(await resp.arrayBuffer());
  }
  return Deno.readFile(url);
}

async function _resolveAndRead(rawPath) {
  let path = rawPath.startsWith('./') ? rawPath.slice(2) : rawPath;

  // If we already know the working prefix, use it directly
  if (_resolvedPrefix !== undefined) {
    return _readFrom((_resolvedPrefix || '') + path);
  }

  // Try candidate prefixes in order until one works, then cache the winner
  const explicit = globalThis.__audioBasePath__;
  const scriptDir = globalThis.__scriptDir__ || '';
  const isUrlMode = scriptDir.startsWith('http://') || scriptDir.startsWith('https://');
  const candidates = [];
  if (explicit) candidates.push(explicit);
  if (isUrlMode) {
    candidates.push(scriptDir);   // e.g. http://localhost:8000/
  } else {
    candidates.push('');           // cwd-relative
    if (scriptDir) {
      candidates.push(scriptDir);
      try {
        const entries = Deno.readDirSync(scriptDir);
        for (const e of entries) {
          if (e.isDirectory && !e.name.startsWith('.')) {
            candidates.push(scriptDir + e.name + '/');
          }
        }
      } catch (_) {}
    }
  }

  for (const prefix of candidates) {
    try {
      const data = await _readFrom(prefix + path);
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
    this._volume = 1;
    this.muted = false;
    this.playbackRate = 1;
    this.readyState = 0;
    this.autoplay = false;
    this.preload = "auto";
    this.crossOrigin = null;
    this._listeners = {};
    this._buffer = null;
    this._source = null;
    this._srcObject = null;
    this._remoteSource = null;
    this._remoteGain = null;
    this._loadPromise = this.src ? this._load() : Promise.resolve();
  }

  get volume() { return this._volume; }
  set volume(v) {
    this._volume = v;
    if (this._remoteGain) {
      this._remoteGain.gain.value = this.muted ? 0 : v;
    }
  }

  get srcObject() { return this._srcObject; }
  set srcObject(stream) {
    // Tear down previous remote playback
    if (this._remoteSource) {
      try { this._remoteSource.disconnect(); } catch (_) {}
      this._remoteSource = null;
    }
    if (this._remoteGain) {
      try { this._remoteGain.disconnect(); } catch (_) {}
      this._remoteGain = null;
    }
    this._srcObject = stream;
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
    // Remote stream playback via srcObject
    if (this._srcObject && this._srcObject._remoteTrackId != null) {
      this._playRemote();
      return Promise.resolve();
    }
    if (!this._buffer) {
      // Not loaded yet — play after load completes
      this._loadPromise?.then(() => { if (this._buffer) this._playNow(); });
      return Promise.resolve();
    }
    this._playNow();
    return Promise.resolve();
  }

  async _playRemote() {
    const stream = this._srcObject;
    if (!stream || stream._remoteTrackId == null) return;
    const __rtc = globalThis.__webrtc;
    if (!__rtc) return;

    const ctx = _getAudioCtx();
    if (!ctx) return;

    try {
      const result = await __rtc.op_rtc_create_remote_audio_source(
        stream._pcId,
        stream._remoteTrackId,
        ctx._id
      );
      // NativeAudioNode wraps a native node by ID
      this._remoteSource = new NativeAudioNode(ctx, result.node_id);
      this._remoteGain = ctx.createGain();
      this._remoteGain.gain.value = this.muted ? 0 : this._volume;
      this._remoteSource.connect(this._remoteGain);
      this._remoteGain.connect(ctx.destination);
      this.paused = false;
    } catch (e) {
      console.warn("[Audio] Failed to set up remote stream playback:", e.message);
    }
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
    gain.gain.value = this.muted ? 0 : this._volume;
    source.connect(gain);
    gain.connect(ctx.destination);
    source.start();
    this._source = source;
    this.paused = false;
    this.ended = false;
  }

  pause() {
    if (this._source) { try { this._source.stop(); } catch (_) {} this._source = null; }
    // Disconnect remote playback
    if (this._remoteSource) {
      try { this._remoteSource.disconnect(); } catch (_) {}
      this._remoteSource = null;
    }
    if (this._remoteGain) {
      try { this._remoteGain.disconnect(); } catch (_) {}
      this._remoteGain = null;
    }
    this.paused = true;
  }

  load() { if (this.src) this._loadPromise = this._load(); }
  remove() { this.pause(); }

  addEventListener(e, cb) { (this._listeners[e] ??= []).push(cb); }
  removeEventListener(e, cb) {
    if (this._listeners[e]) this._listeners[e] = this._listeners[e].filter((c) => c !== cb);
  }
  dispatchEvent(evt) { (this._listeners[evt.type] || []).forEach((cb) => cb(evt)); return true; }
  cloneNode() { return new Audio(this.src); }
}

globalThis.Audio = globalThis.window.Audio = Audio;
