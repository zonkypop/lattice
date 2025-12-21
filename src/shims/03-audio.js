// 03-audio.js - Audio stubs

function createAudioParam(value) {
  return {
    value,
    setTargetAtTime(v) {
      this.value = v;
    },
    linearRampToValueAtTime(v) {
      this.value = v;
    },
  };
}

class AudioNode {
  constructor(ctx) {
    this.context = ctx;
  }
  connect() {}
  disconnect() {}
}

class GainNode extends AudioNode {
  constructor(ctx) {
    super(ctx);
    this.gain = createAudioParam(1);
  }
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
  start() {
    this.onended?.();
  }
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
    this.positionX.value = x;
    this.positionY.value = y;
    this.positionZ.value = z;
  }
  setOrientation(fx, fy, fz, ux, uy, uz) {
    this.forwardX.value = fx;
    this.forwardY.value = fy;
    this.forwardZ.value = fz;
    this.upX.value = ux;
    this.upY.value = uy;
    this.upZ.value = uz;
  }
}

class AudioContext {
  constructor() {
    this.currentTime = 0;
    this.listener = new AudioListener();
    this.destination = new AudioNode(this);
  }
  createGain() {
    return new GainNode(this);
  }
  createBufferSource() {
    return new AudioBufferSourceNode(this);
  }
  createMediaElementSource() {
    return new AudioNode(this);
  }
  createMediaStreamSource() {
    return new AudioNode(this);
  }
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
  }
  play() {
    this.paused = false;
    return Promise.resolve();
  }
  pause() {
    this.paused = true;
  }
  load() {}
  addEventListener(e, cb) {
    (this._listeners[e] ??= []).push(cb);
  }
  removeEventListener(e, cb) {
    if (this._listeners[e])
      this._listeners[e] = this._listeners[e].filter((c) => c !== cb);
  }
  dispatchEvent(evt) {
    (this._listeners[evt.type] || []).forEach((cb) => cb(evt));
    return true;
  }
  cloneNode() {
    return new Audio(this.src);
  }
}

// Install
globalThis.AudioContext = globalThis.window.AudioContext = AudioContext;
globalThis.window.webkitAudioContext = AudioContext;
globalThis.Audio = globalThis.window.Audio = Audio;

console.log("[shims] audio initialized");
