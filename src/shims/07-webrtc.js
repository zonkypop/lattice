// 07-webrtc.js - RTCPeerConnection, RTCDataChannel, and voice chat shims

const rtc = globalThis.__webrtc;
const LOG = () => {}; // set to console.log.bind(console, "[webrtc-shim]") for debugging

// ======================= MediaStream / MediaStreamTrack =======================

class MediaStreamTrack {
  constructor(id, kind) {
    this.id = String(id);
    this.kind = kind || "audio";
    this.enabled = true;
    this.readyState = "live";
    this.muted = false;
    this.label = kind || "audio";
  }
  stop() { this.readyState = "ended"; }
  clone() { return new MediaStreamTrack(this.id, this.kind); }
  getSettings() { return {}; }
  getCapabilities() { return {}; }
  getConstraints() { return {}; }
}

class MediaStream {
  constructor(streamId, tracks) {
    this._id = streamId;
    this._tracks = tracks || [];
    this.id = String(streamId);
    this.active = true;
    // Set by ontrack handler for remote streams
    this._pcId = null;
    this._remoteTrackId = null;
  }
  getTracks() { return [...this._tracks]; }
  getAudioTracks() { return this._tracks.filter(t => t.kind === "audio"); }
  getVideoTracks() { return []; }
  addTrack(track) { this._tracks.push(track); }
  removeTrack(track) { this._tracks = this._tracks.filter(t => t !== track); }
}

globalThis.MediaStream = MediaStream;
globalThis.MediaStreamTrack = MediaStreamTrack;

// ======================= Stubs (no native ops) =======================

if (!rtc) {
  LOG("native ops NOT available - using stubs");
  globalThis.RTCPeerConnection = class RTCPeerConnection {
    createDataChannel() { return null; }
    createOffer() { return Promise.resolve({ type: "offer", sdp: "" }); }
    createAnswer() { return Promise.resolve({ type: "answer", sdp: "" }); }
    setLocalDescription() { return Promise.resolve(); }
    setRemoteDescription() { return Promise.resolve(); }
    addIceCandidate() { return Promise.resolve(); }
    addTrack() { return { track: null }; }
    close() {}
  };
} else {
  LOG("native ops available");

  class RTCDataChannel extends EventTarget {
    constructor(pcId, channelId, label, options) {
      super();
      this._pcId = pcId;
      this._id = channelId;
      this.label = label;
      this.readyState = "connecting";
      this.binaryType = "arraybuffer";
      this.ordered = options?.ordered !== false;
      this.maxPacketLifeTime = options?.maxPacketLifeTime ?? null;
      this.maxRetransmits = options?.maxRetransmits ?? null;
      this.protocol = options?.protocol || "";
      this.negotiated = options?.negotiated || false;
      this.id = options?.id ?? null;
      this.bufferedAmount = 0;
      this._onopen = null;
      this._onclose = null;
      this._onmessage = null;
      this._onerror = null;
      this.onbufferedamountlow = null;
    }

    // Smart setters: if handler is assigned after state change, fire immediately
    get onopen() { return this._onopen; }
    set onopen(fn) {
      this._onopen = fn;
      if (fn && this.readyState === "open") {
        // PeerJS sets onopen after ondatachannel - fire immediately
        queueMicrotask(() => {
          if (this._onopen === fn && this.readyState === "open") {
            fn(new Event("open"));
          }
        });
      }
    }
    get onclose() { return this._onclose; }
    set onclose(fn) { this._onclose = fn; }
    get onmessage() { return this._onmessage; }
    set onmessage(fn) { this._onmessage = fn; }
    get onerror() { return this._onerror; }
    set onerror(fn) { this._onerror = fn; }

    send(data) {
      if (this.readyState !== "open") {
        throw new DOMException("Channel is not open", "InvalidStateError");
      }
      if (typeof data === "string") {
        rtc.op_rtc_send_text(this._pcId, this._id, data);
      } else {
        let bytes;
        if (data instanceof ArrayBuffer) {
          bytes = new Uint8Array(data);
        } else if (ArrayBuffer.isView(data)) {
          bytes = new Uint8Array(data.buffer, data.byteOffset, data.byteLength);
        } else {
          throw new TypeError("Invalid data type for send()");
        }
        rtc.op_rtc_send_binary(this._pcId, this._id, bytes);
      }
    }

    close() {
      this.readyState = "closed";
    }

    _handleOpen() {
      this.readyState = "open";
      LOG("datachannel OPEN:", this.label, "id:", this._id);
      const evt = new Event("open");
      if (this._onopen) this._onopen(evt);
      this.dispatchEvent(evt);
    }

    _handleClose() {
      this.readyState = "closed";
      LOG("datachannel CLOSE:", this.label);
      const evt = new Event("close");
      if (this._onclose) this._onclose(evt);
      this.dispatchEvent(evt);
    }

    _handleMessage(data, binary) {
      let payload;
      if (binary) {
        const str = atob(data);
        const buf = new ArrayBuffer(str.length);
        const view = new Uint8Array(buf);
        for (let i = 0; i < str.length; i++) view[i] = str.charCodeAt(i);
        payload = buf;
      } else {
        payload = data;
      }
      const evt = new MessageEvent("message", { data: payload });
      if (this._onmessage) this._onmessage(evt);
      this.dispatchEvent(evt);
    }
  }

  class RTCSessionDescription {
    constructor(init) {
      this.type = init.type;
      this.sdp = init.sdp;
    }
    toJSON() {
      return { type: this.type, sdp: this.sdp };
    }
  }

  class RTCIceCandidate {
    constructor(init) {
      this.candidate = init.candidate || "";
      this.sdpMid = init.sdpMid || "";
      this.sdpMLineIndex = init.sdpMLineIndex ?? null;
    }
    toJSON() {
      return { candidate: this.candidate, sdpMid: this.sdpMid, sdpMLineIndex: this.sdpMLineIndex };
    }
  }

  class RTCPeerConnection extends EventTarget {
    constructor(config) {
      super();
      LOG("new RTCPeerConnection(", config, ")");
      this._id = null;
      this._channels = new Map();
      this._pendingLabels = [];
      this._polling = false;
      this._closed = false;
      this._ready = false;
      this._pendingCandidates = [];
      this._remoteDescriptionSet = false;

      this.localDescription = null;
      this.remoteDescription = null;
      this.signalingState = "stable";
      this.connectionState = "new";
      this.iceConnectionState = "new";
      this.iceGatheringState = "new";

      this.onicecandidate = null;
      this.ondatachannel = null;
      this.ontrack = null;
      this.onconnectionstatechange = null;
      this.oniceconnectionstatechange = null;

      // Async init — ops that need _id will await _initPromise
      this._initPromise = rtc.op_rtc_new().then(info => {
        this._id = info.id;
        this._ready = true;
        LOG("  peer_id:", this._id);
        // Register on_track handler for receiving remote audio
        rtc.op_rtc_setup_on_track(this._id);
        this._startPolling();
      });
    }

    createDataChannel(label, options) {
      LOG("createDataChannel:", label, options);
      if (this.connectionState === "connected") {
        const channelId = rtc.op_rtc_create_data_channel(this._id, label);
        const dc = new RTCDataChannel(this._id, channelId, label, options);
        this._channels.set(channelId, dc);
        return dc;
      }
      const dc = new RTCDataChannel(this._id, null, label, options);
      this._pendingLabels.push({ label, dc, options });
      return dc;
    }

    async addTrack(track, stream) {
      await this._initPromise;
      LOG("addTrack:", track?.kind, "stream:", stream?.id);
      if (!stream || !stream._id) {
        console.warn("[webrtc-shim] addTrack: no stream provided");
        return { track, replaceTrack: () => Promise.resolve() };
      }
      try {
        const result = await rtc.op_rtc_add_track(this._id, stream._id);
        LOG("  addTrack result:", result);
      } catch (e) {
        console.warn("[webrtc-shim] addTrack error:", e.message);
      }
      // Return minimal RTCRtpSender stub (PeerJS only checks it's truthy)
      return { track, replaceTrack: () => Promise.resolve() };
    }

    async createOffer() {
      await this._initPromise;
      const label = this._pendingLabels.length > 0
        ? this._pendingLabels[0].label
        : "__data";
      LOG("createOffer (channel label:", label, ")");
      try {
        const result = await rtc.op_rtc_create_offer(this._id, label);
        LOG("  offer SDP length:", result.sdp.length);
        const desc = new RTCSessionDescription({ type: "offer", sdp: result.sdp });
        return desc;
      } catch (e) {
        LOG("  createOffer ERROR:", e.message);
        throw e;
      }
    }

    async createAnswer() {
      await this._initPromise;
      LOG("createAnswer");
      if (!this.remoteDescription) {
        throw new DOMException("Remote description not set", "InvalidStateError");
      }
      try {
        const result = await rtc.op_rtc_create_answer(this._id, this.remoteDescription.sdp);
        this._remoteDescriptionSet = true;
        LOG("  answer SDP length:", result.sdp.length);
        // Flush any ICE candidates that arrived before the Rust side had the remote description
        if (this._pendingCandidates.length > 0) {
          LOG("  flushing", this._pendingCandidates.length, "buffered ICE candidates");
          await this._flushPendingCandidates();
        }
        const desc = new RTCSessionDescription({ type: "answer", sdp: result.sdp });
        return desc;
      } catch (e) {
        LOG("  createAnswer ERROR:", e.message);
        throw e;
      }
    }

    async setLocalDescription(desc) {
      await this._initPromise;
      LOG("setLocalDescription:", desc?.type);
      if (!desc) {
        desc = await this.createOffer();
      }
      this.localDescription = desc;
      if (desc.type === "offer") {
        this.signalingState = "have-local-offer";
      } else if (desc.type === "answer") {
        this.signalingState = "stable";
      }
    }

    async setRemoteDescription(desc) {
      await this._initPromise;
      LOG("setRemoteDescription:", desc?.type, "sdp length:", desc?.sdp?.length);
      this.remoteDescription = new RTCSessionDescription(desc);
      if (desc.type === "answer") {
        LOG("  accepting answer...");
        try {
          await rtc.op_rtc_accept_answer(this._id, desc.sdp);
          this.signalingState = "stable";
          this._remoteDescriptionSet = true;
          if (this._pendingCandidates.length > 0) {
            LOG("  flushing", this._pendingCandidates.length, "buffered ICE candidates");
            await this._flushPendingCandidates();
          }
          LOG("  answer accepted OK");
        } catch (e) {
          LOG("  accept_answer ERROR:", e.message);
          throw e;
        }
      } else if (desc.type === "offer") {
        this.signalingState = "have-remote-offer";
        // Note: the actual Rust-side remote description is set when createAnswer
        // is called. ICE candidates are buffered until then.
      }
    }

    async _flushPendingCandidates() {
      const pending = this._pendingCandidates;
      this._pendingCandidates = [];
      for (const c of pending) {
        try {
          await rtc.op_rtc_add_remote_candidate(this._id, c.candidate, c.sdpMid);
        } catch (e) {
          console.warn("[webrtc-shim] failed to add buffered ICE candidate:", e.message);
        }
      }
    }

    async addIceCandidate(candidate) {
      if (!candidate || !candidate.candidate) {
        LOG("addIceCandidate: (empty/null - end of candidates)");
        return;
      }
      await this._initPromise;
      LOG("addIceCandidate:", candidate.candidate.substring(0, 80));
      if (!this._remoteDescriptionSet) {
        LOG("  buffering ICE candidate (remote description not yet set on native)");
        this._pendingCandidates.push({
          candidate: candidate.candidate,
          sdpMid: candidate.sdpMid || "0",
        });
        return;
      }
      try {
        await rtc.op_rtc_add_remote_candidate(
          this._id,
          candidate.candidate,
          candidate.sdpMid || "0"
        );
      } catch (e) {
        console.warn("[webrtc-shim] addIceCandidate error:", e.message);
      }
    }

    close() {
      LOG("close peer:", this._id);
      this._closed = true;
      if (this._id != null) {
        rtc.op_rtc_close(this._id);
      }
      this.signalingState = "closed";
      this.connectionState = "closed";
      for (const dc of this._channels.values()) {
        dc._handleClose();
      }
      this._channels.clear();
    }

    getConfiguration() {
      return {};
    }

    _startPolling() {
      if (this._polling) return;
      this._polling = true;

      const poll = () => {
        if (this._closed) return;

        try {
          const events = rtc.op_rtc_poll(this._id);
          for (const evt of events) {
            try {
              this._handleEvent(evt);
            } catch (e) {
              console.error("[webrtc-shim] event handler error:", e);
            }
          }
        } catch (e) {
          if (!this._closed) {
            console.error("[webrtc-shim] poll error:", e);
          }
        }

        // Poll at 5ms interval for low-latency networking
        // (RAF is too slow - tied to render frame rate)
        setTimeout(poll, 5);
      };

      setTimeout(poll, 0);
    }

    _handleEvent(evt) {
      switch (evt.type) {
        case "channel-open": {
          LOG("event: channel-open, id:", evt.channel_id, "label:", evt.label);
          let dc = this._channels.get(evt.channel_id);
          if (!dc) {
            const pendingIdx = this._pendingLabels.findIndex(p => p.label === evt.label);
            if (pendingIdx >= 0) {
              dc = this._pendingLabels[pendingIdx].dc;
              dc._pcId = this._id;
              dc._id = evt.channel_id;
              this._pendingLabels.splice(pendingIdx, 1);
              this._channels.set(evt.channel_id, dc);
              dc._handleOpen();
            } else {
              dc = new RTCDataChannel(this._id, evt.channel_id, evt.label);
              this._channels.set(evt.channel_id, dc);
              dc._handleOpen();
              const e = new Event("datachannel");
              e.channel = dc;
              if (this.ondatachannel) this.ondatachannel(e);
              this.dispatchEvent(e);
            }
          } else {
            dc._handleOpen();
          }
          break;
        }
        case "channel-close": {
          LOG("event: channel-close, id:", evt.channel_id);
          const dc = this._channels.get(evt.channel_id);
          if (dc) {
            dc._handleClose();
            this._channels.delete(evt.channel_id);
          }
          break;
        }
        case "channel-data": {
          const dc = this._channels.get(evt.channel_id);
          if (dc) {
            dc._handleMessage(evt.data, evt.binary);
          }
          break;
        }
        case "ice-candidate": {
          const candidate = new RTCIceCandidate({
            candidate: evt.candidate,
            sdpMid: evt.sdp_mid,
          });
          const e = new Event("icecandidate");
          e.candidate = candidate;
          if (this.onicecandidate) this.onicecandidate(e);
          this.dispatchEvent(e);
          break;
        }
        case "connection-state": {
          LOG("event: connection-state:", evt.state);
          this.connectionState = evt.state;
          this.iceConnectionState = evt.state;
          const e = new Event("connectionstatechange");
          if (this.onconnectionstatechange) this.onconnectionstatechange(e);
          this.dispatchEvent(e);
          const e2 = new Event("iceconnectionstatechange");
          if (this.oniceconnectionstatechange) this.oniceconnectionstatechange(e2);
          this.dispatchEvent(e2);
          break;
        }
        case "track": {
          LOG("event: track, track_id:", evt.track_id, "stream_id:", evt.stream_id, "kind:", evt.kind);
          const track = new MediaStreamTrack(evt.track_id, evt.kind);
          const stream = new MediaStream(evt.stream_id, [track]);
          stream._pcId = this._id;
          stream._remoteTrackId = evt.track_id;
          const e = new Event("track");
          e.track = track;
          e.streams = [stream];
          e.receiver = { track };
          e.transceiver = { receiver: { track }, sender: { track: null } };
          if (this.ontrack) this.ontrack(e);
          this.dispatchEvent(e);
          break;
        }
        case "error": {
          LOG("event: ERROR:", evt.message);
          break;
        }
      }
    }
  }

  globalThis.RTCPeerConnection = RTCPeerConnection;
  globalThis.RTCSessionDescription = RTCSessionDescription;
  globalThis.RTCIceCandidate = RTCIceCandidate;
  globalThis.RTCDataChannel = RTCDataChannel;
}
