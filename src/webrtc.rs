// src/webrtc.rs - WebRTC data channels + voice chat via webrtc-rs
//
// Uses a dedicated multi-threaded tokio runtime for WebRTC operations.
// Events are collected in a shared queue and polled from the JS thread.

use deno_core::op2;
use deno_error::JsErrorBox;
use serde::Serialize;
use std::collections::{HashMap, VecDeque};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Duration;

use bytes::Bytes;
use webrtc::api::interceptor_registry::register_default_interceptors;
use webrtc::api::media_engine::MediaEngine;
use webrtc::api::APIBuilder;
use webrtc::data_channel::data_channel_message::DataChannelMessage;
use webrtc::data_channel::RTCDataChannel;
use webrtc::ice_transport::ice_candidate::RTCIceCandidateInit;
use webrtc::ice_transport::ice_server::RTCIceServer;
use webrtc::interceptor::registry::Registry;
use webrtc::peer_connection::configuration::RTCConfiguration;
use webrtc::peer_connection::sdp::session_description::RTCSessionDescription;
use webrtc::peer_connection::RTCPeerConnection;
use webrtc::rtp_transceiver::rtp_codec::{RTCRtpCodecCapability, RTPCodecType};
use webrtc::track::track_local::track_local_static_sample::TrackLocalStaticSample;
use webrtc::track::track_remote::TrackRemote;

// ======================= WebRTC Runtime =======================

fn webrtc_runtime() -> &'static tokio::runtime::Runtime {
    static RT: OnceLock<tokio::runtime::Runtime> = OnceLock::new();
    RT.get_or_init(|| {
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .thread_name("webrtc")
            .build()
            .expect("Failed to create WebRTC runtime")
    })
}

// ======================= State =======================

struct RtcPeer {
    pc: Arc<RTCPeerConnection>,
    channels: HashMap<u32, Arc<RTCDataChannel>>,
    next_channel_id: u32,
    events: Arc<Mutex<VecDeque<RtcEventOut>>>,
}

struct WebRtcState {
    peers: HashMap<u32, RtcPeer>,
    next_id: u32,
}

impl WebRtcState {
    fn new() -> Self {
        Self {
            peers: HashMap::new(),
            next_id: 1,
        }
    }

    fn alloc_id(&mut self) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }
}

static WEBRTC: OnceLock<Mutex<WebRtcState>> = OnceLock::new();

fn webrtc() -> &'static Mutex<WebRtcState> {
    WEBRTC.get_or_init(|| Mutex::new(WebRtcState::new()))
}

// ======================= Event types for JS =======================

#[derive(Serialize, Clone)]
#[serde(tag = "type")]
pub enum RtcEventOut {
    #[serde(rename = "channel-open")]
    ChannelOpen { channel_id: u32, label: String },
    #[serde(rename = "channel-close")]
    ChannelClose { channel_id: u32 },
    #[serde(rename = "channel-data")]
    ChannelData {
        channel_id: u32,
        data: String,
        binary: bool,
    },
    #[serde(rename = "connection-state")]
    ConnectionState { state: String },
    #[serde(rename = "ice-candidate")]
    IceCandidate { candidate: String, sdp_mid: String },
    #[serde(rename = "error")]
    Error { message: String },
    #[serde(rename = "track")]
    Track { track_id: u32, stream_id: u32, kind: String },
}

#[derive(Serialize)]
pub struct PeerInfo {
    pub id: u32,
    pub local_addr: String,
}

#[derive(Serialize)]
pub struct OfferResult {
    pub sdp: String,
}

#[derive(Serialize)]
pub struct AnswerResult {
    pub sdp: String,
}

// ======================= Helpers =======================

fn base64_encode(data: &[u8]) -> String {
    const CHARS: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut result = String::with_capacity((data.len() + 2) / 3 * 4);
    for chunk in data.chunks(3) {
        let b0 = chunk[0] as u32;
        let b1 = chunk.get(1).copied().unwrap_or(0) as u32;
        let b2 = chunk.get(2).copied().unwrap_or(0) as u32;
        let triple = (b0 << 16) | (b1 << 8) | b2;
        result.push(CHARS[((triple >> 18) & 0x3F) as usize] as char);
        result.push(CHARS[((triple >> 12) & 0x3F) as usize] as char);
        if chunk.len() > 1 {
            result.push(CHARS[((triple >> 6) & 0x3F) as usize] as char);
        } else {
            result.push('=');
        }
        if chunk.len() > 2 {
            result.push(CHARS[(triple & 0x3F) as usize] as char);
        } else {
            result.push('=');
        }
    }
    result
}

/// Set up data channel callbacks that push events to the shared queue.
fn setup_data_channel_handlers(
    dc: &Arc<RTCDataChannel>,
    peer_id: u32,
    channel_id: u32,
    events: Arc<Mutex<VecDeque<RtcEventOut>>>,
) {
    let events_open = events.clone();
    let label = dc.label().to_string();
    dc.on_open(Box::new(move || {
        let events = events_open.clone();
        let label = label.clone();
        Box::pin(async move {
            events.lock().unwrap().push_back(RtcEventOut::ChannelOpen {
                channel_id,
                label,
            });
        })
    }));

    let events_msg = events.clone();
    dc.on_message(Box::new(move |msg: DataChannelMessage| {
        let events = events_msg.clone();
        Box::pin(async move {
            let (data, binary) = if msg.is_string {
                (String::from_utf8_lossy(&msg.data).to_string(), false)
            } else {
                (base64_encode(&msg.data), true)
            };
            events.lock().unwrap().push_back(RtcEventOut::ChannelData {
                channel_id,
                data,
                binary,
            });
        })
    }));

    let events_close = events;
    dc.on_close(Box::new(move || {
        let events = events_close.clone();
        Box::pin(async move {
            events.lock().unwrap().push_back(RtcEventOut::ChannelClose { channel_id });
        })
    }));
}

// ======================= Ops =======================

/// Create a new RTCPeerConnection.
#[op2]
#[serde]
pub async fn op_rtc_new() -> Result<PeerInfo, JsErrorBox> {
    let peer_id = {
        let mut state = webrtc().lock().unwrap_or_else(|e| e.into_inner());
        state.alloc_id()
    };

    let events = Arc::new(Mutex::new(VecDeque::new()));

    let pc = tokio::task::spawn_blocking(move || {
        webrtc_runtime().block_on(async {
            let mut m = MediaEngine::default();
            m.register_default_codecs()
                .map_err(|e| JsErrorBox::generic(format!("MediaEngine error: {}", e)))?;

            let mut registry = Registry::new();
            registry = register_default_interceptors(registry, &mut m)
                .map_err(|e| JsErrorBox::generic(format!("Interceptor error: {}", e)))?;

            let api = APIBuilder::new()
                .with_media_engine(m)
                .with_interceptor_registry(registry)
                .build();

            let config = RTCConfiguration {
                ice_servers: vec![RTCIceServer {
                    urls: vec!["stun:stun.l.google.com:19302".to_string()],
                    ..Default::default()
                }],
                ..Default::default()
            };

            let pc = api
                .new_peer_connection(config)
                .await
                .map_err(|e| JsErrorBox::generic(format!("PeerConnection error: {}", e)))?;

            Ok::<Arc<RTCPeerConnection>, JsErrorBox>(Arc::new(pc))
        })
    })
    .await
    .map_err(|e| JsErrorBox::generic(format!("spawn error: {}", e)))??;

    // Set up connection state change handler
    let events_state = events.clone();
    pc.on_peer_connection_state_change(Box::new(move |state| {
        let events = events_state.clone();
        Box::pin(async move {
            let s = format!("{}", state).to_lowercase();
            events.lock().unwrap().push_back(RtcEventOut::ConnectionState { state: s });
        })
    }));

    // Set up ICE candidate handler
    let events_ice = events.clone();
    pc.on_ice_candidate(Box::new(move |candidate| {
        let events = events_ice.clone();
        Box::pin(async move {
            if let Some(c) = candidate {
                let json = c.to_json().unwrap_or_default();
                events.lock().unwrap().push_back(RtcEventOut::IceCandidate {
                    candidate: json.candidate,
                    sdp_mid: json.sdp_mid.unwrap_or_default(),
                });
            }
        })
    }));

    // Set up incoming data channel handler
    let events_dc = events.clone();
    pc.on_data_channel(Box::new(move |dc: Arc<RTCDataChannel>| {
        let events = events_dc.clone();
        Box::pin(async move {
            let channel_id = {
                let mut state = webrtc().lock().unwrap_or_else(|e| e.into_inner());
                let peer = state.peers.get_mut(&peer_id).unwrap();
                let id = peer.next_channel_id;
                peer.next_channel_id += 1;
                peer.channels.insert(id, dc.clone());
                id
            };
            setup_data_channel_handlers(&dc, peer_id, channel_id, events);
        })
    }));

    let peer = RtcPeer {
        pc,
        channels: HashMap::new(),
        next_channel_id: 1,
        events,
    };

    {
        let mut state = webrtc().lock().unwrap_or_else(|e| e.into_inner());
        state.peers.insert(peer_id, peer);
    }

    Ok(PeerInfo {
        id: peer_id,
        local_addr: "webrtc-rs".to_string(),
    })
}

/// Create an SDP offer with a data channel.
#[op2]
#[serde]
pub async fn op_rtc_create_offer(
    #[smi] peer_id: u32,
    #[string] label: String,
) -> Result<OfferResult, JsErrorBox> {
    let (pc, events) = {
        let mut state = webrtc().lock().unwrap_or_else(|e| e.into_inner());
        let peer = state
            .peers
            .get_mut(&peer_id)
            .ok_or_else(|| JsErrorBox::generic("Peer not found"))?;

        // Create data channel
        let channel_id = peer.next_channel_id;
        peer.next_channel_id += 1;

        let pc = peer.pc.clone();
        let events = peer.events.clone();

        // We'll store the DC after creation
        (pc, events)
    };

    let sdp = tokio::task::spawn_blocking(move || {
        webrtc_runtime().block_on(async {
            let dc = pc
                .create_data_channel(&label, None)
                .await
                .map_err(|e| JsErrorBox::generic(format!("CreateDataChannel error: {}", e)))?;

            // Store channel and set up handlers
            {
                let mut state = webrtc().lock().unwrap_or_else(|e| e.into_inner());
                let peer = state.peers.get_mut(&peer_id).unwrap();
                let channel_id = peer.next_channel_id - 1;
                peer.channels.insert(channel_id, dc.clone());
                setup_data_channel_handlers(&dc, peer_id, channel_id, events);
            }

            let offer = pc
                .create_offer(None)
                .await
                .map_err(|e| JsErrorBox::generic(format!("CreateOffer error: {}", e)))?;

            pc.set_local_description(offer.clone())
                .await
                .map_err(|e| JsErrorBox::generic(format!("SetLocalDescription error: {}", e)))?;

            Ok::<String, JsErrorBox>(offer.sdp)
        })
    })
    .await
    .map_err(|e| JsErrorBox::generic(format!("spawn error: {}", e)))??;

    Ok(OfferResult { sdp })
}

/// Accept a remote SDP offer and generate an answer.
#[op2]
#[serde]
pub async fn op_rtc_create_answer(
    #[smi] peer_id: u32,
    #[string] offer_sdp: String,
) -> Result<AnswerResult, JsErrorBox> {
    let pc = {
        let state = webrtc().lock().unwrap_or_else(|e| e.into_inner());
        let peer = state
            .peers
            .get(&peer_id)
            .ok_or_else(|| JsErrorBox::generic("Peer not found"))?;
        peer.pc.clone()
    };

    let sdp = tokio::task::spawn_blocking(move || {
        webrtc_runtime().block_on(async {
            let offer = RTCSessionDescription::offer(offer_sdp)
                .map_err(|e| JsErrorBox::generic(format!("Parse offer error: {}", e)))?;

            pc.set_remote_description(offer)
                .await
                .map_err(|e| JsErrorBox::generic(format!("SetRemoteDescription error: {}", e)))?;

            let answer = pc
                .create_answer(None)
                .await
                .map_err(|e| JsErrorBox::generic(format!("CreateAnswer error: {}", e)))?;

            pc.set_local_description(answer.clone())
                .await
                .map_err(|e| JsErrorBox::generic(format!("SetLocalDescription error: {}", e)))?;

            Ok::<String, JsErrorBox>(answer.sdp)
        })
    })
    .await
    .map_err(|e| JsErrorBox::generic(format!("spawn error: {}", e)))??;

    Ok(AnswerResult { sdp })
}

/// Accept a remote SDP answer (caller side).
#[op2]
#[serde]
pub async fn op_rtc_accept_answer(
    #[smi] peer_id: u32,
    #[string] answer_sdp: String,
) -> Result<(), JsErrorBox> {
    let pc = {
        let state = webrtc().lock().unwrap_or_else(|e| e.into_inner());
        let peer = state
            .peers
            .get(&peer_id)
            .ok_or_else(|| JsErrorBox::generic("Peer not found"))?;
        peer.pc.clone()
    };

    tokio::task::spawn_blocking(move || {
        webrtc_runtime().block_on(async {
            let answer = RTCSessionDescription::answer(answer_sdp)
                .map_err(|e| JsErrorBox::generic(format!("Parse answer error: {}", e)))?;

            pc.set_remote_description(answer)
                .await
                .map_err(|e| JsErrorBox::generic(format!("SetRemoteDescription error: {}", e)))?;

            Ok::<(), JsErrorBox>(())
        })
    })
    .await
    .map_err(|e| JsErrorBox::generic(format!("spawn error: {}", e)))??;

    Ok(())
}

/// Add a remote ICE candidate.
#[op2]
#[serde]
pub async fn op_rtc_add_remote_candidate(
    #[smi] peer_id: u32,
    #[string] candidate: String,
    #[string] sdp_mid: String,
) -> Result<(), JsErrorBox> {
    let pc = {
        let state = webrtc().lock().unwrap_or_else(|e| e.into_inner());
        let peer = state
            .peers
            .get(&peer_id)
            .ok_or_else(|| JsErrorBox::generic("Peer not found"))?;
        peer.pc.clone()
    };

    tokio::task::spawn_blocking(move || {
        webrtc_runtime().block_on(async {
            let init = RTCIceCandidateInit {
                candidate,
                sdp_mid: Some(sdp_mid),
                ..Default::default()
            };

            pc.add_ice_candidate(init)
                .await
                .map_err(|e| JsErrorBox::generic(format!("AddIceCandidate error: {}", e)))?;

            Ok::<(), JsErrorBox>(())
        })
    })
    .await
    .map_err(|e| JsErrorBox::generic(format!("spawn error: {}", e)))??;

    Ok(())
}

/// Create a data channel on an already-connected peer.
#[op2(fast)]
pub fn op_rtc_create_data_channel(
    #[smi] peer_id: u32,
    #[string] label: String,
) -> Result<u32, JsErrorBox> {
    let (pc, events) = {
        let mut state = webrtc().lock().unwrap_or_else(|e| e.into_inner());
        let peer = state
            .peers
            .get_mut(&peer_id)
            .ok_or_else(|| JsErrorBox::generic("Peer not found"))?;
        let channel_id = peer.next_channel_id;
        peer.next_channel_id += 1;
        (peer.pc.clone(), peer.events.clone())
    };

    let channel_id = {
        let state = webrtc().lock().unwrap_or_else(|e| e.into_inner());
        state.peers.get(&peer_id).unwrap().next_channel_id - 1
    };

    // Spawn on WebRTC runtime (fire and forget — channel open event will arrive later)
    webrtc_runtime().spawn(async move {
        match pc.create_data_channel(&label, None).await {
            Ok(dc) => {
                {
                    let mut state = webrtc().lock().unwrap_or_else(|e| e.into_inner());
                    if let Some(peer) = state.peers.get_mut(&peer_id) {
                        peer.channels.insert(channel_id, dc.clone());
                    }
                }
                setup_data_channel_handlers(&dc, peer_id, channel_id, events);
            }
            Err(e) => {
                events.lock().unwrap().push_back(RtcEventOut::Error {
                    message: format!("CreateDataChannel error: {}", e),
                });
            }
        }
    });

    Ok(channel_id)
}

/// Send UTF-8 text on a data channel.
#[op2(fast)]
pub fn op_rtc_send_text(
    #[smi] peer_id: u32,
    #[smi] channel_id: u32,
    #[string] text: String,
) -> Result<(), JsErrorBox> {
    let dc = {
        let state = webrtc().lock().unwrap_or_else(|e| e.into_inner());
        let peer = state
            .peers
            .get(&peer_id)
            .ok_or_else(|| JsErrorBox::generic("Peer not found"))?;
        peer.channels
            .get(&channel_id)
            .cloned()
            .ok_or_else(|| JsErrorBox::generic("Channel not found"))?
    };

    webrtc_runtime().spawn(async move {
        let _ = dc.send_text(text).await;
    });

    Ok(())
}

/// Send binary data on a data channel.
#[op2(fast)]
pub fn op_rtc_send_binary(
    #[smi] peer_id: u32,
    #[smi] channel_id: u32,
    #[buffer] data: &[u8],
) -> Result<(), JsErrorBox> {
    let dc = {
        let state = webrtc().lock().unwrap_or_else(|e| e.into_inner());
        let peer = state
            .peers
            .get(&peer_id)
            .ok_or_else(|| JsErrorBox::generic("Peer not found"))?;
        peer.channels
            .get(&channel_id)
            .cloned()
            .ok_or_else(|| JsErrorBox::generic("Channel not found"))?
    };

    let data = Bytes::copy_from_slice(data);
    webrtc_runtime().spawn(async move {
        let _ = dc.send(&data).await;
    });

    Ok(())
}

/// Poll a peer connection for events.
#[op2]
#[serde]
pub fn op_rtc_poll(#[smi] peer_id: u32) -> Result<Vec<RtcEventOut>, JsErrorBox> {
    let state = webrtc().lock().unwrap_or_else(|e| e.into_inner());
    let peer = state
        .peers
        .get(&peer_id)
        .ok_or_else(|| JsErrorBox::generic("Peer not found"))?;

    let events: Vec<RtcEventOut> = peer.events.lock().unwrap().drain(..).collect();
    Ok(events)
}

/// Close and destroy a peer connection.
#[op2(fast)]
pub fn op_rtc_close(#[smi] peer_id: u32) {
    let pc = {
        let mut state = webrtc().lock().unwrap_or_else(|e| e.into_inner());
        state.peers.remove(&peer_id).map(|p| p.pc)
    };

    if let Some(pc) = pc {
        webrtc_runtime().spawn(async move {
            let _ = pc.close().await;
        });
    }
}

/// Get the connection state of a peer.
#[op2]
#[string]
pub fn op_rtc_get_state(#[smi] peer_id: u32) -> Result<String, JsErrorBox> {
    let state = webrtc().lock().unwrap_or_else(|e| e.into_inner());
    let peer = state
        .peers
        .get(&peer_id)
        .ok_or_else(|| JsErrorBox::generic("Peer not found"))?;

    Ok(format!("{}", peer.pc.connection_state()).to_lowercase())
}

// ======================= Voice Chat =======================

struct VoiceState {
    local_streams: HashMap<u32, web_audio_api::media_streams::MediaStream>,
    /// Senders for decoded remote audio (track_id → sender)
    remote_audio_senders: HashMap<u32, crossbeam_channel::Sender<web_audio_api::AudioBuffer>>,
    /// Receivers waiting to be consumed by create_remote_audio_source
    remote_audio_receivers: HashMap<u32, crossbeam_channel::Receiver<web_audio_api::AudioBuffer>>,
    /// Stop flags for capture pump tasks
    capture_flags: HashMap<u32, Arc<AtomicBool>>,
    next_id: u32,
}

impl VoiceState {
    fn new() -> Self {
        Self {
            local_streams: HashMap::new(),
            remote_audio_senders: HashMap::new(),
            remote_audio_receivers: HashMap::new(),
            capture_flags: HashMap::new(),
            next_id: 1,
        }
    }
    fn alloc_id(&mut self) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }
}

static VOICE: OnceLock<Mutex<VoiceState>> = OnceLock::new();

fn voice() -> &'static Mutex<VoiceState> {
    VOICE.get_or_init(|| Mutex::new(VoiceState::new()))
}

/// Capture microphone audio. Returns a stream ID.
#[op2]
#[serde]
pub async fn op_rtc_get_user_media() -> Result<HashMap<String, u32>, JsErrorBox> {
    let stream = tokio::task::spawn_blocking(|| {
        use web_audio_api::media_devices::{get_user_media_sync, MediaStreamConstraints};
        // Catch panics from cpal backend (e.g. Android Quest mic not available)
        std::panic::catch_unwind(|| {
            get_user_media_sync(MediaStreamConstraints::Audio)
        }).map_err(|e| {
            let msg = if let Some(s) = e.downcast_ref::<String>() {
                s.clone()
            } else if let Some(s) = e.downcast_ref::<&str>() {
                s.to_string()
            } else {
                "unknown panic in get_user_media_sync".to_string()
            };
            msg
        })
    })
    .await
    .map_err(|e| JsErrorBox::generic(format!("getUserMedia failed: {}", e)))?
    .map_err(|e| JsErrorBox::generic(format!("getUserMedia not available: {}", e)))?;

    let stream_id = {
        let mut vs = voice().lock().unwrap_or_else(|e| e.into_inner());
        let id = vs.alloc_id();
        vs.local_streams.insert(id, stream);
        id
    };

    let mut result = HashMap::new();
    result.insert("stream_id".to_string(), stream_id);
    Ok(result)
}

/// Add a local audio track to a peer connection. Starts the capture→encode→send pump.
#[op2]
#[serde]
pub async fn op_rtc_add_track(
    #[smi] peer_id: u32,
    #[smi] stream_id: u32,
) -> Result<HashMap<String, u32>, JsErrorBox> {
    let pc = {
        let state = webrtc().lock().unwrap_or_else(|e| e.into_inner());
        let peer = state
            .peers
            .get(&peer_id)
            .ok_or_else(|| JsErrorBox::generic("Peer not found"))?;
        peer.pc.clone()
    };

    // Create the local audio track with Opus codec
    let track = Arc::new(TrackLocalStaticSample::new(
        RTCRtpCodecCapability {
            mime_type: "audio/opus".to_string(),
            clock_rate: 48000,
            channels: 1,
            sdp_fmtp_line: "minptime=10;useinbandfec=1".to_string(),
            ..Default::default()
        },
        format!("audio-{}", stream_id),
        format!("stream-{}", stream_id),
    ));

    // Add track to peer connection
    let track_clone = track.clone();
    tokio::task::spawn_blocking(move || {
        webrtc_runtime().block_on(async {
            pc.add_track(track_clone as Arc<dyn webrtc::track::track_local::TrackLocal + Send + Sync>)
                .await
                .map_err(|e| JsErrorBox::generic(format!("addTrack error: {}", e)))
        })
    })
    .await
    .map_err(|e| JsErrorBox::generic(format!("spawn error: {}", e)))??;

    // Take the MediaStream out to move into the capture task
    let mic_stream = {
        let mut vs = voice().lock().unwrap_or_else(|e| e.into_inner());
        vs.local_streams
            .remove(&stream_id)
            .ok_or_else(|| JsErrorBox::generic("Stream not found"))?
    };

    let capturing = Arc::new(AtomicBool::new(true));
    let capturing_flag = capturing.clone();

    let track_id = {
        let mut vs = voice().lock().unwrap_or_else(|e| e.into_inner());
        let id = vs.alloc_id();
        vs.capture_flags.insert(id, capturing.clone());
        id
    };

    // Spawn capture pump: mic samples → Opus encode → TrackLocalStaticSample
    let track_for_pump = track.clone();
    std::thread::spawn(move || {
        let mut encoder = match crate::opus::Encoder::new(48000, 1) {
            Ok(e) => e,
            Err(e) => {
                log::error!("Failed to create Opus encoder: {}", e);
                return;
            }
        };

        let tracks = mic_stream.get_tracks();
        if tracks.is_empty() {
            log::error!("No tracks in microphone stream");
            return;
        }

        let mut accumulator: Vec<f32> = Vec::with_capacity(960);
        let mut opus_buf = vec![0u8; 4000]; // max Opus frame size
        let mut detected_rate: Option<f32> = None;

        for item in tracks[0].iter() {
            if !capturing_flag.load(Ordering::Relaxed) {
                break;
            }

            let audio_buf = match item {
                Ok(buf) => buf,
                Err(e) => {
                    log::warn!("Mic capture error: {}", e);
                    continue;
                }
            };

            let src_rate = audio_buf.sample_rate();
            if detected_rate.is_none() {
                log::info!("Mic capture sample rate: {} Hz", src_rate);
                detected_rate = Some(src_rate);
            }

            // Get mono samples (channel 0)
            let samples = audio_buf.get_channel_data(0);

            // Resample to 48kHz if device rate differs
            if (src_rate - 48000.0).abs() > 1.0 {
                let ratio = 48000.0 / src_rate;
                let out_len = (samples.len() as f64 * ratio as f64).ceil() as usize;
                let mut resampled = Vec::with_capacity(out_len);
                for i in 0..out_len {
                    let src_pos = i as f64 / ratio as f64;
                    let idx = src_pos as usize;
                    let frac = src_pos - idx as f64;
                    let s0 = samples[idx.min(samples.len() - 1)];
                    let s1 = samples[(idx + 1).min(samples.len() - 1)];
                    resampled.push(s0 + (s1 - s0) * frac as f32);
                }
                accumulator.extend_from_slice(&resampled);
            } else {
                accumulator.extend_from_slice(samples);
            }

            // Encode when we have a full 20ms Opus frame (960 samples at 48kHz)
            while accumulator.len() >= 960 {
                let frame: Vec<f32> = accumulator.drain(..960).collect();

                match encoder.encode_float(&frame, &mut opus_buf) {
                    Ok(encoded_len) => {
                        let sample = webrtc::media::Sample {
                            data: Bytes::copy_from_slice(&opus_buf[..encoded_len]),
                            duration: Duration::from_millis(20),
                            ..Default::default()
                        };

                        // Write sample on the webrtc runtime
                        let track_ref = track_for_pump.clone();
                        webrtc_runtime().block_on(async {
                            if let Err(e) = track_ref.write_sample(&sample).await {
                                log::warn!("write_sample error: {}", e);
                            }
                        });
                    }
                    Err(e) => {
                        log::warn!("Opus encode error: {}", e);
                    }
                }
            }
        }

        log::info!("Capture pump stopped for stream {}", stream_id);
    });

    let mut result = HashMap::new();
    result.insert("track_id".to_string(), track_id);
    Ok(result)
}

/// Register on_track handler for receiving remote audio tracks.
#[op2(fast)]
pub fn op_rtc_setup_on_track(#[smi] peer_id: u32) -> Result<(), JsErrorBox> {
    let (pc, events) = {
        let state = webrtc().lock().unwrap_or_else(|e| e.into_inner());
        let peer = state
            .peers
            .get(&peer_id)
            .ok_or_else(|| JsErrorBox::generic("Peer not found"))?;
        (peer.pc.clone(), peer.events.clone())
    };

    pc.on_track(Box::new(move |track: Arc<TrackRemote>, _receiver, _transceiver| {
        let events = events.clone();
        Box::pin(async move {
            let kind = match track.kind() {
                RTPCodecType::Audio => "audio",
                RTPCodecType::Video => "video",
                _ => "unknown",
            };

            if kind != "audio" {
                log::info!("Ignoring non-audio track: {}", kind);
                return;
            }

            log::info!("Remote audio track received: id={}, codec={:?}",
                track.id(), track.codec());

            // Create a channel for decoded audio buffers
            // Bounded(50) gives ~1s jitter buffer (50 * 20ms Opus frames)
            let (sender, receiver) = crossbeam_channel::bounded::<web_audio_api::AudioBuffer>(50);

            let (track_id, stream_id) = {
                let mut vs = voice().lock().unwrap_or_else(|e| e.into_inner());
                let tid = vs.alloc_id();
                let sid = vs.alloc_id();
                vs.remote_audio_senders.insert(tid, sender);
                vs.remote_audio_receivers.insert(tid, receiver);
                (tid, sid)
            };

            // Push event for JS
            events.lock().unwrap().push_back(RtcEventOut::Track {
                track_id,
                stream_id,
                kind: kind.to_string(),
            });

            // Spawn decode pump: read RTP → Opus decode → AudioBuffer → channel
            let track_remote = track.clone();
            webrtc_runtime().spawn(async move {
                let mut decoder = match crate::opus::Decoder::new(48000, 1) {
                    Ok(d) => d,
                    Err(e) => {
                        log::error!("Failed to create Opus decoder: {}", e);
                        return;
                    }
                };

                let mut decode_buf = vec![0f32; 960]; // 20ms at 48kHz mono

                loop {
                    match track_remote.read_rtp().await {
                        Ok((rtp_packet, _attrs)) => {
                            let payload = &rtp_packet.payload;
                            if payload.is_empty() {
                                continue;
                            }

                            let decoded_len = match decoder.decode_float(
                                payload.as_ref(),
                                &mut decode_buf,
                            ) {
                                Ok(len) => len,
                                Err(e) => {
                                    log::warn!("Opus decode error: {}", e);
                                    continue;
                                }
                            };

                            // Send the full decoded frame as a single AudioBuffer.
                            // web-audio-api's MediaStreamAudioSourceNode handles
                            // resampling and render-quantum splitting internally.
                            let decoded_samples = &decode_buf[..decoded_len];
                            let mut buf = web_audio_api::AudioBuffer::new(web_audio_api::AudioBufferOptions {
                                number_of_channels: 1,
                                length: decoded_len,
                                sample_rate: 48000.0,
                            });
                            buf.copy_to_channel(decoded_samples, 0);

                            let vs = voice().lock().unwrap_or_else(|e| e.into_inner());
                            if let Some(sender) = vs.remote_audio_senders.get(&track_id) {
                                let _ = sender.try_send(buf);
                            } else {
                                break; // sender removed, track closed
                            }
                        }
                        Err(e) => {
                            log::info!("Remote track read ended: {}", e);
                            break;
                        }
                    }
                }

                // Clean up
                let mut vs = voice().lock().unwrap_or_else(|e| e.into_inner());
                vs.remote_audio_senders.remove(&track_id);
                log::info!("Decode pump stopped for track {}", track_id);
            });
        })
    }));

    Ok(())
}

/// Create a native MediaStreamAudioSourceNode from a remote track's decoded audio.
/// Returns a node_id that JS can connect to a GainNode for volume control.
#[op2]
#[serde]
pub fn op_rtc_create_remote_audio_source(
    #[smi] _peer_id: u32,
    #[smi] track_id: u32,
    #[smi] ctx_id: u32,
) -> Result<HashMap<String, u32>, JsErrorBox> {
    // Take the receiver from voice state
    let receiver = {
        let mut vs = voice().lock().unwrap_or_else(|e| e.into_inner());
        vs.remote_audio_receivers
            .remove(&track_id)
            .ok_or_else(|| JsErrorBox::generic("Remote track not found"))?
    };

    // Create a MediaStreamTrack from the receiver iterator
    let track_iter = receiver.into_iter().map(|buf| -> Result<web_audio_api::AudioBuffer, Box<dyn std::error::Error + Send + Sync>> {
        Ok(buf)
    });
    let media_track = web_audio_api::media_streams::MediaStreamTrack::from_iter(track_iter);
    let media_stream = web_audio_api::media_streams::MediaStream::from_tracks(vec![media_track]);

    // Get the AudioContext and create a MediaStreamAudioSourceNode
    let node_id = {
        let mut audio_state = crate::audio::audio_state().lock().unwrap_or_else(|e| e.into_inner());
        let ctx = audio_state
            .contexts
            .get(&ctx_id)
            .ok_or_else(|| JsErrorBox::generic("AudioContext not found"))?;

        let source_node = ctx.create_media_stream_source(&media_stream);

        let id = audio_state.next_id;
        audio_state.next_id += 1;
        audio_state.nodes.insert(id, crate::audio::NodeWrapper::MediaStreamSource(source_node));
        id
    };

    let mut result = HashMap::new();
    result.insert("node_id".to_string(), node_id);
    Ok(result)
}

/// Stop a local stream capture.
#[op2(fast)]
pub fn op_rtc_stop_stream(#[smi] stream_id: u32) {
    let mut vs = voice().lock().unwrap_or_else(|e| e.into_inner());
    // Stop capture if running
    if let Some(flag) = vs.capture_flags.remove(&stream_id) {
        flag.store(false, Ordering::Relaxed);
    }
    // Remove stream if still held
    vs.local_streams.remove(&stream_id);
}
