// src/audio.rs — Native audio ops backed by web-audio-api-rs.
// Follows the same global-state + deno_core ops pattern as sqlite.rs.

use deno_core::op2;
use deno_error::JsErrorBox;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use web_audio_api::context::{AudioContext, BaseAudioContext, OfflineAudioContext};
use web_audio_api::node::{AudioNode, AudioScheduledSourceNode};

// ======================= Global State =======================

struct AudioState {
    contexts: HashMap<u32, AudioContext>,
    offline_contexts: HashMap<u32, OfflineAudioContext>,
    /// Offline contexts that have finished rendering but must stay alive until
    /// all nodes created on them are dropped. web-audio-api-rs nodes send a
    /// deregister message to their parent context on drop — if the context is
    /// already gone, the channel send panics/aborts.
    finished_offline_contexts: Vec<OfflineAudioContext>,
    nodes: HashMap<u32, NodeWrapper>,
    buffers: HashMap<u32, web_audio_api::AudioBuffer>,
    next_id: u32,
}

impl AudioState {
    fn new() -> Self {
        Self {
            contexts: HashMap::new(),
            offline_contexts: HashMap::new(),
            finished_offline_contexts: Vec::new(),
            nodes: HashMap::new(),
            buffers: HashMap::new(),
            next_id: 1,
        }
    }
    fn alloc_id(&mut self) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }
}

static AUDIO: OnceLock<Mutex<AudioState>> = OnceLock::new();

fn audio() -> &'static Mutex<AudioState> {
    AUDIO.get_or_init(|| Mutex::new(AudioState::new()))
}

/// Catch panics from web-audio-api-rs ops that are known to panic
/// (connect, start/stop, drop, convolver set_buffer, context close).
/// Converts the panic into a JsErrorBox so it surfaces as a JS exception
/// instead of crashing the process.
fn catch_audio<F, R>(f: F) -> Result<R, JsErrorBox>
where
    F: FnOnce() -> Result<R, JsErrorBox> + std::panic::UnwindSafe,
{
    match std::panic::catch_unwind(f) {
        Ok(result) => result,
        Err(e) => {
            let msg = if let Some(s) = e.downcast_ref::<&str>() {
                format!("audio internal error: {}", s)
            } else if let Some(s) = e.downcast_ref::<String>() {
                format!("audio internal error: {}", s)
            } else {
                "audio internal error".to_string()
            };
            log::error!("[audio] panic caught: {}", msg);
            Err(JsErrorBox::generic(msg))
        }
    }
}

/// Same as catch_audio but for ops that return () (no Result).
fn catch_audio_void<F>(f: F)
where
    F: FnOnce() + std::panic::UnwindSafe,
{
    if let Err(e) = std::panic::catch_unwind(f) {
        let msg = if let Some(s) = e.downcast_ref::<&str>() {
            s.to_string()
        } else if let Some(s) = e.downcast_ref::<String>() {
            s.clone()
        } else {
            "unknown".to_string()
        };
        log::error!("[audio] panic caught (void op): {}", msg);
    }
}

// ======================= Node Wrapper =======================

#[allow(dead_code)]
enum NodeWrapper {
    Destination(web_audio_api::node::AudioDestinationNode),
    Gain(web_audio_api::node::GainNode),
    BufferSource(web_audio_api::node::AudioBufferSourceNode),
    Oscillator(web_audio_api::node::OscillatorNode),
    Panner(web_audio_api::node::PannerNode),
    BiquadFilter(web_audio_api::node::BiquadFilterNode),
    StereoPanner(web_audio_api::node::StereoPannerNode),
    Delay(web_audio_api::node::DelayNode),
    DynamicsCompressor(web_audio_api::node::DynamicsCompressorNode),
    Analyser(web_audio_api::node::AnalyserNode),
    ConstantSource(web_audio_api::node::ConstantSourceNode),
    Convolver(web_audio_api::node::ConvolverNode),
    WaveShaper(web_audio_api::node::WaveShaperNode),
    ChannelMerger(web_audio_api::node::ChannelMergerNode),
    ChannelSplitter(web_audio_api::node::ChannelSplitterNode),
}

/// Dispatch to the inner AudioNode impl (used for connect / disconnect).
macro_rules! with_node {
    ($w:expr, |$n:ident| $body:expr) => {
        match $w {
            NodeWrapper::Destination($n) => $body,
            NodeWrapper::Gain($n) => $body,
            NodeWrapper::BufferSource($n) => $body,
            NodeWrapper::Oscillator($n) => $body,
            NodeWrapper::Panner($n) => $body,
            NodeWrapper::BiquadFilter($n) => $body,
            NodeWrapper::StereoPanner($n) => $body,
            NodeWrapper::Delay($n) => $body,
            NodeWrapper::DynamicsCompressor($n) => $body,
            NodeWrapper::Analyser($n) => $body,
            NodeWrapper::ConstantSource($n) => $body,
            NodeWrapper::Convolver($n) => $body,
            NodeWrapper::WaveShaper($n) => $body,
            NodeWrapper::ChannelMerger($n) => $body,
            NodeWrapper::ChannelSplitter($n) => $body,
        }
    };
}

/// Dispatch to an AudioParam on the node by JS-side param name.
macro_rules! with_param {
    ($node:expr, $param:expr, |$p:ident| $body:expr) => {
        match $node {
            NodeWrapper::Gain(n) => match $param {
                "gain" => { let $p = n.gain(); $body }
                other => return Err(JsErrorBox::generic(format!("GainNode: unknown param '{}'", other))),
            },
            NodeWrapper::BufferSource(n) => match $param {
                "playbackRate" => { let $p = n.playback_rate(); $body }
                "detune" => { let $p = n.detune(); $body }
                other => return Err(JsErrorBox::generic(format!("BufferSource: unknown param '{}'", other))),
            },
            NodeWrapper::Oscillator(n) => match $param {
                "frequency" => { let $p = n.frequency(); $body }
                "detune" => { let $p = n.detune(); $body }
                other => return Err(JsErrorBox::generic(format!("Oscillator: unknown param '{}'", other))),
            },
            NodeWrapper::BiquadFilter(n) => match $param {
                "frequency" => { let $p = n.frequency(); $body }
                "Q" => { let $p = n.q(); $body }
                "gain" => { let $p = n.gain(); $body }
                "detune" => { let $p = n.detune(); $body }
                other => return Err(JsErrorBox::generic(format!("BiquadFilter: unknown param '{}'", other))),
            },
            NodeWrapper::StereoPanner(n) => match $param {
                "pan" => { let $p = n.pan(); $body }
                other => return Err(JsErrorBox::generic(format!("StereoPanner: unknown param '{}'", other))),
            },
            NodeWrapper::Delay(n) => match $param {
                "delayTime" => { let $p = n.delay_time(); $body }
                other => return Err(JsErrorBox::generic(format!("Delay: unknown param '{}'", other))),
            },
            NodeWrapper::DynamicsCompressor(n) => match $param {
                "threshold" => { let $p = n.threshold(); $body }
                "knee" => { let $p = n.knee(); $body }
                "ratio" => { let $p = n.ratio(); $body }
                "attack" => { let $p = n.attack(); $body }
                "release" => { let $p = n.release(); $body }
                other => return Err(JsErrorBox::generic(format!("DynamicsCompressor: unknown param '{}'", other))),
            },
            NodeWrapper::Panner(n) => match $param {
                "positionX" => { let $p = n.position_x(); $body }
                "positionY" => { let $p = n.position_y(); $body }
                "positionZ" => { let $p = n.position_z(); $body }
                "orientationX" => { let $p = n.orientation_x(); $body }
                "orientationY" => { let $p = n.orientation_y(); $body }
                "orientationZ" => { let $p = n.orientation_z(); $body }
                other => return Err(JsErrorBox::generic(format!("Panner: unknown param '{}'", other))),
            },
            NodeWrapper::ConstantSource(n) => match $param {
                "offset" => { let $p = n.offset(); $body }
                other => return Err(JsErrorBox::generic(format!("ConstantSource: unknown param '{}'", other))),
            },
            _ => return Err(JsErrorBox::generic("Node type has no audio params")),
        }
    };
}

// ======================= Serializable Return Types =======================

#[derive(Serialize)]
pub struct AudioId {
    pub id: u32,
}

#[derive(Serialize)]
pub struct AudioContextInfo {
    pub id: u32,
    pub destination_id: u32,
    pub sample_rate: f32,
}

#[derive(Serialize)]
pub struct AudioBufferInfo {
    pub id: u32,
    pub duration: f64,
    pub length: u32,
    pub sample_rate: f32,
    pub number_of_channels: u32,
}

// ======================= Context Ops =======================

#[op2]
#[serde]
pub fn op_audio_create_context() -> Result<AudioContextInfo, JsErrorBox> {
    let mut s = audio().lock().unwrap_or_else(|e| e.into_inner());
    let ctx = AudioContext::new(web_audio_api::context::AudioContextOptions {
        latency_hint: web_audio_api::context::AudioContextLatencyCategory::Playback,
        ..Default::default()
    });
    let sample_rate = ctx.sample_rate();
    let dest = ctx.destination();
    let ctx_id = s.alloc_id();
    let dest_id = s.alloc_id();
    s.contexts.insert(ctx_id, ctx);
    s.nodes.insert(dest_id, NodeWrapper::Destination(dest));
    log::info!("[audio] Context {} created ({}Hz), destination={}", ctx_id, sample_rate, dest_id);
    Ok(AudioContextInfo { id: ctx_id, destination_id: dest_id, sample_rate })
}

#[op2(fast)]
pub fn op_audio_context_current_time(#[smi] ctx_id: u32) -> f64 {
    let s = audio().lock().unwrap_or_else(|e| e.into_inner());
    if let Some(c) = s.contexts.get(&ctx_id) {
        c.current_time()
    } else if let Some(c) = s.offline_contexts.get(&ctx_id) {
        c.current_time()
    } else {
        0.0
    }
}

#[op2(fast)]
pub fn op_audio_context_close(#[smi] ctx_id: u32) {
    catch_audio_void(std::panic::AssertUnwindSafe(|| {
        if let Some(ctx) = audio().lock().unwrap_or_else(|e| e.into_inner()).contexts.remove(&ctx_id) {
            ctx.close_sync();
        }
    }));
}

// ======================= Node Creation =======================

macro_rules! create_node_op {
    ($fn_name:ident, $create_method:ident, $variant:ident) => {
        #[op2]
        #[serde]
        pub fn $fn_name(#[smi] ctx_id: u32) -> Result<AudioId, JsErrorBox> {
            let mut s = audio().lock().unwrap_or_else(|e| e.into_inner());
            let node = if let Some(ctx) = s.contexts.get(&ctx_id) {
                ctx.$create_method()
            } else if let Some(ctx) = s.offline_contexts.get(&ctx_id) {
                ctx.$create_method()
            } else {
                return Err(JsErrorBox::generic("AudioContext not found"));
            };
            let id = s.alloc_id();
            s.nodes.insert(id, NodeWrapper::$variant(node));
            Ok(AudioId { id })
        }
    };
}

create_node_op!(op_audio_create_gain, create_gain, Gain);
create_node_op!(op_audio_create_buffer_source, create_buffer_source, BufferSource);
create_node_op!(op_audio_create_oscillator, create_oscillator, Oscillator);
create_node_op!(op_audio_create_panner, create_panner, Panner);
create_node_op!(op_audio_create_biquad_filter, create_biquad_filter, BiquadFilter);
create_node_op!(op_audio_create_stereo_panner, create_stereo_panner, StereoPanner);
create_node_op!(op_audio_create_dynamics_compressor, create_dynamics_compressor, DynamicsCompressor);
create_node_op!(op_audio_create_analyser, create_analyser, Analyser);
create_node_op!(op_audio_create_constant_source, create_constant_source, ConstantSource);
create_node_op!(op_audio_create_convolver, create_convolver, Convolver);
create_node_op!(op_audio_create_wave_shaper, create_wave_shaper, WaveShaper);

#[op2]
#[serde]
pub fn op_audio_create_delay(#[smi] ctx_id: u32, max_delay_time: f64) -> Result<AudioId, JsErrorBox> {
    let mut s = audio().lock().unwrap_or_else(|e| e.into_inner());
    let node = if let Some(ctx) = s.contexts.get(&ctx_id) {
        ctx.create_delay(max_delay_time)
    } else if let Some(ctx) = s.offline_contexts.get(&ctx_id) {
        ctx.create_delay(max_delay_time)
    } else {
        return Err(JsErrorBox::generic("AudioContext not found"));
    };
    let id = s.alloc_id();
    s.nodes.insert(id, NodeWrapper::Delay(node));
    Ok(AudioId { id })
}

// ======================= Connections =======================
// catch_audio: connect can panic on cross-context connections

#[op2(fast)]
pub fn op_audio_connect(#[smi] src_id: u32, #[smi] dst_id: u32, #[smi] output: u32, #[smi] input: u32) -> Result<(), JsErrorBox> {
    let state = audio().lock().unwrap_or_else(|e| e.into_inner());
    let src = state.nodes.get(&src_id)
        .ok_or_else(|| JsErrorBox::generic("Source node not found"))?;
    let dst = state.nodes.get(&dst_id)
        .ok_or_else(|| JsErrorBox::generic("Destination node not found"))?;
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        with_node!(src, |s_node| {
            with_node!(dst, |d_node| {
                s_node.connect_from_output_to_input(d_node, output as usize, input as usize);
            });
        });
    }));
    if let Err(_) = result {
        log::debug!("[audio] Cross-context connect: node {} -> node {}", src_id, dst_id);
    }
    Ok(())
}

#[op2(fast)]
pub fn op_audio_connect_param(
    #[smi] src_id: u32,
    #[smi] dst_node_id: u32,
    #[string] param_name: String,
    #[smi] output: u32,
) -> Result<(), JsErrorBox> {
    let state = audio().lock().unwrap_or_else(|e| e.into_inner());
    let src = state.nodes.get(&src_id)
        .ok_or_else(|| JsErrorBox::generic("Source node not found"))?;
    let dst = state.nodes.get(&dst_node_id)
        .ok_or_else(|| JsErrorBox::generic("Destination node not found"))?;
    let src_id_copy = src_id;
    let dst_id_copy = dst_node_id;
    let param_copy = param_name.clone();
    with_node!(src, |s_node| {
        with_param!(dst, param_name.as_str(), |p| {
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                s_node.connect_from_output_to_input(p, output as usize, 0);
            }));
            if result.is_err() {
                log::debug!("[audio] Cross-context connect_param: node {} -> node {}.{}", src_id_copy, dst_id_copy, param_copy);
            }
        });
    });
    Ok(())
}

#[op2(fast)]
pub fn op_audio_disconnect(#[smi] node_id: u32) -> Result<(), JsErrorBox> {
    let state = audio().lock().unwrap_or_else(|e| e.into_inner());
    let node = state.nodes.get(&node_id)
        .ok_or_else(|| JsErrorBox::generic("Node not found"))?;
    with_node!(node, |n| { n.disconnect(); });
    Ok(())
}

// ======================= AudioParam Ops =======================

#[op2(fast)]
pub fn op_audio_param_set_value(
    #[smi] node_id: u32,
    #[string] param: String,
    value: f64,
) -> Result<(), JsErrorBox> {
    let state = audio().lock().unwrap_or_else(|e| e.into_inner());
    let node = state.nodes.get(&node_id)
        .ok_or_else(|| JsErrorBox::generic("Node not found"))?;
    let v = value as f32;
    with_param!(node, param.as_str(), |p| p.set_value(v));
    Ok(())
}

#[op2(fast)]
pub fn op_audio_param_set_value_at_time(
    #[smi] node_id: u32,
    #[string] param: String,
    value: f64,
    start_time: f64,
) -> Result<(), JsErrorBox> {
    let state = audio().lock().unwrap_or_else(|e| e.into_inner());
    let node = state.nodes.get(&node_id)
        .ok_or_else(|| JsErrorBox::generic("Node not found"))?;
    with_param!(node, param.as_str(), |p| p.set_value_at_time(value as f32, start_time));
    Ok(())
}

#[op2(fast)]
pub fn op_audio_param_set_target_at_time(
    #[smi] node_id: u32,
    #[string] param: String,
    value: f64,
    start_time: f64,
    time_constant: f64,
) -> Result<(), JsErrorBox> {
    let state = audio().lock().unwrap_or_else(|e| e.into_inner());
    let node = state.nodes.get(&node_id)
        .ok_or_else(|| JsErrorBox::generic("Node not found"))?;
    with_param!(node, param.as_str(), |p| p.set_target_at_time(value as f32, start_time, time_constant));
    Ok(())
}

#[op2(fast)]
pub fn op_audio_param_linear_ramp(
    #[smi] node_id: u32,
    #[string] param: String,
    value: f64,
    end_time: f64,
) -> Result<(), JsErrorBox> {
    let state = audio().lock().unwrap_or_else(|e| e.into_inner());
    let node = state.nodes.get(&node_id)
        .ok_or_else(|| JsErrorBox::generic("Node not found"))?;
    with_param!(node, param.as_str(), |p| p.linear_ramp_to_value_at_time(value as f32, end_time));
    Ok(())
}

#[op2(fast)]
pub fn op_audio_param_exponential_ramp(
    #[smi] node_id: u32,
    #[string] param: String,
    value: f64,
    end_time: f64,
) -> Result<(), JsErrorBox> {
    let state = audio().lock().unwrap_or_else(|e| e.into_inner());
    let node = state.nodes.get(&node_id)
        .ok_or_else(|| JsErrorBox::generic("Node not found"))?;
    with_param!(node, param.as_str(), |p| p.exponential_ramp_to_value_at_time(value as f32, end_time));
    Ok(())
}

#[op2(fast)]
pub fn op_audio_param_cancel_scheduled_values(
    #[smi] node_id: u32,
    #[string] param: String,
    cancel_time: f64,
) -> Result<(), JsErrorBox> {
    let t = cancel_time.max(0.0);
    let state = audio().lock().unwrap_or_else(|e| e.into_inner());
    let node = state.nodes.get(&node_id)
        .ok_or_else(|| JsErrorBox::generic("Node not found"))?;
    with_param!(node, param.as_str(), |p| p.cancel_scheduled_values(t));
    Ok(())
}

#[op2(fast)]
pub fn op_audio_param_cancel_and_hold(
    #[smi] node_id: u32,
    #[string] param: String,
    cancel_time: f64,
) -> Result<(), JsErrorBox> {
    let t = cancel_time.max(0.0);
    let state = audio().lock().unwrap_or_else(|e| e.into_inner());
    let node = state.nodes.get(&node_id)
        .ok_or_else(|| JsErrorBox::generic("Node not found"))?;
    with_param!(node, param.as_str(), |p| p.cancel_and_hold_at_time(t));
    Ok(())
}

// ======================= AudioBuffer =======================

/// Dedicated context for decoding — avoids holding the main AudioState mutex
/// during CPU work, and avoids opening an ALSA device (OfflineAudioContext).
static DECODE_CTX: OnceLock<OfflineAudioContext> = OnceLock::new();

fn decode_ctx() -> &'static OfflineAudioContext {
    DECODE_CTX.get_or_init(|| OfflineAudioContext::new(2, 1, 44100.0))
}

#[op2]
#[serde]
pub fn op_audio_decode_audio_data(
    #[smi] _ctx_id: u32,
    #[buffer] data: &[u8],
) -> Result<AudioBufferInfo, JsErrorBox> {
    let data_vec = data.to_vec();
    let cursor = std::io::Cursor::new(data_vec);
    let buffer = decode_ctx().decode_audio_data_sync(cursor)
        .map_err(|e| JsErrorBox::generic(format!("Audio decode failed: {}", e)))?;
    let mut s = audio().lock().unwrap_or_else(|e| e.into_inner());
    let id = s.alloc_id();
    let info = AudioBufferInfo {
        id,
        duration: buffer.duration(),
        length: buffer.length() as u32,
        sample_rate: buffer.sample_rate(),
        number_of_channels: buffer.number_of_channels() as u32,
    };
    s.buffers.insert(id, buffer);
    Ok(info)
}

#[op2]
#[serde]
pub fn op_audio_create_buffer(
    #[smi] ctx_id: u32,
    #[smi] channels: u32,
    #[smi] length: u32,
    sample_rate: f64,
) -> Result<AudioBufferInfo, JsErrorBox> {
    let mut s = audio().lock().unwrap_or_else(|e| e.into_inner());
    let buffer = if let Some(ctx) = s.contexts.get(&ctx_id) {
        ctx.create_buffer(channels as usize, length as usize, sample_rate as f32)
    } else if let Some(ctx) = s.offline_contexts.get(&ctx_id) {
        ctx.create_buffer(channels as usize, length as usize, sample_rate as f32)
    } else {
        return Err(JsErrorBox::generic("AudioContext not found"));
    };
    let id = s.alloc_id();
    let info = AudioBufferInfo {
        id,
        duration: buffer.duration(),
        length: buffer.length() as u32,
        sample_rate: buffer.sample_rate(),
        number_of_channels: buffer.number_of_channels() as u32,
    };
    s.buffers.insert(id, buffer);
    Ok(info)
}

#[op2(fast)]
pub fn op_audio_buffer_copy_to_channel(
    #[smi] buffer_id: u32,
    #[buffer] data: &[f32],
    #[smi] channel: u32,
) -> Result<(), JsErrorBox> {
    let mut s = audio().lock().unwrap_or_else(|e| e.into_inner());
    let buffer = s.buffers.get_mut(&buffer_id)
        .ok_or_else(|| JsErrorBox::generic("AudioBuffer not found"))?;
    buffer.copy_to_channel(data, channel as usize);
    Ok(())
}

#[op2]
#[buffer]
pub fn op_audio_buffer_get_channel_data(
    #[smi] buffer_id: u32,
    #[smi] channel: u32,
) -> Result<Vec<f32>, JsErrorBox> {
    let s = audio().lock().unwrap_or_else(|e| e.into_inner());
    let buffer = s.buffers.get(&buffer_id)
        .ok_or_else(|| JsErrorBox::generic("AudioBuffer not found"))?;
    let data = buffer.get_channel_data(channel as usize);
    Ok(data.to_vec())
}

#[op2(fast)]
pub fn op_audio_buffer_drop(#[smi] buffer_id: u32) {
    audio().lock().unwrap_or_else(|e| e.into_inner()).buffers.remove(&buffer_id);
}

// ======================= AudioBufferSourceNode =======================
// web-audio-api-rs requires &mut self for set_buffer/start/stop/set_loop.

#[op2(fast)]
pub fn op_audio_buffer_source_set_buffer(
    #[smi] node_id: u32,
    #[smi] buffer_id: u32,
) -> Result<(), JsErrorBox> {
    let mut state = audio().lock().unwrap_or_else(|e| e.into_inner());
    let buffer = state.buffers.get(&buffer_id)
        .ok_or_else(|| JsErrorBox::generic("AudioBuffer not found"))?
        .clone();
    let node = state.nodes.get_mut(&node_id)
        .ok_or_else(|| JsErrorBox::generic("Node not found"))?;
    match node {
        NodeWrapper::BufferSource(n) => n.set_buffer(buffer),
        _ => return Err(JsErrorBox::generic("Not a BufferSourceNode")),
    }
    Ok(())
}

// catch_audio: start/stop can panic (double start, start after stop, etc.)

#[op2(fast)]
pub fn op_audio_buffer_source_start(#[smi] node_id: u32, when: f64, offset: f64, duration: f64) -> Result<(), JsErrorBox> {
    catch_audio(std::panic::AssertUnwindSafe(|| {
        let mut state = audio().lock().unwrap_or_else(|e| e.into_inner());
        match state.nodes.get_mut(&node_id) {
            Some(NodeWrapper::BufferSource(n)) => {
                let start_time = if when > 0.0 { when } else { 0.0 };
                if duration > 0.0 {
                    n.start_at_with_offset_and_duration(start_time, offset, duration);
                } else if offset > 0.0 {
                    n.start_at_with_offset(start_time, offset);
                } else if when > 0.0 {
                    n.start_at(when);
                } else {
                    n.start();
                }
                Ok(())
            }
            _ => Err(JsErrorBox::generic("Not a BufferSourceNode")),
        }
    }))
}

#[op2(fast)]
pub fn op_audio_buffer_source_stop(#[smi] node_id: u32, when: f64) -> Result<(), JsErrorBox> {
    catch_audio(std::panic::AssertUnwindSafe(|| {
        let mut state = audio().lock().unwrap_or_else(|e| e.into_inner());
        match state.nodes.get_mut(&node_id) {
            Some(NodeWrapper::BufferSource(n)) => {
                if when > 0.0 { n.stop_at(when); } else { n.stop(); }
                Ok(())
            }
            _ => Err(JsErrorBox::generic("Not a BufferSourceNode")),
        }
    }))
}

#[op2(fast)]
pub fn op_audio_buffer_source_set_loop(#[smi] node_id: u32, looping: bool) -> Result<(), JsErrorBox> {
    let mut state = audio().lock().unwrap_or_else(|e| e.into_inner());
    match state.nodes.get_mut(&node_id) {
        Some(NodeWrapper::BufferSource(n)) => { n.set_loop(looping); Ok(()) }
        _ => Err(JsErrorBox::generic("Not a BufferSourceNode")),
    }
}

#[op2(fast)]
pub fn op_audio_buffer_source_set_loop_start(#[smi] node_id: u32, value: f64) -> Result<(), JsErrorBox> {
    let mut state = audio().lock().unwrap_or_else(|e| e.into_inner());
    match state.nodes.get_mut(&node_id) {
        Some(NodeWrapper::BufferSource(n)) => { n.set_loop_start(value); Ok(()) }
        _ => Err(JsErrorBox::generic("Not a BufferSourceNode")),
    }
}

#[op2(fast)]
pub fn op_audio_buffer_source_set_loop_end(#[smi] node_id: u32, value: f64) -> Result<(), JsErrorBox> {
    let mut state = audio().lock().unwrap_or_else(|e| e.into_inner());
    match state.nodes.get_mut(&node_id) {
        Some(NodeWrapper::BufferSource(n)) => { n.set_loop_end(value); Ok(()) }
        _ => Err(JsErrorBox::generic("Not a BufferSourceNode")),
    }
}

// ======================= OscillatorNode =======================

use web_audio_api::node::OscillatorType;
use web_audio_api::node::BiquadFilterType;

#[op2(fast)]
pub fn op_audio_oscillator_set_type(
    #[smi] node_id: u32,
    #[string] osc_type: String,
) -> Result<(), JsErrorBox> {
    let t = match osc_type.as_str() {
        "sine" => OscillatorType::Sine,
        "square" => OscillatorType::Square,
        "sawtooth" => OscillatorType::Sawtooth,
        "triangle" => OscillatorType::Triangle,
        other => return Err(JsErrorBox::generic(format!("Unknown oscillator type: {}", other))),
    };
    let mut state = audio().lock().unwrap_or_else(|e| e.into_inner());
    match state.nodes.get_mut(&node_id) {
        Some(NodeWrapper::Oscillator(n)) => { n.set_type(t); Ok(()) }
        _ => Err(JsErrorBox::generic("Not an OscillatorNode")),
    }
}

// catch_audio: start/stop can panic

#[op2(fast)]
pub fn op_audio_oscillator_start(#[smi] node_id: u32, when: f64) -> Result<(), JsErrorBox> {
    catch_audio(std::panic::AssertUnwindSafe(|| {
        let mut state = audio().lock().unwrap_or_else(|e| e.into_inner());
        match state.nodes.get_mut(&node_id) {
            Some(NodeWrapper::Oscillator(n)) => {
                if when > 0.0 { n.start_at(when); } else { n.start(); }
                Ok(())
            }
            _ => Err(JsErrorBox::generic("Not an OscillatorNode")),
        }
    }))
}

#[op2(fast)]
pub fn op_audio_oscillator_stop(#[smi] node_id: u32, when: f64) -> Result<(), JsErrorBox> {
    catch_audio(std::panic::AssertUnwindSafe(|| {
        let mut state = audio().lock().unwrap_or_else(|e| e.into_inner());
        match state.nodes.get_mut(&node_id) {
            Some(NodeWrapper::Oscillator(n)) => {
                if when > 0.0 { n.stop_at(when); } else { n.stop(); }
                Ok(())
            }
            _ => Err(JsErrorBox::generic("Not an OscillatorNode")),
        }
    }))
}

// ======================= ConstantSourceNode =======================
// catch_audio: start/stop can panic

#[op2(fast)]
pub fn op_audio_constant_source_start(#[smi] node_id: u32, when: f64) -> Result<(), JsErrorBox> {
    catch_audio(std::panic::AssertUnwindSafe(|| {
        let mut state = audio().lock().unwrap_or_else(|e| e.into_inner());
        match state.nodes.get_mut(&node_id) {
            Some(NodeWrapper::ConstantSource(n)) => {
                if when > 0.0 { n.start_at(when); } else { n.start(); }
                Ok(())
            }
            _ => Err(JsErrorBox::generic("Not a ConstantSourceNode")),
        }
    }))
}

#[op2(fast)]
pub fn op_audio_constant_source_stop(#[smi] node_id: u32, when: f64) -> Result<(), JsErrorBox> {
    catch_audio(std::panic::AssertUnwindSafe(|| {
        let mut state = audio().lock().unwrap_or_else(|e| e.into_inner());
        match state.nodes.get_mut(&node_id) {
            Some(NodeWrapper::ConstantSource(n)) => {
                if when > 0.0 { n.stop_at(when); } else { n.stop(); }
                Ok(())
            }
            _ => Err(JsErrorBox::generic("Not a ConstantSourceNode")),
        }
    }))
}

// ======================= PannerNode Configuration =======================

#[derive(Deserialize)]
pub struct PannerConfig {
    pub panning_model: Option<String>,
    pub distance_model: Option<String>,
    pub ref_distance: Option<f64>,
    pub max_distance: Option<f64>,
    pub rolloff_factor: Option<f64>,
    pub cone_inner_angle: Option<f64>,
    pub cone_outer_angle: Option<f64>,
    pub cone_outer_gain: Option<f64>,
}

#[op2]
pub fn op_audio_panner_configure(
    #[smi] node_id: u32,
    #[serde] config: PannerConfig,
) -> Result<(), JsErrorBox> {
    use web_audio_api::node::{PanningModelType, DistanceModelType};
    let mut state = audio().lock().unwrap_or_else(|e| e.into_inner());
    match state.nodes.get_mut(&node_id) {
        Some(NodeWrapper::Panner(n)) => {
            if let Some(ref model) = config.panning_model {
                n.set_panning_model(match model.as_str() {
                    "HRTF" => PanningModelType::HRTF,
                    _ => PanningModelType::EqualPower,
                });
            }
            if let Some(ref model) = config.distance_model {
                n.set_distance_model(match model.as_str() {
                    "linear" => DistanceModelType::Linear,
                    "exponential" => DistanceModelType::Exponential,
                    _ => DistanceModelType::Inverse,
                });
            }
            if let Some(v) = config.ref_distance { n.set_ref_distance(v); }
            if let Some(v) = config.max_distance { n.set_max_distance(v); }
            if let Some(v) = config.rolloff_factor { n.set_rolloff_factor(v); }
            if let Some(v) = config.cone_inner_angle { n.set_cone_inner_angle(v); }
            if let Some(v) = config.cone_outer_angle { n.set_cone_outer_angle(v); }
            if let Some(v) = config.cone_outer_gain { n.set_cone_outer_gain(v); }
            Ok(())
        }
        _ => Err(JsErrorBox::generic("Not a PannerNode")),
    }
}

// ======================= BiquadFilterNode =======================

#[op2(fast)]
pub fn op_audio_biquad_set_type(
    #[smi] node_id: u32,
    #[string] filter_type: String,
) -> Result<(), JsErrorBox> {
    let t = match filter_type.as_str() {
        "lowpass" => BiquadFilterType::Lowpass,
        "highpass" => BiquadFilterType::Highpass,
        "bandpass" => BiquadFilterType::Bandpass,
        "lowshelf" => BiquadFilterType::Lowshelf,
        "highshelf" => BiquadFilterType::Highshelf,
        "peaking" => BiquadFilterType::Peaking,
        "notch" => BiquadFilterType::Notch,
        "allpass" => BiquadFilterType::Allpass,
        other => return Err(JsErrorBox::generic(format!("Unknown filter type: {}", other))),
    };
    let mut state = audio().lock().unwrap_or_else(|e| e.into_inner());
    match state.nodes.get_mut(&node_id) {
        Some(NodeWrapper::BiquadFilter(n)) => { n.set_type(t); Ok(()) }
        _ => Err(JsErrorBox::generic("Not a BiquadFilterNode")),
    }
}

// ======================= AudioListener =======================

#[op2(fast)]
pub fn op_audio_listener_set_position(
    #[smi] ctx_id: u32,
    x: f64, y: f64, z: f64,
) -> Result<(), JsErrorBox> {
    let state = audio().lock().unwrap_or_else(|e| e.into_inner());
    let ctx = state.contexts.get(&ctx_id)
        .ok_or_else(|| JsErrorBox::generic("AudioContext not found"))?;
    let listener = ctx.listener();
    listener.position_x().set_value(x as f32);
    listener.position_y().set_value(y as f32);
    listener.position_z().set_value(z as f32);
    Ok(())
}

#[op2(fast)]
pub fn op_audio_listener_set_orientation(
    #[smi] ctx_id: u32,
    fx: f64, fy: f64, fz: f64,
    ux: f64, uy: f64, uz: f64,
) -> Result<(), JsErrorBox> {
    let state = audio().lock().unwrap_or_else(|e| e.into_inner());
    let ctx = state.contexts.get(&ctx_id)
        .ok_or_else(|| JsErrorBox::generic("AudioContext not found"))?;
    let listener = ctx.listener();
    listener.forward_x().set_value(fx as f32);
    listener.forward_y().set_value(fy as f32);
    listener.forward_z().set_value(fz as f32);
    listener.up_x().set_value(ux as f32);
    listener.up_y().set_value(uy as f32);
    listener.up_z().set_value(uz as f32);
    Ok(())
}

// ======================= ChannelMerger / Splitter =======================

#[op2]
#[serde]
pub fn op_audio_create_channel_merger(#[smi] ctx_id: u32, #[smi] inputs: u32) -> Result<AudioId, JsErrorBox> {
    let mut s = audio().lock().unwrap_or_else(|e| e.into_inner());
    let node = if let Some(ctx) = s.contexts.get(&ctx_id) {
        ctx.create_channel_merger(inputs as usize)
    } else if let Some(ctx) = s.offline_contexts.get(&ctx_id) {
        ctx.create_channel_merger(inputs as usize)
    } else {
        return Err(JsErrorBox::generic("AudioContext not found"));
    };
    let id = s.alloc_id();
    s.nodes.insert(id, NodeWrapper::ChannelMerger(node));
    Ok(AudioId { id })
}

#[op2]
#[serde]
pub fn op_audio_create_channel_splitter(#[smi] ctx_id: u32, #[smi] outputs: u32) -> Result<AudioId, JsErrorBox> {
    let mut s = audio().lock().unwrap_or_else(|e| e.into_inner());
    let node = if let Some(ctx) = s.contexts.get(&ctx_id) {
        ctx.create_channel_splitter(outputs as usize)
    } else if let Some(ctx) = s.offline_contexts.get(&ctx_id) {
        ctx.create_channel_splitter(outputs as usize)
    } else {
        return Err(JsErrorBox::generic("AudioContext not found"));
    };
    let id = s.alloc_id();
    s.nodes.insert(id, NodeWrapper::ChannelSplitter(node));
    Ok(AudioId { id })
}

// ======================= WaveShaperNode =======================

#[op2(fast)]
pub fn op_audio_wave_shaper_set_curve(
    #[smi] node_id: u32,
    #[buffer] curve: &[f32],
) -> Result<(), JsErrorBox> {
    let mut state = audio().lock().unwrap_or_else(|e| e.into_inner());
    let node = state.nodes.get_mut(&node_id)
        .ok_or_else(|| JsErrorBox::generic("Node not found"))?;
    match node {
        NodeWrapper::WaveShaper(n) => {
            n.set_curve(curve.to_vec());
            Ok(())
        }
        _ => Err(JsErrorBox::generic("Not a WaveShaperNode")),
    }
}

// ======================= ConvolverNode =======================
// catch_audio: set_buffer does FFT partitioning which can panic

#[op2(fast)]
pub fn op_audio_convolver_set_buffer(
    #[smi] node_id: u32,
    #[smi] buffer_id: u32,
) -> Result<(), JsErrorBox> {
    catch_audio(std::panic::AssertUnwindSafe(|| {
        let mut state = audio().lock().unwrap_or_else(|e| e.into_inner());
        let buffer = state.buffers.get(&buffer_id)
            .ok_or_else(|| JsErrorBox::generic("AudioBuffer not found"))?
            .clone();
        let node = state.nodes.get_mut(&node_id)
            .ok_or_else(|| JsErrorBox::generic("Node not found"))?;
        match node {
            NodeWrapper::Convolver(n) => { n.set_buffer(buffer); Ok(()) }
            _ => Err(JsErrorBox::generic("Not a ConvolverNode")),
        }
    }))
}

// ======================= OfflineAudioContext =======================

use web_audio_api::node::AudioNode as _;

#[op2]
#[serde]
pub fn op_offline_context_create(
    #[smi] channels: u32,
    #[smi] length: u32,
    sample_rate: f64,
) -> Result<AudioContextInfo, JsErrorBox> {
    let mut s = audio().lock().unwrap_or_else(|e| e.into_inner());
    let ctx = OfflineAudioContext::new(channels as usize, length as usize, sample_rate as f32);
    let sample_rate = ctx.sample_rate();
    let dest = ctx.destination();
    let ctx_id = s.alloc_id();
    let dest_id = s.alloc_id();
    log::info!("[audio] OfflineContext {} created ({}Hz), destination={}", ctx_id, sample_rate, dest_id);
    s.nodes.insert(dest_id, NodeWrapper::Destination(dest));
    s.offline_contexts.insert(ctx_id, ctx);
    Ok(AudioContextInfo { id: ctx_id, destination_id: dest_id, sample_rate })
}

#[op2]
#[serde]
pub fn op_offline_context_start_rendering(
    #[smi] ctx_id: u32,
) -> Result<AudioBufferInfo, JsErrorBox> {
    // Remove context from the active map, render outside the lock.
    let mut ctx = {
        let mut s = audio().lock().unwrap_or_else(|e| e.into_inner());
        s.offline_contexts.remove(&ctx_id)
            .ok_or_else(|| JsErrorBox::generic("OfflineAudioContext not found"))?
    };
    let buffer = ctx.start_rendering_sync();
    // Re-lock to store result AND keep the context alive.
    // Nodes created on this context still hold internal references to its audio
    // graph. If the context is dropped before those nodes, their Drop impl tries
    // to send a deregister message to a dead channel → panic/abort.
    let mut s = audio().lock().unwrap_or_else(|e| e.into_inner());
    s.finished_offline_contexts.push(ctx);
    let id = s.alloc_id();
    let info = AudioBufferInfo {
        id,
        duration: buffer.duration(),
        length: buffer.length() as u32,
        sample_rate: buffer.sample_rate(),
        number_of_channels: buffer.number_of_channels() as u32,
    };
    s.buffers.insert(id, buffer);
    Ok(info)
}

// ======================= Channel Configuration =======================

use web_audio_api::node::{ChannelCountMode, ChannelInterpretation};

#[op2(fast)]
pub fn op_audio_set_channel_count(#[smi] node_id: u32, #[smi] count: u32) -> Result<(), JsErrorBox> {
    let state = audio().lock().unwrap_or_else(|e| e.into_inner());
    let node = state.nodes.get(&node_id)
        .ok_or_else(|| JsErrorBox::generic("Node not found"))?;
    with_node!(node, |n| { n.set_channel_count(count as usize); });
    Ok(())
}

#[op2(fast)]
pub fn op_audio_set_channel_count_mode(
    #[smi] node_id: u32,
    #[string] mode: String,
) -> Result<(), JsErrorBox> {
    let m = match mode.as_str() {
        "max" => ChannelCountMode::Max,
        "clamped-max" => ChannelCountMode::ClampedMax,
        "explicit" => ChannelCountMode::Explicit,
        other => return Err(JsErrorBox::generic(format!("Unknown channelCountMode: {}", other))),
    };
    let state = audio().lock().unwrap_or_else(|e| e.into_inner());
    let node = state.nodes.get(&node_id)
        .ok_or_else(|| JsErrorBox::generic("Node not found"))?;
    with_node!(node, |n| { n.set_channel_count_mode(m); });
    Ok(())
}

#[op2(fast)]
pub fn op_audio_set_channel_interpretation(
    #[smi] node_id: u32,
    #[string] interp: String,
) -> Result<(), JsErrorBox> {
    let i = match interp.as_str() {
        "speakers" => ChannelInterpretation::Speakers,
        "discrete" => ChannelInterpretation::Discrete,
        other => return Err(JsErrorBox::generic(format!("Unknown channelInterpretation: {}", other))),
    };
    let state = audio().lock().unwrap_or_else(|e| e.into_inner());
    let node = state.nodes.get(&node_id)
        .ok_or_else(|| JsErrorBox::generic("Node not found"))?;
    with_node!(node, |n| { n.set_channel_interpretation(i); });
    Ok(())
}

// ======================= Cleanup =======================
// catch_audio_void: node drop can panic if parent context was already dropped

#[op2(fast)]
pub fn op_audio_node_drop(#[smi] node_id: u32) {
    let node = audio().lock().unwrap_or_else(|e| e.into_inner()).nodes.remove(&node_id);
    if let Some(node) = node {
        let _ = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            drop(node);
        }));
    }
}
