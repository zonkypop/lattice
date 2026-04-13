// src/audio.rs — Native audio ops backed by web-audio-api-rs.
// Follows the same global-state + deno_core ops pattern as sqlite.rs.

use deno_core::op2;
use deno_error::JsErrorBox;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::{Mutex, OnceLock};

use web_audio_api::context::{AudioContext, BaseAudioContext};
use web_audio_api::node::{AudioNode, AudioScheduledSourceNode};

// ======================= Global State =======================

struct AudioState {
    contexts: HashMap<u32, AudioContext>,
    nodes: HashMap<u32, NodeWrapper>,
    buffers: HashMap<u32, web_audio_api::AudioBuffer>,
    next_id: u32,
}

impl AudioState {
    fn new() -> Self {
        Self { contexts: HashMap::new(), nodes: HashMap::new(), buffers: HashMap::new(), next_id: 1 }
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
    let mut s = audio().lock().unwrap();
    let ctx = AudioContext::default();
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
    audio().lock().unwrap().contexts.get(&ctx_id).map(|c| c.current_time()).unwrap_or(0.0)
}

#[op2(fast)]
pub fn op_audio_context_close(#[smi] ctx_id: u32) {
    if let Some(ctx) = audio().lock().unwrap().contexts.remove(&ctx_id) {
        ctx.close_sync();
    }
}

// ======================= Node Creation =======================

macro_rules! create_node_op {
    ($fn_name:ident, $create_method:ident, $variant:ident) => {
        #[op2]
        #[serde]
        pub fn $fn_name(#[smi] ctx_id: u32) -> Result<AudioId, JsErrorBox> {
            let mut s = audio().lock().unwrap();
            let node = {
                let ctx = s.contexts.get(&ctx_id)
                    .ok_or_else(|| JsErrorBox::generic("AudioContext not found"))?;
                ctx.$create_method()
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

#[op2]
#[serde]
pub fn op_audio_create_delay(#[smi] ctx_id: u32, max_delay_time: f64) -> Result<AudioId, JsErrorBox> {
    let mut s = audio().lock().unwrap();
    let node = {
        let ctx = s.contexts.get(&ctx_id)
            .ok_or_else(|| JsErrorBox::generic("AudioContext not found"))?;
        ctx.create_delay(max_delay_time)
    };
    let id = s.alloc_id();
    s.nodes.insert(id, NodeWrapper::Delay(node));
    Ok(AudioId { id })
}

// ======================= Connections =======================

#[op2(fast)]
pub fn op_audio_connect(#[smi] src_id: u32, #[smi] dst_id: u32) -> Result<(), JsErrorBox> {
    let state = audio().lock().unwrap();
    let src = state.nodes.get(&src_id)
        .ok_or_else(|| JsErrorBox::generic("Source node not found"))?;
    let dst = state.nodes.get(&dst_id)
        .ok_or_else(|| JsErrorBox::generic("Destination node not found"))?;
    with_node!(src, |s_node| {
        with_node!(dst, |d_node| {
            s_node.connect(d_node);
        });
    });
    Ok(())
}

#[op2(fast)]
pub fn op_audio_disconnect(#[smi] node_id: u32) -> Result<(), JsErrorBox> {
    let state = audio().lock().unwrap();
    let node = state.nodes.get(&node_id)
        .ok_or_else(|| JsErrorBox::generic("Node not found"))?;
    with_node!(node, |n| { n.disconnect(); });
    Ok(())
}

// ======================= AudioParam Ops =======================
// Using #[op2(fast)] — matches sqlite.rs pattern for ops with #[string] params.

#[op2(fast)]
pub fn op_audio_param_set_value(
    #[smi] node_id: u32,
    #[string] param: String,
    value: f64,
) -> Result<(), JsErrorBox> {
    let state = audio().lock().unwrap();
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
    let state = audio().lock().unwrap();
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
    let state = audio().lock().unwrap();
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
    let state = audio().lock().unwrap();
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
    let state = audio().lock().unwrap();
    let node = state.nodes.get(&node_id)
        .ok_or_else(|| JsErrorBox::generic("Node not found"))?;
    with_param!(node, param.as_str(), |p| p.exponential_ramp_to_value_at_time(value as f32, end_time));
    Ok(())
}

// ======================= AudioBuffer =======================

#[op2]
#[serde]
pub fn op_audio_decode_audio_data(
    #[smi] ctx_id: u32,
    #[buffer] data: &[u8],
) -> Result<AudioBufferInfo, JsErrorBox> {
    let data_vec = data.to_vec();
    let mut s = audio().lock().unwrap();
    let buffer = {
        let ctx = s.contexts.get(&ctx_id)
            .ok_or_else(|| JsErrorBox::generic("AudioContext not found"))?;
        let cursor = std::io::Cursor::new(data_vec);
        ctx.decode_audio_data_sync(cursor)
            .map_err(|e| JsErrorBox::generic(format!("Audio decode failed: {}", e)))?
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
pub fn op_audio_buffer_drop(#[smi] buffer_id: u32) {
    audio().lock().unwrap().buffers.remove(&buffer_id);
}

// ======================= AudioBufferSourceNode =======================
// web-audio-api-rs requires &mut self for set_buffer/start/stop/set_loop.

#[op2(fast)]
pub fn op_audio_buffer_source_set_buffer(
    #[smi] node_id: u32,
    #[smi] buffer_id: u32,
) -> Result<(), JsErrorBox> {
    let mut state = audio().lock().unwrap();
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

#[op2(fast)]
pub fn op_audio_buffer_source_start(#[smi] node_id: u32, when: f64) -> Result<(), JsErrorBox> {
    let mut state = audio().lock().unwrap();
    match state.nodes.get_mut(&node_id) {
        Some(NodeWrapper::BufferSource(n)) => {
            if when > 0.0 { n.start_at(when); } else { n.start(); }
            Ok(())
        }
        _ => Err(JsErrorBox::generic("Not a BufferSourceNode")),
    }
}

#[op2(fast)]
pub fn op_audio_buffer_source_stop(#[smi] node_id: u32, when: f64) -> Result<(), JsErrorBox> {
    let mut state = audio().lock().unwrap();
    match state.nodes.get_mut(&node_id) {
        Some(NodeWrapper::BufferSource(n)) => {
            if when > 0.0 { n.stop_at(when); } else { n.stop(); }
            Ok(())
        }
        _ => Err(JsErrorBox::generic("Not a BufferSourceNode")),
    }
}

#[op2(fast)]
pub fn op_audio_buffer_source_set_loop(#[smi] node_id: u32, looping: bool) -> Result<(), JsErrorBox> {
    let mut state = audio().lock().unwrap();
    match state.nodes.get_mut(&node_id) {
        Some(NodeWrapper::BufferSource(n)) => { n.set_loop(looping); Ok(()) }
        _ => Err(JsErrorBox::generic("Not a BufferSourceNode")),
    }
}

#[op2(fast)]
pub fn op_audio_buffer_source_set_loop_start(#[smi] node_id: u32, value: f64) -> Result<(), JsErrorBox> {
    let mut state = audio().lock().unwrap();
    match state.nodes.get_mut(&node_id) {
        Some(NodeWrapper::BufferSource(n)) => { n.set_loop_start(value); Ok(()) }
        _ => Err(JsErrorBox::generic("Not a BufferSourceNode")),
    }
}

#[op2(fast)]
pub fn op_audio_buffer_source_set_loop_end(#[smi] node_id: u32, value: f64) -> Result<(), JsErrorBox> {
    let mut state = audio().lock().unwrap();
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
    let mut state = audio().lock().unwrap();
    match state.nodes.get_mut(&node_id) {
        Some(NodeWrapper::Oscillator(n)) => { n.set_type(t); Ok(()) }
        _ => Err(JsErrorBox::generic("Not an OscillatorNode")),
    }
}

#[op2(fast)]
pub fn op_audio_oscillator_start(#[smi] node_id: u32, when: f64) -> Result<(), JsErrorBox> {
    let mut state = audio().lock().unwrap();
    match state.nodes.get_mut(&node_id) {
        Some(NodeWrapper::Oscillator(n)) => {
            if when > 0.0 { n.start_at(when); } else { n.start(); }
            Ok(())
        }
        _ => Err(JsErrorBox::generic("Not an OscillatorNode")),
    }
}

#[op2(fast)]
pub fn op_audio_oscillator_stop(#[smi] node_id: u32, when: f64) -> Result<(), JsErrorBox> {
    let mut state = audio().lock().unwrap();
    match state.nodes.get_mut(&node_id) {
        Some(NodeWrapper::Oscillator(n)) => {
            if when > 0.0 { n.stop_at(when); } else { n.stop(); }
            Ok(())
        }
        _ => Err(JsErrorBox::generic("Not an OscillatorNode")),
    }
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
    let mut state = audio().lock().unwrap();
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
    let mut state = audio().lock().unwrap();
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
    let state = audio().lock().unwrap();
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
    let state = audio().lock().unwrap();
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

// ======================= Cleanup =======================

#[op2(fast)]
pub fn op_audio_node_drop(#[smi] node_id: u32) {
    audio().lock().unwrap().nodes.remove(&node_id);
}
