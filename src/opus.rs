// src/opus.rs — Minimal safe Opus encoder/decoder wrapper over vendored C libopus.

use std::os::raw::c_int;

// Raw FFI bindings (only what we need)
extern "C" {
    fn opus_encoder_get_size(channels: c_int) -> c_int;
    fn opus_encoder_init(st: *mut u8, fs: i32, channels: c_int, application: c_int) -> c_int;
    fn opus_encode_float(
        st: *mut u8,
        pcm: *const f32,
        frame_size: c_int,
        data: *mut u8,
        max_data_bytes: i32,
    ) -> i32;

    fn opus_decoder_get_size(channels: c_int) -> c_int;
    fn opus_decoder_init(st: *mut u8, fs: i32, channels: c_int) -> c_int;
    fn opus_decode_float(
        st: *mut u8,
        data: *const u8,
        len: i32,
        pcm: *mut f32,
        frame_size: c_int,
        decode_fec: c_int,
    ) -> c_int;
}

const OPUS_APPLICATION_VOIP: c_int = 2048;
const OPUS_OK: c_int = 0;

pub struct Encoder {
    state: Vec<u8>,
}

impl Encoder {
    pub fn new(sample_rate: i32, channels: i32) -> Result<Self, String> {
        let size = unsafe { opus_encoder_get_size(channels) };
        if size <= 0 {
            return Err("opus_encoder_get_size failed".into());
        }
        let mut state = vec![0u8; size as usize];
        let ret = unsafe { opus_encoder_init(state.as_mut_ptr(), sample_rate, channels, OPUS_APPLICATION_VOIP) };
        if ret != OPUS_OK {
            return Err(format!("opus_encoder_init failed: {}", ret));
        }
        Ok(Encoder { state })
    }

    /// Encode float PCM samples into Opus. Returns number of bytes written.
    pub fn encode_float(&mut self, pcm: &[f32], output: &mut [u8]) -> Result<usize, String> {
        let frame_size = pcm.len() as c_int; // mono: samples == frame_size
        let ret = unsafe {
            opus_encode_float(
                self.state.as_mut_ptr(),
                pcm.as_ptr(),
                frame_size,
                output.as_mut_ptr(),
                output.len() as i32,
            )
        };
        if ret < 0 {
            Err(format!("opus_encode_float failed: {}", ret))
        } else {
            Ok(ret as usize)
        }
    }
}

// Safety: Opus encoder state is self-contained, no thread-local or shared mutable state
unsafe impl Send for Encoder {}

pub struct Decoder {
    state: Vec<u8>,
}

impl Decoder {
    pub fn new(sample_rate: i32, channels: i32) -> Result<Self, String> {
        let size = unsafe { opus_decoder_get_size(channels) };
        if size <= 0 {
            return Err("opus_decoder_get_size failed".into());
        }
        let mut state = vec![0u8; size as usize];
        let ret = unsafe { opus_decoder_init(state.as_mut_ptr(), sample_rate, channels) };
        if ret != OPUS_OK {
            return Err(format!("opus_decoder_init failed: {}", ret));
        }
        Ok(Decoder { state })
    }

    /// Decode Opus packet into float PCM. Returns number of samples decoded.
    pub fn decode_float(&mut self, data: &[u8], output: &mut [f32]) -> Result<usize, String> {
        let frame_size = output.len() as c_int;
        let ret = unsafe {
            opus_decode_float(
                self.state.as_mut_ptr(),
                data.as_ptr(),
                data.len() as i32,
                output.as_mut_ptr(),
                frame_size,
                0, // no FEC
            )
        };
        if ret < 0 {
            Err(format!("opus_decode_float failed: {}", ret))
        } else {
            Ok(ret as usize)
        }
    }
}

unsafe impl Send for Decoder {}
