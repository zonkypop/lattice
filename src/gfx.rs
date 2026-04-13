// src/gfx.rs - Graphics module for WebGPU operations

use deno_core::{op2, OpState, Resource, ResourceId, JsBuffer};
use deno_error::JsErrorBox;
use image::{GenericImageView, DynamicImage};
use serde::{Deserialize, Serialize};
use std::sync::{Arc, OnceLock};
use std::{iter};
use wgpu::util::DeviceExt;

use std::sync::Mutex;
use std::collections::HashMap;
use std::sync::atomic::{AtomicU32, AtomicBool, Ordering};

use std::sync::RwLock;

use crate::xr::is_xr_active;

// ======================= Graphics Context =======================

struct MsaaState {
    texture: wgpu::Texture,
    view: wgpu::TextureView,
    width: u32,
    height: u32,
}


pub struct GfxContext {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub surface: Option<wgpu::Surface<'static>>,
    pub format: wgpu::TextureFormat,
}

static GFX_CTX: OnceLock<&'static GfxContext> = OnceLock::new();

pub fn set_gfx_context(ctx: &'static GfxContext) -> Result<(), &'static GfxContext> {
    GFX_CTX.set(ctx)
}

pub fn gfx_ctx() -> Result<&'static GfxContext, JsErrorBox> {
    GFX_CTX
        .get()
        .copied()
        .ok_or_else(|| JsErrorBox::generic("Graphics context not initialized"))
}

// ======================= JS-visible Resources =======================


// Global storage for decoded images (shared across all isolates/workers)
static DECODED_IMAGE_STORE: OnceLock<Mutex<HashMap<u32, GfxDecodedImage>>> = OnceLock::new();
static DECODED_IMAGE_COUNTER: AtomicU32 = AtomicU32::new(1);

fn get_decoded_image_store() -> &'static Mutex<HashMap<u32, GfxDecodedImage>> {
    DECODED_IMAGE_STORE.get_or_init(|| Mutex::new(HashMap::new()))
}

#[derive(Clone)]
pub struct GfxDecodedImage {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>, // RGBA8
}

pub struct GfxShader {
    pub module: wgpu::ShaderModule,
}
impl Resource for GfxShader {}

pub struct GfxBuffer {
    pub buffer: Arc<wgpu::Buffer>,
}
impl Resource for GfxBuffer {}

pub struct GfxTexture {
    pub texture: wgpu::Texture,
}
impl Resource for GfxTexture {}

pub struct GfxTextureView {
    pub view: Arc<wgpu::TextureView>,
    pub width: u32,
    pub height: u32,
}
impl Resource for GfxTextureView {}

pub struct GfxSampler {
    pub sampler: Arc<wgpu::Sampler>,
}
impl Resource for GfxSampler {}

pub struct GfxBindGroupLayout {
    pub layout: Arc<wgpu::BindGroupLayout>,
}
impl Resource for GfxBindGroupLayout {}

pub struct GfxBindGroup {
    pub group: Arc<wgpu::BindGroup>,
}
impl Resource for GfxBindGroup {}

pub struct GfxPipelineLayout {
    pub layout: wgpu::PipelineLayout,
}
impl Resource for GfxPipelineLayout {}

pub struct GfxPipeline {
    pub pipeline: wgpu::RenderPipeline,
}
impl Resource for GfxPipeline {}

pub struct GfxComputePipeline {
    pub pipeline: wgpu::ComputePipeline,
}
impl Resource for GfxComputePipeline {}

pub struct GfxQuerySet {
    pub query_set: wgpu::QuerySet,
    pub count: u32,
}
impl Resource for GfxQuerySet {}

// ======================= GPU Resource ID Registry =======================
// Monotonically increasing IDs prevent stale-reference bugs caused by
// Deno's ResourceTable recycling integer IDs after drop.  JS only sees
// the monotonic ID; the internal ResourceTable ID is an implementation detail.

pub struct GpuIdMap {
    next_id: u32,
    map: HashMap<u32, u32>, // monotonic_id -> resource_table_id
}

impl GpuIdMap {
    pub fn new() -> Self {
        Self {
            next_id: 1,
            map: HashMap::new(),
        }
    }

    fn insert(&mut self, resource_table_rid: ResourceId) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        self.map.insert(id, u32::from(resource_table_rid));
        id
    }

    fn resolve(&self, monotonic_id: u32) -> Option<ResourceId> {
        self.map.get(&monotonic_id).map(|&rid| ResourceId::from(rid))
    }

    fn remove(&mut self, monotonic_id: u32) -> Option<ResourceId> {
        self.map.remove(&monotonic_id).map(|rid| ResourceId::from(rid))
    }
}

pub fn gpu_add<T: Resource>(state: &mut OpState, resource: T) -> ResourceId {
    let internal_rid = state.resource_table.add(resource);
    let monotonic_id = state.borrow_mut::<GpuIdMap>().insert(internal_rid);
    ResourceId::from(monotonic_id)
}

fn gpu_get<T: Resource>(state: &OpState, rid: u32) -> Result<std::rc::Rc<T>, JsErrorBox> {
    let internal_rid = state.borrow::<GpuIdMap>()
        .resolve(rid)
        .ok_or_else(|| JsErrorBox::generic(format!("GPU resource {} not found (stale or invalid ID)", rid)))?;
    state.resource_table.get::<T>(internal_rid)
        .map_err(|e| JsErrorBox::generic(e.to_string()))
}

fn gpu_take<T: Resource>(state: &mut OpState, rid: u32) {
    if let Some(internal_rid) = state.borrow_mut::<GpuIdMap>().remove(rid) {
        let _ = state.resource_table.take::<T>(internal_rid);
    }
}

// ======================= Mapping helpers =======================

fn bytes_per_pixel(format: wgpu::TextureFormat) -> Option<u32> {
    use wgpu::TextureFormat::*;
    match format {
        // 1 byte per pixel
        R8Unorm | R8Snorm | R8Uint | R8Sint => Some(1),
        
        // 2 bytes per pixel
        R16Uint | R16Sint | R16Unorm | R16Snorm | R16Float |
        Rg8Unorm | Rg8Snorm | Rg8Uint | Rg8Sint => Some(2),
        
        // 4 bytes per pixel
        R32Uint | R32Sint | R32Float |
        Rg16Uint | Rg16Sint | Rg16Unorm | Rg16Snorm | Rg16Float |
        Rgba8Unorm | Rgba8UnormSrgb | Rgba8Snorm | Rgba8Uint | Rgba8Sint |
        Bgra8Unorm | Bgra8UnormSrgb => Some(4),
        
        // 8 bytes per pixel
        Rg32Uint | Rg32Sint | Rg32Float |
        Rgba16Uint | Rgba16Sint | Rgba16Unorm | Rgba16Snorm | Rgba16Float => Some(8),
        
        // 16 bytes per pixel
        Rgba32Uint | Rgba32Sint | Rgba32Float => Some(16),
        
        // Depth formats - typically can't be written to directly via write_texture
        Depth16Unorm => Some(2),
        Depth24Plus | Depth24PlusStencil8 => None, // Can't write directly
        Depth32Float => Some(4),
        Depth32FloatStencil8 => None, // Can't write directly
        
        // Stencil only
        Stencil8 => Some(1),
        
        // Compressed formats - can't use write_texture with raw pixels
        _ => None,
    }
}

/// Get the texture format from a GfxTexture resource
fn get_texture_format(texture: &wgpu::Texture) -> wgpu::TextureFormat {
    texture.format()
}

fn map_shader_visibility(mask: u32) -> wgpu::ShaderStages {
    wgpu::ShaderStages::from_bits_truncate(mask)
}

fn map_texture_dimension(dim: &str) -> wgpu::TextureDimension {
    match dim {
        "1d" => wgpu::TextureDimension::D1,
        "3d" => wgpu::TextureDimension::D3,
        _ => wgpu::TextureDimension::D2,
    }
}

fn map_view_dimension(dim: Option<&str>) -> wgpu::TextureViewDimension {
    match dim {
        Some("1d") => wgpu::TextureViewDimension::D1,
        Some("2d") => wgpu::TextureViewDimension::D2,
        Some("2d-array") => wgpu::TextureViewDimension::D2Array,
        Some("cube") => wgpu::TextureViewDimension::Cube,
        Some("cube-array") => wgpu::TextureViewDimension::CubeArray,
        Some("3d") => wgpu::TextureViewDimension::D3,
        _ => wgpu::TextureViewDimension::D2,
    }
}

fn map_texture_sample_type(t: Option<&str>) -> wgpu::TextureSampleType {
    match t {
        Some("unfilterable-float") => wgpu::TextureSampleType::Float { filterable: false },
        Some("depth") => wgpu::TextureSampleType::Depth,
        Some("sint") => wgpu::TextureSampleType::Sint,
        Some("uint") => wgpu::TextureSampleType::Uint,
        _ => wgpu::TextureSampleType::Float { filterable: true },
    }
}

fn map_storage_access(a: Option<&str>) -> wgpu::StorageTextureAccess {
    match a {
        Some("read-only") => wgpu::StorageTextureAccess::ReadOnly,
        Some("read-write") => wgpu::StorageTextureAccess::ReadWrite,
        _ => wgpu::StorageTextureAccess::WriteOnly,
    }
}

fn map_sampler_type(t: Option<&str>) -> wgpu::SamplerBindingType {
    match t {
        Some("comparison") => wgpu::SamplerBindingType::Comparison,
        Some("non-filtering") => wgpu::SamplerBindingType::NonFiltering,
        _ => wgpu::SamplerBindingType::Filtering,
    }
}

fn map_buffer_binding_type(t: Option<&str>) -> wgpu::BufferBindingType {
    match t {
        Some("storage") => wgpu::BufferBindingType::Storage { read_only: false },
        Some("read-only-storage") => wgpu::BufferBindingType::Storage { read_only: true },
        _ => wgpu::BufferBindingType::Uniform,
    }
}

pub fn map_texture_format(fmt: &str) -> wgpu::TextureFormat {
    match fmt {
        "r8unorm" => wgpu::TextureFormat::R8Unorm,
        "r8snorm" => wgpu::TextureFormat::R8Snorm,
        "r8uint" => wgpu::TextureFormat::R8Uint,
        "r8sint" => wgpu::TextureFormat::R8Sint,
        "rgba8unorm" => wgpu::TextureFormat::Rgba8Unorm,
        "rgba8unorm-srgb" => wgpu::TextureFormat::Rgba8UnormSrgb,
        "bgra8unorm" => wgpu::TextureFormat::Bgra8Unorm,
        "bgra8unorm-srgb" => wgpu::TextureFormat::Bgra8UnormSrgb,
        "rgba16float" => wgpu::TextureFormat::Rgba16Float,
        "rgba32float" => wgpu::TextureFormat::Rgba32Float,
        "r16float" => wgpu::TextureFormat::R16Float,
        "r32float" => wgpu::TextureFormat::R32Float,
        "r32uint" => wgpu::TextureFormat::R32Uint,
        "r32sint" => wgpu::TextureFormat::R32Sint,
        "rg32uint" => wgpu::TextureFormat::Rg32Uint,
        "rg32sint" => wgpu::TextureFormat::Rg32Sint,
        "rg32float" => wgpu::TextureFormat::Rg32Float,
        "rgba32uint" => wgpu::TextureFormat::Rgba32Uint,
        "rgba32sint" => wgpu::TextureFormat::Rgba32Sint,
        "rgba8uint" => wgpu::TextureFormat::Rgba8Uint,
        "rgba8sint" => wgpu::TextureFormat::Rgba8Sint,
        "rgba16uint" => wgpu::TextureFormat::Rgba16Uint,
        "rgba16sint" => wgpu::TextureFormat::Rgba16Sint,
        "depth16unorm" => wgpu::TextureFormat::Depth16Unorm,
        "depth24plus" => wgpu::TextureFormat::Depth24Plus,
        "depth24plus-stencil8" => wgpu::TextureFormat::Depth24PlusStencil8,
        "depth32float" => wgpu::TextureFormat::Depth32Float,
        _ => wgpu::TextureFormat::Rgba8Unorm,
    }
}

fn map_filter_mode(s: Option<&str>) -> wgpu::FilterMode {
    match s {
        Some("nearest") => wgpu::FilterMode::Nearest,
        _ => wgpu::FilterMode::Linear,
    }
}

fn map_address_mode(s: Option<&str>) -> wgpu::AddressMode {
    match s {
        Some("repeat") => wgpu::AddressMode::Repeat,
        Some("mirror-repeat") => wgpu::AddressMode::MirrorRepeat,
        Some("clamp-to-edge") => wgpu::AddressMode::ClampToEdge,
        _ => wgpu::AddressMode::ClampToEdge,
    }
}

fn map_vertex_format(fmt: &str) -> Result<wgpu::VertexFormat, JsErrorBox> {
    use wgpu::VertexFormat::*;
    let v = match fmt {
        "float32" => Float32,
        "float32x2" => Float32x2,
        "float32x3" => Float32x3,
        "float32x4" => Float32x4,
        "float16x2" => Float16x2,
        "float16x4" => Float16x4,
        "uint32" => Uint32,
        "uint32x2" => Uint32x2,
        "uint32x3" => Uint32x3,
        "uint32x4" => Uint32x4,
        "sint32" => Sint32,
        "sint32x2" => Sint32x2,
        "sint32x3" => Sint32x3,
        "sint32x4" => Sint32x4,
        "uint8x2" => Uint8x2,
        "uint8x4" => Uint8x4,
        "sint8x2" => Sint8x2,
        "sint8x4" => Sint8x4,
        "unorm8x2" => Unorm8x2,
        "unorm8x4" => Unorm8x4,
        "snorm8x2" => Snorm8x2,
        "snorm8x4" => Snorm8x4,
        "uint16x2" => Uint16x2,
        "uint16x4" => Uint16x4,
        "sint16x2" => Sint16x2,
        "sint16x4" => Sint16x4,
        "unorm16x2" => Unorm16x2,
        "unorm16x4" => Unorm16x4,
        "snorm16x2" => Snorm16x2,
        "snorm16x4" => Snorm16x4,
        _ => {
            return Err(JsErrorBox::generic(format!(
                "Unsupported vertex format in WebGPU pipeline: {fmt}"
            )))
        }
    };
    Ok(v)
}

fn map_step_mode(mode: Option<&str>) -> wgpu::VertexStepMode {
    match mode {
        Some("instance") => wgpu::VertexStepMode::Instance,
        _ => wgpu::VertexStepMode::Vertex,
    }
}

fn map_front_face(s: Option<&str>) -> wgpu::FrontFace {
    match s {
        Some("cw") => wgpu::FrontFace::Cw,
        _ => wgpu::FrontFace::Ccw,
    }
}

fn map_cull_mode(s: Option<&str>) -> Option<wgpu::Face> {
    match s {
        Some("front") => Some(wgpu::Face::Front),
        Some("back") => Some(wgpu::Face::Back),
        Some("none") | None => None,
        _ => None,
    }
}

fn map_compare_function(s: Option<&str>) -> wgpu::CompareFunction {
    use wgpu::CompareFunction::*;
    match s {
        Some("never") => Never,
        Some("less") => Less,
        Some("equal") => Equal,
        Some("less-equal") => LessEqual,
        Some("greater") => Greater,
        Some("not-equal") => NotEqual,
        Some("greater-equal") => GreaterEqual,
        Some("always") => Always,
        _ => Less,
    }
}

fn map_stencil_operation(s: Option<&str>) -> wgpu::StencilOperation {
    match s {
        Some("keep") => wgpu::StencilOperation::Keep,
        Some("zero") => wgpu::StencilOperation::Zero,
        Some("replace") => wgpu::StencilOperation::Replace,
        Some("invert") => wgpu::StencilOperation::Invert,
        Some("increment-clamp") => wgpu::StencilOperation::IncrementClamp,
        Some("decrement-clamp") => wgpu::StencilOperation::DecrementClamp,
        Some("increment-wrap") => wgpu::StencilOperation::IncrementWrap,
        Some("decrement-wrap") => wgpu::StencilOperation::DecrementWrap,
        _ => wgpu::StencilOperation::Keep,
    }
}

// ======================= WGSL Sanitization =======================

fn sanitize_wgsl(code: &str) -> String {
    let mut result: String = code
        .lines()
        .filter(|line| {
            let t = line.trim_start();
            !t.starts_with("diagnostic(")
        })
        .collect::<Vec<_>>()
        .join("\n");


    let re_vecu_i32 = regex::Regex::new(r"(\d+)i(\s*[,\)])").unwrap();
    
    // Only apply within vec*u context - more targeted approach
    let re_vec2u = regex::Regex::new(r"vec2u\([^)]+\)").unwrap();
    result = re_vec2u.replace_all(&result, |caps: &regex::Captures| {
        let matched = &caps[0];
        re_vecu_i32.replace_all(matched, "${1}u${2}").to_string()
    }).to_string();
    
    let re_vec3u = regex::Regex::new(r"vec3u\([^)]+\)").unwrap();
    result = re_vec3u.replace_all(&result, |caps: &regex::Captures| {
        let matched = &caps[0];
        re_vecu_i32.replace_all(matched, "${1}u${2}").to_string()
    }).to_string();
    
    let re_vec4u = regex::Regex::new(r"vec4u\([^)]+\)").unwrap();
    result = re_vec4u.replace_all(&result, |caps: &regex::Captures| {
        let matched = &caps[0];
        re_vecu_i32.replace_all(matched, "${1}u${2}").to_string()
    }).to_string();

    result
}

/// Replace `fn linear_to_srgb` with a pass-through version for XR mode.
/// XR compositors expect linear color output and handle gamma correction themselves,
/// so we skip the sRGB conversion to prevent double-correction.
fn replace_linear_to_srgb_for_xr(code: &str) -> String {
    const FN_NAME: &str = "fn linear_to_srgb";
    const PASSTHROUGH: &str = "fn linear_to_srgb(c: vec4f) -> vec4f { return c; }";

    let mut result = code.to_string();
    let mut search_start = 0;

    while let Some(fn_index) = result[search_start..].find(FN_NAME) {
        let fn_index = search_start + fn_index;

        // Find the opening brace
        if let Some(brace_offset) = result[fn_index..].find('{') {
            let brace_start = fn_index + brace_offset;

            // Match braces to find the end of the function
            let mut depth = 1;
            let mut i = brace_start + 1;
            let bytes = result.as_bytes();

            while i < bytes.len() && depth > 0 {
                match bytes[i] {
                    b'{' => depth += 1,
                    b'}' => depth -= 1,
                    _ => {}
                }
                i += 1;
            }

            if depth == 0 {
                // Replace the entire function with pass-through
                result = format!("{}{}{}", &result[..fn_index], PASSTHROUGH, &result[i..]);
                search_start = fn_index + PASSTHROUGH.len();
            } else {
                break;
            }
        } else {
            break;
        }
    }

    result
}

// ======================= Image Decoding =======================

#[derive(Deserialize)]
pub struct DecodeImageOptions {
    pub resize_width: Option<u32>,
    pub resize_height: Option<u32>,
    pub resize_quality: Option<String>,  // "low", "medium", "high", "pixelated"
    pub image_orientation: Option<String>, // "none", "flipY"
}


/// Decode WebP using libwebp (SIMD-accelerated, with decode-time scaling).
fn decode_webp_native(
    data: &[u8],
    options: Option<&DecodeImageOptions>,
) -> Result<(u32, u32, Vec<u8>), Box<dyn std::error::Error + Send + Sync>> {
    use libwebp_sys::*;
    use std::os::raw::c_int;

    unsafe {
        let mut config: WebPDecoderConfig = std::mem::zeroed();
        if !WebPInitDecoderConfig(&mut config) {
            return Err("Failed to init WebP decoder config".into());
        }

        // Get image dimensions
        if WebPGetFeatures(data.as_ptr(), data.len(), &mut config.input) != VP8StatusCode::VP8_STATUS_OK {
            return Err("Failed to read WebP features".into());
        }

        let orig_w = config.input.width as u32;
        let orig_h = config.input.height as u32;

        // Configure decode-time scaling if resize requested
        if let Some(opts) = options {
            if let (Some(w), Some(h)) = (opts.resize_width, opts.resize_height) {
                if w > 0 && h > 0 && (w != orig_w || h != orig_h) {
                    config.options.use_scaling = 1;
                    config.options.scaled_width = w as c_int;
                    config.options.scaled_height = h as c_int;
                }
            }
            if opts.image_orientation.as_deref() == Some("flipY") {
                config.options.flip = 1;
            }
        }

        // Request RGBA output
        config.output.colorspace = WEBP_CSP_MODE::MODE_RGBA;

        // Decode
        let status = WebPDecode(data.as_ptr(), data.len(), &mut config);
        if status != VP8StatusCode::VP8_STATUS_OK {
            WebPFreeDecBuffer(&mut config.output);
            return Err(format!("WebP decode failed: {:?}", status).into());
        }

        let width = if config.options.use_scaling != 0 {
            config.options.scaled_width as u32
        } else {
            orig_w
        };
        let height = if config.options.use_scaling != 0 {
            config.options.scaled_height as u32
        } else {
            orig_h
        };

        // Copy pixels out of libwebp's buffer into our own Vec
        let rgba = &config.output.u.RGBA;
        let stride = rgba.stride as usize;
        let row_bytes = (width as usize) * 4;
        let mut pixels = Vec::with_capacity((width * height * 4) as usize);

        for y in 0..height as usize {
            let row_start = y * stride;
            let row = std::slice::from_raw_parts(rgba.rgba.add(row_start), row_bytes);
            pixels.extend_from_slice(row);
        }

        WebPFreeDecBuffer(&mut config.output);

        Ok((width, height, pixels))
    }
}

fn decode_image_internal(
    data: &[u8],
    format: &str,
    options: Option<DecodeImageOptions>,
) -> Result<(u32, u32, Vec<u8>), Box<dyn std::error::Error + Send + Sync>> {
    // Use libwebp for WebP (SIMD-accelerated, decode-time scaling)
    if format == "webp" {
        return decode_webp_native(data, options.as_ref());
    }
    // Also try libwebp for auto-detect if data starts with RIFF..WEBP
    if format == "auto" && data.len() >= 12 && &data[0..4] == b"RIFF" && &data[8..12] == b"WEBP" {
        return decode_webp_native(data, options.as_ref());
    }

    use image::{ImageFormat, imageops::FilterType};

    let mut img: DynamicImage = match format {
        "png" => image::load_from_memory_with_format(data, ImageFormat::Png)?,
        "jpeg" | "jpg" => image::load_from_memory_with_format(data, ImageFormat::Jpeg)?,
        _ => image::load_from_memory(data)?,
    };

    if let Some(opts) = options {
        // Handle flipY
        if opts.image_orientation.as_deref() == Some("flipY") {
            img = img.flipv();
        }

        // Handle resize
        if let (Some(w), Some(h)) = (opts.resize_width, opts.resize_height) {
            if w > 0 && h > 0 {
                let filter = match opts.resize_quality.as_deref() {
                    Some("pixelated") | Some("low") => FilterType::Nearest,
                    Some("medium") => FilterType::Triangle,
                    Some("high") => FilterType::Lanczos3,
                    _ => FilterType::Lanczos3, // default to high quality
                };
                img = img.resize_exact(w, h, filter);
            }
        }
    }

    let (width, height) = img.dimensions();
    let rgba = img.to_rgba8();
    let pixels = rgba.into_raw();

    Ok((width, height, pixels))
}

// ======================= Serde Types =======================

#[derive(Deserialize)]
pub struct MultiDrawIndexedIndirectArgs {
    pub depth_view_rid: Option<u32>,
    pub depth_clear_value: Option<f32>,
    pub target_view_rid: Option<u32>,
    pub clear_color: Option<ClearColor>,
    pub pipeline_rid: u32,
    pub vertex_buffer_rids: Vec<u32>,
    pub bind_group_rids: Vec<Option<u32>>,
    pub index_buffer_rid: u32,
    pub index_format: String,
    pub indirect_buffer_rid: u32,
    pub indirect_offset: u64,
    pub count: u32,
}

#[derive(Serialize)]
pub struct DecodeImageStoreResult {
    pub rid: ResourceId,
    pub width: u32,
    pub height: u32,
}

#[derive(Deserialize)]
pub struct SurfaceConfig {
    pub format: String,
    pub width: u32,
    pub height: u32,
    pub alphaMode: Option<String>,
}

#[derive(Deserialize)]
pub struct ShaderCreate {
    pub code: String,
    pub label: Option<String>,
}

#[derive(Serialize)]
pub struct JsGfxShader {
    pub rid: ResourceId,
}

#[derive(Deserialize)]
pub struct BufferCreate {
    pub size: u64,
    pub usage: u32,
    pub label: Option<String>,
}

#[derive(Serialize)]
pub struct JsGfxBuffer {
    pub rid: ResourceId,
}

#[derive(Deserialize)]
pub struct TextureSizeDesc {
    pub width: u32,
    pub height: u32,
    pub depth_or_array_layers: u32,
}

#[derive(Deserialize)]
pub struct TextureCreateDesc {
    pub label: Option<String>,
    pub size: TextureSizeDesc,
    pub mip_level_count: u32,
    pub sample_count: u32,
    pub dimension: String,
    pub format: String,
    pub usage: u32,
    #[serde(default)]
    pub view_formats: Vec<String>,
}

#[derive(Serialize)]
pub struct JsGfxTexture {
    pub rid: ResourceId,
}

#[derive(Deserialize)]
pub struct TextureViewCreateDesc {
    pub texture_rid: u32,
    pub label: Option<String>,
    pub format: Option<String>,
    pub dimension: Option<String>,
    pub usage: Option<u32>,
    pub base_mip_level: Option<u32>,
    pub mip_level_count: Option<u32>,
    pub base_array_layer: Option<u32>,
    pub array_layer_count: Option<u32>,
}

#[derive(Serialize)]
pub struct JsGfxTextureView {
    pub rid: ResourceId,
}

#[derive(Deserialize)]
pub struct SamplerCreateDesc {
    pub label: Option<String>,
    pub mag_filter: Option<String>,
    pub min_filter: Option<String>,
    pub mipmap_filter: Option<String>,
    pub address_mode_u: Option<String>,
    pub address_mode_v: Option<String>,
    pub address_mode_w: Option<String>,
    pub compare: Option<String>,
}


#[derive(Serialize)]
pub struct JsGfxSampler {
    pub rid: ResourceId,
}

#[derive(Deserialize)]
pub struct BufferLayoutDesc {
    #[serde(rename = "type")]
    pub r#type: Option<String>,
    pub hasDynamicOffset: Option<bool>,
    pub minBindingSize: Option<u64>,
}

#[derive(Deserialize)]
pub struct SamplerLayoutDesc {
    #[serde(rename = "type")]
    pub r#type: Option<String>,
}

#[derive(Deserialize)]
pub struct TextureLayoutDesc {
    pub sampleType: Option<String>,
    pub viewDimension: Option<String>,
    pub multisampled: Option<bool>,
}

#[derive(Deserialize)]
pub struct StorageTextureLayoutDesc {
    pub access: Option<String>,
    pub format: Option<String>,
    pub viewDimension: Option<String>,
}

#[derive(Deserialize)]
pub struct BindGroupLayoutEntryDesc {
    pub binding: u32,
    pub visibility: u32,
    pub buffer: Option<BufferLayoutDesc>,
    pub sampler: Option<SamplerLayoutDesc>,
    pub texture: Option<TextureLayoutDesc>,
    pub storageTexture: Option<StorageTextureLayoutDesc>,
}

#[derive(Deserialize)]
pub struct BindGroupLayoutCreate {
    pub label: Option<String>,
    pub entries: Vec<BindGroupLayoutEntryDesc>,
}

#[derive(Serialize)]
pub struct JsGfxBindGroupLayout {
    pub rid: ResourceId,
}

#[derive(Deserialize)]
pub struct PipelineLayoutCreate {
    pub label: Option<String>,
    pub layout_rids: Vec<u32>,
}

#[derive(Serialize)]
pub struct JsGfxPipelineLayout {
    pub rid: ResourceId,
}

#[derive(Deserialize)]
pub struct BindGroupEntryDesc {
    pub binding: u32,
    pub buffer_rid: Option<u32>,
    pub offset: u64,
    pub size: Option<u64>,
    pub sampler_rid: Option<u32>,
    pub texture_view_rid: Option<u32>,
}

#[derive(Deserialize)]
pub struct BindGroupCreate {
    pub label: Option<String>,
    pub layout_rid: u32,
    pub entries: Vec<BindGroupEntryDesc>,
}

#[derive(Serialize)]
pub struct JsGfxBindGroup {
    pub rid: ResourceId,
}

#[derive(Deserialize)]
pub struct VertexAttributeDesc {
    pub format: String,
    pub offset: u64,
    pub shader_location: u32,
}

#[derive(Deserialize)]
pub struct VertexBufferLayoutDesc {
    pub array_stride: u64,
    pub step_mode: Option<String>,
    pub attributes: Vec<VertexAttributeDesc>,
}

#[derive(Deserialize)]
pub struct PipelineCreate {
    pub vertex_module_rid: u32,
    pub vertex_entry: String,
    pub fragment_module_rid: Option<u32>,
    pub fragment_entry: Option<String>,
    pub vertex_buffers: Vec<VertexBufferLayoutDesc>,
    pub pipeline_layout_rid: Option<u32>,
    pub primitive_topology: Option<String>,
    pub primitive_front_face: Option<String>,
    pub primitive_cull_mode: Option<String>,
    pub depth_format: Option<String>,
    pub depth_write_enabled: Option<bool>,
    pub depth_compare: Option<String>,
    pub depth_bias: Option<i32>,
    pub depth_bias_slope_scale: Option<f32>,
    pub depth_bias_clamp: Option<f32>,
    pub stencil_front_compare: Option<String>,
    pub stencil_front_pass_op: Option<String>,
    pub stencil_front_fail_op: Option<String>,
    pub stencil_front_depth_fail_op: Option<String>,
    pub stencil_back_compare: Option<String>,
    pub stencil_back_pass_op: Option<String>,
    pub stencil_back_fail_op: Option<String>,
    pub stencil_back_depth_fail_op: Option<String>,
    pub stencil_read_mask: Option<u32>,
    pub stencil_write_mask: Option<u32>,
    pub color_format: Option<String>,
    pub color_write_mask: Option<u32>,
    pub sample_count: Option<u32>,
    pub alpha_to_coverage_enabled: Option<bool>,
}

#[derive(Serialize)]
pub struct JsGfxPipeline {
    pub rid: ResourceId,
}

#[derive(Deserialize)]
pub struct ComputePipelineCreate {
    pub shader_module_rid: u32,
    pub entry_point: String,
    pub pipeline_layout_rid: Option<u32>,
}

#[derive(Serialize)]
pub struct JsGfxComputePipeline {
    pub rid: ResourceId,
}

#[derive(Deserialize)]
pub struct ComputeDispatchArgs {
    pub pipeline_rid: u32,
    pub bind_group_rids: Vec<Option<u32>>,
    pub workgroup_count_x: u32,
    pub workgroup_count_y: u32,
    pub workgroup_count_z: u32,
}

#[derive(Deserialize)]
pub struct ComputeDispatchIndirectArgs {
    pub pipeline_rid: u32,
    pub bind_group_rids: Vec<Option<u32>>,
    pub indirect_buffer_rid: u32,
    pub indirect_offset: u64,
}

#[derive(Deserialize)]
pub struct TimestampWritesDesc {
    pub query_set_rid: u32,
    pub beginning_of_pass_write_index: Option<u32>,
    pub end_of_pass_write_index: Option<u32>,
}

#[derive(Deserialize)]
pub struct BatchedComputeCommand {
    pub cmd: String, // "clear_buffer", "dispatch", "dispatch_indirect", "copy_buffer_to_texture", "copy_texture_to_texture"
    // clear_buffer fields
    pub buffer_rid: Option<u32>,
    pub offset: Option<u64>,
    pub size: Option<u64>,
    // dispatch fields
    pub pipeline_rid: Option<u32>,
    pub bind_group_rids: Option<Vec<Option<u32>>>,
    pub workgroup_count_x: Option<u32>,
    pub workgroup_count_y: Option<u32>,
    pub workgroup_count_z: Option<u32>,
    // timestamp writes for compute pass
    pub timestamp_writes: Option<TimestampWritesDesc>,
    // dispatch_indirect fields
    pub indirect_buffer_rid: Option<u32>,
    pub indirect_offset: Option<u64>,
    // copy_buffer_to_texture fields
    pub src_buffer_rid: Option<u32>,
    pub buffer_offset: Option<u64>,
    pub bytes_per_row: Option<u32>,
    pub rows_per_image: Option<u32>,
    pub texture_rid: Option<u32>,
    pub mip_level: Option<u32>,
    pub origin_x: Option<u32>,
    pub origin_y: Option<u32>,
    pub origin_z: Option<u32>,
    pub width: Option<u32>,
    pub height: Option<u32>,
    pub depth_or_array_layers: Option<u32>,
    // copy_texture_to_texture fields (src)
    pub src_texture_rid: Option<u32>,
    pub src_mip_level: Option<u32>,
    pub src_origin_x: Option<u32>,
    pub src_origin_y: Option<u32>,
    pub src_origin_z: Option<u32>,
    // copy_texture_to_texture fields (dst uses texture_rid, mip_level, origin_x/y/z above)
}

#[derive(Deserialize)]
pub struct BatchedComputeArgs {
    pub commands: Vec<BatchedComputeCommand>,
}

#[derive(Deserialize)]
pub struct ClearColor {
    pub r: f64,
    pub g: f64,
    pub b: f64,
    pub a: f64,
}

#[derive(Deserialize)]
pub struct DrawCall {
    pub pipeline_rid: u32,
    pub vertex_buffer_rids: Vec<u32>,
    pub bind_group_rids: Vec<Option<u32>>,
    pub clear_color: Option<ClearColor>,
    pub vertex_count: u32,
    pub instance_count: u32,
    pub first_vertex: u32,
    pub first_instance: u32,
    pub index_buffer_rid: Option<u32>,
    pub index_format: Option<String>,
    pub index_count: u32,
    pub first_index: u32,
    pub base_vertex: i32,
    pub depth_view_rid: Option<u32>,
    pub depth_clear_value: Option<f32>,
    pub depth_load_op: Option<String>,
    pub depth_store_op: Option<String>,
    pub depth_read_only: Option<bool>,
    pub stencil_clear_value: Option<u32>,
    pub stencil_load_op: Option<String>,
    pub stencil_store_op: Option<String>,
    pub stencil_read_only: Option<bool>,
    pub stencil_reference: Option<u32>,
    // Multi draw
    pub is_multi_draw: Option<bool>,
    pub indirect_buffer_rid: Option<u32>,
    pub indirect_offset: Option<u64>,
    pub draw_count: Option<u32>,
    // Indirect indexed
    pub is_indirect: Option<bool>,
    // Scissor rect [x, y, width, height]
    pub scissor_rect: Option<[u32; 4]>,
}

#[derive(Deserialize)]
pub struct SurfaceDrawArgs {
    pub draw_calls: Vec<DrawCall>,
    // Compute commands to execute BEFORE render pass (in same command buffer)
    pub compute_commands: Option<Vec<BatchedComputeCommand>>,
    // Timestamp support for frame timing - written at START and END of all work
    pub timestamp_writes: Option<TimestampWritesDesc>,
    // Operations to perform after render pass (resolve queries, copy buffers)
    pub resolve_query_sets: Option<Vec<ResolveQuerySetDesc>>,
    pub copy_buffers: Option<Vec<CopyBufferDesc>>,
}

#[derive(Deserialize)]
pub struct ResolveQuerySetDesc {
    pub query_set_rid: u32,
    pub first_query: u32,
    pub query_count: u32,
    pub destination_rid: u32,
    pub destination_offset: u64,
}

#[derive(Deserialize)]
pub struct CopyBufferDesc {
    pub src_rid: u32,
    pub src_offset: u64,
    pub dst_rid: u32,
    pub dst_offset: u64,
    pub size: u64,
}

#[derive(Serialize)]
pub struct DecodeImageResult {
    pub width: u32,
    pub height: u32,
    pub data: Vec<u8>,
    pub error: Option<String>,
}

#[derive(Deserialize)]
pub struct GetBindGroupLayoutArgs {
    pub pipeline_rid: u32,
    pub index: u32,
}

#[derive(Deserialize)]
pub struct RenderToTextureArgs {
    pub target_view_rid: u32,
    pub draw_calls: Vec<DrawCall>,
}

#[derive(Deserialize)]
pub struct CopyTextureToTextureArgs {
    pub src_texture_rid: u32,
    pub src_mip_level: u32,
    pub src_origin_x: u32,
    pub src_origin_y: u32,
    pub src_origin_z: u32,
    pub dst_texture_rid: u32,
    pub dst_mip_level: u32,
    pub dst_origin_x: u32,
    pub dst_origin_y: u32,
    pub dst_origin_z: u32,
    pub width: u32,
    pub height: u32,
    pub depth_or_array_layers: u32,
}

#[derive(Deserialize)]
pub struct CopyBufferToTextureArgs {
    pub buffer_rid: u32,
    pub buffer_offset: u64,
    pub bytes_per_row: u32,
    pub rows_per_image: u32,
    pub texture_rid: u32,
    pub mip_level: u32,
    pub origin_x: u32,
    pub origin_y: u32,
    pub origin_z: u32,
    pub width: u32,
    pub height: u32,
    pub depth_or_array_layers: u32,
}

// ======================= OPS =======================
static MSAA_STATE: OnceLock<RwLock<Option<MsaaState>>> = OnceLock::new();

fn get_msaa_state() -> &'static RwLock<Option<MsaaState>> {
    MSAA_STATE.get_or_init(|| RwLock::new(None))
}

const SAMPLE_COUNT: u32 = 4;

// ======================= Command Buffer Batching =======================
// Accumulates command buffers for batched submission to reduce CPU overhead
static PENDING_COMMAND_BUFFERS: OnceLock<Mutex<Vec<wgpu::CommandBuffer>>> = OnceLock::new();

fn get_pending_commands() -> &'static Mutex<Vec<wgpu::CommandBuffer>> {
    PENDING_COMMAND_BUFFERS.get_or_init(|| Mutex::new(Vec::new()))
}

fn queue_command_buffer(cmd: wgpu::CommandBuffer) {
    get_pending_commands().lock().unwrap().push(cmd);
}

fn flush_pending_commands(queue: &wgpu::Queue) {
    let mut pending = get_pending_commands().lock().unwrap();
    if !pending.is_empty() {
        queue.submit(pending.drain(..));
    }
}

/// Public function to flush pending command buffers - called from XR module before frame end
pub fn flush_pending_command_buffers() -> Result<(), JsErrorBox> {
    let ctx = gfx_ctx()?;
    flush_pending_commands(&ctx.queue);
    Ok(())
}

fn ensure_msaa_texture(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
    width: u32,
    height: u32,
) {
    let state_lock = get_msaa_state();

    // Check if we need to recreate
    {
        let state = state_lock.read().unwrap();
        if let Some(ref s) = *state {
            if s.width == width && s.height == height {
                return;
            }
        }
    }

    // Need to create/recreate
    // TRANSIENT hint tells tile-based GPUs this texture never needs system memory backing
    // since we discard it after resolve. Saves memory bandwidth on Quest 3.
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("MSAA Texture"),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: SAMPLE_COUNT,
        dimension: wgpu::TextureDimension::D2,
        format,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TRANSIENT,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor::default());

    {
        let mut state = state_lock.write().unwrap();
        *state = Some(MsaaState {
            texture,
            view,
            width,
            height,
        });
    }
}

#[op2]
pub fn op_gfx_multi_draw_indexed_indirect(
    state: &mut OpState,
    #[serde] args: MultiDrawIndexedIndirectArgs,
) -> Result<(), JsErrorBox> {
    let ctx = gfx_ctx()?;

    let pipeline = gpu_get::<GfxPipeline>(state, args.pipeline_rid)?;

    let mut vbufs: Vec<Arc<wgpu::Buffer>> = Vec::new();
    for rid in args.vertex_buffer_rids.iter() {
        let buf = gpu_get::<GfxBuffer>(state, *rid)?;
        vbufs.push(buf.buffer.clone());
    }

    let mut bgs: Vec<Option<Arc<wgpu::BindGroup>>> = Vec::new();
    for rid_opt in args.bind_group_rids.iter() {
        if let Some(rid) = rid_opt {
            let bg = gpu_get::<GfxBindGroup>(state, *rid)?;
            bgs.push(Some(bg.group.clone()));
        } else {
            bgs.push(None);
        }
    }

    let index_buf = gpu_get::<GfxBuffer>(state, args.index_buffer_rid)?;

    let indirect_buf = gpu_get::<GfxBuffer>(state, args.indirect_buffer_rid)?;

    let depth_view = if let Some(rid) = args.depth_view_rid {
        Some(
            gpu_get::<GfxTextureView>(state, rid)?,
        )
    } else {
        None
    };

    let target_view = if let Some(rid) = args.target_view_rid {
        Some(
            gpu_get::<GfxTextureView>(state, rid)?,
        )
    } else {
        None
    };

    let index_format = match args.index_format.as_str() {
        "uint16" => wgpu::IndexFormat::Uint16,
        _ => wgpu::IndexFormat::Uint32,
    };

    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("MultiDrawIndexedIndirect Encoder"),
    });

    {
        let color_attachment = if let Some(ref tv) = target_view {
            Some(wgpu::RenderPassColorAttachment {
                view: &tv.view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: if let Some(ref c) = args.clear_color {
                        wgpu::LoadOp::Clear(wgpu::Color { r: c.r, g: c.g, b: c.b, a: c.a })
                    } else {
                        wgpu::LoadOp::Load
                    },
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })
        } else {
            None
        };

        let depth_attachment = depth_view.as_ref().map(|dv| {
            wgpu::RenderPassDepthStencilAttachment {
                view: &dv.view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(args.depth_clear_value.unwrap_or(1.0)),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }
        });

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("MultiDrawIndexedIndirect Pass"),
            color_attachments: &[color_attachment],
            depth_stencil_attachment: depth_attachment,
            occlusion_query_set: None,
            timestamp_writes: None,
            multiview_mask: None, 
        });

        pass.set_pipeline(&pipeline.pipeline);

        for (slot, buf) in vbufs.iter().enumerate() {
            pass.set_vertex_buffer(slot as u32, buf.slice(..));
        }

        for (idx, bg_opt) in bgs.iter().enumerate() {
            if let Some(bg) = bg_opt {
                pass.set_bind_group(idx as u32, bg.as_ref(), &[]);
            }
        }

        pass.set_index_buffer(index_buf.buffer.slice(..), index_format);
        pass.multi_draw_indexed_indirect(&indirect_buf.buffer, args.indirect_offset, args.count);
    }

    // Queue for batched submission instead of immediate submit
    queue_command_buffer(encoder.finish());
    Ok(())
}

#[op2(fast)]
pub fn op_gfx_queue_submit_empty() -> Result<(), JsErrorBox> {
    let ctx = gfx_ctx()?;
    // Flush any pending command buffers so JS queue.submit() properly synchronizes
    // all prior GPU work (render-to-texture, compute, mipmap gen, etc.)
    flush_pending_commands(&ctx.queue);
    Ok(())
}


#[derive(Deserialize)]
pub struct DecodeImageStoreArgs {
    pub format: String,
    pub resize_width: Option<u32>,
    pub resize_height: Option<u32>,
    pub resize_quality: Option<String>,
    pub image_orientation: Option<String>,
}


#[op2]
#[serde]
pub async fn op_gfx_decode_image_store(
    #[buffer] data: JsBuffer,
    #[serde] args: DecodeImageStoreArgs,
) -> Result<DecodeImageStoreResult, JsErrorBox> {
    let owned_data: Vec<u8> = data.to_vec();

    let result = tokio::task::spawn_blocking(move || {
        let options = Some(DecodeImageOptions {
            resize_width: args.resize_width,
            resize_height: args.resize_height,
            resize_quality: args.resize_quality,
            image_orientation: args.image_orientation,
        });

        let (width, height, pixels) = decode_image_internal(&owned_data, &args.format, options)
            .map_err(|e| e.to_string())?;

        let img = GfxDecodedImage { width, height, pixels };
        let rid = DECODED_IMAGE_COUNTER.fetch_add(1, Ordering::SeqCst);
        {
            let mut store = get_decoded_image_store()
                .lock()
                .map_err(|e| format!("Failed to lock image store: {}", e))?;
            store.insert(rid, img);
        }

        Ok::<_, String>((rid, width, height))
    })
    .await
    .map_err(|e| JsErrorBox::generic(format!("Decode task panicked: {}", e)))?
    .map_err(|e| JsErrorBox::generic(e))?;

    Ok(DecodeImageStoreResult { rid: result.0.into(), width: result.1, height: result.2 })
}

#[derive(Deserialize)]
pub struct ResizeDecodedImageArgs {
    pub source_rid: u32,
    pub width: u32,
    pub height: u32,
    pub quality: Option<String>,
}

#[op2]
#[serde]
pub fn op_gfx_resize_decoded_image(
    _state: &mut OpState,
    #[serde] args: ResizeDecodedImageArgs,
) -> Result<DecodeImageStoreResult, JsErrorBox> {
    use image::{RgbaImage, imageops::FilterType};

    let (src_w, src_h, src_pixels) = {
        let store = get_decoded_image_store()
            .lock()
            .map_err(|e| JsErrorBox::generic(format!("Failed to lock image store: {}", e)))?;
        let img = store.get(&args.source_rid)
            .ok_or_else(|| JsErrorBox::generic(format!("Decoded image {} not found", args.source_rid)))?;
        (img.width, img.height, img.pixels.clone())
    };

    let src_img = RgbaImage::from_raw(src_w, src_h, src_pixels)
        .ok_or_else(|| JsErrorBox::generic("Failed to create image from stored pixels"))?;

    let filter = match args.quality.as_deref() {
        Some("pixelated") | Some("low") => FilterType::Nearest,
        Some("medium") => FilterType::Triangle,
        _ => FilterType::Lanczos3,
    };

    let resized = image::imageops::resize(&src_img, args.width, args.height, filter);
    let (w, h) = resized.dimensions();
    let pixels = resized.into_raw();

    let rid = DECODED_IMAGE_COUNTER.fetch_add(1, Ordering::SeqCst);
    {
        let mut store = get_decoded_image_store()
            .lock()
            .map_err(|e| JsErrorBox::generic(format!("Failed to lock image store: {}", e)))?;
        store.insert(rid, GfxDecodedImage { width: w, height: h, pixels });
    }

    Ok(DecodeImageStoreResult { rid: rid.into(), width: w, height: h })
}

static MIPMAP_RESOURCES: OnceLock<MipmapResources> = OnceLock::new();

struct MipmapResources {
    shader: wgpu::ShaderModule,
    sampler: wgpu::Sampler,
    bind_group_layout: wgpu::BindGroupLayout,
    pipeline_layout: wgpu::PipelineLayout,
    pipelines: std::sync::Mutex<std::collections::HashMap<wgpu::TextureFormat, wgpu::RenderPipeline>>,
}

const MIPMAP_SHADER: &str = r#"
struct VOut { 
    @builtin(position) pos: vec4f, 
    @location(0) uv: vec2f 
}

@vertex fn vs(@builtin(vertex_index) i: u32) -> VOut {
    var pos = array<vec2f, 3>(vec2f(-1,-1), vec2f(3,-1), vec2f(-1,3));
    var uv = array<vec2f, 3>(vec2f(0,1), vec2f(2,1), vec2f(0,-1));
    var out: VOut;
    out.pos = vec4f(pos[i], 0, 1);
    out.uv = uv[i];
    return out;
}

@group(0) @binding(0) var srcTex: texture_2d<f32>;
@group(0) @binding(1) var samp: sampler;

@fragment fn fs(v: VOut) -> @location(0) vec4f {
    return textureSample(srcTex, samp, v.uv);
}
"#;

fn get_mipmap_resources(device: &wgpu::Device) -> &'static MipmapResources {
    MIPMAP_RESOURCES.get_or_init(|| {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Mipmap Shader"),
            source: wgpu::ShaderSource::Wgsl(MIPMAP_SHADER.into()),
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            min_filter: wgpu::FilterMode::Linear,
            mag_filter: wgpu::FilterMode::Linear,
            ..Default::default()
        });
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Mipmap Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Mipmap Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            ..Default::default()
        });
        MipmapResources {
            shader,
            sampler,
            bind_group_layout,
            pipeline_layout,
            pipelines: std::sync::Mutex::new(std::collections::HashMap::new()),
        }
    })
}

fn get_mipmap_pipeline(device: &wgpu::Device, resources: &'static MipmapResources, format: wgpu::TextureFormat) -> &'static wgpu::RenderPipeline {
    let mut pipelines = resources.pipelines.lock().unwrap();
    if !pipelines.contains_key(&format) {
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Mipmap Pipeline"),
            layout: Some(&resources.pipeline_layout),
            vertex: wgpu::VertexState {
                module: &resources.shader,
                entry_point: Some("vs"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            fragment: Some(wgpu::FragmentState {
                module: &resources.shader,
                entry_point: Some("fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            primitive: wgpu::PrimitiveState::default(),
            depth_stencil: None,
            multisample: wgpu::MultisampleState::default(),
            multiview_mask: None,
            cache: None,
        });
        pipelines.insert(format, pipeline);
    }
    // SAFETY: We're returning a reference to a pipeline that lives in a static OnceLock.
    // The HashMap is only ever inserted into, never removed from, so the reference is stable.
    unsafe { &*(pipelines.get(&format).unwrap() as *const _) }
}

pub fn generate_mipmaps(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    width: u32,
    height: u32,
    layer: u32,
) {
    let resources = get_mipmap_resources(device);
    let format = texture.format();
    let pipeline = get_mipmap_pipeline(device, resources, format);

    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Mipmap Encoder"),
    });

    let mut mip_width = width;
    let mut mip_height = height;
    let mut mip_level = 0u32;

    while mip_width > 1 || mip_height > 1 {
        let src_view = texture.create_view(&wgpu::TextureViewDescriptor {
            base_mip_level: mip_level,
            mip_level_count: Some(1),
            base_array_layer: layer,
            array_layer_count: Some(1),
            dimension: Some(wgpu::TextureViewDimension::D2),
            ..Default::default()
        });

        mip_width = (mip_width / 2).max(1);
        mip_height = (mip_height / 2).max(1);
        mip_level += 1;

        let dst_view = texture.create_view(&wgpu::TextureViewDescriptor {
            base_mip_level: mip_level,
            mip_level_count: Some(1),
            base_array_layer: layer,
            array_layer_count: Some(1),
            dimension: Some(wgpu::TextureViewDimension::D2),
            ..Default::default()
        });

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Mipmap Bind Group"),
            layout: &resources.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(&src_view),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Sampler(&resources.sampler),
                },
            ],
        });

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("Mipmap Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &dst_view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: None,
            ..Default::default()
        });

        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.draw(0..3, 0..1);
    }

    // Deferred submission: encoder.finish() produces a CommandBuffer that holds
    // Arc references to all recorded resources (views, bind groups), so they
    // survive even after local variables are dropped. Batching avoids per-image
    // GPU sync points which serializes texture loading.
    queue_command_buffer(encoder.finish());
}

#[op2(fast)]
pub fn op_gfx_upload_decoded_image_to_texture(
    state: &mut OpState,
    image_rid: u32,
    texture_rid: u32,
    mip_level: u32,
    origin_x: u32,
    origin_y: u32,
    origin_z: u32,  
) -> Result<(), JsErrorBox> {
    let ctx = gfx_ctx()?;

    // Borrow image from GLOBAL storage (kept alive until ImageBitmap.close() calls drop)
    let img = {
        let store = get_decoded_image_store()
            .lock()
            .map_err(|e| JsErrorBox::generic(format!("Failed to lock image store: {}", e)))?;
        store.get(&image_rid)
            .ok_or_else(|| JsErrorBox::generic(format!("Failed to get decoded image: Bad resource ID {}", image_rid)))?
            .clone()
    };

    // Get texture from per-isolate resource table (textures ARE per-isolate, that's fine)
    let texture = gpu_get::<GfxTexture>(state, texture_rid)
        .map_err(|e| JsErrorBox::generic(format!("Failed to get texture: {}", e)))?;

    

    // Validate image dimensions
    if img.width == 0 || img.height == 0 {
        return Err(JsErrorBox::generic("upload_decoded_image: image has zero dimensions"));
    }

    let expected_size = (4 * img.width * img.height) as usize;
    if img.pixels.len() < expected_size {
        return Err(JsErrorBox::generic(format!(
            "upload_decoded_image: pixel data too small. Expected {} bytes for {}x{} RGBA8, got {} bytes",
            expected_size, img.width, img.height, img.pixels.len()
        )));
    }

    let format = texture.texture.format();
    let compatible = matches!(
        format,
        wgpu::TextureFormat::Rgba8Unorm
            | wgpu::TextureFormat::Rgba8UnormSrgb
            | wgpu::TextureFormat::Bgra8Unorm
            | wgpu::TextureFormat::Bgra8UnormSrgb
    );
    
    if !compatible {
        return Err(JsErrorBox::generic(format!(
            "upload_decoded_image: texture format {:?} is not compatible with RGBA8 image data.",
            format
        )));
    }

    let tex_size = texture.texture.size();
    let mip_width = (tex_size.width >> mip_level).max(1);
    let mip_height = (tex_size.height >> mip_level).max(1);
    
    if origin_x + img.width > mip_width || origin_y + img.height > mip_height {
        return Err(JsErrorBox::generic(format!(
            "upload_decoded_image: image ({}x{}) at origin ({}, {}) exceeds texture bounds ({}x{}) at mip level {}",
            img.width, img.height, origin_x, origin_y, mip_width, mip_height, mip_level
        )));
    }

    ctx.queue.write_texture(
        wgpu::TexelCopyTextureInfo {
            texture: &texture.texture,
            mip_level,
            origin: wgpu::Origin3d {
                x: origin_x,
                y: origin_y,
                z: origin_z,
            },
            aspect: wgpu::TextureAspect::All,
        },
        &img.pixels,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(4 * img.width),
            rows_per_image: Some(img.height),
        },
        wgpu::Extent3d {
            width: img.width,
            height: img.height,
            depth_or_array_layers: 1,
        },
    );

    // Mipmap generation is now handled by JS compute shaders (generateMipmapsCompute)
    // which avoids per-mip render pass overhead on tile-based GPUs (Quest 3).

    Ok(())
}

#[op2(fast)]
pub fn op_gfx_decoded_image_drop(
    _state: &mut OpState,
    image_rid: u32,
) -> Result<(), JsErrorBox> {
    let mut store = get_decoded_image_store()
        .lock()
        .map_err(|e| JsErrorBox::generic(format!("Failed to lock image store: {}", e)))?;
    
    store.remove(&image_rid);
    Ok(())
}


#[op2]
#[string]
pub fn op_gfx_get_preferred_surface_format() -> String {
    "bgra8unorm".to_string()
}



#[op2]
pub fn op_gfx_surface_configure(
    _state: &mut OpState,
    #[serde] _cfg: SurfaceConfig,
) -> Result<(), JsErrorBox> {
    Ok(())
}

#[op2]
#[serde]
pub fn op_gfx_device_create_shader(
    state: &mut OpState,
    #[serde] desc: ShaderCreate,
) -> Result<JsGfxShader, JsErrorBox> {
    let ctx = gfx_ctx()?;

    let sanitized = sanitize_wgsl(&desc.code);

    // In XR mode, replace linear_to_srgb with pass-through to avoid double gamma correction
    let final_code = if is_xr_active() {
        replace_linear_to_srgb_for_xr(&sanitized)
    } else {
        sanitized
    };

    let module = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: desc.label.as_deref(),
        source: wgpu::ShaderSource::Wgsl(final_code.into()),
    });

    let rid = gpu_add(state, GfxShader { module });
    Ok(JsGfxShader { rid })
}

#[op2]
#[serde]
pub fn op_gfx_device_create_buffer_init(
    state: &mut OpState,
    usage: u32,
    #[string] label: Option<String>,
    #[buffer] data: JsBuffer,
) -> Result<JsGfxBuffer, JsErrorBox> {
    let ctx = gfx_ctx()?;

    let usage = wgpu::BufferUsages::from_bits(usage)
        .ok_or_else(|| JsErrorBox::generic("invalid buffer usage flags"))?;

    let bytes: &[u8] = &data;

    let raw = ctx
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: label.as_deref(),
            contents: bytes,
            usage,
        });

    let buffer = Arc::new(raw);
    let rid = gpu_add(state, GfxBuffer { buffer });
    Ok(JsGfxBuffer { rid })
}

#[op2]
#[serde]
pub fn op_gfx_device_create_buffer(
    state: &mut OpState,
    #[serde] desc: BufferCreate,
) -> Result<JsGfxBuffer, JsErrorBox> {
    let ctx = gfx_ctx()?;
    let usage = wgpu::BufferUsages::from_bits(desc.usage)
        .ok_or_else(|| JsErrorBox::generic("invalid buffer usage flags"))?;

    let raw = ctx.device.create_buffer(&wgpu::BufferDescriptor {
        label: desc.label.as_deref(),
        size: desc.size,
        usage,
        mapped_at_creation: false,
    });

    let buffer = Arc::new(raw);
    let rid = gpu_add(state, GfxBuffer { buffer });
    Ok(JsGfxBuffer { rid })
}

#[op2]
pub fn op_gfx_queue_write_buffer(
    state: &mut OpState,
    buffer_rid: u32,
    dst_offset: u32,
    #[buffer] data: JsBuffer,
) -> Result<(), JsErrorBox> {
    let ctx = gfx_ctx()?;

    let buffer = gpu_get::<GfxBuffer>(state, buffer_rid)?;

    let bytes: &[u8] = &data;
    
    // Debug: log first 16 floats if buffer is large enough
    if bytes.len() >= 64 {
        let floats: &[f32] = unsafe {
            std::slice::from_raw_parts(bytes.as_ptr() as *const f32, 16.min(bytes.len() / 4))
        };
    }

    ctx.queue
        .write_buffer(&buffer.buffer, dst_offset as u64, bytes);

    Ok(())
}

#[derive(Deserialize)]
pub struct ClearBufferArgs {
    buffer_rid: u32,
    offset: u64,
    size: u64,
}

#[op2]
pub fn op_gfx_clear_buffer(
    state: &mut OpState,
    #[serde] args: ClearBufferArgs,
) -> Result<(), JsErrorBox> {
    let ctx = gfx_ctx()?;

    let buffer = gpu_get::<GfxBuffer>(state, args.buffer_rid)?;

    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("clear_buffer_encoder"),
    });

    encoder.clear_buffer(&buffer.buffer, args.offset, Some(args.size));

    ctx.queue.submit(std::iter::once(encoder.finish()));

    Ok(())
}

#[op2]
#[serde]
pub fn op_gfx_device_create_texture(
    state: &mut OpState,
    #[serde] desc: TextureCreateDesc,
) -> Result<JsGfxTexture, JsErrorBox> {
    let ctx = gfx_ctx()?;
    let usage = wgpu::TextureUsages::from_bits(desc.usage)
        .ok_or_else(|| JsErrorBox::generic("invalid texture usage flags"))?;

    let format = map_texture_format(&desc.format);
    let mapped_view_formats: Vec<wgpu::TextureFormat> = desc.view_formats
        .iter()
        .map(|f| map_texture_format(f))
        .collect();
    let error_guard = ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let texture = ctx.device.create_texture(&wgpu::TextureDescriptor {
        label: desc.label.as_deref(),
        size: wgpu::Extent3d {
            width: desc.size.width,
            height: desc.size.height,
            depth_or_array_layers: desc.size.depth_or_array_layers,
        },
        mip_level_count: desc.mip_level_count,
        sample_count: desc.sample_count,
        dimension: map_texture_dimension(&desc.dimension),
        format,
        usage,
        view_formats: &mapped_view_formats,
    });
    let maybe_err = pollster::block_on(error_guard.pop());
    if let Some(err) = maybe_err {
        eprintln!("[TEX-DIAG] ERROR creating texture label={:?} size={}x{}x{} format={:?} usage={:?} mips={} samples={}: {}",
            desc.label, desc.size.width, desc.size.height, desc.size.depth_or_array_layers,
            format, usage, desc.mip_level_count, desc.sample_count, err);
    }

    let rid = gpu_add(state, GfxTexture { texture });
    Ok(JsGfxTexture { rid })
}

#[op2]
#[serde]
pub fn op_gfx_texture_create_view(
    state: &mut OpState,
    #[serde] desc: TextureViewCreateDesc,
) -> Result<JsGfxTextureView, JsErrorBox> {
    let tex = gpu_get::<GfxTexture>(state, desc.texture_rid)?;

    let tex_size = tex.texture.size();
    let base_mip = desc.base_mip_level.unwrap_or(0);
    let width = (tex_size.width >> base_mip).max(1);
    let height = (tex_size.height >> base_mip).max(1);

    let view_desc = wgpu::TextureViewDescriptor {
        label: desc.label.as_deref(),
        format: desc.format.as_deref().map(map_texture_format),
        dimension: desc.dimension.as_deref().map(|d| map_view_dimension(Some(d))),
        aspect: wgpu::TextureAspect::All,
        base_mip_level: desc.base_mip_level.unwrap_or(0),
        mip_level_count: desc.mip_level_count,
        base_array_layer: desc.base_array_layer.unwrap_or(0),
        array_layer_count: desc.array_layer_count,
        usage: desc.usage
            .and_then(|u| wgpu::TextureUsages::from_bits(u)),
    };

    let view = tex.texture.create_view(&view_desc);

    let rid = gpu_add(state, GfxTextureView { 
            view: Arc::new(view),
            width,
            height,
        });
    Ok(JsGfxTextureView { rid })
}

fn map_mipmap_filter_mode(s: Option<&str>) -> wgpu::MipmapFilterMode {
    match s {
        Some("nearest") => wgpu::MipmapFilterMode::Nearest,
        _ => wgpu::MipmapFilterMode::Linear,
    }
}

#[op2]
#[serde]
pub fn op_gfx_device_create_sampler(
    state: &mut OpState,
    #[serde] desc: SamplerCreateDesc,
) -> Result<JsGfxSampler, JsErrorBox> {
    let ctx = gfx_ctx()?;

    let compare = if desc.compare.is_some() {
        Some(map_compare_function(desc.compare.as_deref()))
    } else {
        None
    };


    let sampler = ctx.device.create_sampler(&wgpu::SamplerDescriptor {
        label: desc.label.as_deref(),
        mag_filter: map_filter_mode(desc.mag_filter.as_deref()),
        min_filter: map_filter_mode(desc.min_filter.as_deref()),
        mipmap_filter: map_mipmap_filter_mode(desc.mipmap_filter.as_deref()),
        address_mode_u: map_address_mode(desc.address_mode_u.as_deref()),
        address_mode_v: map_address_mode(desc.address_mode_v.as_deref()),
        address_mode_w: map_address_mode(desc.address_mode_w.as_deref()),
        compare,  
        ..Default::default()
    });

    let rid = gpu_add(state, GfxSampler { sampler: Arc::new(sampler) });
    Ok(JsGfxSampler { rid })
}

#[op2]
#[serde]
pub fn op_gfx_device_create_bind_group_layout(
    state: &mut OpState,
    #[serde] desc: BindGroupLayoutCreate,
) -> Result<JsGfxBindGroupLayout, JsErrorBox> {
    let ctx = gfx_ctx()?;

    let mut entries: Vec<wgpu::BindGroupLayoutEntry> = Vec::new();

    for e in desc.entries.iter() {
        let mut ty_opt: Option<wgpu::BindingType> = None;

        if let Some(buf) = e.buffer.as_ref() {
            let min_size = buf
                .minBindingSize
                .and_then(std::num::NonZeroU64::new);
            ty_opt = Some(wgpu::BindingType::Buffer {
                ty: map_buffer_binding_type(buf.r#type.as_deref()),
                has_dynamic_offset: buf.hasDynamicOffset.unwrap_or(false),
                min_binding_size: min_size,
            });
        } else if let Some(s) = e.sampler.as_ref() {
            ty_opt = Some(wgpu::BindingType::Sampler(map_sampler_type(
                s.r#type.as_deref(),
            )));
        } else if let Some(t) = e.texture.as_ref() {
            ty_opt = Some(wgpu::BindingType::Texture {
                sample_type: map_texture_sample_type(t.sampleType.as_deref()),
                view_dimension: map_view_dimension(t.viewDimension.as_deref()),
                multisampled: t.multisampled.unwrap_or(false),
            });
        } else if let Some(st) = e.storageTexture.as_ref() {
            let fmt = st
                .format
                .as_ref()
                .map(|s| map_texture_format(s))
                .unwrap_or(wgpu::TextureFormat::Rgba8Unorm);
            ty_opt = Some(wgpu::BindingType::StorageTexture {
                access: map_storage_access(st.access.as_deref()),
                format: fmt,
                view_dimension: map_view_dimension(st.viewDimension.as_deref()),
            });
        }

        let ty = ty_opt.ok_or_else(|| JsErrorBox::generic("invalid bind group layout entry"))?;

        entries.push(wgpu::BindGroupLayoutEntry {
            binding: e.binding,
            visibility: map_shader_visibility(e.visibility),
            ty,
            count: None,
        });
    }

    let layout = ctx.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: desc.label.as_deref(),
        entries: &entries,
    });

    let rid = gpu_add(state, GfxBindGroupLayout { layout: Arc::new(layout) });
    Ok(JsGfxBindGroupLayout { rid })
}

#[op2]
#[serde]
pub fn op_gfx_device_create_pipeline_layout(
    state: &mut OpState,
    #[serde] desc: PipelineLayoutCreate,
) -> Result<JsGfxPipelineLayout, JsErrorBox> {
    let ctx = gfx_ctx()?;

    let mut layout_arcs: Vec<Arc<wgpu::BindGroupLayout>> = Vec::new();
    for rid in desc.layout_rids.iter() {
        let l = gpu_get::<GfxBindGroupLayout>(state, *rid)?;
        layout_arcs.push(l.layout.clone());
    }

    let layout_refs: Vec<&wgpu::BindGroupLayout> =
        layout_arcs.iter().map(|a| a.as_ref()).collect();

    let pipeline_layout =
        ctx.device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: desc.label.as_deref(),
                bind_group_layouts: &layout_refs,
                immediate_size: 0,
            });

    let rid = gpu_add(state, GfxPipelineLayout { layout: pipeline_layout });
    Ok(JsGfxPipelineLayout { rid })
}

#[op2]
#[serde]
pub fn op_gfx_device_create_bind_group(
    state: &mut OpState,
    #[serde] desc: BindGroupCreate,
) -> Result<JsGfxBindGroup, JsErrorBox> {
    let ctx = gfx_ctx()?;

    let layout = gpu_get::<GfxBindGroupLayout>(state, desc.layout_rid)?;

    let mut buffers: Vec<Arc<wgpu::Buffer>> = Vec::new();
    let mut samplers: Vec<Arc<wgpu::Sampler>> = Vec::new();
    let mut views: Vec<Arc<wgpu::TextureView>> = Vec::new();

    enum EntryKind {
        Buffer {
            idx: usize,
            offset: u64,
            size: Option<std::num::NonZeroU64>,
        },
        Sampler {
            idx: usize,
        },
        TextureView {
            idx: usize,
        },
    }

    let mut plans: Vec<(u32, EntryKind)> = Vec::new();

    for e in desc.entries.iter() {
        if let Some(buf_rid) = e.buffer_rid {
            let buf_res = gpu_get::<GfxBuffer>(state, buf_rid)?;
            let idx = buffers.len();
            buffers.push(buf_res.buffer.clone());

            let size_nz = e.size.and_then(std::num::NonZeroU64::new);

            plans.push((
                e.binding,
                EntryKind::Buffer {
                    idx,
                    offset: e.offset,
                    size: size_nz,
                },
            ));
        } else if let Some(sam_rid) = e.sampler_rid {
            let s_res = gpu_get::<GfxSampler>(state, sam_rid)?;
            let idx = samplers.len();
            samplers.push(s_res.sampler.clone());

            plans.push((e.binding, EntryKind::Sampler { idx }));
        } else if let Some(view_rid) = e.texture_view_rid {
            let v_res = gpu_get::<GfxTextureView>(state, view_rid)?;
            let idx = views.len();
            views.push(v_res.view.clone());

            plans.push((e.binding, EntryKind::TextureView { idx }));
        } else {
            return Err(JsErrorBox::generic("bind group entry missing resource"));
        }
    }

    let mut entries: Vec<wgpu::BindGroupEntry> = Vec::new();
    for (binding, kind) in plans.into_iter() {
        match kind {
            EntryKind::Buffer { idx, offset, size } => {
                entries.push(wgpu::BindGroupEntry {
                    binding,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &buffers[idx],
                        offset,
                        size,
                    }),
                });
            }
            EntryKind::Sampler { idx } => {
                entries.push(wgpu::BindGroupEntry {
                    binding,
                    resource: wgpu::BindingResource::Sampler(&*samplers[idx]),
                });
            }
            EntryKind::TextureView { idx } => {
                entries.push(wgpu::BindGroupEntry {
                    binding,
                    resource: wgpu::BindingResource::TextureView(&*views[idx]),
                });
            }
        }
    }

    // Auto-label bind groups for debugging (so we can identify which one is invalid)
    static BG_COUNTER: AtomicU32 = AtomicU32::new(0);
    let bg_id = BG_COUNTER.fetch_add(1, Ordering::Relaxed);
    let auto_label = if let Some(ref l) = desc.label {
        format!("BG#{} {}", bg_id, l)
    } else {
        format!("BG#{} layout_rid={} entries={}", bg_id, desc.layout_rid, desc.entries.len())
    };

    // Log detailed entry info for large bind groups (targeting the problematic 20-entry one)
    if desc.entries.len() >= 15 {
        eprintln!("[BG-DIAG] Creating {} with layout_rid={}", auto_label, desc.layout_rid);
        for e in desc.entries.iter() {
            if let Some(buf_rid) = e.buffer_rid {
                eprintln!("  binding={} -> buffer_rid={} offset={} size={:?}", e.binding, buf_rid, e.offset, e.size);
            } else if let Some(sam_rid) = e.sampler_rid {
                eprintln!("  binding={} -> sampler_rid={}", e.binding, sam_rid);
            } else if let Some(view_rid) = e.texture_view_rid {
                eprintln!("  binding={} -> texture_view_rid={}", e.binding, view_rid);
            } else {
                eprintln!("  binding={} -> NO RESOURCE!", e.binding);
            }
        }
    }

    let error_guard = ctx.device.push_error_scope(wgpu::ErrorFilter::Validation);
    let group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: Some(&auto_label),
        layout: &layout.layout,
        entries: &entries,
    });
    let maybe_err = pollster::block_on(error_guard.pop());
    if let Some(err) = maybe_err {
        eprintln!("[BG-DIAG] ERROR creating {}: {}", auto_label, err);
    }

    let rid = gpu_add(state, GfxBindGroup { group: Arc::new(group) });
    Ok(JsGfxBindGroup { rid })
}


#[op2]
#[serde]
pub fn op_gfx_device_create_pipeline(
    state: &mut OpState,
    #[serde] desc: PipelineCreate,
) -> Result<JsGfxPipeline, JsErrorBox> {
    let ctx = gfx_ctx()?;

    let v_mod = gpu_get::<GfxShader>(state, desc.vertex_module_rid)?;

    // Fragment shader is optional for depth-only passes
    let f_mod = if let Some(frag_rid) = desc.fragment_module_rid {
        Some(
            gpu_get::<GfxShader>(state, frag_rid)?,
        )
    } else {
        None
    };

    let mut vertex_attributes_storage: Vec<Vec<wgpu::VertexAttribute>> = Vec::new();
    let mut vertex_buffer_layouts: Vec<wgpu::VertexBufferLayout> = Vec::new();

    for vb in desc.vertex_buffers.iter() {
        let mut attrs: Vec<wgpu::VertexAttribute> = Vec::new();
        for a in vb.attributes.iter() {
            attrs.push(wgpu::VertexAttribute {
                offset: a.offset,
                shader_location: a.shader_location,
                format: map_vertex_format(&a.format)?,
            });
        }
        vertex_attributes_storage.push(attrs);
    }

    for (i, vb) in desc.vertex_buffers.iter().enumerate() {
        vertex_buffer_layouts.push(wgpu::VertexBufferLayout {
            array_stride: vb.array_stride,
            step_mode: map_step_mode(vb.step_mode.as_deref()),
            attributes: &vertex_attributes_storage[i],
        });
    }

    let topology = match desc.primitive_topology.as_deref() {
        Some("point-list") => wgpu::PrimitiveTopology::PointList,
        Some("line-list") => wgpu::PrimitiveTopology::LineList,
        Some("line-strip") => wgpu::PrimitiveTopology::LineStrip,
        Some("triangle-strip") => wgpu::PrimitiveTopology::TriangleStrip,
        _ => wgpu::PrimitiveTopology::TriangleList,
    };

    let primitive = wgpu::PrimitiveState {
        topology,
        strip_index_format: if topology == wgpu::PrimitiveTopology::TriangleStrip
            || topology == wgpu::PrimitiveTopology::LineStrip
        {
            Some(wgpu::IndexFormat::Uint32)
        } else {
            None
        },
        front_face: map_front_face(desc.primitive_front_face.as_deref()),
        cull_mode: map_cull_mode(desc.primitive_cull_mode.as_deref()),
        unclipped_depth: false,
        polygon_mode: wgpu::PolygonMode::Fill,
        conservative: false,
    };

    let depth_stencil = if let Some(fmt_str) = desc.depth_format.as_deref() {
        let tfmt = map_texture_format(fmt_str);
        let has_stencil = desc.stencil_front_compare.is_some() || desc.stencil_back_compare.is_some();
        let stencil = if has_stencil {
            wgpu::StencilState {
                front: wgpu::StencilFaceState {
                    compare: map_compare_function(desc.stencil_front_compare.as_deref()),
                    pass_op: map_stencil_operation(desc.stencil_front_pass_op.as_deref()),
                    fail_op: map_stencil_operation(desc.stencil_front_fail_op.as_deref()),
                    depth_fail_op: map_stencil_operation(desc.stencil_front_depth_fail_op.as_deref()),
                },
                back: wgpu::StencilFaceState {
                    compare: map_compare_function(desc.stencil_back_compare.as_deref()),
                    pass_op: map_stencil_operation(desc.stencil_back_pass_op.as_deref()),
                    fail_op: map_stencil_operation(desc.stencil_back_fail_op.as_deref()),
                    depth_fail_op: map_stencil_operation(desc.stencil_back_depth_fail_op.as_deref()),
                },
                read_mask: desc.stencil_read_mask.unwrap_or(0xFF),
                write_mask: desc.stencil_write_mask.unwrap_or(0xFF),
            }
        } else {
            wgpu::StencilState::default()
        };
        Some(wgpu::DepthStencilState {
            format: tfmt,
            depth_write_enabled: desc.depth_write_enabled.unwrap_or(true),
            depth_compare: map_compare_function(desc.depth_compare.as_deref()),
            stencil,
            bias: wgpu::DepthBiasState {
                constant: desc.depth_bias.unwrap_or(0),
                slope_scale: desc.depth_bias_slope_scale.unwrap_or(0.0),
                clamp: desc.depth_bias_clamp.unwrap_or(0.0),
            },
        })
    } else {
        None
    };

    // Build fragment state only if we have a fragment shader
    let fragment_state = if let Some(ref f) = f_mod {
        let color_format = desc
            .color_format
            .as_deref()
            .map(map_texture_format)
            .unwrap_or(ctx.format);

        let entry_point = desc.fragment_entry.as_deref().unwrap_or("fs_main");

        let color_write_mask = wgpu::ColorWrites::from_bits(desc.color_write_mask.unwrap_or(0xF))
            .unwrap_or(wgpu::ColorWrites::ALL);
        Some(wgpu::FragmentState {
            module: &f.module,
            entry_point: Some(entry_point),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: color_format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: color_write_mask,
            })],
        })
    } else {
        None
    };

    let multisample = wgpu::MultisampleState {
        count: desc.sample_count.unwrap_or(1),
        mask: !0,
        alpha_to_coverage_enabled: desc.alpha_to_coverage_enabled.unwrap_or(false),
    };

    let pipeline = if let Some(layout_rid) = desc.pipeline_layout_rid {
        let pl = gpu_get::<GfxPipelineLayout>(state, layout_rid)?;

        ctx.device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("GFX Pipeline"),
                layout: Some(&pl.layout),
                vertex: wgpu::VertexState {
                    module: &v_mod.module,
                    entry_point: Some(&desc.vertex_entry),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &vertex_buffer_layouts,
                },
                fragment: fragment_state,
                primitive,
                depth_stencil,
                multisample,
                multiview_mask: None,
                cache: None, 
            })
    } else {
        ctx.device
            .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("GFX Pipeline (auto layout)"),
                layout: None,
                vertex: wgpu::VertexState {
                    module: &v_mod.module,
                    entry_point: Some(&desc.vertex_entry),
                    compilation_options: wgpu::PipelineCompilationOptions::default(),
                    buffers: &vertex_buffer_layouts,
                },
                fragment: fragment_state,
                primitive,
                depth_stencil,
                multisample,
                multiview_mask: None,
                cache: None, 
            })
    };

    let rid = gpu_add(state, GfxPipeline { pipeline });
    Ok(JsGfxPipeline { rid })
}
#[op2]
pub fn op_gfx_surface_draw(
    state: &mut OpState,
    #[serde] args: SurfaceDrawArgs,
) -> Result<(), JsErrorBox> {
    let ctx = gfx_ctx()?;

    if args.draw_calls.is_empty() {
        return Ok(());
    }

    for (call_idx, dc) in args.draw_calls.iter().enumerate() {
        if gpu_get::<GfxPipeline>(state, dc.pipeline_rid)
            .is_err()
        {
            log::error!(
                "Draw call {}: invalid pipeline rid {}",
                call_idx,
                dc.pipeline_rid
            );
            return Ok(());
        }

        for (slot, rid) in dc.vertex_buffer_rids.iter().enumerate() {
            if gpu_get::<GfxBuffer>(state, *rid)
                .is_err()
            {
                log::error!(
                    "Draw call {}: invalid vertex buffer rid {} at slot {}",
                    call_idx,
                    rid,
                    slot
                );
                return Ok(());
            }
        }

        for (idx, rid_opt) in dc.bind_group_rids.iter().enumerate() {
            if let Some(rid) = rid_opt {
                if gpu_get::<GfxBindGroup>(state, *rid)
                    .is_err()
                {
                    log::error!(
                        "Draw call {}: invalid bind group rid {} at index {}",
                        call_idx,
                        rid,
                        idx
                    );
                    return Ok(());
                }
            }
        }

        if let Some(idx_rid) = dc.index_buffer_rid {
            if gpu_get::<GfxBuffer>(state, idx_rid)
                .is_err()
            {
                log::error!(
                    "Draw call {}: invalid index buffer rid {}",
                    call_idx,
                    idx_rid
                );
                return Ok(());
            }
        }

        if let Some(view_rid) = dc.depth_view_rid {
            if gpu_get::<GfxTextureView>(state, view_rid)
                .is_err()
            {
                log::error!(
                    "Draw call {}: invalid depth view rid {}",
                    call_idx,
                    view_rid
                );
                return Ok(());
            }
        }

        // Validate indirect buffer if multi-draw
        if dc.is_multi_draw.unwrap_or(false) || dc.is_indirect.unwrap_or(false) {
            if let Some(indirect_rid) = dc.indirect_buffer_rid {
                if gpu_get::<GfxBuffer>(state, indirect_rid)
                    .is_err()
                {
                    log::error!(
                        "Draw call {}: invalid indirect buffer rid {}",
                        call_idx,
                        indirect_rid
                    );
                    return Ok(());
                }
            }
        }
    }

    let surface = ctx.surface.as_ref()
        .ok_or_else(|| JsErrorBox::generic("No surface in XR mode"))?;
    let output = match surface.get_current_texture() {
        Ok(t) => t,
        Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
            log::warn!("Surface lost or outdated, skipping frame");
            return Ok(());
        }
        Err(wgpu::SurfaceError::OutOfMemory) => {
            return Err(JsErrorBox::generic("GPU out of memory"));
        }
        Err(wgpu::SurfaceError::Timeout) => {
            log::warn!("Surface timeout, skipping frame");
            return Ok(());
        }
        Err(e) => {
            log::error!("Surface error: {:?}", e);
            return Ok(());
        }
    };

    if output.texture.width() == 0 || output.texture.height() == 0 {
        log::warn!("Surface texture has zero dimensions, skipping frame");
        return Ok(());
    }

    let surface_width = output.texture.width();
    let surface_height = output.texture.height();

    // Ensure MSAA texture exists and matches surface size
    ensure_msaa_texture(&ctx.device, ctx.format, surface_width, surface_height);
    
    let msaa_state_lock = get_msaa_state();
    let msaa_state = msaa_state_lock.read().unwrap();
    let msaa_view = &msaa_state.as_ref().unwrap().view;

    // The surface texture is the resolve target
    let resolve_view = output
        .texture
        .create_view(&wgpu::TextureViewDescriptor::default());

    // Validate depth views match surface size
    for (i, dc) in args.draw_calls.iter().enumerate() {
        if let Some(view_rid) = dc.depth_view_rid {
            let v_res = gpu_get::<GfxTextureView>(state, view_rid)
                .map_err(|e| {
                    JsErrorBox::generic(format!(
                        "Draw call {}: failed to get depth view: {}",
                        i, e
                    ))
                })?;

            if v_res.width != surface_width || v_res.height != surface_height {
                log::warn!(
                    "Draw call {}: depth view size ({}x{}) doesn't match surface ({}x{}), skipping frame",
                    i, v_res.width, v_res.height, surface_width, surface_height
                );
                output.present();
                return Ok(());
            }
        }
    }

    // Pre-collect depth view, depth ops, and stencil ops from first draw call that has depth
    let mut depth_view: Option<Arc<wgpu::TextureView>> = None;
    let mut depth_ops: Option<wgpu::Operations<f32>> = None;
    let mut stencil_ops: Option<wgpu::Operations<u32>> = None;

    for dc in args.draw_calls.iter() {
        if depth_view.is_none() {
            if let Some(view_rid) = dc.depth_view_rid {
                let v_res = gpu_get::<GfxTextureView>(state, view_rid)?;
                depth_view = Some(v_res.view.clone());

                let clear = dc.depth_clear_value.unwrap_or(1.0);
                let load = match dc.depth_load_op.as_deref() {
                    Some("load") => wgpu::LoadOp::Load,
                    _ => wgpu::LoadOp::Clear(clear),
                };
                let store = match dc.depth_store_op.as_deref() {
                    Some("discard") => wgpu::StoreOp::Discard,
                    _ => wgpu::StoreOp::Store,
                };
                depth_ops = Some(wgpu::Operations { load, store });

                if dc.stencil_load_op.is_some() || dc.stencil_store_op.is_some() {
                    let s_clear = dc.stencil_clear_value.unwrap_or(0);
                    let s_load = match dc.stencil_load_op.as_deref() {
                        Some("load") => wgpu::LoadOp::Load,
                        _ => wgpu::LoadOp::Clear(s_clear),
                    };
                    let s_store = match dc.stencil_store_op.as_deref() {
                        Some("discard") => wgpu::StoreOp::Discard,
                        _ => wgpu::StoreOp::Store,
                    };
                    stencil_ops = Some(wgpu::Operations { load: s_load, store: s_store });
                }

                break;
            }
        }
    }

    // Determine color clear from first draw call
    let color_load_op = if let Some(c) = args.draw_calls.first().and_then(|dc| dc.clear_color.as_ref()) {
        wgpu::LoadOp::Clear(wgpu::Color {
            r: c.r,
            g: c.g,
            b: c.b,
            a: c.a,
        })
    } else {
        wgpu::LoadOp::Load
    };

    // Pre-collect all draw data before starting the render pass
    struct SurfaceDrawData {
        pipeline_rid: u32,
        vbufs: Vec<Arc<wgpu::Buffer>>,
        bgs: Vec<Option<Arc<wgpu::BindGroup>>>,
        index_buf: Option<Arc<wgpu::Buffer>>,
        indirect_buf: Option<Arc<wgpu::Buffer>>,
        index_format: wgpu::IndexFormat,
        is_multi_draw: bool,
        is_indirect: bool,
        indirect_offset: u64,
        draw_count: u32,
        index_count: u32,
        first_index: u32,
        base_vertex: i32,
        instance_count: u32,
        first_instance: u32,
        vertex_count: u32,
        first_vertex: u32,
        stencil_reference: u32,
    }

    let mut draw_data_list: Vec<SurfaceDrawData> = Vec::with_capacity(args.draw_calls.len());

    for (call_idx, dc) in args.draw_calls.iter().enumerate() {
        // Validate pipeline exists
        let _ = gpu_get::<GfxPipeline>(state, dc.pipeline_rid)
            .map_err(|e| {
                JsErrorBox::generic(format!(
                    "Draw call {}: invalid pipeline rid {}: {}",
                    call_idx, dc.pipeline_rid, e
                ))
            })?;

        let mut vbufs: Vec<Arc<wgpu::Buffer>> = Vec::new();
        for (slot, rid) in dc.vertex_buffer_rids.iter().enumerate() {
            let buf_res = gpu_get::<GfxBuffer>(state, *rid)
                .map_err(|e| {
                    JsErrorBox::generic(format!(
                        "Draw call {}: invalid vertex buffer at slot {}: {}",
                        call_idx, slot, e
                    ))
                })?;
            vbufs.push(buf_res.buffer.clone());
        }

        let mut bgs: Vec<Option<Arc<wgpu::BindGroup>>> = Vec::new();
        for (idx, rid_opt) in dc.bind_group_rids.iter().enumerate() {
            if let Some(rid) = rid_opt {
                let bg_res = gpu_get::<GfxBindGroup>(state, *rid)
                    .map_err(|e| {
                        JsErrorBox::generic(format!(
                            "Draw call {}: invalid bind group at index {}: {}",
                            call_idx, idx, e
                        ))
                    })?;
                bgs.push(Some(bg_res.group.clone()));
            } else {
                bgs.push(None);
            }
        }

        let index_buf: Option<Arc<wgpu::Buffer>> = if let Some(rid) = dc.index_buffer_rid {
            let buf_res = gpu_get::<GfxBuffer>(state, rid)
                .map_err(|e| {
                    JsErrorBox::generic(format!(
                        "Draw call {}: invalid index buffer: {}",
                        call_idx, e
                    ))
                })?;
            Some(buf_res.buffer.clone())
        } else {
            None
        };

        let is_multi_draw = dc.is_multi_draw.unwrap_or(false);
        let is_indirect = dc.is_indirect.unwrap_or(false);

        let indirect_buf: Option<Arc<wgpu::Buffer>> = if is_multi_draw || is_indirect {
            if let Some(rid) = dc.indirect_buffer_rid {
                let buf_res = gpu_get::<GfxBuffer>(state, rid)
                    .map_err(|e| {
                        JsErrorBox::generic(format!(
                            "Draw call {}: invalid indirect buffer: {}",
                            call_idx, e
                        ))
                    })?;
                Some(buf_res.buffer.clone())
            } else {
                None
            }
        } else {
            None
        };

        let index_format = match dc.index_format.as_deref() {
            Some("uint16") => wgpu::IndexFormat::Uint16,
            _ => wgpu::IndexFormat::Uint32,
        };

        draw_data_list.push(SurfaceDrawData {
            pipeline_rid: dc.pipeline_rid,
            vbufs,
            bgs,
            index_buf,
            indirect_buf,
            index_format,
            is_multi_draw,
            is_indirect,
            indirect_offset: dc.indirect_offset.unwrap_or(0),
            draw_count: dc.draw_count.unwrap_or(0),
            index_count: dc.index_count,
            first_index: dc.first_index,
            base_vertex: dc.base_vertex,
            instance_count: dc.instance_count,
            first_instance: dc.first_instance,
            vertex_count: dc.vertex_count,
            first_vertex: dc.first_vertex,
            stencil_reference: dc.stencil_reference.unwrap_or(0),
        });
    }

    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("GFX Encoder"),
    });

    // Get query set for timestamp writes if provided
    let qs_ref = args.timestamp_writes.as_ref().and_then(|tw| {
        gpu_get::<GfxQuerySet>(state, tw.query_set_rid).ok()
    });

    // Write START timestamp BEFORE any work (compute or render) at encoder level
    // This ensures we measure ALL GPU work including compute passes
    if let (Some(tw), Some(qs)) = (&args.timestamp_writes, &qs_ref) {
        if let Some(start_idx) = tw.beginning_of_pass_write_index {
            encoder.write_timestamp(&qs.query_set, start_idx);
        }
    }

    // Execute compute commands BEFORE render pass (all in same command buffer)
    if let Some(ref compute_cmds) = args.compute_commands {
        for cmd in compute_cmds.iter() {
            match cmd.cmd.as_str() {
                "clear_buffer" => {
                    if let Some(buf_rid) = cmd.buffer_rid {
                        if let Ok(buf) = gpu_get::<GfxBuffer>(state, buf_rid) {
                            encoder.clear_buffer(&buf.buffer, cmd.offset.unwrap_or(0), Some(cmd.size.unwrap_or(0)));
                        }
                    }
                }
                "dispatch" => {
                    if let Some(p_rid) = cmd.pipeline_rid {
                        if let Ok(pipeline) = gpu_get::<GfxComputePipeline>(state, p_rid) {
                            // Get query set if timestamp writes provided for this compute pass
                            let compute_qs_ref = cmd.timestamp_writes.as_ref().and_then(|tw| {
                                gpu_get::<GfxQuerySet>(state, tw.query_set_rid).ok()
                            });
                            let compute_ts_writes = match (&cmd.timestamp_writes, &compute_qs_ref) {
                                (Some(tw), Some(qs)) => Some(wgpu::ComputePassTimestampWrites {
                                    query_set: &qs.query_set,
                                    beginning_of_pass_write_index: tw.beginning_of_pass_write_index,
                                    end_of_pass_write_index: tw.end_of_pass_write_index,
                                }),
                                _ => None,
                            };

                            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                                label: Some("Batched Compute Pass"),
                                timestamp_writes: compute_ts_writes,
                            });
                            pass.set_pipeline(&pipeline.pipeline);
                            if let Some(ref rids) = cmd.bind_group_rids {
                                for (idx, rid_opt) in rids.iter().enumerate() {
                                    if let Some(rid) = rid_opt {
                                        if let Ok(bg) = gpu_get::<GfxBindGroup>(state, *rid) {
                                            pass.set_bind_group(idx as u32, bg.group.as_ref(), &[]);
                                        }
                                    }
                                }
                            }
                            pass.dispatch_workgroups(
                                cmd.workgroup_count_x.unwrap_or(1),
                                cmd.workgroup_count_y.unwrap_or(1),
                                cmd.workgroup_count_z.unwrap_or(1),
                            );
                        }
                    }
                }
                "dispatch_indirect" => {
                    if let (Some(p_rid), Some(ib_rid)) = (cmd.pipeline_rid, cmd.indirect_buffer_rid) {
                        if let (Ok(pipeline), Ok(indirect_buf)) = (
                            gpu_get::<GfxComputePipeline>(state, p_rid),
                            gpu_get::<GfxBuffer>(state, ib_rid)
                        ) {
                            let compute_qs_ref = cmd.timestamp_writes.as_ref().and_then(|tw| {
                                gpu_get::<GfxQuerySet>(state, tw.query_set_rid).ok()
                            });
                            let compute_ts_writes = match (&cmd.timestamp_writes, &compute_qs_ref) {
                                (Some(tw), Some(qs)) => Some(wgpu::ComputePassTimestampWrites {
                                    query_set: &qs.query_set,
                                    beginning_of_pass_write_index: tw.beginning_of_pass_write_index,
                                    end_of_pass_write_index: tw.end_of_pass_write_index,
                                }),
                                _ => None,
                            };

                            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                                label: Some("Batched Compute Pass Indirect"),
                                timestamp_writes: compute_ts_writes,
                            });
                            pass.set_pipeline(&pipeline.pipeline);
                            if let Some(ref rids) = cmd.bind_group_rids {
                                for (idx, rid_opt) in rids.iter().enumerate() {
                                    if let Some(rid) = rid_opt {
                                        if let Ok(bg) = gpu_get::<GfxBindGroup>(state, *rid) {
                                            pass.set_bind_group(idx as u32, bg.group.as_ref(), &[]);
                                        }
                                    }
                                }
                            }
                            pass.dispatch_workgroups_indirect(&indirect_buf.buffer, cmd.indirect_offset.unwrap_or(0));
                        }
                    }
                }
                "copy_buffer_to_texture" => {
                    if let (Some(buf_rid), Some(tex_rid)) = (cmd.src_buffer_rid, cmd.texture_rid) {
                        if let (Ok(buf), Ok(tex)) = (
                            gpu_get::<GfxBuffer>(state, buf_rid),
                            gpu_get::<GfxTexture>(state, tex_rid)
                        ) {
                            encoder.copy_buffer_to_texture(
                                wgpu::TexelCopyBufferInfo {
                                    buffer: &buf.buffer,
                                    layout: wgpu::TexelCopyBufferLayout {
                                        offset: cmd.buffer_offset.unwrap_or(0),
                                        bytes_per_row: Some(cmd.bytes_per_row.unwrap_or(256)),
                                        rows_per_image: Some(cmd.rows_per_image.unwrap_or(1)),
                                    },
                                },
                                wgpu::TexelCopyTextureInfo {
                                    texture: &tex.texture,
                                    mip_level: cmd.mip_level.unwrap_or(0),
                                    origin: wgpu::Origin3d {
                                        x: cmd.origin_x.unwrap_or(0),
                                        y: cmd.origin_y.unwrap_or(0),
                                        z: cmd.origin_z.unwrap_or(0),
                                    },
                                    aspect: wgpu::TextureAspect::All,
                                },
                                wgpu::Extent3d {
                                    width: cmd.width.unwrap_or(1),
                                    height: cmd.height.unwrap_or(1),
                                    depth_or_array_layers: cmd.depth_or_array_layers.unwrap_or(1),
                                },
                            );
                        }
                    }
                }
                "copy_texture_to_texture" => {
                    if let (Some(src_rid), Some(dst_rid)) = (cmd.src_texture_rid, cmd.texture_rid) {
                        if let (Ok(src_tex), Ok(dst_tex)) = (
                            gpu_get::<GfxTexture>(state, src_rid),
                            gpu_get::<GfxTexture>(state, dst_rid)
                        ) {
                            encoder.copy_texture_to_texture(
                                wgpu::TexelCopyTextureInfo {
                                    texture: &src_tex.texture,
                                    mip_level: cmd.src_mip_level.unwrap_or(0),
                                    origin: wgpu::Origin3d {
                                        x: cmd.src_origin_x.unwrap_or(0),
                                        y: cmd.src_origin_y.unwrap_or(0),
                                        z: cmd.src_origin_z.unwrap_or(0),
                                    },
                                    aspect: wgpu::TextureAspect::All,
                                },
                                wgpu::TexelCopyTextureInfo {
                                    texture: &dst_tex.texture,
                                    mip_level: cmd.mip_level.unwrap_or(0),
                                    origin: wgpu::Origin3d {
                                        x: cmd.origin_x.unwrap_or(0),
                                        y: cmd.origin_y.unwrap_or(0),
                                        z: cmd.origin_z.unwrap_or(0),
                                    },
                                    aspect: wgpu::TextureAspect::All,
                                },
                                wgpu::Extent3d {
                                    width: cmd.width.unwrap_or(1),
                                    height: cmd.height.unwrap_or(1),
                                    depth_or_array_layers: cmd.depth_or_array_layers.unwrap_or(1),
                                },
                            );
                        }
                    }
                }
                _ => {}
            }
        }
    }

    // Render pass for all draw calls
    {
        let depth_attachment = match (&depth_view, &depth_ops) {
            (Some(dv), Some(ops)) => Some(wgpu::RenderPassDepthStencilAttachment {
                view: dv.as_ref(),
                depth_ops: Some(*ops),
                stencil_ops: stencil_ops,
            }),
            _ => None,
        };

        // No pass-level timestamps - we use encoder-level timestamps to capture
        // ALL GPU work including compute passes that run before the render pass
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("GFX Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: msaa_view,
                resolve_target: Some(&resolve_view),
                ops: wgpu::Operations {
                    load: color_load_op,
                    store: wgpu::StoreOp::Discard, // MSAA not needed after resolve
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: depth_attachment,
            occlusion_query_set: None,
            timestamp_writes: None,
            multiview_mask: None,
        });

        // Issue all draw calls within the single pass
        for data in draw_data_list.iter() {
            let pipeline_res = gpu_get::<GfxPipeline>(state, data.pipeline_rid)
                .unwrap();

            pass.set_pipeline(&pipeline_res.pipeline);
            pass.set_stencil_reference(data.stencil_reference);

            for (slot, buf) in data.vbufs.iter().enumerate() {
                pass.set_vertex_buffer(slot as u32, buf.slice(..));
            }

            for (idx, bg_opt) in data.bgs.iter().enumerate() {
                if let Some(bg) = bg_opt {
                    pass.set_bind_group(idx as u32, bg.as_ref(), &[]);
                }
            }

            if data.is_multi_draw {
                if let (Some(ref idx_buf), Some(ref indirect)) = (&data.index_buf, &data.indirect_buf) {
                    pass.set_index_buffer(idx_buf.slice(..), data.index_format);
                    pass.multi_draw_indexed_indirect(indirect, data.indirect_offset, data.draw_count);
                }
            } else if data.is_indirect {
                if let (Some(ref idx_buf), Some(ref indirect)) = (&data.index_buf, &data.indirect_buf) {
                    pass.set_index_buffer(idx_buf.slice(..), data.index_format);
                    pass.draw_indexed_indirect(indirect, data.indirect_offset);
                }
            } else if let Some(ref idx_buf) = data.index_buf {
                if data.index_count > 0 {
                    pass.set_index_buffer(idx_buf.slice(..), data.index_format);
                    pass.draw_indexed(
                        data.first_index..(data.first_index + data.index_count),
                        data.base_vertex,
                        data.first_instance..(data.first_instance + data.instance_count),
                    );
                }
            } else if data.vertex_count > 0 && data.instance_count > 0 {
                pass.draw(
                    data.first_vertex..(data.first_vertex + data.vertex_count),
                    data.first_instance..(data.first_instance + data.instance_count),
                );
            }
        }
    }

    // Write END timestamp AFTER all work (compute + render) at encoder level
    if let (Some(tw), Some(qs)) = (&args.timestamp_writes, &qs_ref) {
        if let Some(end_idx) = tw.end_of_pass_write_index {
            encoder.write_timestamp(&qs.query_set, end_idx);
        }
    }

    // Resolve query sets after render pass
    if let Some(ref resolve_ops) = args.resolve_query_sets {
        for op in resolve_ops {
            if let Ok(qs) = gpu_get::<GfxQuerySet>(state, op.query_set_rid) {
                if let Ok(dst) = gpu_get::<GfxBuffer>(state, op.destination_rid) {
                    encoder.resolve_query_set(
                        &qs.query_set,
                        op.first_query..op.first_query + op.query_count,
                        &dst.buffer,
                        op.destination_offset,
                    );
                }
            }
        }
    }

    // Copy buffers after resolve
    if let Some(ref copy_ops) = args.copy_buffers {
        for op in copy_ops {
            if let Ok(src) = gpu_get::<GfxBuffer>(state, op.src_rid) {
                if let Ok(dst) = gpu_get::<GfxBuffer>(state, op.dst_rid) {
                    encoder.copy_buffer_to_buffer(
                        &src.buffer,
                        op.src_offset,
                        &dst.buffer,
                        op.dst_offset,
                        op.size,
                    );
                }
            }
        }
    }

    // Flush any pending command buffers before submitting the main frame
    flush_pending_commands(&ctx.queue);

    ctx.queue.submit(iter::once(encoder.finish()));
    output.present();

    Ok(())
}

#[op2]
#[serde]
pub fn op_gfx_decode_image(
    #[buffer] data: JsBuffer,
    #[string] format: String,
) -> DecodeImageResult {
    let slice: &[u8] = &data;
    let result = decode_image_internal(slice, &format, None);
    match result {
        Ok((width, height, pixels)) => DecodeImageResult {
            width,
            height,
            data: pixels,
            error: None,
        },
        Err(e) => DecodeImageResult {
            width: 0,
            height: 0,
            data: vec![],
            error: Some(e.to_string()),
        },
    }
}

#[op2]
pub fn op_gfx_write_texture_image(
    state: &mut OpState,
    texture_rid: u32,
    width: u32,
    height: u32,
    depth: u32,
    origin_x: u32,
    origin_y: u32,
    origin_z: u32,
    mip_level: u32,
    #[buffer] data: JsBuffer,
) -> Result<(), JsErrorBox> {
    let ctx = gfx_ctx()?;

    let texture = gpu_get::<GfxTexture>(state, texture_rid)
        .map_err(|e| JsErrorBox::generic(format!("Failed to get texture: {}", e)))?;

    // Validate dimensions
    if width == 0 || height == 0 {
        return Err(JsErrorBox::generic("write_texture: width and height must be > 0"));
    }

    let depth_layers = if depth == 0 { 1 } else { depth };

    // Get the texture format and calculate bytes per pixel
    let format = texture.texture.format();
    let bpp = bytes_per_pixel(format).ok_or_else(|| {
        JsErrorBox::generic(format!(
            "write_texture: unsupported texture format {:?} for direct pixel writes",
            format
        ))
    })?;

    // Calculate expected data size
    let bytes_per_row = bpp * width;
    let expected_size = (bytes_per_row * height * depth_layers) as usize;

    let bytes: &[u8] = &data;

    // Validate data size
    if bytes.len() < expected_size {
        return Err(JsErrorBox::generic(format!(
            "write_texture: data buffer too small. Expected at least {} bytes for {}x{}x{} {:?} texture, got {} bytes",
            expected_size, width, height, depth_layers, format, bytes.len()
        )));
    }

    // Validate that the write region fits within the texture at the given mip level
    let tex_size = texture.texture.size();
    let mip_width = (tex_size.width >> mip_level).max(1);
    let mip_height = (tex_size.height >> mip_level).max(1);

    if origin_x + width > mip_width || origin_y + height > mip_height {
        return Err(JsErrorBox::generic(format!(
            "write_texture: write region (origin: ({}, {}), size: {}x{}) exceeds texture bounds ({}x{}) at mip level {}",
            origin_x, origin_y, width, height, mip_width, mip_height, mip_level
        )));
    }

    ctx.queue.write_texture(
        wgpu::TexelCopyTextureInfo  {
            texture: &texture.texture,
            mip_level,
            origin: wgpu::Origin3d {
                x: origin_x,
                y: origin_y,
                z: origin_z,
            },
            aspect: wgpu::TextureAspect::All,
        },
        bytes,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(bytes_per_row),
            rows_per_image: Some(height),
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: depth_layers,
        },
    );

    Ok(())
}


#[op2]
#[serde]
pub fn op_gfx_pipeline_get_bind_group_layout(
    state: &mut OpState,
    #[serde] args: GetBindGroupLayoutArgs,
) -> Result<JsGfxBindGroupLayout, JsErrorBox> {
    let pipeline = gpu_get::<GfxPipeline>(state, args.pipeline_rid)?;

    let layout = pipeline.pipeline.get_bind_group_layout(args.index);

    let rid = gpu_add(state, GfxBindGroupLayout { layout: Arc::new(layout) });
    Ok(JsGfxBindGroupLayout { rid })
}

#[derive(Deserialize)]
pub struct RenderXrFrameArgs {
    pub msaa_view_rid: u32,
    pub resolve_view_rid: u32,
    pub draw_calls: Vec<DrawCall>,
    pub depth_view_rid: Option<u32>,
    pub depth_clear_value: Option<f32>,
    pub depth_store_op: Option<String>,
    pub color_store_op: Option<String>,
    pub clear_color: Option<ClearColor>,
}


// Pre-collected draw call data for single-pass rendering (avoids borrow issues)
struct XrDrawData {
    pipeline_rid: u32,
    vbufs: Vec<Arc<wgpu::Buffer>>,
    bgs: Vec<Option<Arc<wgpu::BindGroup>>>,
    index_buf: Option<Arc<wgpu::Buffer>>,
    indirect_buf: Option<Arc<wgpu::Buffer>>,
    index_format: wgpu::IndexFormat,
    is_multi_draw: bool,
    is_indirect: bool,
    indirect_offset: u64,
    draw_count: u32,
    index_count: u32,
    first_index: u32,
    base_vertex: i32,
    instance_count: u32,
    first_instance: u32,
    vertex_count: u32,
    first_vertex: u32,
    stencil_reference: u32,
}

#[op2]
pub fn op_gfx_render_xr_frame(
    state: &mut OpState,
    #[serde] args: RenderXrFrameArgs,
) -> Result<(), JsErrorBox> {
    let ctx = gfx_ctx()?;

    if args.draw_calls.is_empty() {
        return Ok(());
    }

    let msaa_view = gpu_get::<GfxTextureView>(state, args.msaa_view_rid)
        .map_err(|e| JsErrorBox::generic(format!("Failed to get MSAA view: {}", e)))?;

    let resolve_view = gpu_get::<GfxTextureView>(state, args.resolve_view_rid)
        .map_err(|e| JsErrorBox::generic(format!("Failed to get resolve view: {}", e)))?;

    let depth_view = if let Some(rid) = args.depth_view_rid {
        Some(
            gpu_get::<GfxTextureView>(state, rid)
                .map_err(|e| JsErrorBox::generic(format!("Failed to get depth view: {}", e)))?,
        )
    } else {
        None
    };

    // Pre-collect all resources before starting the render pass
    // This avoids multiple tile flushes on tile-based GPUs like Quest 3
    let mut draw_data_list: Vec<XrDrawData> = Vec::with_capacity(args.draw_calls.len());

    for dc in args.draw_calls.iter() {
        let _ = gpu_get::<GfxPipeline>(state, dc.pipeline_rid)?;

        let mut vbufs: Vec<Arc<wgpu::Buffer>> = Vec::new();
        for rid in dc.vertex_buffer_rids.iter() {
            let buf_res = gpu_get::<GfxBuffer>(state, *rid)?;
            vbufs.push(buf_res.buffer.clone());
        }

        let mut bgs: Vec<Option<Arc<wgpu::BindGroup>>> = Vec::new();
        for rid_opt in dc.bind_group_rids.iter() {
            if let Some(rid) = rid_opt {
                let bg_res = gpu_get::<GfxBindGroup>(state, *rid)?;
                bgs.push(Some(bg_res.group.clone()));
            } else {
                bgs.push(None);
            }
        }

        let index_buf: Option<Arc<wgpu::Buffer>> = if let Some(rid) = dc.index_buffer_rid {
            let buf_res = gpu_get::<GfxBuffer>(state, rid)?;
            Some(buf_res.buffer.clone())
        } else {
            None
        };

        let is_multi_draw = dc.is_multi_draw.unwrap_or(false);
        let is_indirect = dc.is_indirect.unwrap_or(false);

        let indirect_buf: Option<Arc<wgpu::Buffer>> = if is_multi_draw || is_indirect {
            if let Some(rid) = dc.indirect_buffer_rid {
                let buf_res = gpu_get::<GfxBuffer>(state, rid)?;
                Some(buf_res.buffer.clone())
            } else {
                None
            }
        } else {
            None
        };

        let index_format = match dc.index_format.as_deref() {
            Some("uint16") => wgpu::IndexFormat::Uint16,
            _ => wgpu::IndexFormat::Uint32,
        };

        draw_data_list.push(XrDrawData {
            pipeline_rid: dc.pipeline_rid,
            vbufs,
            bgs,
            index_buf,
            indirect_buf,
            index_format,
            is_multi_draw,
            is_indirect,
            indirect_offset: dc.indirect_offset.unwrap_or(0),
            draw_count: dc.draw_count.unwrap_or(0),
            index_count: dc.index_count,
            first_index: dc.first_index,
            base_vertex: dc.base_vertex,
            instance_count: dc.instance_count,
            first_instance: dc.first_instance,
            vertex_count: dc.vertex_count,
            first_vertex: dc.first_vertex,
            stencil_reference: dc.stencil_reference.unwrap_or(0),
        });
    }

    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("XR Frame Encoder"),
    });

    let clear_color = args.clear_color.as_ref().map(|c| wgpu::Color {
        r: c.r,
        g: c.g,
        b: c.b,
        a: c.a,
    });

    // Convert store ops from JS strings to wgpu types
    let depth_store = match args.depth_store_op.as_deref() {
        Some("discard") => wgpu::StoreOp::Discard,
        _ => wgpu::StoreOp::Store,
    };
    let color_store = match args.color_store_op.as_deref() {
        Some("discard") => wgpu::StoreOp::Discard,
        _ => wgpu::StoreOp::Store,
    };

    // Extract stencil ops from draw calls (matching surface draw path)
    let mut stencil_ops: Option<wgpu::Operations<u32>> = None;
    for dc in args.draw_calls.iter() {
        if dc.stencil_load_op.is_some() || dc.stencil_store_op.is_some() {
            let s_clear = dc.stencil_clear_value.unwrap_or(0);
            let s_load = match dc.stencil_load_op.as_deref() {
                Some("load") => wgpu::LoadOp::Load,
                _ => wgpu::LoadOp::Clear(s_clear),
            };
            let s_store = match dc.stencil_store_op.as_deref() {
                Some("discard") => wgpu::StoreOp::Discard,
                _ => wgpu::StoreOp::Store,
            };
            stencil_ops = Some(wgpu::Operations { load: s_load, store: s_store });
            break;
        }
    }

    // single render pass for all draw calls - good for TBRs
    {
        let depth_attachment = depth_view.as_ref().map(|dv| {
            wgpu::RenderPassDepthStencilAttachment {
                view: &dv.view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(args.depth_clear_value.unwrap_or(1.0)),
                    store: depth_store,
                }),
                stencil_ops: stencil_ops,
            }
        });

        let color_load_op = match clear_color {
            Some(c) => wgpu::LoadOp::Clear(c),
            None => wgpu::LoadOp::Load,
        };

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("XR Frame Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &msaa_view.view,
                resolve_target: Some(&resolve_view.view),
                ops: wgpu::Operations {
                    load: color_load_op,
                    store: color_store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: depth_attachment,
            occlusion_query_set: None,
            timestamp_writes: None,
            multiview_mask: None,
        });

        // Issue all draw calls within the single pass
        for data in draw_data_list.iter() {
            // Get pipeline reference
            let pipeline_res = gpu_get::<GfxPipeline>(state, data.pipeline_rid)
                .unwrap();

            pass.set_pipeline(&pipeline_res.pipeline);
            pass.set_stencil_reference(data.stencil_reference);

            for (slot, buf) in data.vbufs.iter().enumerate() {
                pass.set_vertex_buffer(slot as u32, buf.slice(..));
            }

            for (idx, bg_opt) in data.bgs.iter().enumerate() {
                if let Some(bg) = bg_opt {
                    pass.set_bind_group(idx as u32, bg.as_ref(), &[]);
                }
            }

            if data.is_multi_draw {
                if let Some(ref idx_buf) = data.index_buf {
                    pass.set_index_buffer(idx_buf.slice(..), data.index_format);

                    if let Some(ref indirect) = data.indirect_buf {
                        pass.multi_draw_indexed_indirect(
                            indirect,
                            data.indirect_offset,
                            data.draw_count,
                        );
                    }
                }
            } else if data.is_indirect {
                if let (Some(ref idx_buf), Some(ref indirect)) = (&data.index_buf, &data.indirect_buf) {
                    pass.set_index_buffer(idx_buf.slice(..), data.index_format);
                    pass.draw_indexed_indirect(indirect, data.indirect_offset);
                }
            } else if let Some(ref idx_buf) = data.index_buf {
                if data.index_count > 0 {
                    pass.set_index_buffer(idx_buf.slice(..), data.index_format);
                    pass.draw_indexed(
                        data.first_index..(data.first_index + data.index_count),
                        data.base_vertex,
                        data.first_instance..(data.first_instance + data.instance_count),
                    );
                }
            } else if data.vertex_count > 0 && data.instance_count > 0 {
                pass.draw(
                    data.first_vertex..(data.first_vertex + data.vertex_count),
                    data.first_instance..(data.first_instance + data.instance_count),
                );
            }
        }
    }

    // Queue this command buffer for deferred submission.
    // All XR eye passes + prior compute/shadow work get flushed together
    // in a single queue.submit via op_gfx_queue_submit_empty at end of frame.
    queue_command_buffer(encoder.finish());
    Ok(())
}

#[op2]
pub fn op_gfx_render_to_texture(
    state: &mut OpState,
    #[serde] args: RenderToTextureArgs,
) -> Result<(), JsErrorBox> {
    let ctx = gfx_ctx()?;

    if args.draw_calls.is_empty() {
        return Ok(());
    }

    let target_view = gpu_get::<GfxTextureView>(state, args.target_view_rid)?;

    // Pre-collect depth view, depth ops, and stencil ops from first draw call that has depth
    let mut depth_view: Option<Arc<wgpu::TextureView>> = None;
    let mut depth_ops: Option<wgpu::Operations<f32>> = None;
    let mut stencil_ops: Option<wgpu::Operations<u32>> = None;

    for dc in args.draw_calls.iter() {
        if depth_view.is_none() {
            if let Some(view_rid) = dc.depth_view_rid {
                let v_res = gpu_get::<GfxTextureView>(state, view_rid)?;
                depth_view = Some(v_res.view.clone());

                let clear = dc.depth_clear_value.unwrap_or(1.0);
                let load = match dc.depth_load_op.as_deref() {
                    Some("load") => wgpu::LoadOp::Load,
                    _ => wgpu::LoadOp::Clear(clear),
                };
                let store = match dc.depth_store_op.as_deref() {
                    Some("discard") => wgpu::StoreOp::Discard,
                    _ => wgpu::StoreOp::Store,
                };
                depth_ops = Some(wgpu::Operations { load, store });

                if dc.stencil_load_op.is_some() || dc.stencil_store_op.is_some() {
                    let s_clear = dc.stencil_clear_value.unwrap_or(0);
                    let s_load = match dc.stencil_load_op.as_deref() {
                        Some("load") => wgpu::LoadOp::Load,
                        _ => wgpu::LoadOp::Clear(s_clear),
                    };
                    let s_store = match dc.stencil_store_op.as_deref() {
                        Some("discard") => wgpu::StoreOp::Discard,
                        _ => wgpu::StoreOp::Store,
                    };
                    stencil_ops = Some(wgpu::Operations { load: s_load, store: s_store });
                }

                break;
            }
        }
    }

    // Determine color clear from first draw call
    let color_load_op = if let Some(c) = args.draw_calls.first().and_then(|dc| dc.clear_color.as_ref()) {
        wgpu::LoadOp::Clear(wgpu::Color {
            r: c.r,
            g: c.g,
            b: c.b,
            a: c.a,
        })
    } else {
        wgpu::LoadOp::Load
    };

    // Pre-collect all draw data BEFORE starting the render pass
    struct TextureDrawData {
        pipeline_rid: u32,
        vbufs: Vec<Arc<wgpu::Buffer>>,
        bgs: Vec<Option<Arc<wgpu::BindGroup>>>,
        index_buf: Option<Arc<wgpu::Buffer>>,
        index_format: wgpu::IndexFormat,
        index_count: u32,
        first_index: u32,
        base_vertex: i32,
        instance_count: u32,
        first_instance: u32,
        vertex_count: u32,
        first_vertex: u32,
        stencil_reference: u32,
        scissor_rect: Option<[u32; 4]>,
    }

    let mut draw_data_list: Vec<TextureDrawData> = Vec::with_capacity(args.draw_calls.len());

    for dc in args.draw_calls.iter() {
        // Validate pipeline exists
        let _ = gpu_get::<GfxPipeline>(state, dc.pipeline_rid)?;

        let mut vbufs: Vec<Arc<wgpu::Buffer>> = Vec::new();
        for rid in dc.vertex_buffer_rids.iter() {
            let buf_res = gpu_get::<GfxBuffer>(state, *rid)?;
            vbufs.push(buf_res.buffer.clone());
        }

        let mut bgs: Vec<Option<Arc<wgpu::BindGroup>>> = Vec::new();
        for rid_opt in dc.bind_group_rids.iter() {
            if let Some(rid) = rid_opt {
                let bg_res = gpu_get::<GfxBindGroup>(state, *rid)?;
                bgs.push(Some(bg_res.group.clone()));
            } else {
                bgs.push(None);
            }
        }

        let index_buf: Option<Arc<wgpu::Buffer>> = if let Some(rid) = dc.index_buffer_rid {
            let buf_res = gpu_get::<GfxBuffer>(state, rid)?;
            Some(buf_res.buffer.clone())
        } else {
            None
        };

        let index_format = match dc.index_format.as_deref() {
            Some("uint16") => wgpu::IndexFormat::Uint16,
            _ => wgpu::IndexFormat::Uint32,
        };

        draw_data_list.push(TextureDrawData {
            pipeline_rid: dc.pipeline_rid,
            vbufs,
            bgs,
            index_buf,
            index_format,
            index_count: dc.index_count,
            first_index: dc.first_index,
            base_vertex: dc.base_vertex,
            instance_count: dc.instance_count,
            first_instance: dc.first_instance,
            vertex_count: dc.vertex_count,
            first_vertex: dc.first_vertex,
            stencil_reference: dc.stencil_reference.unwrap_or(0),
            scissor_rect: dc.scissor_rect,
        });
    }

    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("GFX Texture Encoder"),
    });

    {
        let depth_attachment = match (&depth_view, &depth_ops) {
            (Some(dv), Some(ops)) => Some(wgpu::RenderPassDepthStencilAttachment {
                view: dv.as_ref(),
                depth_ops: Some(*ops),
                stencil_ops: stencil_ops,
            }),
            _ => None,
        };

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("GFX Texture Pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: &target_view.view,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: color_load_op,
                    store: wgpu::StoreOp::Store,
                },
                depth_slice: None,
            })],
            depth_stencil_attachment: depth_attachment,
            occlusion_query_set: None,
            timestamp_writes: None,
            multiview_mask: None,
        });

        // Issue all draw calls within the single pass
        for data in draw_data_list.iter() {
            let pipeline_res = gpu_get::<GfxPipeline>(state, data.pipeline_rid)
                .unwrap();

            pass.set_pipeline(&pipeline_res.pipeline);
            pass.set_stencil_reference(data.stencil_reference);

            if let Some([sx, sy, sw, sh]) = data.scissor_rect {
                pass.set_scissor_rect(sx, sy, sw, sh);
            }

            for (slot, buf) in data.vbufs.iter().enumerate() {
                pass.set_vertex_buffer(slot as u32, buf.slice(..));
            }

            for (idx, bg_opt) in data.bgs.iter().enumerate() {
                if let Some(bg) = bg_opt {
                    pass.set_bind_group(idx as u32, bg.as_ref(), &[]);
                }
            }

            if let Some(ref idx_buf) = data.index_buf {
                if data.index_count > 0 {
                    pass.set_index_buffer(idx_buf.slice(..), data.index_format);
                    pass.draw_indexed(
                        data.first_index..(data.first_index + data.index_count),
                        data.base_vertex,
                        data.first_instance..(data.first_instance + data.instance_count),
                    );
                }
            } else if data.vertex_count > 0 && data.instance_count > 0 {
                pass.draw(
                    data.first_vertex..(data.first_vertex + data.vertex_count),
                    data.first_instance..(data.first_instance + data.instance_count),
                );
            }
        }
    }

    // Queue for batched submission - will be flushed before main render
    queue_command_buffer(encoder.finish());

    Ok(())
}

#[op2]
pub fn op_gfx_copy_texture_to_texture(
    state: &mut OpState,
    #[serde] args: CopyTextureToTextureArgs,
) -> Result<(), JsErrorBox> {
    let ctx = gfx_ctx()?;

    let src_tex = gpu_get::<GfxTexture>(state, args.src_texture_rid)?;

    let dst_tex = gpu_get::<GfxTexture>(state, args.dst_texture_rid)?;

    let mut encoder =
        ctx.device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("GFX CopyTextureToTexture"),
            });

    encoder.copy_texture_to_texture(
        wgpu::TexelCopyTextureInfo  {
            texture: &src_tex.texture,
            mip_level: args.src_mip_level,
            origin: wgpu::Origin3d {
                x: args.src_origin_x,
                y: args.src_origin_y,
                z: args.src_origin_z,
            },
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::TexelCopyTextureInfo  {
            texture: &dst_tex.texture,
            mip_level: args.dst_mip_level,
            origin: wgpu::Origin3d {
                x: args.dst_origin_x,
                y: args.dst_origin_y,
                z: args.dst_origin_z,
            },
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::Extent3d {
            width: args.width,
            height: args.height,
            depth_or_array_layers: args.depth_or_array_layers,
        },
    );

    // Queue for batched submission instead of immediate submit
    queue_command_buffer(encoder.finish());
    Ok(())
}

#[op2]
pub fn op_gfx_copy_buffer_to_texture(
    state: &mut OpState,
    #[serde] args: CopyBufferToTextureArgs,
) -> Result<(), JsErrorBox> {
    let ctx = gfx_ctx()?;

    let buffer = gpu_get::<GfxBuffer>(state, args.buffer_rid)?;

    let texture = gpu_get::<GfxTexture>(state, args.texture_rid)?;

    let mut encoder =
        ctx.device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("GFX CopyBufferToTexture"),
            });

    encoder.copy_buffer_to_texture(
        wgpu::TexelCopyBufferInfo {
            buffer: &buffer.buffer,
            layout: wgpu::TexelCopyBufferLayout {
                offset: args.buffer_offset,
                bytes_per_row: Some(args.bytes_per_row),
                rows_per_image: Some(args.rows_per_image),
            },
        },
        wgpu::TexelCopyTextureInfo {
            texture: &texture.texture,
            mip_level: args.mip_level,
            origin: wgpu::Origin3d {
                x: args.origin_x,
                y: args.origin_y,
                z: args.origin_z,
            },
            aspect: wgpu::TextureAspect::All,
        },
        wgpu::Extent3d {
            width: args.width,
            height: args.height,
            depth_or_array_layers: args.depth_or_array_layers,
        },
    );

    // Queue for batched submission instead of immediate submit
    queue_command_buffer(encoder.finish());
    Ok(())
}


// depth / shadows passes

#[derive(Deserialize)]
pub struct RenderDepthOnlyArgs {
    pub depth_view_rid: u32,
    pub draw_calls: Vec<DrawCall>,
}

#[op2]
pub fn op_gfx_render_depth_only(
    state: &mut OpState,
    #[serde] args: RenderDepthOnlyArgs,
) -> Result<(), JsErrorBox> {
    let ctx = gfx_ctx()?;

    if args.draw_calls.is_empty() {
        return Ok(());
    }

    let depth_view = gpu_get::<GfxTextureView>(state, args.depth_view_rid)?;

    // Get depth clear value from first draw call
    let depth_clear_value = args.draw_calls.first()
        .and_then(|dc| dc.depth_clear_value)
        .unwrap_or(1.0);

    // Pre-collect all draw data BEFORE starting the render pass
    struct DepthDrawData {
        pipeline_rid: u32,
        vbufs: Vec<Arc<wgpu::Buffer>>,
        bgs: Vec<Option<Arc<wgpu::BindGroup>>>,
        index_buf: Option<Arc<wgpu::Buffer>>,
        index_format: wgpu::IndexFormat,
        index_count: u32,
        first_index: u32,
        base_vertex: i32,
        instance_count: u32,
        first_instance: u32,
        vertex_count: u32,
        first_vertex: u32,
        stencil_reference: u32,
    }

    let mut draw_data_list: Vec<DepthDrawData> = Vec::with_capacity(args.draw_calls.len());

    for dc in args.draw_calls.iter() {
        // Validate pipeline exists
        let _ = gpu_get::<GfxPipeline>(state, dc.pipeline_rid)?;

        let mut vbufs: Vec<Arc<wgpu::Buffer>> = Vec::new();
        for rid in dc.vertex_buffer_rids.iter() {
            let buf_res = gpu_get::<GfxBuffer>(state, *rid)?;
            vbufs.push(buf_res.buffer.clone());
        }

        let mut bgs: Vec<Option<Arc<wgpu::BindGroup>>> = Vec::new();
        for rid_opt in dc.bind_group_rids.iter() {
            if let Some(rid) = rid_opt {
                let bg_res = gpu_get::<GfxBindGroup>(state, *rid)?;
                bgs.push(Some(bg_res.group.clone()));
            } else {
                bgs.push(None);
            }
        }

        let index_buf: Option<Arc<wgpu::Buffer>> = if let Some(rid) = dc.index_buffer_rid {
            let buf_res = gpu_get::<GfxBuffer>(state, rid)?;
            Some(buf_res.buffer.clone())
        } else {
            None
        };

        let index_format = match dc.index_format.as_deref() {
            Some("uint16") => wgpu::IndexFormat::Uint16,
            _ => wgpu::IndexFormat::Uint32,
        };

        draw_data_list.push(DepthDrawData {
            pipeline_rid: dc.pipeline_rid,
            vbufs,
            bgs,
            index_buf,
            index_format,
            index_count: dc.index_count,
            first_index: dc.first_index,
            base_vertex: dc.base_vertex,
            instance_count: dc.instance_count,
            first_instance: dc.first_instance,
            vertex_count: dc.vertex_count,
            first_vertex: dc.first_vertex,
            stencil_reference: dc.stencil_reference.unwrap_or(0),
        });
    }

    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("GFX Depth Only Encoder"),
    });

    {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("GFX Depth Only Pass"),
            color_attachments: &[],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &depth_view.view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(depth_clear_value),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            occlusion_query_set: None,
            timestamp_writes: None,
            multiview_mask: None,
        });

        // Issue all draw calls within the single pass
        for data in draw_data_list.iter() {
            let pipeline_res = gpu_get::<GfxPipeline>(state, data.pipeline_rid)
                .unwrap();

            pass.set_pipeline(&pipeline_res.pipeline);
            pass.set_stencil_reference(data.stencil_reference);

            for (slot, buf) in data.vbufs.iter().enumerate() {
                pass.set_vertex_buffer(slot as u32, buf.slice(..));
            }

            for (idx, bg_opt) in data.bgs.iter().enumerate() {
                if let Some(bg) = bg_opt {
                    pass.set_bind_group(idx as u32, bg.as_ref(), &[]);
                }
            }

            if let Some(ref idx_buf) = data.index_buf {
                if data.index_count > 0 {
                    pass.set_index_buffer(idx_buf.slice(..), data.index_format);
                    pass.draw_indexed(
                        data.first_index..(data.first_index + data.index_count),
                        data.base_vertex,
                        data.first_instance..(data.first_instance + data.instance_count),
                    );
                }
            } else if data.vertex_count > 0 && data.instance_count > 0 {
                pass.draw(
                    data.first_vertex..(data.first_vertex + data.vertex_count),
                    data.first_instance..(data.first_instance + data.instance_count),
                );
            }
        }
    }

    // Queue for batched submission - will be flushed before main render
    queue_command_buffer(encoder.finish());
    Ok(())
}

#[op2(fast)]
pub fn op_gfx_resource_drop(
    state: &mut OpState,
    rid: u32,
) -> Result<(), JsErrorBox> {
    gpu_take::<GfxTextureView>(state, rid);
    Ok(())
}

#[op2(fast)]
pub fn op_gfx_texture_drop(
    state: &mut OpState,
    rid: u32,
) -> Result<(), JsErrorBox> {
    gpu_take::<GfxTexture>(state, rid);
    Ok(())
}

#[op2(fast)]
pub fn op_gfx_buffer_drop(
    state: &mut OpState,
    rid: u32,
) -> Result<(), JsErrorBox> {
    gpu_take::<GfxBuffer>(state, rid);
    Ok(())
}

#[op2(fast)]
pub fn op_gfx_bind_group_drop(
    state: &mut OpState,
    rid: u32,
) -> Result<(), JsErrorBox> {
    gpu_take::<GfxBindGroup>(state, rid);
    Ok(())
}

/// Flush all pending command buffers to the GPU.
/// Call this when you need to ensure all prior render commands are submitted,
/// e.g., before a buffer readback or at end of frame.
#[op2(fast)]
pub fn op_gfx_flush_commands() -> Result<(), JsErrorBox> {
    let ctx = gfx_ctx()?;
    flush_pending_commands(&ctx.queue);
    Ok(())
}

// ======================= Compute Shader Support =======================

#[op2]
#[serde]
pub fn op_gfx_device_create_compute_pipeline(
    state: &mut OpState,
    #[serde] desc: ComputePipelineCreate,
) -> Result<JsGfxComputePipeline, JsErrorBox> {
    let ctx = gfx_ctx()?;

    let shader = gpu_get::<GfxShader>(state, desc.shader_module_rid)?;

    let pipeline = if let Some(layout_rid) = desc.pipeline_layout_rid {
        let pl = gpu_get::<GfxPipelineLayout>(state, layout_rid)?;

        ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("GFX Compute Pipeline"),
            layout: Some(&pl.layout),
            module: &shader.module,
            entry_point: Some(&desc.entry_point),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        })
    } else {
        ctx.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("GFX Compute Pipeline (auto layout)"),
            layout: None,
            module: &shader.module,
            entry_point: Some(&desc.entry_point),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            cache: None,
        })
    };

    let rid = gpu_add(state, GfxComputePipeline { pipeline });
    Ok(JsGfxComputePipeline { rid })
}

#[op2]
#[serde]
pub fn op_gfx_compute_pipeline_get_bind_group_layout(
    state: &mut OpState,
    #[serde] args: GetBindGroupLayoutArgs,
) -> Result<JsGfxBindGroupLayout, JsErrorBox> {
    let pipeline = gpu_get::<GfxComputePipeline>(state, args.pipeline_rid)?;

    let layout = pipeline.pipeline.get_bind_group_layout(args.index);

    let rid = gpu_add(state, GfxBindGroupLayout { layout: Arc::new(layout) });
    Ok(JsGfxBindGroupLayout { rid })
}

#[op2]
pub fn op_gfx_compute_dispatch(
    state: &mut OpState,
    #[serde] args: ComputeDispatchArgs,
) -> Result<(), JsErrorBox> {
    let ctx = gfx_ctx()?;

    let pipeline = gpu_get::<GfxComputePipeline>(state, args.pipeline_rid)?;

    let mut bgs: Vec<Option<Arc<wgpu::BindGroup>>> = Vec::new();
    for rid_opt in args.bind_group_rids.iter() {
        if let Some(rid) = rid_opt {
            let bg = gpu_get::<GfxBindGroup>(state, *rid)?;
            bgs.push(Some(bg.group.clone()));
        } else {
            bgs.push(None);
        }
    }

    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Compute Dispatch Encoder"),
    });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Compute Pass"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&pipeline.pipeline);

        for (idx, bg_opt) in bgs.iter().enumerate() {
            if let Some(bg) = bg_opt {
                pass.set_bind_group(idx as u32, bg.as_ref(), &[]);
            }
        }

        pass.dispatch_workgroups(
            args.workgroup_count_x,
            args.workgroup_count_y,
            args.workgroup_count_z,
        );
    }

    // Queue for batched submission instead of immediate submit
    queue_command_buffer(encoder.finish());
    Ok(())
}

#[op2]
pub fn op_gfx_compute_dispatch_indirect(
    state: &mut OpState,
    #[serde] args: ComputeDispatchIndirectArgs,
) -> Result<(), JsErrorBox> {
    let ctx = gfx_ctx()?;

    let pipeline = gpu_get::<GfxComputePipeline>(state, args.pipeline_rid)?;

    let indirect_buf = gpu_get::<GfxBuffer>(state, args.indirect_buffer_rid)?;

    let mut bgs: Vec<Option<Arc<wgpu::BindGroup>>> = Vec::new();
    for rid_opt in args.bind_group_rids.iter() {
        if let Some(rid) = rid_opt {
            let bg = gpu_get::<GfxBindGroup>(state, *rid)?;
            bgs.push(Some(bg.group.clone()));
        } else {
            bgs.push(None);
        }
    }

    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Compute Dispatch Indirect Encoder"),
    });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Compute Pass Indirect"),
            timestamp_writes: None,
        });

        pass.set_pipeline(&pipeline.pipeline);

        for (idx, bg_opt) in bgs.iter().enumerate() {
            if let Some(bg) = bg_opt {
                pass.set_bind_group(idx as u32, bg.as_ref(), &[]);
            }
        }

        pass.dispatch_workgroups_indirect(&indirect_buf.buffer, args.indirect_offset);
    }

    // Queue for batched submission instead of immediate submit
    queue_command_buffer(encoder.finish());
    Ok(())
}

#[op2]
pub fn op_gfx_compute_batch(
    state: &mut OpState,
    #[serde] args: BatchedComputeArgs,
) -> Result<(), JsErrorBox> {
    let ctx = gfx_ctx()?;

    // Validate all resource IDs upfront before encoding
    for (i, cmd) in args.commands.iter().enumerate() {
        match cmd.cmd.as_str() {
            "clear_buffer" => {
                let buf_rid = cmd.buffer_rid.ok_or_else(|| JsErrorBox::generic("clear_buffer: missing buffer_rid"))?;
                gpu_get::<GfxBuffer>(state, buf_rid)
                    .map_err(|e| JsErrorBox::generic(format!("batch cmd {}: {}", i, e)))?;
            }
            "dispatch" => {
                let p_rid = cmd.pipeline_rid.ok_or_else(|| JsErrorBox::generic("dispatch: missing pipeline_rid"))?;
                gpu_get::<GfxComputePipeline>(state, p_rid)
                    .map_err(|e| JsErrorBox::generic(format!("batch cmd {}: {}", i, e)))?;
                if let Some(ref rids) = cmd.bind_group_rids {
                    for rid_opt in rids.iter() {
                        if let Some(rid) = rid_opt {
                            gpu_get::<GfxBindGroup>(state, *rid)
                                .map_err(|e| JsErrorBox::generic(format!("batch cmd {}: {}", i, e)))?;
                        }
                    }
                }
            }
            "dispatch_indirect" => {
                let p_rid = cmd.pipeline_rid.ok_or_else(|| JsErrorBox::generic("dispatch_indirect: missing pipeline_rid"))?;
                gpu_get::<GfxComputePipeline>(state, p_rid)
                    .map_err(|e| JsErrorBox::generic(format!("batch cmd {}: {}", i, e)))?;
                let ib_rid = cmd.indirect_buffer_rid.ok_or_else(|| JsErrorBox::generic("dispatch_indirect: missing indirect_buffer_rid"))?;
                gpu_get::<GfxBuffer>(state, ib_rid)
                    .map_err(|e| JsErrorBox::generic(format!("batch cmd {}: {}", i, e)))?;
                if let Some(ref rids) = cmd.bind_group_rids {
                    for rid_opt in rids.iter() {
                        if let Some(rid) = rid_opt {
                            gpu_get::<GfxBindGroup>(state, *rid)
                                .map_err(|e| JsErrorBox::generic(format!("batch cmd {}: {}", i, e)))?;
                        }
                    }
                }
            }
            "copy_buffer_to_texture" => {
                let buf_rid = cmd.src_buffer_rid.ok_or_else(|| JsErrorBox::generic("copy: missing src_buffer_rid"))?;
                gpu_get::<GfxBuffer>(state, buf_rid)
                    .map_err(|e| JsErrorBox::generic(format!("batch cmd {}: {}", i, e)))?;
                let tex_rid = cmd.texture_rid.ok_or_else(|| JsErrorBox::generic("copy: missing texture_rid"))?;
                gpu_get::<GfxTexture>(state, tex_rid)
                    .map_err(|e| JsErrorBox::generic(format!("batch cmd {}: {}", i, e)))?;
            }
            "copy_texture_to_texture" => {
                let src_rid = cmd.src_texture_rid.ok_or_else(|| JsErrorBox::generic("copy_t2t: missing src_texture_rid"))?;
                gpu_get::<GfxTexture>(state, src_rid)
                    .map_err(|e| JsErrorBox::generic(format!("batch cmd {}: {}", i, e)))?;
                let dst_rid = cmd.texture_rid.ok_or_else(|| JsErrorBox::generic("copy_t2t: missing texture_rid"))?;
                gpu_get::<GfxTexture>(state, dst_rid)
                    .map_err(|e| JsErrorBox::generic(format!("batch cmd {}: {}", i, e)))?;
            }
            other => {
                return Err(JsErrorBox::generic(format!("Unknown batch command: {}", other)));
            }
        }
    }

    // Now encode everything in a single command buffer
    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Batched Compute Encoder"),
    });

    for cmd in args.commands.iter() {
        match cmd.cmd.as_str() {
            "clear_buffer" => {
                let buf = gpu_get::<GfxBuffer>(state, cmd.buffer_rid.unwrap()).unwrap();
                encoder.clear_buffer(&buf.buffer, cmd.offset.unwrap_or(0), Some(cmd.size.unwrap()));
            }
            "dispatch" => {
                let pipeline = gpu_get::<GfxComputePipeline>(state, cmd.pipeline_rid.unwrap()).unwrap();

                // Get query set if timestamp writes provided
                let qs_ref = cmd.timestamp_writes.as_ref().and_then(|tw| {
                    gpu_get::<GfxQuerySet>(state, tw.query_set_rid).ok()
                });

                // Build timestamp writes descriptor
                let ts_writes = match (&cmd.timestamp_writes, &qs_ref) {
                    (Some(tw), Some(qs)) => Some(wgpu::ComputePassTimestampWrites {
                        query_set: &qs.query_set,
                        beginning_of_pass_write_index: tw.beginning_of_pass_write_index,
                        end_of_pass_write_index: tw.end_of_pass_write_index,
                    }),
                    _ => None,
                };

                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Batched Compute Pass"),
                    timestamp_writes: ts_writes,
                });
                pass.set_pipeline(&pipeline.pipeline);
                if let Some(ref rids) = cmd.bind_group_rids {
                    for (idx, rid_opt) in rids.iter().enumerate() {
                        if let Some(rid) = rid_opt {
                            let bg = gpu_get::<GfxBindGroup>(state, *rid).unwrap();
                            pass.set_bind_group(idx as u32, bg.group.as_ref(), &[]);
                        }
                    }
                }
                pass.dispatch_workgroups(
                    cmd.workgroup_count_x.unwrap_or(1),
                    cmd.workgroup_count_y.unwrap_or(1),
                    cmd.workgroup_count_z.unwrap_or(1),
                );
            }
            "dispatch_indirect" => {
                let pipeline = gpu_get::<GfxComputePipeline>(state, cmd.pipeline_rid.unwrap()).unwrap();
                let indirect_buf = gpu_get::<GfxBuffer>(state, cmd.indirect_buffer_rid.unwrap()).unwrap();

                // Get query set if timestamp writes provided
                let qs_ref = cmd.timestamp_writes.as_ref().and_then(|tw| {
                    gpu_get::<GfxQuerySet>(state, tw.query_set_rid).ok()
                });

                // Build timestamp writes descriptor
                let ts_writes = match (&cmd.timestamp_writes, &qs_ref) {
                    (Some(tw), Some(qs)) => Some(wgpu::ComputePassTimestampWrites {
                        query_set: &qs.query_set,
                        beginning_of_pass_write_index: tw.beginning_of_pass_write_index,
                        end_of_pass_write_index: tw.end_of_pass_write_index,
                    }),
                    _ => None,
                };

                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("Batched Compute Pass Indirect"),
                    timestamp_writes: ts_writes,
                });
                pass.set_pipeline(&pipeline.pipeline);
                if let Some(ref rids) = cmd.bind_group_rids {
                    for (idx, rid_opt) in rids.iter().enumerate() {
                        if let Some(rid) = rid_opt {
                            let bg = gpu_get::<GfxBindGroup>(state, *rid).unwrap();
                            pass.set_bind_group(idx as u32, bg.group.as_ref(), &[]);
                        }
                    }
                }
                pass.dispatch_workgroups_indirect(&indirect_buf.buffer, cmd.indirect_offset.unwrap_or(0));
            }
            "copy_buffer_to_texture" => {
                let buf = gpu_get::<GfxBuffer>(state, cmd.src_buffer_rid.unwrap()).unwrap();
                let tex = gpu_get::<GfxTexture>(state, cmd.texture_rid.unwrap()).unwrap();
                encoder.copy_buffer_to_texture(
                    wgpu::TexelCopyBufferInfo {
                        buffer: &buf.buffer,
                        layout: wgpu::TexelCopyBufferLayout {
                            offset: cmd.buffer_offset.unwrap_or(0),
                            bytes_per_row: Some(cmd.bytes_per_row.unwrap_or(256)),
                            rows_per_image: Some(cmd.rows_per_image.unwrap_or(1)),
                        },
                    },
                    wgpu::TexelCopyTextureInfo {
                        texture: &tex.texture,
                        mip_level: cmd.mip_level.unwrap_or(0),
                        origin: wgpu::Origin3d {
                            x: cmd.origin_x.unwrap_or(0),
                            y: cmd.origin_y.unwrap_or(0),
                            z: cmd.origin_z.unwrap_or(0),
                        },
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::Extent3d {
                        width: cmd.width.unwrap_or(1),
                        height: cmd.height.unwrap_or(1),
                        depth_or_array_layers: cmd.depth_or_array_layers.unwrap_or(1),
                    },
                );
            }
            "copy_texture_to_texture" => {
                let src_tex = gpu_get::<GfxTexture>(state, cmd.src_texture_rid.unwrap()).unwrap();
                let dst_tex = gpu_get::<GfxTexture>(state, cmd.texture_rid.unwrap()).unwrap();
                encoder.copy_texture_to_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &src_tex.texture,
                        mip_level: cmd.src_mip_level.unwrap_or(0),
                        origin: wgpu::Origin3d {
                            x: cmd.src_origin_x.unwrap_or(0),
                            y: cmd.src_origin_y.unwrap_or(0),
                            z: cmd.src_origin_z.unwrap_or(0),
                        },
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::TexelCopyTextureInfo {
                        texture: &dst_tex.texture,
                        mip_level: cmd.mip_level.unwrap_or(0),
                        origin: wgpu::Origin3d {
                            x: cmd.origin_x.unwrap_or(0),
                            y: cmd.origin_y.unwrap_or(0),
                            z: cmd.origin_z.unwrap_or(0),
                        },
                        aspect: wgpu::TextureAspect::All,
                    },
                    wgpu::Extent3d {
                        width: cmd.width.unwrap_or(1),
                        height: cmd.height.unwrap_or(1),
                        depth_or_array_layers: cmd.depth_or_array_layers.unwrap_or(1),
                    },
                );
            }
            _ => {} // already validated above
        }
    }

    // Queue for batched submission instead of immediate submit
    queue_command_buffer(encoder.finish());
    Ok(())
}

// ======================= GPUTimer / Timestamp Query Support =======================

#[derive(Deserialize)]
pub struct QuerySetCreateDesc {
    pub query_type: String, // "timestamp" or "occlusion"
    pub count: u32,
}

#[derive(Serialize)]
pub struct JsGfxQuerySet {
    pub rid: ResourceId,
}

#[op2]
#[serde]
pub fn op_gfx_device_create_query_set(
    state: &mut OpState,
    #[serde] desc: QuerySetCreateDesc,
) -> Result<JsGfxQuerySet, JsErrorBox> {
    let ctx = gfx_ctx()?;

    let query_type = match desc.query_type.as_str() {
        "timestamp" => wgpu::QueryType::Timestamp,
        "occlusion" => wgpu::QueryType::Occlusion,
        _ => return Err(JsErrorBox::generic(format!("Unknown query type: {}", desc.query_type))),
    };

    let query_set = ctx.device.create_query_set(&wgpu::QuerySetDescriptor {
        label: Some("GPUTimer QuerySet"),
        ty: query_type,
        count: desc.count,
    });

    let rid = gpu_add(state, GfxQuerySet {
        query_set,
        count: desc.count,
    });

    Ok(JsGfxQuerySet { rid })
}

#[op2(fast)]
pub fn op_gfx_query_set_destroy(
    state: &mut OpState,
    #[smi] query_set_rid: u32,
) -> Result<(), JsErrorBox> {
    gpu_take::<GfxQuerySet>(state, query_set_rid);
    Ok(())
}

/// Check if timestamp-query feature is available
#[op2(fast)]
pub fn op_gfx_has_timestamp_query() -> Result<bool, JsErrorBox> {
    let ctx = gfx_ctx()?;
    Ok(ctx.device.features().contains(wgpu::Features::TIMESTAMP_QUERY))
}

/// Get the timestamp period in nanoseconds per tick.
/// Timestamps from query sets must be multiplied by this value to get nanoseconds.
#[op2(fast)]
pub fn op_gfx_get_timestamp_period() -> Result<f32, JsErrorBox> {
    let ctx = gfx_ctx()?;
    Ok(ctx.queue.get_timestamp_period())
}

// ======================= Timestamp Command Recording =======================
// These operations build a list of timestamp commands that get executed together

#[derive(Deserialize)]
pub struct TimestampWriteDesc {
    pub query_set_rid: u32,
    pub query_index: u32,
}

#[derive(Deserialize)]
pub struct TimestampBatchArgs {
    pub write_timestamps: Vec<TimestampWriteDesc>,
    pub resolve_query_sets: Vec<ResolveQuerySetDesc>,
    pub copy_buffers: Vec<CopyBufferDesc>,
}

/// Execute a batch of timestamp operations (writeTimestamp, resolveQuerySet, copyBufferToBuffer)
/// This is designed to be called at the end of each frame to collect timing data.
#[op2]
pub fn op_gfx_timestamp_batch(
    state: &mut OpState,
    #[serde] args: TimestampBatchArgs,
) -> Result<(), JsErrorBox> {
    let ctx = gfx_ctx()?;

    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Timestamp Batch Encoder"),
    });

    // Write timestamps
    for ts in &args.write_timestamps {
        let qs = gpu_get::<GfxQuerySet>(state, ts.query_set_rid)?;
        encoder.write_timestamp(&qs.query_set, ts.query_index);
    }

    // Resolve query sets to buffers
    for rq in &args.resolve_query_sets {
        let qs = gpu_get::<GfxQuerySet>(state, rq.query_set_rid)?;
        let dest_buf = gpu_get::<GfxBuffer>(state, rq.destination_rid)?;
        encoder.resolve_query_set(
            &qs.query_set,
            rq.first_query..rq.first_query + rq.query_count,
            &dest_buf.buffer,
            rq.destination_offset,
        );
    }

    // Copy buffers
    for cb in &args.copy_buffers {
        let src_buf = gpu_get::<GfxBuffer>(state, cb.src_rid)?;
        let dst_buf = gpu_get::<GfxBuffer>(state, cb.dst_rid)?;
        encoder.copy_buffer_to_buffer(
            &src_buf.buffer,
            cb.src_offset,
            &dst_buf.buffer,
            cb.dst_offset,
            cb.size,
        );
    }

    // Queue for batched submission instead of immediate submit
    queue_command_buffer(encoder.finish());
    Ok(())
}

// ======================= Buffer Mapping =======================

// Track pending buffer maps for async polling
struct PendingMapState {
    ready: Arc<AtomicBool>,
    result: Arc<Mutex<Option<Result<(), wgpu::BufferAsyncError>>>>,
}

static PENDING_MAPS: OnceLock<Mutex<HashMap<u32, PendingMapState>>> = OnceLock::new();

fn get_pending_maps() -> &'static Mutex<HashMap<u32, PendingMapState>> {
    PENDING_MAPS.get_or_init(|| Mutex::new(HashMap::new()))
}

/// Start mapping a buffer (non-blocking). Call op_gfx_buffer_map_poll to check completion.
#[op2(nofast)]
pub fn op_gfx_buffer_map_async(
    state: &mut OpState,
    #[smi] buffer_rid: u32,
    #[smi] mode: u32, // 1 = READ, 2 = WRITE
    #[bigint] offset: u64,
    #[bigint] size: u64,
) -> Result<(), JsErrorBox> {
    let buffer = gpu_get::<GfxBuffer>(state, buffer_rid)?
        .buffer
        .clone();

    let map_mode = if mode == 1 {
        wgpu::MapMode::Read
    } else {
        wgpu::MapMode::Write
    };

    let slice = buffer.slice(offset..offset + size);

    // Set up async tracking
    let ready = Arc::new(AtomicBool::new(false));
    let result_holder: Arc<Mutex<Option<Result<(), wgpu::BufferAsyncError>>>> = Arc::new(Mutex::new(None));

    let ready_clone = ready.clone();
    let result_clone = result_holder.clone();

    slice.map_async(map_mode, move |result| {
        *result_clone.lock().unwrap() = Some(result);
        ready_clone.store(true, Ordering::SeqCst);
    });

    // Store pending state
    get_pending_maps().lock().unwrap().insert(buffer_rid, PendingMapState {
        ready,
        result: result_holder,
    });

    // Do a quick non-blocking poll to kick off the operation
    let ctx = gfx_ctx()?;
    let _ = ctx.device.poll(wgpu::PollType::Poll);

    Ok(())
}

/// Poll for buffer map completion. Returns true if ready, false if still pending.
#[op2(fast)]
pub fn op_gfx_buffer_map_poll(
    #[smi] buffer_rid: u32,
) -> Result<bool, JsErrorBox> {
    let ctx = gfx_ctx()?;

    // Do a non-blocking poll
    let _ = ctx.device.poll(wgpu::PollType::Poll);

    // Check if this buffer's map is ready
    let pending = get_pending_maps().lock().unwrap();
    if let Some(state) = pending.get(&buffer_rid) {
        if state.ready.load(Ordering::SeqCst) {
            // Check the result
            let result = state.result.lock().unwrap();
            if let Some(ref r) = *result {
                if r.is_err() {
                    return Err(JsErrorBox::generic("Buffer map failed"));
                }
            }
            return Ok(true);
        }
    }

    Ok(false)
}

/// Block until buffer map completes. Calls device.poll(Wait) to avoid spinning.
#[op2(fast)]
pub fn op_gfx_buffer_map_wait(
    #[smi] buffer_rid: u32,
) -> Result<(), JsErrorBox> {
    let ctx = gfx_ctx()?;

    // Check if already ready (fast path)
    {
        let pending = get_pending_maps().lock().unwrap();
        if let Some(state) = pending.get(&buffer_rid) {
            if state.ready.load(Ordering::SeqCst) {
                let result = state.result.lock().unwrap();
                if let Some(ref r) = *result {
                    if r.is_err() {
                        return Err(JsErrorBox::generic("Buffer map failed"));
                    }
                }
                return Ok(());
            }
        } else {
            return Ok(());
        }
    }

    // Block until GPU work completes
    let _ = ctx.device.poll(wgpu::PollType::wait_indefinitely());

    // Verify completion
    let pending = get_pending_maps().lock().unwrap();
    if let Some(state) = pending.get(&buffer_rid) {
        let result = state.result.lock().unwrap();
        if let Some(ref r) = *result {
            if r.is_err() {
                return Err(JsErrorBox::generic("Buffer map failed"));
            }
        }
    }

    Ok(())
}

/// Finish buffer mapping - call after map_poll returns true
#[op2(fast)]
pub fn op_gfx_buffer_map_finish(
    #[smi] buffer_rid: u32,
) -> Result<(), JsErrorBox> {
    // Remove from pending
    get_pending_maps().lock().unwrap().remove(&buffer_rid);
    Ok(())
}

/// Get the mapped range of a buffer as a Uint8Array
#[op2]
#[buffer]
pub fn op_gfx_buffer_get_mapped_range(
    state: &mut OpState,
    #[smi] buffer_rid: u32,
    #[bigint] offset: u64,
    #[bigint] size: u64,
) -> Result<Vec<u8>, JsErrorBox> {
    let buffer = gpu_get::<GfxBuffer>(state, buffer_rid)?;

    let slice = buffer.buffer.slice(offset..offset + size);
    let view = slice.get_mapped_range();
    let data = view.to_vec();

    Ok(data)
}

/// Unmap a previously mapped buffer
#[op2(fast)]
pub fn op_gfx_buffer_unmap(
    state: &mut OpState,
    #[smi] buffer_rid: u32,
) -> Result<(), JsErrorBox> {
    let buffer = gpu_get::<GfxBuffer>(state, buffer_rid)?;

    buffer.buffer.unmap();
    Ok(())
}

// ======================= Immediate Timestamp Write =======================

/// Write a timestamp immediately (outside of any pass).
/// This creates a command encoder, writes the timestamp, and submits it.
/// Used for frame-level timing that spans multiple passes.
#[op2(fast)]
pub fn op_gfx_write_timestamp(
    state: &mut OpState,
    #[smi] query_set_rid: u32,
    #[smi] query_index: u32,
) -> Result<(), JsErrorBox> {
    let ctx = gfx_ctx()?;

    let qs = gpu_get::<GfxQuerySet>(state, query_set_rid)?;

    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("Timestamp Write Encoder"),
    });

    encoder.write_timestamp(&qs.query_set, query_index);

    // Queue for batched submission instead of immediate submit
    queue_command_buffer(encoder.finish());
    Ok(())
}