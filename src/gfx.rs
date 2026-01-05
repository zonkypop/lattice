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
use std::sync::atomic::{AtomicU32, Ordering};

use std::sync::RwLock;

use wgpu::BufferAsyncError;

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
        "rgba8unorm" => wgpu::TextureFormat::Rgba8Unorm,
        "rgba8unorm-srgb" => wgpu::TextureFormat::Rgba8UnormSrgb,
        "bgra8unorm" => wgpu::TextureFormat::Bgra8Unorm,
        "bgra8unorm-srgb" => wgpu::TextureFormat::Bgra8UnormSrgb,
        "rgba16float" => wgpu::TextureFormat::Rgba16Float,
        "rgba32float" => wgpu::TextureFormat::Rgba32Float,
        "r16float" => wgpu::TextureFormat::R16Float,
        "r32float" => wgpu::TextureFormat::R32Float,
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

// ======================= Image Decoding =======================

#[derive(Deserialize)]
pub struct DecodeImageOptions {
    pub resize_width: Option<u32>,
    pub resize_height: Option<u32>,
    pub resize_quality: Option<String>,  // "low", "medium", "high", "pixelated"
    pub image_orientation: Option<String>, // "none", "flipY"
}


fn decode_image_internal(
    data: &[u8], 
    format: &str,
    options: Option<DecodeImageOptions>,
) -> Result<(u32, u32, Vec<u8>), Box<dyn std::error::Error + Send + Sync>> {
    use image::{ImageFormat, imageops::FilterType};
    
    let mut img: DynamicImage = match format {
        "png" => image::load_from_memory_with_format(data, ImageFormat::Png)?,
        "jpeg" | "jpg" => image::load_from_memory_with_format(data, ImageFormat::Jpeg)?,
        "webp" => image::load_from_memory_with_format(data, ImageFormat::WebP)?,
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
    pub color_format: Option<String>,
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
    // Multi draw
    pub is_multi_draw: Option<bool>,
    pub indirect_buffer_rid: Option<u32>,
    pub indirect_offset: Option<u64>,
    pub draw_count: Option<u32>,
    // Indirect indexed
    pub is_indirect: Option<bool>,
}

#[derive(Deserialize)]
pub struct SurfaceDrawArgs {
    pub draw_calls: Vec<DrawCall>,
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

    let pipeline = state
        .resource_table
        .get::<GfxPipeline>(ResourceId::from(args.pipeline_rid))
        .map_err(|e| JsErrorBox::generic(e.to_string()))?;

    let mut vbufs: Vec<Arc<wgpu::Buffer>> = Vec::new();
    for rid in args.vertex_buffer_rids.iter() {
        let buf = state
            .resource_table
            .get::<GfxBuffer>(ResourceId::from(*rid))
            .map_err(|e| JsErrorBox::generic(e.to_string()))?;
        vbufs.push(buf.buffer.clone());
    }

    let mut bgs: Vec<Option<Arc<wgpu::BindGroup>>> = Vec::new();
    for rid_opt in args.bind_group_rids.iter() {
        if let Some(rid) = rid_opt {
            let bg = state
                .resource_table
                .get::<GfxBindGroup>(ResourceId::from(*rid))
                .map_err(|e| JsErrorBox::generic(e.to_string()))?;
            bgs.push(Some(bg.group.clone()));
        } else {
            bgs.push(None);
        }
    }

    let index_buf = state
        .resource_table
        .get::<GfxBuffer>(ResourceId::from(args.index_buffer_rid))
        .map_err(|e| JsErrorBox::generic(e.to_string()))?;

    let indirect_buf = state
        .resource_table
        .get::<GfxBuffer>(ResourceId::from(args.indirect_buffer_rid))
        .map_err(|e| JsErrorBox::generic(e.to_string()))?;

    let depth_view = if let Some(rid) = args.depth_view_rid {
        Some(
            state
                .resource_table
                .get::<GfxTextureView>(ResourceId::from(rid))
                .map_err(|e| JsErrorBox::generic(e.to_string()))?,
        )
    } else {
        None
    };

    let target_view = if let Some(rid) = args.target_view_rid {
        Some(
            state
                .resource_table
                .get::<GfxTextureView>(ResourceId::from(rid))
                .map_err(|e| JsErrorBox::generic(e.to_string()))?,
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

    ctx.queue.submit(std::iter::once(encoder.finish()));
    Ok(())
}

#[op2(fast)]
pub fn op_gfx_queue_submit_empty() -> Result<(), JsErrorBox> {
    let ctx = gfx_ctx()?;
    ctx.queue.submit(std::iter::empty());
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
pub fn op_gfx_decode_image_store(
    _state: &mut OpState, 
    #[buffer] data: JsBuffer,
    #[serde] args: DecodeImageStoreArgs,
) -> Result<DecodeImageStoreResult, JsErrorBox> {
    let slice: &[u8] = &data;
    
    let options = Some(DecodeImageOptions {
        resize_width: args.resize_width,
        resize_height: args.resize_height,
        resize_quality: args.resize_quality,
        image_orientation: args.image_orientation,
    });
    
    let (width, height, pixels) = decode_image_internal(slice, &args.format, options)
        .map_err(|e| JsErrorBox::generic(e.to_string()))?;

    let img = GfxDecodedImage {
        width,
        height,
        pixels,
    };

    // Store in global storage instead of per-isolate resource table
    let rid = DECODED_IMAGE_COUNTER.fetch_add(1, Ordering::SeqCst);
    
    {
        let mut store = get_decoded_image_store()
            .lock()
            .map_err(|e| JsErrorBox::generic(format!("Failed to lock image store: {}", e)))?;
        store.insert(rid, img);
    }
    
    Ok(DecodeImageStoreResult { rid: rid.into(), width, height })
}

static MIPMAP_RESOURCES: OnceLock<MipmapResources> = OnceLock::new();

struct MipmapResources {
    shader: wgpu::ShaderModule,
    sampler: wgpu::Sampler,
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
        MipmapResources { shader, sampler }
    })
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

        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Mipmap Pipeline"),
            layout: None,
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

        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Mipmap Bind Group"),
            layout: &pipeline.get_bind_group_layout(0),
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

        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.draw(0..3, 0..1);
    }

    queue.submit(std::iter::once(encoder.finish()));
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

    // Get image from GLOBAL storage
    let store = get_decoded_image_store()
        .lock()
        .map_err(|e| JsErrorBox::generic(format!("Failed to lock image store: {}", e)))?;
    
    let img = store
        .get(&image_rid)
        .ok_or_else(|| JsErrorBox::generic(format!("Failed to get decoded image: Bad resource ID {}", image_rid)))?;

    // Get texture from per-isolate resource table (textures ARE per-isolate, that's fine)
    let texture = state
        .resource_table
        .get::<GfxTexture>(ResourceId::from(texture_rid))
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

    generate_mipmaps(&ctx.device, &ctx.queue, &texture.texture, img.width, img.height, origin_z);

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

    let module = ctx.device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: desc.label.as_deref(),
        source: wgpu::ShaderSource::Wgsl(sanitized.into()),
    });

    let rid = state.resource_table.add(GfxShader { module });
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
    let rid = state.resource_table.add(GfxBuffer { buffer });
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
    let rid = state.resource_table.add(GfxBuffer { buffer });
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

    let buffer = state
        .resource_table
        .get::<GfxBuffer>(ResourceId::from(buffer_rid))
        .map_err(|e| JsErrorBox::generic(e.to_string()))?;

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

#[op2]
#[serde]
pub fn op_gfx_device_create_texture(
    state: &mut OpState,
    #[serde] desc: TextureCreateDesc,
) -> Result<JsGfxTexture, JsErrorBox> {
    let ctx = gfx_ctx()?;
    let usage = wgpu::TextureUsages::from_bits(desc.usage)
        .ok_or_else(|| JsErrorBox::generic("invalid texture usage flags"))?;

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
        format: map_texture_format(&desc.format),
        usage,
        view_formats: &[],
    });

    let rid = state.resource_table.add(GfxTexture { texture });
    Ok(JsGfxTexture { rid })
}

#[op2]
#[serde]
pub fn op_gfx_texture_create_view(
    state: &mut OpState,
    #[serde] desc: TextureViewCreateDesc,
) -> Result<JsGfxTextureView, JsErrorBox> {
    let tex = state
        .resource_table
        .get::<GfxTexture>(ResourceId::from(desc.texture_rid))
        .map_err(|e| JsErrorBox::generic(e.to_string()))?;

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
        usage: None, 
    };

    let view = tex.texture.create_view(&view_desc);

    let rid = state
        .resource_table
        .add(GfxTextureView { 
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

    let rid = state
        .resource_table
        .add(GfxSampler { sampler: Arc::new(sampler) });
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

    let rid = state
        .resource_table
        .add(GfxBindGroupLayout { layout: Arc::new(layout) });
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
        let l = state
            .resource_table
            .get::<GfxBindGroupLayout>(ResourceId::from(*rid))
            .map_err(|e| JsErrorBox::generic(e.to_string()))?;
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

    let rid = state
        .resource_table
        .add(GfxPipelineLayout { layout: pipeline_layout });
    Ok(JsGfxPipelineLayout { rid })
}

#[op2]
#[serde]
pub fn op_gfx_device_create_bind_group(
    state: &mut OpState,
    #[serde] desc: BindGroupCreate,
) -> Result<JsGfxBindGroup, JsErrorBox> {
    let ctx = gfx_ctx()?;

    let layout = state
        .resource_table
        .get::<GfxBindGroupLayout>(ResourceId::from(desc.layout_rid))
        .map_err(|e| JsErrorBox::generic(e.to_string()))?;

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
            let buf_res = state
                .resource_table
                .get::<GfxBuffer>(ResourceId::from(buf_rid))
                .map_err(|err| JsErrorBox::generic(err.to_string()))?;
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
            let s_res = state
                .resource_table
                .get::<GfxSampler>(ResourceId::from(sam_rid))
                .map_err(|err| JsErrorBox::generic(err.to_string()))?;
            let idx = samplers.len();
            samplers.push(s_res.sampler.clone());

            plans.push((e.binding, EntryKind::Sampler { idx }));
        } else if let Some(view_rid) = e.texture_view_rid {
            let v_res = state
                .resource_table
                .get::<GfxTextureView>(ResourceId::from(view_rid))
                .map_err(|err| JsErrorBox::generic(err.to_string()))?;
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

    let group = ctx.device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: desc.label.as_deref(),
        layout: &layout.layout,
        entries: &entries,
    });

    let rid = state
        .resource_table
        .add(GfxBindGroup { group: Arc::new(group) });
    Ok(JsGfxBindGroup { rid })
}


#[op2]
#[serde]
pub fn op_gfx_device_create_pipeline(
    state: &mut OpState,
    #[serde] desc: PipelineCreate,
) -> Result<JsGfxPipeline, JsErrorBox> {
    let ctx = gfx_ctx()?;

    let v_mod = state
        .resource_table
        .get::<GfxShader>(ResourceId::from(desc.vertex_module_rid))
        .map_err(|e| JsErrorBox::generic(e.to_string()))?;

    // Fragment shader is optional for depth-only passes
    let f_mod = if let Some(frag_rid) = desc.fragment_module_rid {
        Some(
            state
                .resource_table
                .get::<GfxShader>(ResourceId::from(frag_rid))
                .map_err(|e| JsErrorBox::generic(e.to_string()))?,
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
        Some(wgpu::DepthStencilState {
            format: tfmt,
            depth_write_enabled: desc.depth_write_enabled.unwrap_or(true),
            depth_compare: map_compare_function(desc.depth_compare.as_deref()),
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
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

        Some(wgpu::FragmentState {
            module: &f.module,
            entry_point: Some(entry_point),
            compilation_options: wgpu::PipelineCompilationOptions::default(),
            targets: &[Some(wgpu::ColorTargetState {
                format: color_format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
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
        let pl = state
            .resource_table
            .get::<GfxPipelineLayout>(ResourceId::from(layout_rid))
            .map_err(|e| JsErrorBox::generic(e.to_string()))?;

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

    let rid = state.resource_table.add(GfxPipeline { pipeline });
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
        if state
            .resource_table
            .get::<GfxPipeline>(ResourceId::from(dc.pipeline_rid))
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
            if state
                .resource_table
                .get::<GfxBuffer>(ResourceId::from(*rid))
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
                if state
                    .resource_table
                    .get::<GfxBindGroup>(ResourceId::from(*rid))
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
            if state
                .resource_table
                .get::<GfxBuffer>(ResourceId::from(idx_rid))
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
            if state
                .resource_table
                .get::<GfxTextureView>(ResourceId::from(view_rid))
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
                if state
                    .resource_table
                    .get::<GfxBuffer>(ResourceId::from(indirect_rid))
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
            let v_res = state
                .resource_table
                .get::<GfxTextureView>(ResourceId::from(view_rid))
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

    // Pre-collect depth view and ops from first draw call that has depth
    let mut depth_view: Option<Arc<wgpu::TextureView>> = None;
    let mut depth_ops: Option<wgpu::Operations<f32>> = None;

    for dc in args.draw_calls.iter() {
        if depth_view.is_none() {
            if let Some(view_rid) = dc.depth_view_rid {
                let v_res = state
                    .resource_table
                    .get::<GfxTextureView>(ResourceId::from(view_rid))
                    .map_err(|e| JsErrorBox::generic(e.to_string()))?;
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
    }

    let mut draw_data_list: Vec<SurfaceDrawData> = Vec::with_capacity(args.draw_calls.len());

    for (call_idx, dc) in args.draw_calls.iter().enumerate() {
        // Validate pipeline exists
        let _ = state
            .resource_table
            .get::<GfxPipeline>(ResourceId::from(dc.pipeline_rid))
            .map_err(|e| {
                JsErrorBox::generic(format!(
                    "Draw call {}: invalid pipeline rid {}: {}",
                    call_idx, dc.pipeline_rid, e
                ))
            })?;

        let mut vbufs: Vec<Arc<wgpu::Buffer>> = Vec::new();
        for (slot, rid) in dc.vertex_buffer_rids.iter().enumerate() {
            let buf_res = state
                .resource_table
                .get::<GfxBuffer>(ResourceId::from(*rid))
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
                let bg_res = state
                    .resource_table
                    .get::<GfxBindGroup>(ResourceId::from(*rid))
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
            let buf_res = state
                .resource_table
                .get::<GfxBuffer>(ResourceId::from(rid))
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
                let buf_res = state
                    .resource_table
                    .get::<GfxBuffer>(ResourceId::from(rid))
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
        });
    }

    let mut encoder = ctx.device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("GFX Encoder"),
    });

    // single render pass for all draw calls 
    {
        let depth_attachment = match (&depth_view, &depth_ops) {
            (Some(dv), Some(ops)) => Some(wgpu::RenderPassDepthStencilAttachment {
                view: dv.as_ref(),
                depth_ops: Some(*ops),
                stencil_ops: None,
            }),
            _ => None,
        };

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
            let pipeline_res = state
                .resource_table
                .get::<GfxPipeline>(ResourceId::from(data.pipeline_rid))
                .unwrap();

            pass.set_pipeline(&pipeline_res.pipeline);

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
    origin_x: u32,
    origin_y: u32,
    origin_z: u32,  
    mip_level: u32,
    #[buffer] data: JsBuffer,
) -> Result<(), JsErrorBox> {
    let ctx = gfx_ctx()?;


    let texture = state
        .resource_table
        .get::<GfxTexture>(ResourceId::from(texture_rid))
        .map_err(|e| JsErrorBox::generic(format!("Failed to get texture: {}", e)))?;

    // Validate dimensions
    if width == 0 || height == 0 {
        return Err(JsErrorBox::generic("write_texture: width and height must be > 0"));
    }

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
    let expected_size = (bytes_per_row * height) as usize;
    
    let bytes: &[u8] = &data;
    
    // Validate data size
    if bytes.len() < expected_size {
        return Err(JsErrorBox::generic(format!(
            "write_texture: data buffer too small. Expected at least {} bytes for {}x{} {:?} texture, got {} bytes",
            expected_size, width, height, format, bytes.len()
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
            depth_or_array_layers: 1,
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
    let pipeline = state
        .resource_table
        .get::<GfxPipeline>(ResourceId::from(args.pipeline_rid))
        .map_err(|e| JsErrorBox::generic(e.to_string()))?;

    let layout = pipeline.pipeline.get_bind_group_layout(args.index);

    let rid = state
        .resource_table
        .add(GfxBindGroupLayout { layout: Arc::new(layout) });
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

    let msaa_view = state
        .resource_table
        .get::<GfxTextureView>(ResourceId::from(args.msaa_view_rid))
        .map_err(|e| JsErrorBox::generic(format!("Failed to get MSAA view: {}", e)))?;

    let resolve_view = state
        .resource_table
        .get::<GfxTextureView>(ResourceId::from(args.resolve_view_rid))
        .map_err(|e| JsErrorBox::generic(format!("Failed to get resolve view: {}", e)))?;

    let depth_view = if let Some(rid) = args.depth_view_rid {
        Some(
            state
                .resource_table
                .get::<GfxTextureView>(ResourceId::from(rid))
                .map_err(|e| JsErrorBox::generic(format!("Failed to get depth view: {}", e)))?,
        )
    } else {
        None
    };

    // Pre-collect all resources before starting the render pass
    // This avoids multiple tile flushes on tile-based GPUs like Quest 3
    let mut draw_data_list: Vec<XrDrawData> = Vec::with_capacity(args.draw_calls.len());

    for dc in args.draw_calls.iter() {
        let _ = state
            .resource_table
            .get::<GfxPipeline>(ResourceId::from(dc.pipeline_rid))
            .map_err(|e| JsErrorBox::generic(e.to_string()))?;

        let mut vbufs: Vec<Arc<wgpu::Buffer>> = Vec::new();
        for rid in dc.vertex_buffer_rids.iter() {
            let buf_res = state
                .resource_table
                .get::<GfxBuffer>(ResourceId::from(*rid))
                .map_err(|e| JsErrorBox::generic(e.to_string()))?;
            vbufs.push(buf_res.buffer.clone());
        }

        let mut bgs: Vec<Option<Arc<wgpu::BindGroup>>> = Vec::new();
        for rid_opt in dc.bind_group_rids.iter() {
            if let Some(rid) = rid_opt {
                let bg_res = state
                    .resource_table
                    .get::<GfxBindGroup>(ResourceId::from(*rid))
                    .map_err(|e| JsErrorBox::generic(e.to_string()))?;
                bgs.push(Some(bg_res.group.clone()));
            } else {
                bgs.push(None);
            }
        }

        let index_buf: Option<Arc<wgpu::Buffer>> = if let Some(rid) = dc.index_buffer_rid {
            let buf_res = state
                .resource_table
                .get::<GfxBuffer>(ResourceId::from(rid))
                .map_err(|e| JsErrorBox::generic(e.to_string()))?;
            Some(buf_res.buffer.clone())
        } else {
            None
        };

        let is_multi_draw = dc.is_multi_draw.unwrap_or(false);
        let is_indirect = dc.is_indirect.unwrap_or(false);

        let indirect_buf: Option<Arc<wgpu::Buffer>> = if is_multi_draw || is_indirect {
            if let Some(rid) = dc.indirect_buffer_rid {
                let buf_res = state
                    .resource_table
                    .get::<GfxBuffer>(ResourceId::from(rid))
                    .map_err(|e| JsErrorBox::generic(e.to_string()))?;
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

    // single render pass for all draw calls - good for TBRs
    {
        let depth_attachment = depth_view.as_ref().map(|dv| {
            wgpu::RenderPassDepthStencilAttachment {
                view: &dv.view,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(args.depth_clear_value.unwrap_or(1.0)),
                    store: depth_store,
                }),
                stencil_ops: None,
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
            let pipeline_res = state
                .resource_table
                .get::<GfxPipeline>(ResourceId::from(data.pipeline_rid))
                .unwrap();

            pass.set_pipeline(&pipeline_res.pipeline);

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

    // Queue this command buffer and flush all pending commands
    // XR frame is the final pass, so we submit everything together
    queue_command_buffer(encoder.finish());
    flush_pending_commands(&ctx.queue);
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

    let target_view = state
        .resource_table
        .get::<GfxTextureView>(ResourceId::from(args.target_view_rid))
        .map_err(|e| JsErrorBox::generic(e.to_string()))?;

    // Pre-collect depth view and ops from first draw call that has depth
    let mut depth_view: Option<Arc<wgpu::TextureView>> = None;
    let mut depth_ops: Option<wgpu::Operations<f32>> = None;

    for dc in args.draw_calls.iter() {
        if depth_view.is_none() {
            if let Some(view_rid) = dc.depth_view_rid {
                let v_res = state
                    .resource_table
                    .get::<GfxTextureView>(ResourceId::from(view_rid))
                    .map_err(|e| JsErrorBox::generic(e.to_string()))?;
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
    }

    let mut draw_data_list: Vec<TextureDrawData> = Vec::with_capacity(args.draw_calls.len());

    for dc in args.draw_calls.iter() {
        // Validate pipeline exists
        let _ = state
            .resource_table
            .get::<GfxPipeline>(ResourceId::from(dc.pipeline_rid))
            .map_err(|e| JsErrorBox::generic(e.to_string()))?;

        let mut vbufs: Vec<Arc<wgpu::Buffer>> = Vec::new();
        for rid in dc.vertex_buffer_rids.iter() {
            let buf_res = state
                .resource_table
                .get::<GfxBuffer>(ResourceId::from(*rid))
                .map_err(|e| JsErrorBox::generic(e.to_string()))?;
            vbufs.push(buf_res.buffer.clone());
        }

        let mut bgs: Vec<Option<Arc<wgpu::BindGroup>>> = Vec::new();
        for rid_opt in dc.bind_group_rids.iter() {
            if let Some(rid) = rid_opt {
                let bg_res = state
                    .resource_table
                    .get::<GfxBindGroup>(ResourceId::from(*rid))
                    .map_err(|e| JsErrorBox::generic(e.to_string()))?;
                bgs.push(Some(bg_res.group.clone()));
            } else {
                bgs.push(None);
            }
        }

        let index_buf: Option<Arc<wgpu::Buffer>> = if let Some(rid) = dc.index_buffer_rid {
            let buf_res = state
                .resource_table
                .get::<GfxBuffer>(ResourceId::from(rid))
                .map_err(|e| JsErrorBox::generic(e.to_string()))?;
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
                stencil_ops: None,
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
            let pipeline_res = state
                .resource_table
                .get::<GfxPipeline>(ResourceId::from(data.pipeline_rid))
                .unwrap();

            pass.set_pipeline(&pipeline_res.pipeline);

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

    // Submit immediately - these render-to-texture ops must complete before main render
    ctx.queue.submit(iter::once(encoder.finish()));

    Ok(())
}

#[op2]
pub fn op_gfx_copy_texture_to_texture(
    state: &mut OpState,
    #[serde] args: CopyTextureToTextureArgs,
) -> Result<(), JsErrorBox> {
    let ctx = gfx_ctx()?;

    let src_tex = state
        .resource_table
        .get::<GfxTexture>(ResourceId::from(args.src_texture_rid))
        .map_err(|e| JsErrorBox::generic(e.to_string()))?;

    let dst_tex = state
        .resource_table
        .get::<GfxTexture>(ResourceId::from(args.dst_texture_rid))
        .map_err(|e| JsErrorBox::generic(e.to_string()))?;

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

    ctx.queue.submit(iter::once(encoder.finish()));
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

    let depth_view = state
        .resource_table
        .get::<GfxTextureView>(ResourceId::from(args.depth_view_rid))
        .map_err(|e| JsErrorBox::generic(e.to_string()))?;

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
    }

    let mut draw_data_list: Vec<DepthDrawData> = Vec::with_capacity(args.draw_calls.len());

    for dc in args.draw_calls.iter() {
        // Validate pipeline exists
        let _ = state
            .resource_table
            .get::<GfxPipeline>(ResourceId::from(dc.pipeline_rid))
            .map_err(|e| JsErrorBox::generic(e.to_string()))?;

        let mut vbufs: Vec<Arc<wgpu::Buffer>> = Vec::new();
        for rid in dc.vertex_buffer_rids.iter() {
            let buf_res = state
                .resource_table
                .get::<GfxBuffer>(ResourceId::from(*rid))
                .map_err(|e| JsErrorBox::generic(e.to_string()))?;
            vbufs.push(buf_res.buffer.clone());
        }

        let mut bgs: Vec<Option<Arc<wgpu::BindGroup>>> = Vec::new();
        for rid_opt in dc.bind_group_rids.iter() {
            if let Some(rid) = rid_opt {
                let bg_res = state
                    .resource_table
                    .get::<GfxBindGroup>(ResourceId::from(*rid))
                    .map_err(|e| JsErrorBox::generic(e.to_string()))?;
                bgs.push(Some(bg_res.group.clone()));
            } else {
                bgs.push(None);
            }
        }

        let index_buf: Option<Arc<wgpu::Buffer>> = if let Some(rid) = dc.index_buffer_rid {
            let buf_res = state
                .resource_table
                .get::<GfxBuffer>(ResourceId::from(rid))
                .map_err(|e| JsErrorBox::generic(e.to_string()))?;
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
            let pipeline_res = state
                .resource_table
                .get::<GfxPipeline>(ResourceId::from(data.pipeline_rid))
                .unwrap();

            pass.set_pipeline(&pipeline_res.pipeline);

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

    // Submit immediately - depth-only renders must complete before main render
    ctx.queue.submit(iter::once(encoder.finish()));
    Ok(())
}

#[op2(fast)]
pub fn op_gfx_resource_drop(
    state: &mut OpState,
    rid: u32,
) -> Result<(), JsErrorBox> {
    let _ = state.resource_table.take::<GfxTextureView>(ResourceId::from(rid));
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

    let shader = state
        .resource_table
        .get::<GfxShader>(ResourceId::from(desc.shader_module_rid))
        .map_err(|e| JsErrorBox::generic(e.to_string()))?;

    let pipeline = if let Some(layout_rid) = desc.pipeline_layout_rid {
        let pl = state
            .resource_table
            .get::<GfxPipelineLayout>(ResourceId::from(layout_rid))
            .map_err(|e| JsErrorBox::generic(e.to_string()))?;

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

    let rid = state.resource_table.add(GfxComputePipeline { pipeline });
    Ok(JsGfxComputePipeline { rid })
}

#[op2]
#[serde]
pub fn op_gfx_compute_pipeline_get_bind_group_layout(
    state: &mut OpState,
    #[serde] args: GetBindGroupLayoutArgs,
) -> Result<JsGfxBindGroupLayout, JsErrorBox> {
    let pipeline = state
        .resource_table
        .get::<GfxComputePipeline>(ResourceId::from(args.pipeline_rid))
        .map_err(|e| JsErrorBox::generic(e.to_string()))?;

    let layout = pipeline.pipeline.get_bind_group_layout(args.index);

    let rid = state
        .resource_table
        .add(GfxBindGroupLayout { layout: Arc::new(layout) });
    Ok(JsGfxBindGroupLayout { rid })
}

#[op2]
pub fn op_gfx_compute_dispatch(
    state: &mut OpState,
    #[serde] args: ComputeDispatchArgs,
) -> Result<(), JsErrorBox> {
    let ctx = gfx_ctx()?;

    let pipeline = state
        .resource_table
        .get::<GfxComputePipeline>(ResourceId::from(args.pipeline_rid))
        .map_err(|e| JsErrorBox::generic(e.to_string()))?;

    let mut bgs: Vec<Option<Arc<wgpu::BindGroup>>> = Vec::new();
    for rid_opt in args.bind_group_rids.iter() {
        if let Some(rid) = rid_opt {
            let bg = state
                .resource_table
                .get::<GfxBindGroup>(ResourceId::from(*rid))
                .map_err(|e| JsErrorBox::generic(e.to_string()))?;
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

    ctx.queue.submit(iter::once(encoder.finish()));
    Ok(())
}

#[op2]
pub fn op_gfx_compute_dispatch_indirect(
    state: &mut OpState,
    #[serde] args: ComputeDispatchIndirectArgs,
) -> Result<(), JsErrorBox> {
    let ctx = gfx_ctx()?;

    let pipeline = state
        .resource_table
        .get::<GfxComputePipeline>(ResourceId::from(args.pipeline_rid))
        .map_err(|e| JsErrorBox::generic(e.to_string()))?;

    let indirect_buf = state
        .resource_table
        .get::<GfxBuffer>(ResourceId::from(args.indirect_buffer_rid))
        .map_err(|e| JsErrorBox::generic(e.to_string()))?;

    let mut bgs: Vec<Option<Arc<wgpu::BindGroup>>> = Vec::new();
    for rid_opt in args.bind_group_rids.iter() {
        if let Some(rid) = rid_opt {
            let bg = state
                .resource_table
                .get::<GfxBindGroup>(ResourceId::from(*rid))
                .map_err(|e| JsErrorBox::generic(e.to_string()))?;
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

    ctx.queue.submit(iter::once(encoder.finish()));
    Ok(())
}