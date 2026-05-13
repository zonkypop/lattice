// src/xr.rs - OpenXR integration with controller support

use deno_error::JsErrorBox;
use serde::Serialize;
use std::sync::{Arc, Mutex, OnceLock};
use std::sync::atomic::{AtomicBool, Ordering};
use deno_core::{op2, OpState};

use openxr as xr;
use ash::vk::{self, Handle};
use std::ffi::CString;
use crate::gfx::{GfxContext, set_gfx_context, flush_pending_command_buffers};
use wgpu_hal::vulkan::TextureMemory;


use openxr::sys;



#[cfg(target_os = "android")]
fn init_android_loader(entry: &xr::Entry) -> Result<(), JsErrorBox> {
    let native_activity = ndk_glue::native_activity();
    let vm = native_activity.vm() as *mut std::ffi::c_void;
    let activity = native_activity.activity() as *mut std::ffi::c_void;
    
    let mut init_fn: Option<xr::sys::pfn::VoidFunction> = None;
    unsafe {
        (entry.fp().get_instance_proc_addr)(
            xr::sys::Instance::NULL,
            b"xrInitializeLoaderKHR\0".as_ptr(),
            &mut init_fn,
        )
    };
    
    if let Some(func) = init_fn {
        type InitLoaderFn = unsafe extern "system" fn(*const xr::sys::LoaderInitInfoBaseHeaderKHR) -> xr::sys::Result;
        let init_loader: InitLoaderFn = unsafe { std::mem::transmute(func) };
        
        let loader_init = xr::sys::LoaderInitInfoAndroidKHR {
            ty: xr::sys::LoaderInitInfoAndroidKHR::TYPE,
            next: std::ptr::null(),
            application_vm: vm,
            application_context: activity,
        };
        
        unsafe {
            init_loader(&loader_init as *const _ as *const xr::sys::LoaderInitInfoBaseHeaderKHR)
        };
    }
    Ok(())
}

#[cfg(not(target_os = "android"))]
fn init_android_loader(_entry: &xr::Entry) -> Result<(), JsErrorBox> {
    Ok(())
}

// ======================= Serde Types =======================


#[derive(Serialize)]
pub struct XrSessionInfo {
    pub width: u32,
    pub height: u32,
    pub sample_count: u32,
    pub format: String,
    pub swapchain_length: u32,
}

#[derive(Serialize)]
pub struct XrFrameState {
    pub predicted_display_time: i64,
    pub should_render: bool,
}

#[derive(Serialize, Clone)]
pub struct XrPose {
    pub position: [f32; 3],
    pub orientation: [f32; 4],
    pub matrix: [f32; 16],
}

#[derive(Serialize)]
pub struct XrView {
    pub projection_matrix: [f32; 16],
    pub transform: XrPose,
    pub view_index: u32,
    pub viewport_x: u32,
    pub viewport_y: u32,
    pub viewport_width: u32,
    pub viewport_height: u32,
}

#[derive(Serialize)]
pub struct XrViewerPose {
    pub views: Vec<XrView>,
}

#[derive(Serialize)]
pub struct XrSwapchainTextureInfo {
    pub index: u32,
    pub width: u32,
    pub height: u32,
    pub array_layers: u32,
}

// Controller types
#[derive(Serialize)]
pub struct XrInputSourceData {
    pub handedness: String,
    pub target_ray_mode: String,
    pub profiles: Vec<String>,
    pub grip_space_pose: Option<XrPose>,
    pub target_ray_pose: Option<XrPose>,
    pub gamepad: Option<XrGamepadData>,
    pub hand: Option<XrHandData>,
}

#[derive(Serialize)]
pub struct XrGamepadData {
    pub buttons: Vec<XrButtonData>,
    pub axes: Vec<f32>,
}

#[derive(Serialize)]
pub struct XrButtonData {
    pub pressed: bool,
    pub touched: bool,
    pub value: f32,
}

#[derive(Serialize)]
pub struct XrHandData {
    pub joints: Vec<XrJointData>,
}

#[derive(Serialize)]
pub struct XrJointData {
    pub name: String,
    pub pose: XrPose,
    pub radius: f32,
}

#[derive(Serialize)]
pub struct XrInputSourcesState {
    pub sources: Vec<XrInputSourceData>,
}

/// Combined frame begin result - reduces 4 separate lock acquisitions to 1
#[derive(Serialize)]
pub struct XrFrameBeginResult {
    pub frame_state: XrFrameState,
    pub viewer_pose: Option<XrViewerPose>,
    pub input_sources: XrInputSourcesState,
    pub swapchain_info: Option<XrSwapchainTextureInfo>,
}

// ======================= XR State =======================

pub struct XrState {
    pub _vk_entry: ash::Entry,
    pub _vk_instance: ash::Instance,
    pub _vk_device: ash::Device,
    pub instance: xr::Instance,
    pub system: xr::SystemId,
    pub session: xr::Session<xr::Vulkan>,
    pub frame_waiter: xr::FrameWaiter,
    pub frame_stream: xr::FrameStream<xr::Vulkan>,
    pub stage_space: xr::Space,
    pub view_config_type: xr::ViewConfigurationType,
    pub views: Vec<xr::ViewConfigurationView>,
    pub swapchain: xr::Swapchain<xr::Vulkan>,
    pub swapchain_textures: Vec<wgpu::Texture>,
    pub session_running: bool,
    pub session_state: xr::SessionState,
    pub frame_state: Option<xr::FrameState>,
    pub current_swapchain_index: Option<u32>,
    pub render_width: u32,
    pub render_height: u32,
    pub cached_views: Option<Vec<xr::View>>,
    
    // Controller actions
    pub action_set: xr::ActionSet,
    pub left_grip_action: xr::Action<xr::Posef>,
    pub right_grip_action: xr::Action<xr::Posef>,
    pub left_aim_action: xr::Action<xr::Posef>,
    pub right_aim_action: xr::Action<xr::Posef>,
    pub left_grip_space: xr::Space,
    pub right_grip_space: xr::Space,
    pub left_aim_space: xr::Space,
    pub right_aim_space: xr::Space,
    pub left_trigger_action: xr::Action<f32>,
    pub right_trigger_action: xr::Action<f32>,
    pub left_squeeze_action: xr::Action<f32>,
    pub right_squeeze_action: xr::Action<f32>,
    pub left_thumbstick_action: xr::Action<xr::Vector2f>,
    pub right_thumbstick_action: xr::Action<xr::Vector2f>,
    pub left_thumbstick_click_action: xr::Action<bool>,
    pub right_thumbstick_click_action: xr::Action<bool>,
    pub left_x_button_action: xr::Action<bool>,
    pub left_y_button_action: xr::Action<bool>,
    pub right_a_button_action: xr::Action<bool>,
    pub right_b_button_action: xr::Action<bool>,
    pub left_hand_path: xr::Path,
    pub right_hand_path: xr::Path,

    // Haptic output
    pub left_haptic_action: xr::Action<xr::Haptic>,
    pub right_haptic_action: xr::Action<xr::Haptic>,

    // Clip planes (set from JS via updateRenderState)
    pub depth_near: f32,
    pub depth_far: f32,
}

unsafe impl Send for XrState {}
unsafe impl Sync for XrState {}

static XR_STATE: OnceLock<Mutex<Option<XrState>>> = OnceLock::new();

pub fn get_xr_state() -> &'static Mutex<Option<XrState>> {
    XR_STATE.get_or_init(|| Mutex::new(None))
}

/// Returns true if an XR session is currently active
pub fn is_xr_active() -> bool {
    if let Some(state_lock) = XR_STATE.get() {
        if let Ok(guard) = state_lock.lock() {
            return guard.is_some();
        }
    }
    false
}

pub static XR_SWAPCHAIN_TEXTURES: OnceLock<Vec<wgpu::Texture>> = OnceLock::new();

/// Cached texture views for XR swapchain - indexed as [swapchain_index][view_index (eye)]
/// Pre-created during session init to avoid per-frame allocation
pub static XR_SWAPCHAIN_VIEWS: OnceLock<Vec<[std::sync::Arc<wgpu::TextureView>; 2]>> = OnceLock::new();

// ======================= Persistent Frame Wait Worker =======================
// A single long-lived thread handles the blocking xrWaitFrame call.
// This avoids spawning a new thread every frame (72 spawns/s on Quest).

struct FrameWaitWorker {
    /// Main→worker: "start a new frame wait"
    request: Arc<(Mutex<bool>, std::sync::Condvar)>,
    /// Shutdown flag so the thread exits cleanly
    shutdown: Arc<AtomicBool>,
}

/// Worker→main: result is ready (lock-free poll path)
static FRAME_WAIT_READY: AtomicBool = AtomicBool::new(false);

/// The FrameState result produced by the worker thread
static FRAME_WAIT_RESULT: OnceLock<Mutex<Option<Result<xr::FrameState, String>>>> = OnceLock::new();

fn get_frame_wait_result() -> &'static Mutex<Option<Result<xr::FrameState, String>>> {
    FRAME_WAIT_RESULT.get_or_init(|| Mutex::new(None))
}

static FRAME_WAIT_WORKER: OnceLock<Mutex<Option<FrameWaitWorker>>> = OnceLock::new();

fn get_frame_wait_worker() -> &'static Mutex<Option<FrameWaitWorker>> {
    FRAME_WAIT_WORKER.get_or_init(|| Mutex::new(None))
}

/// Lazily spawn the persistent frame-wait worker thread.
fn ensure_frame_wait_worker() {
    let mut guard = get_frame_wait_worker().lock().unwrap();
    if guard.is_some() {
        return;
    }

    let request = Arc::new((Mutex::new(false), std::sync::Condvar::new()));
    let shutdown = Arc::new(AtomicBool::new(false));

    let req = request.clone();
    let shut = shutdown.clone();

    std::thread::Builder::new()
        .name("xr-frame-wait".into())
        .spawn(move || {
            log::info!("Frame-wait worker thread started");
            loop {
                // Park until signalled to start or shut down
                {
                    let (lock, cvar) = &*req;
                    let mut requested = lock.lock().unwrap();
                    while !*requested && !shut.load(Ordering::SeqCst) {
                        requested = cvar.wait(requested).unwrap();
                    }
                    if shut.load(Ordering::SeqCst) {
                        log::info!("Frame-wait worker shutting down");
                        return;
                    }
                    *requested = false;
                }

                // Blocking OpenXR wait — this is the whole reason the thread exists
                let wait_result = {
                    let mut xr_guard = get_xr_state().lock().unwrap();
                    match xr_guard.as_mut() {
                        Some(state) => {
                            state.frame_waiter.wait()
                                .map_err(|e| format!("Wait failed: {}", e))
                        }
                        None => Err("No XR session".to_string()),
                    }
                };

                // Publish result (lock-free ready flag for fast polling)
                *get_frame_wait_result().lock().unwrap() = Some(wait_result);
                FRAME_WAIT_READY.store(true, Ordering::SeqCst);
            }
        })
        .expect("Failed to spawn xr-frame-wait thread");

    *guard = Some(FrameWaitWorker { request, shutdown });
}

/// Shut down the persistent worker (called on session end).
fn shutdown_frame_wait_worker() {
    let mut guard = get_frame_wait_worker().lock().unwrap();
    if let Some(worker) = guard.take() {
        worker.shutdown.store(true, Ordering::SeqCst);
        let (_, cvar) = &*worker.request;
        cvar.notify_one();
    }
}

// ======================= Helper Functions =======================

const VK_TARGET_VERSION: u32 = vk::make_api_version(0, 1, 1, 0);

fn fov_to_projection_matrix(fov: &xr::Fovf, near: f32, far: f32) -> [f32; 16] {
    let tan_left = fov.angle_left.tan();
    let tan_right = fov.angle_right.tan();
    let tan_up = fov.angle_up.tan();
    let tan_down = fov.angle_down.tan();
    
    let tan_width = tan_right - tan_left;
    let tan_height = tan_up - tan_down;
    
    let a11 = 2.0 / tan_width;
    let a22 = 2.0 / tan_height;
    let a31 = (tan_right + tan_left) / tan_width;
    let a32 = (tan_up + tan_down) / tan_height;
    let a33 = -far / (far - near);
    let a43 = -(far * near) / (far - near);
    
    [
        a11, 0.0, 0.0, 0.0,
        0.0, a22, 0.0, 0.0,
        a31, a32, a33, -1.0,
        0.0, 0.0, a43, 0.0,
    ]
}

fn pose_to_matrix(pose: &xr::Posef) -> [f32; 16] {
    let p = pose.position;
    let q = pose.orientation;
    
    let x = q.x;
    let y = q.y;
    let z = q.z;
    let w = q.w;
    
    let x2 = x + x;
    let y2 = y + y;
    let z2 = z + z;
    
    let xx = x * x2;
    let xy = x * y2;
    let xz = x * z2;
    let yy = y * y2;
    let yz = y * z2;
    let zz = z * z2;
    let wx = w * x2;
    let wy = w * y2;
    let wz = w * z2;
    
    [
        1.0 - (yy + zz), xy + wz,         xz - wy,         0.0,
        xy - wz,         1.0 - (xx + zz), yz + wx,         0.0,
        xz + wy,         yz - wx,         1.0 - (xx + yy), 0.0,
        p.x,             p.y,             p.z,             1.0,
    ]
}

pub fn get_xr_swapchain_texture(index: u32) -> Option<&'static wgpu::Texture> {
    XR_SWAPCHAIN_TEXTURES.get().and_then(|textures| textures.get(index as usize))
}

// ======================= OPS =======================

#[op2]
pub async fn op_xr_is_supported() -> Result<bool, JsErrorBox> {
    match unsafe { xr::Entry::load() }.ok().and_then(|e| e.enumerate_extensions().ok()) {
        Some(exts) => Ok(exts.khr_vulkan_enable2),
        None => Ok(false),
    }
}

pub async fn init_xr_session_internal() -> Result<XrSessionInfo, JsErrorBox> {
    log::info!("Requesting XR session...");
    
    #[cfg(target_os = "android")]
    let xr_entry = {
        let entry = unsafe { xr::Entry::load() }
            .map_err(|e| JsErrorBox::generic(format!("Failed to load OpenXR: {}", e)))?;
        init_android_loader(&entry)?;
        entry
    };

    #[cfg(not(target_os = "android"))]
    let xr_entry = unsafe { xr::Entry::load() }
        .map_err(|e| JsErrorBox::generic(format!("Failed to load OpenXR: {}", e)))?;

    #[cfg(target_os = "android")]
    let vk_entry = unsafe { ash::Entry::load() }.expect("Failed to load Vulkan");

    #[cfg(not(target_os = "android"))]
    let vk_entry = unsafe { ash::Entry::load() }
        .map_err(|e| JsErrorBox::generic(format!("Failed to load Vulkan: {}", e)))?;
    
    let mut enabled_extensions = xr::ExtensionSet::default();
    enabled_extensions.khr_vulkan_enable2 = true;

    let instance = xr_entry.create_instance(
        &xr::ApplicationInfo {
            application_name: "App",
            application_version: 1,
            engine_name: "Engine",
            engine_version: 1,
            api_version: xr::Version::new(1, 0, 0),
        },
        &enabled_extensions,
        &[],
    ).map_err(|e| JsErrorBox::generic(format!("Failed to create XR instance: {}", e)))?;
    
    let system = instance.system(xr::FormFactor::HEAD_MOUNTED_DISPLAY)
        .map_err(|e| JsErrorBox::generic(format!("No HMD found: {}", e)))?;
    
    let view_config_type = xr::ViewConfigurationType::PRIMARY_STEREO;
    let views = instance.enumerate_view_configuration_views(system, view_config_type)
        .map_err(|e| JsErrorBox::generic(format!("Failed to get view config: {}", e)))?;
    
    let width = views[0].recommended_image_rect_width;
    let height = views[0].recommended_image_rect_height;
    
    let _reqs = instance.graphics_requirements::<xr::Vulkan>(system)
        .map_err(|e| JsErrorBox::generic(format!("Graphics requirements error: {}", e)))?;
    
    let vk_entry = unsafe { ash::Entry::load() }
        .map_err(|e| JsErrorBox::generic(format!("Failed to load Vulkan: {}", e)))?;
    
    let flags = wgpu::InstanceFlags::default();

    let instance_exts = <wgpu_hal::vulkan::Api as wgpu_hal::Api>::Instance::desired_extensions(
        &vk_entry,
        VK_TARGET_VERSION,
        flags,
    ).map_err(|e| JsErrorBox::generic(format!("Failed to get instance extensions: {}", e)))?;
    
    let extensions_cchar: Vec<_> = instance_exts.iter().map(|s| s.as_ptr()).collect();
    
    let app_name = CString::new("App").unwrap();
    
    let vk_app_info = vk::ApplicationInfo::default()
        .application_name(&app_name)
        .application_version(1)
        .engine_name(&app_name)
        .engine_version(1)
        .api_version(VK_TARGET_VERSION);
    
    let vk_create_info = vk::InstanceCreateInfo::default()
        .application_info(&vk_app_info)
        .enabled_extension_names(&extensions_cchar);
    
    let vk_instance = unsafe {
        let get_instance_proc_addr = vk_entry.static_fn().get_instance_proc_addr;
        let vk_instance_raw = instance
            .create_vulkan_instance(
                system,
                std::mem::transmute(get_instance_proc_addr),
                &vk_create_info as *const _ as *const std::ffi::c_void,
            )
            .map_err(|e| JsErrorBox::generic(format!("Failed to create Vulkan instance: {}", e)))?
            .map_err(|e| JsErrorBox::generic(format!("Vulkan error: {:?}", e)))?;
        
        ash::Instance::load(
            vk_entry.static_fn(),
            vk::Instance::from_raw(vk_instance_raw as _),
        )
    };
    
    log::info!("Vulkan instance created via OpenXR");
    
    let vk_physical_device = vk::PhysicalDevice::from_raw(unsafe {
        instance.vulkan_graphics_device(system, vk_instance.handle().as_raw() as _)
            .map_err(|e| JsErrorBox::generic(format!("Failed to get Vulkan physical device: {}", e)))? as _
    });
    
    let vk_device_properties = unsafe { 
        vk_instance.get_physical_device_properties(vk_physical_device) 
    };
    
    let wgpu_vk_instance = unsafe {
        <wgpu_hal::vulkan::Api as wgpu_hal::Api>::Instance::from_raw(
            vk_entry.clone(),
            vk_instance.clone(),
            vk_device_properties.api_version,
            0,
            None,
            instance_exts.clone(),
            flags,
            wgpu::MemoryBudgetThresholds::default(),
            false,
            None,
        )
    }.map_err(|e| JsErrorBox::generic(format!("Failed to create wgpu-hal instance: {}", e)))?;
    
    let wgpu_exposed_adapter = wgpu_vk_instance
        .expose_adapter(vk_physical_device)
        .ok_or_else(|| JsErrorBox::generic("Failed to expose adapter"))?;
    
    let wgpu_features = wgpu_exposed_adapter.features | wgpu::Features::PASSTHROUGH_SHADERS;

    let enabled_device_extensions = wgpu_exposed_adapter
        .adapter
        .required_device_extensions(wgpu_features);
    
    let queue_families = unsafe { 
        vk_instance.get_physical_device_queue_family_properties(vk_physical_device) 
    };
    let queue_family_index = queue_families
        .iter()
        .enumerate()
        .find(|(_, props)| props.queue_flags.contains(vk::QueueFlags::GRAPHICS))
        .map(|(i, _)| i as u32)
        .ok_or_else(|| JsErrorBox::generic("No graphics queue family"))?;
    
    let queue_priorities = [1.0f32];
    let queue_create_info = vk::DeviceQueueCreateInfo::default()
        .queue_family_index(queue_family_index)
        .queue_priorities(&queue_priorities);
    
    let mut enabled_phd_features = wgpu_exposed_adapter
        .adapter
        .physical_device_features(&enabled_device_extensions, wgpu_features);
    
    let family_infos = [queue_create_info];
    
    let mut multiview_features = vk::PhysicalDeviceMultiviewFeatures {
        multiview: vk::TRUE,
        ..Default::default()
    };
    
    let device_extensions: Vec<_> = enabled_device_extensions.iter().map(|s| s.as_ptr()).collect();
    
    let device_create_info = enabled_phd_features
    .add_to_device_create(
        vk::DeviceCreateInfo::default()
            .queue_create_infos(&family_infos)
            .push_next(&mut multiview_features),
    )
    .enabled_extension_names(&device_extensions);
    
    let vk_device = unsafe {
        let get_instance_proc_addr = vk_entry.static_fn().get_instance_proc_addr;
        let vk_device_raw = instance
            .create_vulkan_device(
                system,
                std::mem::transmute(get_instance_proc_addr),
                vk_physical_device.as_raw() as _,
                &device_create_info as *const _ as *const std::ffi::c_void,
            )
            .map_err(|e| JsErrorBox::generic(format!("Failed to create Vulkan device: {}", e)))?
            .map_err(|e| JsErrorBox::generic(format!("Vulkan device error: {:?}", e)))?;
        
        ash::Device::load(vk_instance.fp_v1_0(), vk::Device::from_raw(vk_device_raw as _))
    };
    
    let wgpu_open_device = unsafe {
        wgpu_exposed_adapter.adapter.device_from_raw(
            vk_device.clone(),
            None,
            &enabled_device_extensions,
            wgpu_features,
            &wgpu::Limits::default(),
            &wgpu::MemoryHints::Performance,
            queue_family_index,
            0,
        )
    }.map_err(|e| JsErrorBox::generic(format!("Failed to create wgpu device: {}", e)))?;
    
    let wgpu_instance = unsafe { 
        wgpu::Instance::from_hal::<wgpu_hal::api::Vulkan>(wgpu_vk_instance) 
    };
    let wgpu_adapter = unsafe { 
        wgpu_instance.create_adapter_from_hal(wgpu_exposed_adapter) 
    };

    let limits = wgpu_adapter.limits();
    let (wgpu_device, wgpu_queue) = unsafe {
        wgpu_adapter.create_device_from_hal(
            wgpu_open_device,
            &wgpu::DeviceDescriptor {
                label: Some("XR Device"),
                required_features: wgpu_features,
                required_limits: limits,
                memory_hints: wgpu::MemoryHints::Performance,
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
                trace: wgpu::Trace::Off,
            },
        )
    }.map_err(|e| JsErrorBox::generic(format!("Failed to finalize wgpu device: {}", e)))?;
    
    let (session, frame_waiter, frame_stream) = unsafe {
        instance.create_session::<xr::Vulkan>(
            system,
            &xr::vulkan::SessionCreateInfo {
                instance: vk_instance.handle().as_raw() as *const std::ffi::c_void,
                physical_device: vk_physical_device.as_raw() as *const std::ffi::c_void,
                device: vk_device.handle().as_raw() as *const std::ffi::c_void,
                queue_family_index,
                queue_index: 0,
            },
        )
    }.map_err(|e| JsErrorBox::generic(format!("Failed to create XR session: {}", e)))?;
    
    let stage_space = session.create_reference_space(
        xr::ReferenceSpaceType::STAGE,
        xr::Posef::IDENTITY,
    ).or_else(|_| {
        session.create_reference_space(xr::ReferenceSpaceType::LOCAL, xr::Posef::IDENTITY)
    }).map_err(|e| JsErrorBox::generic(format!("Failed to create reference space: {}", e)))?;
    
    // ======================= Controller Actions =======================
    let left_hand_path = instance.string_to_path("/user/hand/left").unwrap();
    let right_hand_path = instance.string_to_path("/user/hand/right").unwrap();
    
    let action_set = instance.create_action_set("gameplay", "Gameplay", 0)
        .map_err(|e| JsErrorBox::generic(format!("Failed to create action set: {}", e)))?;
    
    let left_grip_action = action_set.create_action::<xr::Posef>("left_grip", "Left Grip Pose", &[left_hand_path]).map_err(|e| JsErrorBox::generic(format!("Action error: {}", e)))?;
    let right_grip_action = action_set.create_action::<xr::Posef>("right_grip", "Right Grip Pose", &[right_hand_path]).map_err(|e| JsErrorBox::generic(format!("Action error: {}", e)))?;
    let left_aim_action = action_set.create_action::<xr::Posef>("left_aim", "Left Aim Pose", &[left_hand_path]).map_err(|e| JsErrorBox::generic(format!("Action error: {}", e)))?;
    let right_aim_action = action_set.create_action::<xr::Posef>("right_aim", "Right Aim Pose", &[right_hand_path]).map_err(|e| JsErrorBox::generic(format!("Action error: {}", e)))?;
    let left_trigger_action = action_set.create_action::<f32>("left_trigger", "Left Trigger", &[left_hand_path]).map_err(|e| JsErrorBox::generic(format!("Action error: {}", e)))?;
    let right_trigger_action = action_set.create_action::<f32>("right_trigger", "Right Trigger", &[right_hand_path]).map_err(|e| JsErrorBox::generic(format!("Action error: {}", e)))?;
    let left_squeeze_action = action_set.create_action::<f32>("left_squeeze", "Left Squeeze", &[left_hand_path]).map_err(|e| JsErrorBox::generic(format!("Action error: {}", e)))?;
    let right_squeeze_action = action_set.create_action::<f32>("right_squeeze", "Right Squeeze", &[right_hand_path]).map_err(|e| JsErrorBox::generic(format!("Action error: {}", e)))?;
    let left_thumbstick_action = action_set.create_action::<xr::Vector2f>("left_thumbstick", "Left Thumbstick", &[left_hand_path]).map_err(|e| JsErrorBox::generic(format!("Action error: {}", e)))?;
    let right_thumbstick_action = action_set.create_action::<xr::Vector2f>("right_thumbstick", "Right Thumbstick", &[right_hand_path]).map_err(|e| JsErrorBox::generic(format!("Action error: {}", e)))?;
    let left_thumbstick_click_action = action_set.create_action::<bool>("left_thumbstick_click", "Left Thumbstick Click", &[left_hand_path]).map_err(|e| JsErrorBox::generic(format!("Action error: {}", e)))?;
    let right_thumbstick_click_action = action_set.create_action::<bool>("right_thumbstick_click", "Right Thumbstick Click", &[right_hand_path]).map_err(|e| JsErrorBox::generic(format!("Action error: {}", e)))?;
    let left_x_button_action = action_set.create_action::<bool>("left_x", "Left X Button", &[left_hand_path]).map_err(|e| JsErrorBox::generic(format!("Action error: {}", e)))?;
    let left_y_button_action = action_set.create_action::<bool>("left_y", "Left Y Button", &[left_hand_path]).map_err(|e| JsErrorBox::generic(format!("Action error: {}", e)))?;
    let right_a_button_action = action_set.create_action::<bool>("right_a", "Right A Button", &[right_hand_path]).map_err(|e| JsErrorBox::generic(format!("Action error: {}", e)))?;
    let right_b_button_action = action_set.create_action::<bool>("right_b", "Right B Button", &[right_hand_path]).map_err(|e| JsErrorBox::generic(format!("Action error: {}", e)))?;
    let left_haptic_action = action_set.create_action::<xr::Haptic>("left_haptic", "Left Haptic", &[left_hand_path]).map_err(|e| JsErrorBox::generic(format!("Action error: {}", e)))?;
    let right_haptic_action = action_set.create_action::<xr::Haptic>("right_haptic", "Right Haptic", &[right_hand_path]).map_err(|e| JsErrorBox::generic(format!("Action error: {}", e)))?;

    let oculus_profile = instance.string_to_path("/interaction_profiles/oculus/touch_controller").unwrap();
    
    instance.suggest_interaction_profile_bindings(oculus_profile, &[
        xr::Binding::new(&left_grip_action, instance.string_to_path("/user/hand/left/input/grip/pose").unwrap()),
        xr::Binding::new(&right_grip_action, instance.string_to_path("/user/hand/right/input/grip/pose").unwrap()),
        xr::Binding::new(&left_aim_action, instance.string_to_path("/user/hand/left/input/aim/pose").unwrap()),
        xr::Binding::new(&right_aim_action, instance.string_to_path("/user/hand/right/input/aim/pose").unwrap()),
        xr::Binding::new(&left_trigger_action, instance.string_to_path("/user/hand/left/input/trigger/value").unwrap()),
        xr::Binding::new(&right_trigger_action, instance.string_to_path("/user/hand/right/input/trigger/value").unwrap()),
        xr::Binding::new(&left_squeeze_action, instance.string_to_path("/user/hand/left/input/squeeze/value").unwrap()),
        xr::Binding::new(&right_squeeze_action, instance.string_to_path("/user/hand/right/input/squeeze/value").unwrap()),
        xr::Binding::new(&left_thumbstick_action, instance.string_to_path("/user/hand/left/input/thumbstick").unwrap()),
        xr::Binding::new(&right_thumbstick_action, instance.string_to_path("/user/hand/right/input/thumbstick").unwrap()),
        xr::Binding::new(&left_thumbstick_click_action, instance.string_to_path("/user/hand/left/input/thumbstick/click").unwrap()),
        xr::Binding::new(&right_thumbstick_click_action, instance.string_to_path("/user/hand/right/input/thumbstick/click").unwrap()),
        xr::Binding::new(&left_x_button_action, instance.string_to_path("/user/hand/left/input/x/click").unwrap()),
        xr::Binding::new(&left_y_button_action, instance.string_to_path("/user/hand/left/input/y/click").unwrap()),
        xr::Binding::new(&right_a_button_action, instance.string_to_path("/user/hand/right/input/a/click").unwrap()),
        xr::Binding::new(&right_b_button_action, instance.string_to_path("/user/hand/right/input/b/click").unwrap()),
        xr::Binding::new(&left_haptic_action, instance.string_to_path("/user/hand/left/output/haptic").unwrap()),
        xr::Binding::new(&right_haptic_action, instance.string_to_path("/user/hand/right/output/haptic").unwrap()),
    ]).map_err(|e| JsErrorBox::generic(format!("Failed to suggest bindings: {}", e)))?;
    
    let left_grip_space = left_grip_action.create_space(session.clone(), left_hand_path, xr::Posef::IDENTITY).map_err(|e| JsErrorBox::generic(format!("Space error: {}", e)))?;
    let right_grip_space = right_grip_action.create_space(session.clone(), right_hand_path, xr::Posef::IDENTITY).map_err(|e| JsErrorBox::generic(format!("Space error: {}", e)))?;
    let left_aim_space = left_aim_action.create_space(session.clone(), left_hand_path, xr::Posef::IDENTITY).map_err(|e| JsErrorBox::generic(format!("Space error: {}", e)))?;
    let right_aim_space = right_aim_action.create_space(session.clone(), right_hand_path, xr::Posef::IDENTITY).map_err(|e| JsErrorBox::generic(format!("Space error: {}", e)))?;
    
    session.attach_action_sets(&[&action_set])
        .map_err(|e| JsErrorBox::generic(format!("Failed to attach action sets: {}", e)))?;
    
    // ======================= Swapchain =======================
    let swapchain = session.create_swapchain(&openxr::SwapchainCreateInfo {
        create_flags: openxr::SwapchainCreateFlags::EMPTY,
        usage_flags: openxr::SwapchainUsageFlags::COLOR_ATTACHMENT | openxr::SwapchainUsageFlags::SAMPLED,
        format: vk::Format::R8G8B8A8_SRGB.as_raw() as u32,
        sample_count: 1,
        width,
        height,
        face_count: 1,
        array_size: 2,
        mip_count: 1,
    }).map_err(|e| JsErrorBox::generic(format!("Failed to create swapchain: {}", e)))?;

    // ======================= Enumerate Images =======================
    let swapchain_images = swapchain.enumerate_images()
        .map_err(|e| JsErrorBox::generic(format!("Failed to enumerate swapchain images: {}", e)))?;
    let swapchain_length = swapchain_images.len() as u32;

    // ======================= Create wgpu Textures =======================
    let swapchain_textures: Vec<wgpu::Texture> = swapchain_images
        .iter()
        .map(|&vk_image_raw| {
            let vk_image = vk::Image::from_raw(vk_image_raw);
            
            let wgpu_hal_texture = unsafe {
                wgpu_device
                    .as_hal::<wgpu_hal::vulkan::Api>()
                    .ok_or_else(|| JsErrorBox::generic("Failed to get HAL device"))?
                    .texture_from_raw(
                        vk_image,
                        &wgpu_hal::TextureDescriptor {
                            label: Some("XR Swapchain"),
                            size: wgpu::Extent3d { width, height, depth_or_array_layers: 2 },
                            mip_level_count: 1,
                            sample_count: 1,
                            dimension: wgpu::TextureDimension::D2,
                            format: wgpu::TextureFormat::Rgba8UnormSrgb,
                            usage: wgpu::TextureUses::COLOR_TARGET | wgpu::TextureUses::COPY_DST,
                            memory_flags: wgpu_hal::MemoryFlags::empty(),
                            view_formats: vec![],
                        },
                        None,
                        TextureMemory::External,
                    )
            };
            
            let texture = unsafe {
                wgpu_device.create_texture_from_hal::<wgpu_hal::vulkan::Api>(
                    wgpu_hal_texture,
                    &wgpu::TextureDescriptor {
                        label: Some("XR Swapchain"),
                        size: wgpu::Extent3d { width, height, depth_or_array_layers: 2 },
                        mip_level_count: 1,
                        sample_count: 1,
                        dimension: wgpu::TextureDimension::D2,
                        format: wgpu::TextureFormat::Rgba8UnormSrgb,
                        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_DST,
                        view_formats: &[],
                    },
                )
            };
            
            Ok(texture)
        })
        .collect::<Result<Vec<_>, JsErrorBox>>()?;
  

    // ======================= Set GfxContext =======================
    let format = wgpu::TextureFormat::Rgba8UnormSrgb;
    let gfx_box = Box::new(GfxContext {
        device: wgpu_device,
        queue: wgpu_queue,
        surface: None,
        format,
    });
    let gfx_static: &'static GfxContext = Box::leak(gfx_box);
    set_gfx_context(gfx_static)
        .map_err(|_| JsErrorBox::generic("GfxContext already set"))?;
    
    let state = XrState {
        _vk_entry: vk_entry,
        _vk_instance: vk_instance,
        _vk_device: vk_device,
        instance,
        system,
        session,
        frame_waiter,
        frame_stream,
        stage_space,
        view_config_type,
        views,
        swapchain,
        swapchain_textures: vec![],
        session_running: false,
        session_state: xr::SessionState::IDLE,
        frame_state: None,
        current_swapchain_index: None,
        render_width: width,
        render_height: height,
        cached_views: None,
        action_set,
        left_grip_action,
        right_grip_action,
        left_aim_action,
        right_aim_action,
        left_grip_space,
        right_grip_space,
        left_aim_space,
        right_aim_space,
        left_trigger_action,
        right_trigger_action,
        left_squeeze_action,
        right_squeeze_action,
        left_thumbstick_action,
        right_thumbstick_action,
        left_thumbstick_click_action,
        right_thumbstick_click_action,
        left_x_button_action,
        left_y_button_action,
        right_a_button_action,
        right_b_button_action,
        left_hand_path,
        right_hand_path,
        left_haptic_action,
        right_haptic_action,
        depth_near: 0.2,
        depth_far: 550.0,
    };
    
    *get_xr_state().lock().unwrap() = Some(state);

    // Pre-create texture views for all swapchain images and both eyes
    // This avoids per-frame view creation overhead
    let swapchain_views: Vec<[std::sync::Arc<wgpu::TextureView>; 2]> = swapchain_textures
        .iter()
        .map(|texture| {
            let left_view = std::sync::Arc::new(texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("XR Swapchain View Left"),
                format: Some(wgpu::TextureFormat::Rgba8UnormSrgb),
                dimension: Some(wgpu::TextureViewDimension::D2),
                aspect: wgpu::TextureAspect::All,
                base_mip_level: 0,
                mip_level_count: Some(1),
                base_array_layer: 0,
                array_layer_count: Some(1),
                usage: None,
            }));
            let right_view = std::sync::Arc::new(texture.create_view(&wgpu::TextureViewDescriptor {
                label: Some("XR Swapchain View Right"),
                format: Some(wgpu::TextureFormat::Rgba8UnormSrgb),
                dimension: Some(wgpu::TextureViewDimension::D2),
                aspect: wgpu::TextureAspect::All,
                base_mip_level: 0,
                mip_level_count: Some(1),
                base_array_layer: 1,
                array_layer_count: Some(1),
                usage: None,
            }));
            [left_view, right_view]
        })
        .collect();

    let _ = XR_SWAPCHAIN_TEXTURES.set(swapchain_textures);
    let _ = XR_SWAPCHAIN_VIEWS.set(swapchain_views);

    log::info!("XR session created successfully");
    
    Ok(XrSessionInfo {
        width,
        height,
        sample_count: 1,
        format: "rgba8unorm-srgb".to_string(),
        swapchain_length,
    })
}

#[op2]
#[serde]
pub async fn op_xr_request_session() -> Result<XrSessionInfo, JsErrorBox> {
    init_xr_session_internal().await
}

#[op2(fast)]
pub fn op_xr_poll_events() -> Result<bool, JsErrorBox> {
    let mut guard = get_xr_state().lock().unwrap();
    let state = guard.as_mut().ok_or_else(|| JsErrorBox::generic("No XR session"))?;
    
    let mut event_buffer = xr::EventDataBuffer::new();
    
    while let Some(event) = state.instance.poll_event(&mut event_buffer)
        .map_err(|e| JsErrorBox::generic(format!("Poll error: {}", e)))? 
    {
        if let xr::Event::SessionStateChanged(e) = event {
            state.session_state = e.state();
            log::info!("XR state: {:?}", e.state());
            
            match e.state() {
                xr::SessionState::READY => {
                    state.session.begin(state.view_config_type)
                        .map_err(|e| JsErrorBox::generic(format!("Begin failed: {}", e)))?;
                    state.session_running = true;
                }
                xr::SessionState::STOPPING => {
                    shutdown_frame_wait_worker();
                    state.session.end()
                        .map_err(|e| JsErrorBox::generic(format!("End failed: {}", e)))?;
                    state.session_running = false;
                }
                xr::SessionState::EXITING | xr::SessionState::LOSS_PENDING => {
                    shutdown_frame_wait_worker();
                    log::info!("XR session exiting, terminating process immediately");
                    std::process::exit(0);
                }
                _ => {}
            }
        }
    }
    
    Ok(!matches!(state.session_state, xr::SessionState::EXITING | xr::SessionState::LOSS_PENDING))
}


#[op2]
#[serde]
pub fn op_xr_wait_frame() -> Result<XrFrameState, JsErrorBox> {
    let mut guard = get_xr_state().lock().unwrap();
    let state = guard.as_mut().ok_or_else(|| JsErrorBox::generic("No XR session"))?;
    
    if !state.session_running {
        return Ok(XrFrameState { predicted_display_time: 0, should_render: false });
    }
    
    let frame_state = state.frame_waiter.wait()
        .map_err(|e| JsErrorBox::generic(format!("Wait failed: {}", e)))?;
    
    state.frame_stream.begin()
        .map_err(|e| JsErrorBox::generic(format!("Begin failed: {}", e)))?;
    
    state.frame_state = Some(frame_state);
    
    Ok(XrFrameState {
        predicted_display_time: frame_state.predicted_display_time.as_nanos(),
        should_render: frame_state.should_render,
    })
}

#[op2(fast)]
pub fn op_xr_set_clip_planes(near: f32, far: f32) -> Result<(), JsErrorBox> {
    let mut guard = get_xr_state().lock().unwrap();
    let state = guard.as_mut().ok_or_else(|| JsErrorBox::generic("No XR session"))?;
    state.depth_near = near;
    state.depth_far = far;
    Ok(())
}

#[op2]
#[serde]
pub fn op_xr_get_viewer_pose() -> Result<XrViewerPose, JsErrorBox> {
    let mut guard = get_xr_state().lock().unwrap();
    let state = guard.as_mut().ok_or_else(|| JsErrorBox::generic("No XR session"))?;
    
    let frame_state = state.frame_state.as_ref()
        .ok_or_else(|| JsErrorBox::generic("No frame state"))?;
    
    let (_, views) = state.session.locate_views(
        state.view_config_type,
        frame_state.predicted_display_time,
        &state.stage_space,
    ).map_err(|e| JsErrorBox::generic(format!("Locate views failed: {}", e)))?;
    
    let w = state.render_width;
    let h = state.render_height;
    
    let xr_views: Vec<XrView> = views.iter().enumerate().map(|(i, view)| {
        XrView {
            projection_matrix: fov_to_projection_matrix(&view.fov, state.depth_near, state.depth_far),
            transform: XrPose {
                position: [view.pose.position.x, view.pose.position.y, view.pose.position.z],
                orientation: [view.pose.orientation.x, view.pose.orientation.y, view.pose.orientation.z, view.pose.orientation.w],
                matrix: pose_to_matrix(&view.pose),
            },
            view_index: i as u32,
            viewport_x: 0,
            viewport_y: 0,
            viewport_width: w,
            viewport_height: h,
        }
    }).collect();
    
    state.cached_views = Some(views);
    
    Ok(XrViewerPose { views: xr_views })
}

fn get_pose(space: &xr::Space, stage: &xr::Space, time: xr::Time) -> Option<XrPose> {
    space.locate(stage, time).ok()
        .filter(|loc| loc.location_flags.contains(
            xr::SpaceLocationFlags::POSITION_VALID | xr::SpaceLocationFlags::ORIENTATION_VALID
        ))
        .map(|loc| XrPose {
            position: [loc.pose.position.x, loc.pose.position.y, loc.pose.position.z],
            orientation: [loc.pose.orientation.x, loc.pose.orientation.y, loc.pose.orientation.z, loc.pose.orientation.w],
            matrix: pose_to_matrix(&loc.pose),
        })
}

fn build_controller_source(
    session: &xr::Session<xr::Vulkan>,
    grip_space: &xr::Space,
    aim_space: &xr::Space,
    stage_space: &xr::Space,
    time: xr::Time,
    trigger_action: &xr::Action<f32>,
    squeeze_action: &xr::Action<f32>,
    thumbstick_action: &xr::Action<xr::Vector2f>,
    stick_click_action: &xr::Action<bool>,
    face_button_1: &xr::Action<bool>,
    face_button_2: &xr::Action<bool>,
    handedness: &str,
) -> Option<XrInputSourceData> {
    let grip = get_pose(grip_space, stage_space, time);
    let aim = get_pose(aim_space, stage_space, time);

    if grip.is_none() && aim.is_none() {
        return None;
    }

    let trigger = trigger_action.state(session, xr::Path::NULL)
        .map(|s| s.current_state).unwrap_or(0.0);
    let squeeze = squeeze_action.state(session, xr::Path::NULL)
        .map(|s| s.current_state).unwrap_or(0.0);
    let thumbstick = thumbstick_action.state(session, xr::Path::NULL)
        .map(|s| (s.current_state.x, s.current_state.y)).unwrap_or((0.0, 0.0));
    let stick_click = stick_click_action.state(session, xr::Path::NULL)
        .map(|s| s.current_state).unwrap_or(false);
    let btn1 = face_button_1.state(session, xr::Path::NULL)
        .map(|s| s.current_state).unwrap_or(false);
    let btn2 = face_button_2.state(session, xr::Path::NULL)
        .map(|s| s.current_state).unwrap_or(false);

    Some(XrInputSourceData {
        handedness: handedness.to_string(),
        target_ray_mode: "tracked-pointer".to_string(),
        profiles: vec!["oculus-touch-v3".to_string(), "oculus-touch".to_string(), "generic-trigger-squeeze-thumbstick".to_string()],
        grip_space_pose: grip,
        target_ray_pose: aim,
        gamepad: Some(XrGamepadData {
            buttons: vec![
                XrButtonData { pressed: trigger > 0.9, touched: trigger > 0.0, value: trigger },
                XrButtonData { pressed: squeeze > 0.9, touched: squeeze > 0.0, value: squeeze },
                XrButtonData { pressed: false, touched: false, value: 0.0 },
                XrButtonData { pressed: stick_click, touched: thumbstick.0.abs() > 0.01 || thumbstick.1.abs() > 0.01, value: if stick_click { 1.0 } else { 0.0 } },
                XrButtonData { pressed: btn1, touched: btn1, value: if btn1 { 1.0 } else { 0.0 } },
                XrButtonData { pressed: btn2, touched: btn2, value: if btn2 { 1.0 } else { 0.0 } },
            ],
            axes: vec![0.0, 0.0, thumbstick.0, -thumbstick.1],
        }),
        hand: None,
    })
}

fn collect_input_sources(state: &XrState, time: xr::Time) -> Vec<XrInputSourceData> {
    let active_action_set = xr::ActiveActionSet::new(&state.action_set);
    if state.session.sync_actions(&[active_action_set]).is_err() {
        return vec![];
    }

    let mut sources = Vec::with_capacity(2);
    if let Some(src) = build_controller_source(
        &state.session, &state.left_grip_space, &state.left_aim_space, &state.stage_space, time,
        &state.left_trigger_action, &state.left_squeeze_action, &state.left_thumbstick_action,
        &state.left_thumbstick_click_action, &state.left_x_button_action, &state.left_y_button_action, "left",
    ) { sources.push(src); }
    if let Some(src) = build_controller_source(
        &state.session, &state.right_grip_space, &state.right_aim_space, &state.stage_space, time,
        &state.right_trigger_action, &state.right_squeeze_action, &state.right_thumbstick_action,
        &state.right_thumbstick_click_action, &state.right_a_button_action, &state.right_b_button_action, "right",
    ) { sources.push(src); }
    sources
}

#[op2]
#[serde]
pub fn op_xr_get_input_sources() -> Result<XrInputSourcesState, JsErrorBox> {
    let mut guard = get_xr_state().lock().unwrap();
    let state = guard.as_mut().ok_or_else(|| JsErrorBox::generic("No XR session"))?;

    let frame_state = state.frame_state.as_ref()
        .ok_or_else(|| JsErrorBox::generic("No frame state"))?;

    let sources = collect_input_sources(state, frame_state.predicted_display_time);

    Ok(XrInputSourcesState { sources })
}

/// Combined frame begin operation - performs wait_frame, get_viewer_pose,
/// get_input_sources, and acquire_swapchain_image in a single lock acquisition.
/// This reduces mutex overhead from 4 separate locks to 1 per frame.
#[op2]
#[serde]
pub fn op_xr_frame_begin() -> Result<XrFrameBeginResult, JsErrorBox> {
    let mut guard = get_xr_state().lock().unwrap();
    let state = guard.as_mut().ok_or_else(|| JsErrorBox::generic("No XR session"))?;

    // === wait_frame ===
    if !state.session_running {
        return Ok(XrFrameBeginResult {
            frame_state: XrFrameState { predicted_display_time: 0, should_render: false },
            viewer_pose: None,
            input_sources: XrInputSourcesState { sources: Vec::new() },
            swapchain_info: None,
        });
    }

    let frame_state = state.frame_waiter.wait()
        .map_err(|e| JsErrorBox::generic(format!("Wait failed: {}", e)))?;

    state.frame_stream.begin()
        .map_err(|e| JsErrorBox::generic(format!("Begin failed: {}", e)))?;

    state.frame_state = Some(frame_state);

    let js_frame_state = XrFrameState {
        predicted_display_time: frame_state.predicted_display_time.as_nanos(),
        should_render: frame_state.should_render,
    };

    // If we shouldn't render, return early with minimal data
    if !frame_state.should_render {
        return Ok(XrFrameBeginResult {
            frame_state: js_frame_state,
            viewer_pose: None,
            input_sources: XrInputSourcesState { sources: Vec::new() },
            swapchain_info: None,
        });
    }

    // === get_viewer_pose ===
    let (_, views) = state.session.locate_views(
        state.view_config_type,
        frame_state.predicted_display_time,
        &state.stage_space,
    ).map_err(|e| JsErrorBox::generic(format!("Locate views failed: {}", e)))?;

    let w = state.render_width;
    let h = state.render_height;

    let xr_views: Vec<XrView> = views.iter().enumerate().map(|(i, view)| {
        XrView {
            projection_matrix: fov_to_projection_matrix(&view.fov, state.depth_near, state.depth_far),
            transform: XrPose {
                position: [view.pose.position.x, view.pose.position.y, view.pose.position.z],
                orientation: [view.pose.orientation.x, view.pose.orientation.y, view.pose.orientation.z, view.pose.orientation.w],
                matrix: pose_to_matrix(&view.pose),
            },
            view_index: i as u32,
            viewport_x: 0,
            viewport_y: 0,
            viewport_width: w,
            viewport_height: h,
        }
    }).collect();

    state.cached_views = Some(views);
    let viewer_pose = Some(XrViewerPose { views: xr_views });

    // === get_input_sources ===
    let input_sources = XrInputSourcesState {
        sources: collect_input_sources(state, frame_state.predicted_display_time),
    };

    // === acquire_swapchain_image ===
    let index = state.swapchain.acquire_image()
        .map_err(|e| JsErrorBox::generic(format!("Acquire failed: {}", e)))?;

    state.swapchain.wait_image(xr::Duration::INFINITE)
        .map_err(|e| JsErrorBox::generic(format!("Wait failed: {}", e)))?;

    state.current_swapchain_index = Some(index);

    let swapchain_info = Some(XrSwapchainTextureInfo {
        index,
        width: state.render_width,
        height: state.render_height,
        array_layers: 2,
    });

    Ok(XrFrameBeginResult {
        frame_state: js_frame_state,
        viewer_pose,
        input_sources,
        swapchain_info,
    })
}

// ======================= Haptic Feedback =======================

/// Apply haptic vibration to a controller.
/// handedness: 0 = left, 1 = right
/// duration_ms: vibration duration in milliseconds (0 = minimum pulse)
/// frequency: vibration frequency in Hz (0 = default)
/// amplitude: vibration strength 0.0–1.0
#[op2(fast)]
pub fn op_xr_haptic_pulse(handedness: u32, duration_ms: f64, frequency: f32, amplitude: f32) -> Result<(), JsErrorBox> {
    let guard = get_xr_state().lock().unwrap();
    let state = guard.as_ref().ok_or_else(|| JsErrorBox::generic("No XR session"))?;

    let (action, subaction_path) = match handedness {
        0 => (&state.left_haptic_action, state.left_hand_path),
        1 => (&state.right_haptic_action, state.right_hand_path),
        _ => return Err(JsErrorBox::generic("Invalid handedness (0=left, 1=right)")),
    };

    let duration_nanos = if duration_ms <= 0.0 {
        xr::Duration::MIN_HAPTIC
    } else {
        xr::Duration::from_nanos((duration_ms * 1_000_000.0) as i64)
    };

    let vibration = xr::HapticVibration::new()
        .duration(duration_nanos)
        .frequency(frequency)
        .amplitude(amplitude.clamp(0.0, 1.0));

    action.apply_feedback(&state.session, subaction_path, &vibration)
        .map_err(|e| JsErrorBox::generic(format!("Haptic feedback failed: {}", e)))?;

    Ok(())
}

/// Stop haptic feedback on a controller.
/// handedness: 0 = left, 1 = right
#[op2(fast)]
pub fn op_xr_haptic_stop(handedness: u32) -> Result<(), JsErrorBox> {
    let guard = get_xr_state().lock().unwrap();
    let state = guard.as_ref().ok_or_else(|| JsErrorBox::generic("No XR session"))?;

    let (action, subaction_path) = match handedness {
        0 => (&state.left_haptic_action, state.left_hand_path),
        1 => (&state.right_haptic_action, state.right_hand_path),
        _ => return Err(JsErrorBox::generic("Invalid handedness (0=left, 1=right)")),
    };

    action.stop_feedback(&state.session, subaction_path)
        .map_err(|e| JsErrorBox::generic(format!("Stop haptic failed: {}", e)))?;

    Ok(())
}

#[op2]
#[serde]
pub fn op_xr_acquire_swapchain_image() -> Result<XrSwapchainTextureInfo, JsErrorBox> {
    let mut guard = get_xr_state().lock().unwrap();
    let state = guard.as_mut().ok_or_else(|| JsErrorBox::generic("No XR session"))?;
    
    let index = state.swapchain.acquire_image()
        .map_err(|e| JsErrorBox::generic(format!("Acquire failed: {}", e)))?;
    
    state.swapchain.wait_image(xr::Duration::INFINITE)
        .map_err(|e| JsErrorBox::generic(format!("Wait failed: {}", e)))?;
    
    state.current_swapchain_index = Some(index);


    
    Ok(XrSwapchainTextureInfo {
        index,
        width: state.render_width,
        height: state.render_height,
        array_layers: 2,
    })
}

#[op2(fast)]
pub fn op_xr_release_swapchain_image() -> Result<(), JsErrorBox> {
    let mut guard = get_xr_state().lock().unwrap();
    let state = guard.as_mut().ok_or_else(|| JsErrorBox::generic("No XR session"))?;
    
    
    state.swapchain.release_image()
        .map_err(|e| JsErrorBox::generic(format!("Release failed: {}", e)))?;
    
    state.current_swapchain_index = None;
    Ok(())
}


#[op2(fast)]
pub fn op_xr_end_frame() -> Result<(), JsErrorBox> {
    // Flush any pending command buffers before ending the frame
    // This ensures all GPU work is submitted before we release the swapchain
    flush_pending_command_buffers()?;

    let mut guard = get_xr_state().lock().unwrap();
    let state = guard.as_mut().ok_or_else(|| JsErrorBox::generic("No XR session"))?;
    
    let frame_state = state.frame_state.take()
        .ok_or_else(|| JsErrorBox::generic("No frame state"))?;
    
    if !frame_state.should_render {
        state.frame_stream.end(frame_state.predicted_display_time, xr::EnvironmentBlendMode::OPAQUE, &[])
            .map_err(|e| JsErrorBox::generic(format!("End frame failed: {}", e)))?;
        state.cached_views = None;
        return Ok(());
    }
    
    let views = match state.cached_views.take() {
        Some(v) => v,
        None => {
            // No views cached, submit empty frame
            state.frame_stream.end(frame_state.predicted_display_time, xr::EnvironmentBlendMode::OPAQUE, &[])
                .map_err(|e| JsErrorBox::generic(format!("End frame failed: {}", e)))?;
            return Ok(());
        }
    };
    
    // Validate poses - check quaternion is normalized
    for view in &views {
        let q = &view.pose.orientation;
        let len_sq = q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w;
        if len_sq < 0.9 || len_sq > 1.1 {
            // Invalid pose, submit empty frame
            log::warn!("Invalid pose orientation, submitting empty frame");
            state.frame_stream.end(frame_state.predicted_display_time, xr::EnvironmentBlendMode::OPAQUE, &[])
                .map_err(|e| JsErrorBox::generic(format!("End frame failed: {}", e)))?;
            return Ok(());
        }
    }
    
    let w = state.render_width as i32;
    let h = state.render_height as i32;
    
    let projection_views: Vec<xr::CompositionLayerProjectionView<xr::Vulkan>> = views
        .iter()
        .enumerate()
        .map(|(i, view)| {
            xr::CompositionLayerProjectionView::new()
                .pose(view.pose)
                .fov(view.fov)
                .sub_image(xr::SwapchainSubImage::new()
                    .swapchain(&state.swapchain)
                    .image_rect(xr::Rect2Di {
                        offset: xr::Offset2Di { x: 0, y: 0 },
                        extent: xr::Extent2Di { width: w, height: h },
                    })
                    .image_array_index(i as u32))
        })
        .collect();
    
    let layer = xr::CompositionLayerProjection::new()
        .space(&state.stage_space)
        .views(&projection_views);
    
    state.frame_stream.end(frame_state.predicted_display_time, xr::EnvironmentBlendMode::OPAQUE, &[&layer])
        .map_err(|e| JsErrorBox::generic(format!("End frame failed: {}", e)))?;
    
    Ok(())
}

#[op2(fast)]
pub fn op_xr_end_session() -> Result<(), JsErrorBox> {
    let mut guard = get_xr_state().lock().unwrap();
    if let Some(state) = guard.as_mut() {
        if state.session_running {
            let _ = state.session.request_exit();
        }
    }
    Ok(())
}

#[op2]
#[serde]
pub fn op_xr_get_swapchain_texture_view(
    state: &mut OpState,
    view_index: u32,
) -> Result<crate::gfx::JsGfxTextureView, JsErrorBox> {
    let guard = get_xr_state().lock().unwrap();
    let xr_state = guard.as_ref().ok_or_else(|| JsErrorBox::generic("No XR session"))?;

    let swapchain_index = xr_state.current_swapchain_index
        .ok_or_else(|| JsErrorBox::generic("No swapchain image acquired"))?;

    // Use pre-cached texture views instead of creating new ones each frame
    let cached_views = XR_SWAPCHAIN_VIEWS.get()
        .ok_or_else(|| JsErrorBox::generic("Swapchain views not initialized"))?;

    let views_for_image = cached_views.get(swapchain_index as usize)
        .ok_or_else(|| JsErrorBox::generic("Invalid swapchain index"))?;

    let view = views_for_image.get(view_index as usize)
        .ok_or_else(|| JsErrorBox::generic("Invalid view index (must be 0 or 1)"))?;

    let rid = crate::gfx::gpu_add(state, crate::gfx::GfxTextureView {
        view: view.clone(),
        width: xr_state.render_width,
        height: xr_state.render_height,
    });

    Ok(crate::gfx::JsGfxTextureView { rid })
}

// ======================= Async Frame Wait Ops =======================

/// Signal the persistent worker thread to start a new frame wait.
/// Returns true if started successfully, false if session not running or already waiting.
#[op2(fast)]
pub fn op_xr_frame_wait_start() -> Result<bool, JsErrorBox> {
    // Check if session is running
    {
        let guard = get_xr_state().lock().unwrap();
        match guard.as_ref() {
            Some(state) if state.session_running => {}
            _ => return Ok(false),
        }
    }

    // Previous result not yet consumed
    if FRAME_WAIT_READY.load(Ordering::SeqCst) {
        return Ok(false);
    }

    // Lazily create the worker thread on first call
    ensure_frame_wait_worker();

    // Signal the worker to start a new blocking wait
    let worker_guard = get_frame_wait_worker().lock().unwrap();
    if let Some(worker) = worker_guard.as_ref() {
        let (lock, cvar) = &*worker.request;
        let mut requested = lock.lock().unwrap();
        *requested = true;
        cvar.notify_one();
        Ok(true)
    } else {
        Ok(false)
    }
}

/// Poll for frame wait completion - returns true if ready.
/// Lock-free: just reads an atomic, no mutex on the hot path.
#[op2(fast)]
pub fn op_xr_frame_wait_poll() -> bool {
    FRAME_WAIT_READY.load(Ordering::SeqCst)
}

/// Finish the async frame wait and get the result.
/// This completes the frame_stream.begin() call and returns the frame begin result.
#[op2]
#[serde]
pub fn op_xr_frame_wait_finish() -> Result<XrFrameBeginResult, JsErrorBox> {
    // Take the result produced by the worker thread
    let frame_state_result = get_frame_wait_result().lock().unwrap().take()
        .ok_or_else(|| JsErrorBox::generic("Frame wait not complete"))?;

    // Reset ready flag so the next frame can start
    FRAME_WAIT_READY.store(false, Ordering::SeqCst);

    let frame_state = frame_state_result
        .map_err(|e| JsErrorBox::generic(e))?;

    // Now do the rest of the frame begin (same as op_xr_frame_begin but without wait)
    let mut guard = get_xr_state().lock().unwrap();
    let state = guard.as_mut().ok_or_else(|| JsErrorBox::generic("No XR session"))?;

    // Begin the frame stream
    state.frame_stream.begin()
        .map_err(|e| JsErrorBox::generic(format!("Begin failed: {}", e)))?;

    state.frame_state = Some(frame_state);

    let js_frame_state = XrFrameState {
        predicted_display_time: frame_state.predicted_display_time.as_nanos(),
        should_render: frame_state.should_render,
    };

    // If we shouldn't render, return early with minimal data
    if !frame_state.should_render {
        return Ok(XrFrameBeginResult {
            frame_state: js_frame_state,
            viewer_pose: None,
            input_sources: XrInputSourcesState { sources: Vec::new() },
            swapchain_info: None,
        });
    }

    // === get_viewer_pose ===
    let (_, views) = state.session.locate_views(
        state.view_config_type,
        frame_state.predicted_display_time,
        &state.stage_space,
    ).map_err(|e| JsErrorBox::generic(format!("Locate views failed: {}", e)))?;

    let w = state.render_width;
    let h = state.render_height;

    let xr_views: Vec<XrView> = views.iter().enumerate().map(|(i, view)| {
        XrView {
            projection_matrix: fov_to_projection_matrix(&view.fov, state.depth_near, state.depth_far),
            transform: XrPose {
                position: [view.pose.position.x, view.pose.position.y, view.pose.position.z],
                orientation: [view.pose.orientation.x, view.pose.orientation.y, view.pose.orientation.z, view.pose.orientation.w],
                matrix: pose_to_matrix(&view.pose),
            },
            view_index: i as u32,
            viewport_x: 0,
            viewport_y: 0,
            viewport_width: w,
            viewport_height: h,
        }
    }).collect();

    state.cached_views = Some(views);
    let viewer_pose = Some(XrViewerPose { views: xr_views });

    // === get_input_sources ===
    let input_sources = XrInputSourcesState {
        sources: collect_input_sources(state, frame_state.predicted_display_time),
    };

    // === acquire_swapchain_image ===
    let index = state.swapchain.acquire_image()
        .map_err(|e| JsErrorBox::generic(format!("Acquire failed: {}", e)))?;

    state.swapchain.wait_image(xr::Duration::INFINITE)
        .map_err(|e| JsErrorBox::generic(format!("Wait image failed: {}", e)))?;

    state.current_swapchain_index = Some(index);

    let swapchain_info = Some(XrSwapchainTextureInfo {
        index,
        width: state.render_width,
        height: state.render_height,
        array_layers: 2,
    });

    Ok(XrFrameBeginResult {
        frame_state: js_frame_state,
        viewer_pose,
        input_sources,
        swapchain_info,
    })
}