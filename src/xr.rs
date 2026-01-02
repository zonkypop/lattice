// src/xr.rs - OpenXR integration with controller support

use deno_error::JsErrorBox;
use serde::Serialize;
use std::sync::{Mutex, OnceLock};
use deno_core::{op2, OpState};

use openxr as xr;
use ash::vk::{self, Handle};
use std::ffi::CString;
use crate::gfx::{GfxContext, set_gfx_context};
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
    
}

unsafe impl Send for XrState {}
unsafe impl Sync for XrState {}

static XR_STATE: OnceLock<Mutex<Option<XrState>>> = OnceLock::new();

pub fn get_xr_state() -> &'static Mutex<Option<XrState>> {
    XR_STATE.get_or_init(|| Mutex::new(None))
}

pub static XR_SWAPCHAIN_TEXTURES: OnceLock<Vec<wgpu::Texture>> = OnceLock::new();

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

#[op2(async)]
pub async fn op_xr_is_supported() -> Result<bool, JsErrorBox> {
    match unsafe { xr::Entry::load() }.ok().and_then(|e| e.enumerate_extensions().ok()) {
        Some(exts) => Ok(exts.khr_vulkan_enable2),
        None => Ok(false),
    }
}

unsafe fn enable_foveation(
    instance: &xr::Instance,
    session: &xr::Session<xr::Vulkan>,
    swapchain: &xr::Swapchain<xr::Vulkan>,
) -> Result<(), JsErrorBox> {
    eprintln!("Attempting to enable foveation...");
    
    // Get function pointers
    let mut create_foveation_profile: Option<sys::pfn::CreateFoveationProfileFB> = None;
    let mut update_swapchain: Option<sys::pfn::UpdateSwapchainFB> = None;

    let _ = (instance.fp().get_instance_proc_addr)(
        instance.as_raw(),
        c"xrCreateFoveationProfileFB".as_ptr(),
        &mut create_foveation_profile as *mut _ as *mut _,
    );
    
    let _ = (instance.fp().get_instance_proc_addr)(
        instance.as_raw(),
        c"xrUpdateSwapchainFB".as_ptr(),
        &mut update_swapchain as *mut _ as *mut _,
    );

    eprintln!("create_foveation_profile fn: {}", create_foveation_profile.is_some());
    eprintln!("update_swapchain fn: {}", update_swapchain.is_some());

    let create_fn = create_foveation_profile
        .ok_or_else(|| JsErrorBox::generic("xrCreateFoveationProfileFB not found"))?;
    let update_fn = update_swapchain
        .ok_or_else(|| JsErrorBox::generic("xrUpdateSwapchainFB not found"))?;

    // Create fixed foveation profile (high quality setting)
    let level_profile = sys::FoveationLevelProfileCreateInfoFB {
        ty: sys::FoveationLevelProfileCreateInfoFB::TYPE,
        next: std::ptr::null_mut(),
        level: sys::FoveationLevelFB::HIGH,
        vertical_offset: 0.0,
        dynamic: sys::FoveationDynamicFB::LEVEL_ENABLED,
    };

    let profile_create_info = sys::FoveationProfileCreateInfoFB {
        ty: sys::FoveationProfileCreateInfoFB::TYPE,
        next: &level_profile as *const _ as *mut _,
    };

    let mut foveation_profile = sys::FoveationProfileFB::NULL;
    let result = (create_fn)(
        session.as_raw(),
        &profile_create_info,
        &mut foveation_profile,
    );
    
    if result != sys::Result::SUCCESS {
        return Err(JsErrorBox::generic(format!("Failed to create foveation profile: {:?}", result)));
    }

    // Apply foveation to swapchain
    let swapchain_state = sys::SwapchainStateFoveationFB {
        ty: sys::SwapchainStateFoveationFB::TYPE,
        next: std::ptr::null_mut(),
        flags: sys::SwapchainStateFoveationFlagsFB::EMPTY,
        profile: foveation_profile,
    };

    let result = (update_fn)(
        swapchain.as_raw(),
        &swapchain_state as *const _ as *const sys::SwapchainStateBaseHeaderFB,
    );

    if result != sys::Result::SUCCESS {
        return Err(JsErrorBox::generic(format!("Failed to update swapchain foveation: {:?}", result)));
    }

    log::info!("Foveation enabled (HIGH level, dynamic)");
    Ok(())
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

    // Also replace ash::Entry::load with:
    #[cfg(target_os = "android")]
    let vk_entry = unsafe { ash::Entry::load() }.expect("Failed to load Vulkan");

    #[cfg(not(target_os = "android"))]
    let vk_entry = unsafe { ash::Entry::load() }
        .map_err(|e| JsErrorBox::generic(format!("Failed to load Vulkan: {}", e)))?;
    
    let mut enabled_extensions = xr::ExtensionSet::default();
    enabled_extensions.khr_vulkan_enable2 = true;

    // enabled_extensions.fb_foveation = true;
    // enabled_extensions.fb_foveation_configuration = true;
    // enabled_extensions.fb_swapchain_update_state = true;


    
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
    
    // 1680x1760 on Quest 3
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
    
    let wgpu_features = wgpu_exposed_adapter.features;
    let hal_adapter_features = wgpu_exposed_adapter.features;  // Save before move



    let mut enabled_device_extensions = wgpu_exposed_adapter
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
    
    // Pose actions
    let left_grip_action = action_set.create_action::<xr::Posef>(
        "left_grip", "Left Grip Pose", &[left_hand_path]
    ).map_err(|e| JsErrorBox::generic(format!("Action error: {}", e)))?;
    
    let right_grip_action = action_set.create_action::<xr::Posef>(
        "right_grip", "Right Grip Pose", &[right_hand_path]
    ).map_err(|e| JsErrorBox::generic(format!("Action error: {}", e)))?;
    
    let left_aim_action = action_set.create_action::<xr::Posef>(
        "left_aim", "Left Aim Pose", &[left_hand_path]
    ).map_err(|e| JsErrorBox::generic(format!("Action error: {}", e)))?;
    
    let right_aim_action = action_set.create_action::<xr::Posef>(
        "right_aim", "Right Aim Pose", &[right_hand_path]
    ).map_err(|e| JsErrorBox::generic(format!("Action error: {}", e)))?;
    
    // Float actions
    let left_trigger_action = action_set.create_action::<f32>(
        "left_trigger", "Left Trigger", &[left_hand_path]
    ).map_err(|e| JsErrorBox::generic(format!("Action error: {}", e)))?;
    
    let right_trigger_action = action_set.create_action::<f32>(
        "right_trigger", "Right Trigger", &[right_hand_path]
    ).map_err(|e| JsErrorBox::generic(format!("Action error: {}", e)))?;
    
    let left_squeeze_action = action_set.create_action::<f32>(
        "left_squeeze", "Left Squeeze", &[left_hand_path]
    ).map_err(|e| JsErrorBox::generic(format!("Action error: {}", e)))?;
    
    let right_squeeze_action = action_set.create_action::<f32>(
        "right_squeeze", "Right Squeeze", &[right_hand_path]
    ).map_err(|e| JsErrorBox::generic(format!("Action error: {}", e)))?;
    
    // Thumbstick actions
    let left_thumbstick_action = action_set.create_action::<xr::Vector2f>(
        "left_thumbstick", "Left Thumbstick", &[left_hand_path]
    ).map_err(|e| JsErrorBox::generic(format!("Action error: {}", e)))?;
    
    let right_thumbstick_action = action_set.create_action::<xr::Vector2f>(
        "right_thumbstick", "Right Thumbstick", &[right_hand_path]
    ).map_err(|e| JsErrorBox::generic(format!("Action error: {}", e)))?;
    
    let left_thumbstick_click_action = action_set.create_action::<bool>(
        "left_thumbstick_click", "Left Thumbstick Click", &[left_hand_path]
    ).map_err(|e| JsErrorBox::generic(format!("Action error: {}", e)))?;
    
    let right_thumbstick_click_action = action_set.create_action::<bool>(
        "right_thumbstick_click", "Right Thumbstick Click", &[right_hand_path]
    ).map_err(|e| JsErrorBox::generic(format!("Action error: {}", e)))?;
    
    // Button actions
    let left_x_button_action = action_set.create_action::<bool>(
        "left_x", "Left X Button", &[left_hand_path]
    ).map_err(|e| JsErrorBox::generic(format!("Action error: {}", e)))?;
    
    let left_y_button_action = action_set.create_action::<bool>(
        "left_y", "Left Y Button", &[left_hand_path]
    ).map_err(|e| JsErrorBox::generic(format!("Action error: {}", e)))?;
    
    let right_a_button_action = action_set.create_action::<bool>(
        "right_a", "Right A Button", &[right_hand_path]
    ).map_err(|e| JsErrorBox::generic(format!("Action error: {}", e)))?;
    
    let right_b_button_action = action_set.create_action::<bool>(
        "right_b", "Right B Button", &[right_hand_path]
    ).map_err(|e| JsErrorBox::generic(format!("Action error: {}", e)))?;
    
    // Suggest bindings for Oculus Touch
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
    ]).map_err(|e| JsErrorBox::generic(format!("Failed to suggest bindings: {}", e)))?;
    
    // Create action spaces
    let left_grip_space = left_grip_action.create_space(
        session.clone(), left_hand_path, xr::Posef::IDENTITY
    ).map_err(|e| JsErrorBox::generic(format!("Space error: {}", e)))?;
    
    let right_grip_space = right_grip_action.create_space(
        session.clone(), right_hand_path, xr::Posef::IDENTITY
    ).map_err(|e| JsErrorBox::generic(format!("Space error: {}", e)))?;
    
    let left_aim_space = left_aim_action.create_space(
        session.clone(), left_hand_path, xr::Posef::IDENTITY
    ).map_err(|e| JsErrorBox::generic(format!("Space error: {}", e)))?;
    
    let right_aim_space = right_aim_action.create_space(
        session.clone(), right_hand_path, xr::Posef::IDENTITY
    ).map_err(|e| JsErrorBox::generic(format!("Space error: {}", e)))?;
    
    // Attach action set
    session.attach_action_sets(&[&action_set])
        .map_err(|e| JsErrorBox::generic(format!("Failed to attach action sets: {}", e)))?;
    
    
    // ======================= Swapchain =======================
    let swapchain = session.create_swapchain(&xr::SwapchainCreateInfo {
        create_flags: xr::SwapchainCreateFlags::EMPTY,
        usage_flags: xr::SwapchainUsageFlags::COLOR_ATTACHMENT 
            | xr::SwapchainUsageFlags::SAMPLED,
        format: vk::Format::R8G8B8A8_SRGB.as_raw() as _,
        sample_count: 1,
        width,
        height,
        face_count: 1,
        array_size: 2,
        mip_count: 1,
    }).map_err(|e| JsErrorBox::generic(format!("Failed to create swapchain: {}", e)))?;


    // Not working  / part of the WebGPU spec yet :/
    // unsafe {
    //     match enable_foveation(&instance, &session, &swapchain) {
    //         Ok(()) => eprintln!("Foveation setup succeeded"),
    //         Err(e) => eprintln!("Failed to enable foveation: {}", e),
    //     }
    // }

    let mut get_foveation: Option<sys::pfn::GetSwapchainStateFB> = None;
    let _ = unsafe {
        (instance.fp().get_instance_proc_addr)(
            instance.as_raw(),
            c"xrGetSwapchainStateFB".as_ptr(),
            &mut get_foveation as *mut _ as *mut _,
        )
    };

    if let Some(get_fn) = get_foveation {
        let mut foveation_state = sys::SwapchainStateFoveationFB {
            ty: sys::SwapchainStateFoveationFB::TYPE,
            next: std::ptr::null_mut(),
            flags: sys::SwapchainStateFoveationFlagsFB::EMPTY,
            profile: sys::FoveationProfileFB::NULL,
        };
        
        let result = unsafe {
            (get_fn)(
                swapchain.as_raw(),
                &mut foveation_state as *mut _ as *mut sys::SwapchainStateBaseHeaderFB,
            )
        };
        
        eprintln!("Get foveation state result: {:?}", result);
        eprintln!("Foveation profile is null: {}", foveation_state.profile == sys::FoveationProfileFB::NULL);
    } else {
        eprintln!("xrGetSwapchainStateFB not found");
    }
    
    let swapchain_images = swapchain.enumerate_images()
        .map_err(|e| JsErrorBox::generic(format!("Failed to enumerate swapchain images: {}", e)))?;
    
    let swapchain_length = swapchain_images.len() as u32;
    
    let swapchain_textures: Vec<wgpu::Texture> = swapchain_images
        .into_iter()
        .map(|vk_image| {
            let vk_image = vk::Image::from_raw(vk_image);
            
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
    
    // Set GfxContext
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
    };
    
    *get_xr_state().lock().unwrap() = Some(state);
    let _ = XR_SWAPCHAIN_TEXTURES.set(swapchain_textures);
    
    log::info!("XR session created successfully");
    
    Ok(XrSessionInfo {
        width,
        height,
        sample_count: 1,
        format: "rgba8unorm-srgb".to_string(),
        swapchain_length,
    })
}


#[op2(async)]
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
                    state.session.end()
                        .map_err(|e| JsErrorBox::generic(format!("End failed: {}", e)))?;
                    state.session_running = false;
                }
                xr::SessionState::EXITING | xr::SessionState::LOSS_PENDING => {
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
            projection_matrix: fov_to_projection_matrix(&view.fov, 0.05, 500.0),
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

#[op2]
#[serde]
pub fn op_xr_get_input_sources() -> Result<XrInputSourcesState, JsErrorBox> {
    let mut guard = get_xr_state().lock().unwrap();
    let state = guard.as_mut().ok_or_else(|| JsErrorBox::generic("No XR session"))?;
    
    let frame_state = state.frame_state.as_ref()
        .ok_or_else(|| JsErrorBox::generic("No frame state"))?;
    
    // Sync actions
    let active_action_set = xr::ActiveActionSet::new(&state.action_set);
    if let Err(e) = state.session.sync_actions(&[active_action_set]) {
        log::warn!("Sync actions failed: {}", e);
        return Ok(XrInputSourcesState { sources: vec![] });
    }
    
    let mut sources = Vec::new();
    let time = frame_state.predicted_display_time;
    
    // Helper to get pose
    let get_pose = |space: &xr::Space, stage: &xr::Space, time: xr::Time| -> Option<XrPose> {
        space.locate(stage, time).ok()
            .filter(|loc| loc.location_flags.contains(
                xr::SpaceLocationFlags::POSITION_VALID | xr::SpaceLocationFlags::ORIENTATION_VALID
            ))
            .map(|loc| XrPose {
                position: [loc.pose.position.x, loc.pose.position.y, loc.pose.position.z],
                orientation: [loc.pose.orientation.x, loc.pose.orientation.y, loc.pose.orientation.z, loc.pose.orientation.w],
                matrix: pose_to_matrix(&loc.pose),
            })
    };
    
    // Left controller
    let left_grip = get_pose(&state.left_grip_space, &state.stage_space, time);
    let left_aim = get_pose(&state.left_aim_space, &state.stage_space, time);
    
    if left_grip.is_some() || left_aim.is_some() {
        let trigger = state.left_trigger_action.state(&state.session, xr::Path::NULL)
            .map(|s| s.current_state).unwrap_or(0.0);
        let squeeze = state.left_squeeze_action.state(&state.session, xr::Path::NULL)
            .map(|s| s.current_state).unwrap_or(0.0);
        let thumbstick = state.left_thumbstick_action.state(&state.session, xr::Path::NULL)
            .map(|s| (s.current_state.x, s.current_state.y)).unwrap_or((0.0, 0.0));
        let stick_click = state.left_thumbstick_click_action.state(&state.session, xr::Path::NULL)
            .map(|s| s.current_state).unwrap_or(false);
        let x_btn = state.left_x_button_action.state(&state.session, xr::Path::NULL)
            .map(|s| s.current_state).unwrap_or(false);
        let y_btn = state.left_y_button_action.state(&state.session, xr::Path::NULL)
            .map(|s| s.current_state).unwrap_or(false);
        
        sources.push(XrInputSourceData {
            handedness: "left".to_string(),
            target_ray_mode: "tracked-pointer".to_string(),
            profiles: vec!["oculus-touch-v3".to_string(), "oculus-touch".to_string(), "generic-trigger-squeeze-thumbstick".to_string()],
            grip_space_pose: left_grip,
            target_ray_pose: left_aim,
            gamepad: Some(XrGamepadData {
                buttons: vec![
                    XrButtonData { pressed: trigger > 0.9, touched: trigger > 0.0, value: trigger },
                    XrButtonData { pressed: squeeze > 0.9, touched: squeeze > 0.0, value: squeeze },
                    XrButtonData { pressed: false, touched: false, value: 0.0 },
                    XrButtonData { pressed: stick_click, touched: thumbstick.0.abs() > 0.01 || thumbstick.1.abs() > 0.01, value: if stick_click { 1.0 } else { 0.0 } },
                    XrButtonData { pressed: x_btn, touched: x_btn, value: if x_btn { 1.0 } else { 0.0 } },
                    XrButtonData { pressed: y_btn, touched: y_btn, value: if y_btn { 1.0 } else { 0.0 } },
                ],
                axes: vec![0.0, 0.0, thumbstick.0, -thumbstick.1],
            }),
            hand: None,
        });
    }
    
    // Right controller
    let right_grip = get_pose(&state.right_grip_space, &state.stage_space, time);
    let right_aim = get_pose(&state.right_aim_space, &state.stage_space, time);
    
    if right_grip.is_some() || right_aim.is_some() {
        let trigger = state.right_trigger_action.state(&state.session, xr::Path::NULL)
            .map(|s| s.current_state).unwrap_or(0.0);
        let squeeze = state.right_squeeze_action.state(&state.session, xr::Path::NULL)
            .map(|s| s.current_state).unwrap_or(0.0);
        let thumbstick = state.right_thumbstick_action.state(&state.session, xr::Path::NULL)
            .map(|s| (s.current_state.x, s.current_state.y)).unwrap_or((0.0, 0.0));
        let stick_click = state.right_thumbstick_click_action.state(&state.session, xr::Path::NULL)
            .map(|s| s.current_state).unwrap_or(false);
        let a_btn = state.right_a_button_action.state(&state.session, xr::Path::NULL)
            .map(|s| s.current_state).unwrap_or(false);
        let b_btn = state.right_b_button_action.state(&state.session, xr::Path::NULL)
            .map(|s| s.current_state).unwrap_or(false);
        
        sources.push(XrInputSourceData {
            handedness: "right".to_string(),
            target_ray_mode: "tracked-pointer".to_string(),
            profiles: vec!["oculus-touch-v3".to_string(), "oculus-touch".to_string(), "generic-trigger-squeeze-thumbstick".to_string()],
            grip_space_pose: right_grip,
            target_ray_pose: right_aim,
            gamepad: Some(XrGamepadData {
                buttons: vec![
                    XrButtonData { pressed: trigger > 0.9, touched: trigger > 0.0, value: trigger },
                    XrButtonData { pressed: squeeze > 0.9, touched: squeeze > 0.0, value: squeeze },
                    XrButtonData { pressed: false, touched: false, value: 0.0 },
                    XrButtonData { pressed: stick_click, touched: thumbstick.0.abs() > 0.01 || thumbstick.1.abs() > 0.01, value: if stick_click { 1.0 } else { 0.0 } },
                    XrButtonData { pressed: a_btn, touched: a_btn, value: if a_btn { 1.0 } else { 0.0 } },
                    XrButtonData { pressed: b_btn, touched: b_btn, value: if b_btn { 1.0 } else { 0.0 } },
                ],
                axes: vec![0.0, 0.0, thumbstick.0, -thumbstick.1],
            }),
            hand: None,
        });
    }
    
    Ok(XrInputSourcesState { sources })
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
    
    let textures = XR_SWAPCHAIN_TEXTURES.get()
        .ok_or_else(|| JsErrorBox::generic("Swapchain textures not initialized"))?;
    
    let texture = textures.get(swapchain_index as usize)
        .ok_or_else(|| JsErrorBox::generic("Invalid swapchain index"))?;
    
    let view = texture.create_view(&wgpu::TextureViewDescriptor {
        label: Some("XR Swapchain View"),
        format: Some(wgpu::TextureFormat::Rgba8UnormSrgb),
        dimension: Some(wgpu::TextureViewDimension::D2),
        aspect: wgpu::TextureAspect::All,
        base_mip_level: 0,
        mip_level_count: Some(1),
        base_array_layer: view_index,
        array_layer_count: Some(1),
        usage: None,
    });
    
    let rid = state.resource_table.add(crate::gfx::GfxTextureView {
        view: std::sync::Arc::new(view),
        width: xr_state.render_width,
        height: xr_state.render_height,
    });
    
    Ok(crate::gfx::JsGfxTextureView { rid })
}