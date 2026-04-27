// src/lib.rs 

use anyhow::Result;
use deno_core::error::AnyError;
use deno_core::{extension, ModuleSpecifier};

use deno_fs::RealFs;
use deno_resolver::npm::{DenoInNpmPackageChecker, NpmResolver};
use deno_runtime::deno_permissions::PermissionsContainer;
use deno_runtime::permissions::RuntimePermissionDescriptorParser;
use deno_runtime::worker::{MainWorker, WorkerOptions, WorkerServiceOptions};

use env_logger;
use log::{error, info};
use std::rc::Rc;
use std::sync::Arc;


#[cfg(not(target_os = "android"))]
use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, ControlFlow, EventLoop},
    window::{Window, WindowId},
};
use deno_runtime::ops::bootstrap::SnapshotOptions;

mod gfx;
mod input;
mod module_loader;

use gfx::{GfxContext, set_gfx_context};
use module_loader::{ImportMapModuleLoader, create_web_worker_callback};

use deno_runtime::BootstrapOptions;

mod sqlite;
mod audio;
use tokio::runtime::Handle;

use deno_core::{SharedArrayBufferStore, CompiledWasmModuleStore};

use input::{get_input_queue, op_input_poll_events, op_input_get_window_size, 
    op_input_request_pointer_lock, op_input_exit_pointer_lock, 
    op_input_is_pointer_locked, op_input_set_cursor_style, InputEvent};

mod xr;





// ======================= Runtime Mode =======================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RuntimeMode {
    Desktop,
    XR,
}

static RUNTIME_MODE: std::sync::OnceLock<RuntimeMode> = std::sync::OnceLock::new();

pub fn get_runtime_mode() -> RuntimeMode {
    *RUNTIME_MODE.get().unwrap_or(&RuntimeMode::Desktop)
}

pub fn set_runtime_mode(mode: RuntimeMode) {
    let _ = RUNTIME_MODE.set(mode);
}

/// Drain all currently-ready async work from the JS event loop.
///
/// Polls `poll_event_loop` directly (single-tick, safe to call repeatedly)
/// with yields between ticks so tokio tasks can make progress. This allows
/// multi-step promise chains involving async ops (e.g. fetch→parse→process)
/// to fully resolve within a single frame rather than one step per frame.
///
/// Pure microtask chains (.then/.catch) resolve in a single poll. The extra
/// iterations only matter when promises depend on async op completions.
/// Max 8 poll-yield cycles. Stops early if the event loop reports Ready
/// (all work done, though `device.lost = new Promise(()=>{})` keeps it Pending).
async fn drain_event_loop(runtime: &mut deno_core::JsRuntime) {
    let opts = deno_core::PollEventLoopOptions { wait_for_inspector: false };
    for _ in 0..8 {
        let done = std::future::poll_fn(|cx| {
            match runtime.poll_event_loop(cx, opts) {
                std::task::Poll::Ready(result) => {
                    if let Err(e) = result {
                        log::error!("Event loop error: {:?}", e);
                    }
                    std::task::Poll::Ready(true)
                }
                std::task::Poll::Pending => std::task::Poll::Ready(false),
            }
        }).await;
        if done { break; }
        tokio::task::yield_now().await;
    }
}

// ======================= XR Exit Flag =======================

static XR_SHOULD_EXIT: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);

pub fn signal_xr_exit() {
    XR_SHOULD_EXIT.store(true, std::sync::atomic::Ordering::SeqCst);
}

fn should_exit_xr() -> bool {
    XR_SHOULD_EXIT.load(std::sync::atomic::Ordering::SeqCst)
}

// ======================= Extension =======================

extension!(
    gfx_host,
    ops = [
        gfx::op_gfx_get_preferred_surface_format,
        gfx::op_gfx_surface_configure,
        gfx::op_gfx_device_create_shader,
        gfx::op_gfx_device_create_buffer_init,
        gfx::op_gfx_device_create_buffer,
        gfx::op_gfx_queue_write_buffer,
        gfx::op_gfx_clear_buffer,
        gfx::op_gfx_device_create_texture,
        gfx::op_gfx_texture_create_view,
        gfx::op_gfx_device_create_sampler,
        gfx::op_gfx_device_create_bind_group_layout,
        gfx::op_gfx_device_create_pipeline_layout,
        gfx::op_gfx_device_create_bind_group,
        gfx::op_gfx_device_create_pipeline,
        gfx::op_gfx_surface_draw,
        gfx::op_gfx_write_texture_image,
        gfx::op_gfx_pipeline_get_bind_group_layout,
        gfx::op_gfx_render_to_texture,
        gfx::op_gfx_copy_texture_to_texture,
        gfx::op_gfx_copy_buffer_to_texture,
        gfx::op_gfx_decode_image_store,
        gfx::op_gfx_upload_decoded_image_to_texture,
        gfx::op_gfx_decoded_image_drop,
        gfx::op_gfx_decoded_image_stats,
        gfx::op_gfx_decoded_image_stats_detailed,
        gfx::op_gfx_resource_stats,
        gfx::op_gfx_resize_decoded_image,
        gfx::op_gfx_multi_draw_indexed_indirect,
        gfx::op_gfx_queue_submit_empty,

        op_input_poll_events,
        op_input_get_window_size,
        op_input_request_pointer_lock,
        op_input_exit_pointer_lock,
        op_input_is_pointer_locked,
        op_input_set_cursor_style,

        sqlite::op_indexeddb_open,
        sqlite::op_indexeddb_get,
        sqlite::op_indexeddb_put,
        sqlite::op_indexeddb_delete,
        sqlite::op_indexeddb_get_all_keys,
        sqlite::op_indexeddb_clear,
        sqlite::op_indexeddb_store_exists,

        gfx::op_gfx_render_depth_only,

        gfx::op_gfx_resource_drop,
        gfx::op_gfx_texture_drop,
        gfx::op_gfx_buffer_drop,
        gfx::op_gfx_bind_group_drop,
        gfx::op_gfx_shader_drop,
        gfx::op_gfx_sampler_drop,
        gfx::op_gfx_bind_group_layout_drop,
        gfx::op_gfx_pipeline_layout_drop,
        gfx::op_gfx_pipeline_drop,
        gfx::op_gfx_compute_pipeline_drop,
        gfx::op_gfx_flush_commands,
        gfx::op_gfx_render_xr_frame,

        // Compute shader ops
        gfx::op_gfx_device_create_compute_pipeline,
        gfx::op_gfx_compute_pipeline_get_bind_group_layout,
        gfx::op_gfx_compute_dispatch,
        gfx::op_gfx_compute_dispatch_indirect,
        gfx::op_gfx_compute_batch,

        // GPUTimer / Timestamp query ops
        gfx::op_gfx_device_create_query_set,
        gfx::op_gfx_query_set_destroy,
        gfx::op_gfx_has_timestamp_query,
        gfx::op_gfx_has_feature,
        gfx::op_gfx_get_timestamp_period,
        gfx::op_gfx_timestamp_batch,
        gfx::op_gfx_buffer_map_async,
        gfx::op_gfx_buffer_map_poll,
        gfx::op_gfx_buffer_map_wait,
        gfx::op_gfx_buffer_map_finish,
        gfx::op_gfx_buffer_get_mapped_range,
        gfx::op_gfx_buffer_unmap,
        gfx::op_gfx_write_timestamp,

        // XR ops
        xr::op_xr_is_supported,
        xr::op_xr_request_session,
        xr::op_xr_poll_events,
        xr::op_xr_wait_frame,
        xr::op_xr_set_clip_planes,
        xr::op_xr_get_viewer_pose,
        xr::op_xr_acquire_swapchain_image,
        xr::op_xr_release_swapchain_image,
        xr::op_xr_end_frame,
        xr::op_xr_end_session,
        xr::op_xr_get_swapchain_texture_view,
        xr::op_xr_get_input_sources,
        xr::op_xr_frame_begin,
        xr::op_xr_frame_wait_start,
        xr::op_xr_frame_wait_poll,
        xr::op_xr_frame_wait_finish,

        // Audio ops
        audio::op_audio_create_context,
        audio::op_audio_context_current_time,
        audio::op_audio_context_close,
        audio::op_audio_create_gain,
        audio::op_audio_create_buffer_source,
        audio::op_audio_create_oscillator,
        audio::op_audio_create_panner,
        audio::op_audio_create_biquad_filter,
        audio::op_audio_create_stereo_panner,
        audio::op_audio_create_delay,
        audio::op_audio_create_dynamics_compressor,
        audio::op_audio_create_analyser,
        audio::op_audio_connect,
        audio::op_audio_disconnect,
        audio::op_audio_param_set_value,
        audio::op_audio_param_set_value_at_time,
        audio::op_audio_param_set_target_at_time,
        audio::op_audio_param_linear_ramp,
        audio::op_audio_param_exponential_ramp,
        audio::op_audio_decode_audio_data,
        audio::op_audio_buffer_drop,
        audio::op_audio_buffer_source_set_buffer,
        audio::op_audio_buffer_source_start,
        audio::op_audio_buffer_source_stop,
        audio::op_audio_buffer_source_set_loop,
        audio::op_audio_buffer_source_set_loop_start,
        audio::op_audio_buffer_source_set_loop_end,
        audio::op_audio_oscillator_set_type,
        audio::op_audio_oscillator_start,
        audio::op_audio_oscillator_stop,
        audio::op_audio_panner_configure,
        audio::op_audio_biquad_set_type,
        audio::op_audio_listener_set_position,
        audio::op_audio_listener_set_orientation,
        audio::op_audio_node_drop,

        // event loop
        op_yield_to_runtime

    ],
    esm_entry_point = "ext:gfx_host/bootstrap.js",
    esm = [dir "src", "bootstrap.js"],
    state = |state| {
        state.put(gfx::GpuIdMap::new());
    },
    customizer = |ext: &mut deno_core::Extension| {
        ext.needs_lazy_init = false;
    }
);

deno_core::extension!(
    snapshot_options_ext,
    state = |state| {
        state.put(SnapshotOptions::default());
    }
);

// ======================= GPU State (Desktop Mode) =======================

#[cfg(not(target_os = "android"))]
struct GpuState {
    window: Arc<Window>,
    ctx: &'static GfxContext,
    config: wgpu::SurfaceConfiguration,
}

#[cfg(not(target_os = "android"))]
impl GpuState {
    async fn new(window: Arc<Window>) -> Self {
        let size = window.inner_size();

        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..Default::default()
        });

        let surface = instance
            .create_surface(window.clone())
            .expect("Failed to create surface");

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect("Failed to find adapter");

        // Request timestamp-query features if available (for GPUTimer support)
        let mut features = wgpu::Features::empty();
        let adapter_features = adapter.features();
        info!("Adapter features: {:?}", adapter_features);
        if adapter_features.contains(wgpu::Features::TIMESTAMP_QUERY) {
            features |= wgpu::Features::TIMESTAMP_QUERY;
            info!("Enabling TIMESTAMP_QUERY feature for GPU timing");
        } else {
            info!("TIMESTAMP_QUERY not available on this adapter");
        }
        // Also need TIMESTAMP_QUERY_INSIDE_ENCODERS for writeTimestamp in command encoders
        if adapter_features.contains(wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS) {
            features |= wgpu::Features::TIMESTAMP_QUERY_INSIDE_ENCODERS;
            info!("Enabling TIMESTAMP_QUERY_INSIDE_ENCODERS feature");
        }
        // Compressed texture formats (BC7 for desktop, ASTC for mobile/Apple Silicon)
        if adapter_features.contains(wgpu::Features::TEXTURE_COMPRESSION_BC) {
            features |= wgpu::Features::TEXTURE_COMPRESSION_BC;
            info!("Enabling TEXTURE_COMPRESSION_BC feature");
        }
        if adapter_features.contains(wgpu::Features::TEXTURE_COMPRESSION_ASTC) {
            features |= wgpu::Features::TEXTURE_COMPRESSION_ASTC;
            info!("Enabling TEXTURE_COMPRESSION_ASTC feature");
        }

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: features,
                required_limits: adapter.limits(),
                memory_hints: wgpu::MemoryHints::default(),
                trace: wgpu::Trace::Off,
                experimental_features: wgpu::ExperimentalFeatures::default(),
            })
            .await
            .expect("Failed to create device");
        
        device.on_uncaptured_error(Arc::new(|error| {
            eprintln!("!!! UNCAPTURED WGPU ERROR: {:?}", error);
            log::error!("Uncaptured wgpu error: {:?}", error);
        }));

        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| *f == wgpu::TextureFormat::Bgra8Unorm)
            .unwrap_or(caps.formats[0]);

        let mut config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: caps.present_modes[0],
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        surface.configure(&device, &config);

        let gfx_box = Box::new(GfxContext {
            device,
            queue,
            surface: Some(surface),
            format,
        });
        let gfx_static: &'static GfxContext = Box::leak(gfx_box);
        let _ = set_gfx_context(gfx_static);

        {
            let input_queue = get_input_queue();
            if let Ok(mut q) = input_queue.lock() {
                q.set_window_size(size.width.max(1), size.height.max(1));
            };
        }

        GpuState {
            window,
            ctx: gfx_static,
            config: {
                config.width = size.width.max(1);
                config.height = size.height.max(1);
                config
            },
        }
    }

    fn resize(&mut self, w: u32, h: u32) {
        if w == 0 || h == 0 {
            return;
        }
        self.config.width = w;
        self.config.height = h;
        self.ctx.surface.as_ref().unwrap().configure(&self.ctx.device, &self.config);
    }
}

// ======================= JS Runtime Bootstrap =======================

async fn create_main_worker(script_path: &str) -> Result<(MainWorker, ModuleSpecifier, SharedArrayBufferStore), AnyError> {
    use std::fs;

    let js_path = std::env::current_dir()?.join(script_path);
    let script_dir = js_path.parent().unwrap().to_path_buf();
    let import_map_path = script_dir.join("import_map.json");

    let main_module = ModuleSpecifier::from_file_path(&js_path)
        .map_err(|_| AnyError::msg("Failed to create ModuleSpecifier from script path"))?;

    let fs_arc = Arc::new(RealFs);
    let permission_desc_parser = Arc::new(RuntimePermissionDescriptorParser::new(sys_traits::impls::RealSys));

    let module_loader = ImportMapModuleLoader::new(
        script_dir.clone(),
        if import_map_path.exists() { Some(import_map_path) } else { None },
    )?;

    let shared_array_buffer_store = SharedArrayBufferStore::default();
    let compiled_wasm_module_store = CompiledWasmModuleStore::default();

    let create_web_worker_cb = create_web_worker_callback(
        module_loader.clone(),
        fs_arc.clone(),
        shared_array_buffer_store.clone(),
        compiled_wasm_module_store.clone(),
        Handle::current(),
    );


    #[cfg(target_os = "android")]
    let data_dir = std::path::PathBuf::from("/data/data/com.yourcompany.combinedapp/files");
    #[cfg(not(target_os = "android"))]
    let data_dir = std::env::current_dir()?.join("data");

    let origins_dir = data_dir.join("origins");
    let cache_dir = data_dir.join("cache");
    fs::create_dir_all(&origins_dir)?;
    fs::create_dir_all(&cache_dir)?;

    let service_options: WorkerServiceOptions<DenoInNpmPackageChecker, NpmResolver<sys_traits::impls::RealSys>, sys_traits::impls::RealSys> = WorkerServiceOptions {
        deno_rt_native_addon_loader: None,
        module_loader: Rc::new(module_loader),
        permissions: PermissionsContainer::allow_all(permission_desc_parser),
        blob_store: Default::default(),
        broadcast_channel: Default::default(),
        feature_checker: Default::default(),
        node_services: Default::default(),
        npm_process_state_provider: Default::default(),
        root_cert_store_provider: Default::default(),
        fetch_dns_resolver: Default::default(),
        shared_array_buffer_store: Some(shared_array_buffer_store.clone()),
        compiled_wasm_module_store: Some(compiled_wasm_module_store.clone()),
        v8_code_cache: Default::default(),
        fs: fs_arc,
        bundle_provider: Default::default(),
    };

    let worker_options = WorkerOptions {
        bootstrap: BootstrapOptions {
            location: Some(main_module.clone()),
            ..Default::default()
        },
        extensions: vec![
            snapshot_options_ext::init(),
            gfx_host::init(),
        ],
        skip_op_registration: false,
        startup_snapshot: None,
        create_params: Some(
            deno_core::v8::CreateParams::default()
                .heap_limits(0, 4 * 1024 * 1024 * 1024) // 4GB max heap (matches browser)
        ),
        unsafely_ignore_certificate_errors: None,
        seed: None,
        create_web_worker_cb,
        format_js_error_fn: None,
        should_break_on_first_statement: false,
        should_wait_for_inspector_session: false,
        trace_ops: None,
        cache_storage_dir: Some(cache_dir),
        origin_storage_dir: Some(origins_dir),
        stdio: Default::default(),
        enable_stack_trace_arg_in_ops: false,
        enable_raw_imports: false,
        unconfigured_runtime: None,
    };

    let mut worker = MainWorker::bootstrap_from_options(
        &main_module,
        service_options,
        worker_options,
    );

    #[cfg(target_os = "android")]
    let xr_enabled = true;
    #[cfg(not(target_os = "android"))]
    let xr_enabled = std::env::args().any(|a| a == "--xr");

    let script_dir_str = script_dir.to_string_lossy().replace('\\', "/");
    let script_dir_slash = if script_dir_str.ends_with('/') {
        script_dir_str.to_string()
    } else {
        format!("{}/", script_dir_str)
    };

    worker.js_runtime.execute_script(
        "<native_flags>",
        deno_core::ModuleCodeString::from(format!(
            "globalThis.__isNative__ = true; globalThis.__nativeXR__ = {}; globalThis.__scriptDir__ = '{}';",
            xr_enabled,
            script_dir_slash
        )),
    )?;

    Ok((worker, main_module, shared_array_buffer_store))
}

async fn run_js_main(worker: &mut MainWorker, main_module: &ModuleSpecifier) -> Result<(), AnyError> {
    worker.execute_main_module(main_module).await?;
    Ok(())
}

// ======================= Desktop App =======================

#[cfg(not(target_os = "android"))]
struct App {
    gpu: Option<GpuState>,
    worker: Option<MainWorker>,
    main_module: ModuleSpecifier,
    js_ran: bool,
    tokio_rt: tokio::runtime::Runtime,
}

#[cfg(not(target_os = "android"))]
impl ApplicationHandler for App {
    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _device_id: winit::event::DeviceId,
        event: winit::event::DeviceEvent,
    ) {
        if let winit::event::DeviceEvent::MouseMotion { delta } = event {
            let queue = get_input_queue();
            if let Ok(mut q) = queue.lock() {
                if q.pointer_locked {
                    q.handle_raw_mouse_motion(delta.0, delta.1);
                }
            };
        }
    }
    
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.gpu.is_none() {
            let window = Arc::new(
                event_loop
                    .create_window(Window::default_attributes())
                    .expect("Failed to create window"),
            );
            self.gpu = Some(self.tokio_rt.block_on(GpuState::new(window)));
            info!("Graphics context initialized");
        }
    
        if !self.js_ran {
            if let Some(worker) = self.worker.as_mut() {
                if let Err(e) = self.tokio_rt.block_on(run_js_main(worker, &self.main_module)) {
                    error!("Error running JS main: {e:?}");
                } else {
                    info!("JS main executed");
                }
                
                self.tokio_rt.block_on(drain_event_loop(&mut worker.js_runtime));
                
            } else {
                error!("MainWorker missing in App");
            }
            self.js_ran = true;
        }
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        window_id: WindowId,
        event: WindowEvent,
    ) {
        let Some(gpu) = self.gpu.as_mut() else { return; };
        if window_id != gpu.window.id() {
            return;
        }
    
        let input_queue = get_input_queue();

        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                gpu.resize(size.width, size.height);
                if let Ok(mut q) = input_queue.lock() {
                    q.handle_resize(size.width, size.height);
                }
            }
            WindowEvent::Focused(focused) => {
                if let Ok(mut q) = input_queue.lock() {
                    q.handle_focus(focused);
                }
            }
            WindowEvent::ModifiersChanged(modifiers) => {
                if let Ok(mut q) = input_queue.lock() {
                    q.set_modifiers(modifiers.state());
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                if let Ok(mut q) = input_queue.lock() {
                    if !q.pointer_locked {
                        q.handle_cursor_moved(position.x, position.y);
                    }
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if let Ok(mut q) = input_queue.lock() {
                    q.handle_mouse_input(state, button);
                }
            }
            WindowEvent::MouseWheel { delta, phase, .. } => {
                if let Ok(mut q) = input_queue.lock() {
                    q.handle_mouse_wheel(delta, phase);
                }
            }
            WindowEvent::KeyboardInput { event: key_event, is_synthetic, .. } => {
                if !is_synthetic {
                    if let Ok(mut q) = input_queue.lock() {
                        q.handle_keyboard_input(
                            key_event.state,
                            key_event.physical_key,
                            &key_event.logical_key,
                            key_event.repeat,
                        );
                    }
                }
            }
            WindowEvent::Touch(touch) => {
                if let Ok(mut q) = input_queue.lock() {
                    q.handle_touch(touch);
                }
            }
            WindowEvent::RedrawRequested => {
                gpu.window.request_redraw();
            }
            _ => {}
        }
        // Input events are batched in the queue and dispatched once per frame
        // from __runAnimationFrames → __dispatchInputEvents, matching browser
        // behavior where events are processed between tasks, not synchronously.
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(gpu) = &self.gpu {
            let queue = get_input_queue();
            if let Ok(mut q) = queue.lock() {
                if let Some(style) = q.cursor_style_requested.take() {
                    let cursor = match style.as_str() {
                        "pointer" => winit::window::CursorIcon::Pointer,
                        "grab" => winit::window::CursorIcon::Grab,
                        _ => winit::window::CursorIcon::Default,
                    };
                    gpu.window.set_cursor(cursor);
                }
            };
        }

        if let Some(gpu) = &self.gpu {
            let queue = get_input_queue();
            if let Ok(mut q) = queue.lock() {
                if q.pointer_lock_requested {
                    q.pointer_lock_requested = false;
                    let _ = gpu.window.set_cursor_grab(winit::window::CursorGrabMode::Locked)
                        .or_else(|_| gpu.window.set_cursor_grab(winit::window::CursorGrabMode::Confined));
                    gpu.window.set_cursor_visible(false);
                    q.pointer_locked = true;
                    q.push_event(InputEvent::PointerLockChange { locked: true });
                }
                if q.pointer_lock_exit_requested {
                    q.pointer_lock_exit_requested = false;
                    let _ = gpu.window.set_cursor_grab(winit::window::CursorGrabMode::None);
                    gpu.window.set_cursor_visible(true);
                    q.pointer_locked = false;
                    q.push_event(InputEvent::PointerLockChange { locked: false });
                }
            };
        }

        if let Some(ref mut worker) = self.worker {
            self.tokio_rt.block_on(drain_event_loop(&mut worker.js_runtime));
            
            let _guard = self.tokio_rt.enter();
            let result = worker.js_runtime.execute_script(
                "<animation_frame>",
                deno_core::ModuleCodeString::from_static(
                    "if (globalThis.__runAnimationFrames) { globalThis.__runAnimationFrames(); }"
                ),
            );
            if let Err(e) = result {
                log::error!("execute_script error: {:?}", e);
            }
        }
    
        if let Some(gpu) = &self.gpu {
            gpu.window.request_redraw();
        }
    }
}


// ======================= XR Mode =======================

#[deno_core::op2]
pub async fn op_yield_to_runtime() -> Result<(), deno_error::JsErrorBox> {
    tokio::task::yield_now().await;
    Ok(())
}

/// Drain pending Android input events to prevent ANR.
/// Android's InputDispatcher triggers ANR if events aren't acknowledged within 5s.
/// We just consume and acknowledge them — VR input comes from OpenXR, not here.
#[cfg(target_os = "android")]
fn drain_android_events() {
    if let Some(queue) = ndk_glue::input_queue() {
        while let Ok(Some(event)) = queue.get_event() {
            // Acknowledge immediately, not handled by us (VR uses OpenXR input)
            queue.finish_event(event, false);
        }
    }
}

#[cfg(not(target_os = "android"))]
fn drain_android_events() {}

fn run_xr_mode(script_path: &str) -> Result<(), AnyError> {
    info!("Starting XR mode...");

    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("Failed to create tokio runtime");

    rt.block_on(async {
        // Initialize XR session and GfxContext BEFORE JS runs
        // This matches desktop mode which sets up GPU context before JS execution
        // Workers spawned at JS module load time need GfxContext to be ready
        info!("Initializing XR session before JS...");
        let xr_info = xr::init_xr_session_internal().await
            .map_err(|e| anyhow::anyhow!("Failed to init XR session: {}", e))?;
        info!("XR session initialized: {}x{}, format: {}", xr_info.width, xr_info.height, xr_info.format);

        let (mut worker, main_module, _) = create_main_worker(script_path).await?;

        // Store XR session info in JS global so requestSession doesn't try to init again
        worker.js_runtime.execute_script(
            "<xr_session_info>",
            deno_core::ModuleCodeString::from(format!(
                "globalThis.__xrSessionInfo = {{ width: {}, height: {}, sampleCount: {}, format: '{}', swapchainLength: {} }};",
                xr_info.width, xr_info.height, xr_info.sample_count, xr_info.format, xr_info.swapchain_length
            )),
        )?;

        worker.execute_main_module(&main_module).await?;
        info!("JS main executed for XR");
        
        loop {
            if should_exit_xr() {
                info!("XR session ended, exiting...");
                break;
            }

            // Drain Android looper events (input queue + lifecycle) to prevent ANR
            drain_android_events();

            drain_event_loop(&mut worker.js_runtime).await;
        }

        Ok::<(), AnyError>(())
    })
}

// ======================= Main =======================

#[cfg(not(target_os = "android"))]
pub fn main() -> Result<(), AnyError> {
    env_logger::init();

    // Set V8 heap limit before any isolate is created
    deno_core::v8::V8::set_flags_from_string("--max-old-space-size=4096");

    // Parse command line args
    let args: Vec<String> = std::env::args().collect();
    let xr_mode = args.iter().any(|a| a == "--xr");
    let clear_db = args.iter().any(|a| a == "--clear-db");
    let script_path = args.iter()
        .find(|a| !a.starts_with('-') && *a != &args[0])
        .map(|s| s.as_str())
        .unwrap_or("js/entry.js");

    if clear_db {
        let db_dir = std::env::current_dir().unwrap_or_default().join("data").join("indexeddb");
        if db_dir.exists() {
            info!("Clearing IndexedDB at {:?}", db_dir);
            std::fs::remove_dir_all(&db_dir).ok();
        }
    }
    
    rustls::crypto::ring::default_provider()
        .install_default()
        .expect("Failed to install rustls crypto provider");
    
    if xr_mode {
        set_runtime_mode(RuntimeMode::XR);
        info!("Starting in XR mode with script: {}", script_path);
        run_xr_mode(script_path)?;
    } else {
        set_runtime_mode(RuntimeMode::Desktop);
        info!("Starting in desktop mode with script: {}", script_path);
        
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .expect("Failed to create tokio runtime");
        
        let (worker, main_module, _) = rt.block_on(create_main_worker(script_path))?;
        
        let event_loop = EventLoop::new().expect("Failed to create event loop");
        event_loop.set_control_flow(ControlFlow::Poll);
        
        let mut app = App {
            gpu: None,
            worker: Some(worker),
            main_module,
            js_ran: false,
            tokio_rt: rt,
        };
        
        event_loop.run_app(&mut app).expect("Event loop failed");
    }
    
    Ok(())
}



#[cfg(target_os = "android")]
#[ndk_glue::main(backtrace = "on")]
fn android_main() {
    android_logger::init_once(
        android_logger::Config::default().with_max_level(log::LevelFilter::Info)
    );

    let _ = rustls::crypto::ring::default_provider().install_default();
    
    log::info!("Android XR starting...");
    
    // Set working directory, jank for now
    std::env::set_current_dir("/data/local/tmp/combined_app").ok();
    
    if let Err(e) = run_xr_mode("/data/local/tmp/combined_app/js/entry.js") {
        log::error!("XR error: {:?}", e);
    }
}
