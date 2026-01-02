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
use tokio::runtime::Handle;

use deno_core::SharedArrayBufferStore;

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
        gfx::op_gfx_device_create_texture,
        gfx::op_gfx_texture_create_view,
        gfx::op_gfx_device_create_sampler,
        gfx::op_gfx_device_create_bind_group_layout,
        gfx::op_gfx_device_create_pipeline_layout,
        gfx::op_gfx_device_create_bind_group,
        gfx::op_gfx_device_create_pipeline,
        gfx::op_gfx_surface_draw,
        gfx::op_gfx_decode_image,
        gfx::op_gfx_write_texture_image,
        gfx::op_gfx_pipeline_get_bind_group_layout,
        gfx::op_gfx_render_to_texture,
        gfx::op_gfx_copy_texture_to_texture,
        gfx::op_gfx_decode_image_store,
        gfx::op_gfx_upload_decoded_image_to_texture,
        gfx::op_gfx_decoded_image_drop,
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
        gfx::op_gfx_render_xr_frame,

        // XR ops
        xr::op_xr_is_supported,
        xr::op_xr_request_session,
        xr::op_xr_poll_events,
        xr::op_xr_wait_frame,
        xr::op_xr_get_viewer_pose,
        xr::op_xr_acquire_swapchain_image,
        xr::op_xr_release_swapchain_image,
        xr::op_xr_end_frame,
        xr::op_xr_end_session,
        xr::op_xr_get_swapchain_texture_view,
        xr::op_xr_get_input_sources,

        // event loop
        op_yield_to_runtime
       
    ],
    esm_entry_point = "ext:gfx_host/bootstrap.js",
    esm = [dir "src", "bootstrap.js"],
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
    msaa_texture: wgpu::Texture,      
    msaa_view: wgpu::TextureView,     
    sample_count: u32,                 
}

#[cfg(not(target_os = "android"))]
impl GpuState {
    async fn new(window: Arc<Window>) -> Self {
        let size = window.inner_size();
        let sample_count = 4u32; 

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

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
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

        let msaa_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("MSAA Texture"),
            size: wgpu::Extent3d {
                width: size.width.max(1),
                height: size.height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        let msaa_view = msaa_texture.create_view(&wgpu::TextureViewDescriptor::default());

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
            msaa_texture,   
            msaa_view,      
            sample_count,   
        }
    }

    fn resize(&mut self, w: u32, h: u32) {
        if w == 0 || h == 0 {
            return;
        }
        self.config.width = w;
        self.config.height = h;
        self.ctx.surface.as_ref().unwrap().configure(&self.ctx.device, &self.config);
        
        self.msaa_texture = self.ctx.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("MSAA Texture"),
            size: wgpu::Extent3d {
                width: w,
                height: h,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: self.sample_count,
            dimension: wgpu::TextureDimension::D2,
            format: self.config.format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            view_formats: &[],
        });
        self.msaa_view = self.msaa_texture.create_view(&wgpu::TextureViewDescriptor::default());
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

    let create_web_worker_cb = create_web_worker_callback(
        module_loader.clone(),
        fs_arc.clone(),
        shared_array_buffer_store.clone(),
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
        compiled_wasm_module_store: Default::default(),
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
        create_params: None,
        unsafely_ignore_certificate_errors: None,
        seed: None,
        create_web_worker_cb,
        format_js_error_fn: None,
        maybe_inspector_server: None,
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

    worker.js_runtime.execute_script(
        "<xr_flag>",
        deno_core::ModuleCodeString::from(format!("globalThis.__nativeXR = {};", xr_enabled)),
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
                
                self.tokio_rt.block_on(async {
                    tokio::select! {
                        biased;
                        result = worker.js_runtime.run_event_loop(deno_core::PollEventLoopOptions {
                            wait_for_inspector: false,
                            pump_v8_message_loop: true,
                        }) => {
                            if let Err(e) = result {
                                log::error!("Event loop error: {:?}", e);
                            }
                        }
                        _ = tokio::time::sleep(std::time::Duration::from_millis(10)) => {}
                    }
                });
                
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
        let mut needs_immediate_dispatch = false;
    
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(size) => {
                gpu.resize(size.width, size.height);
                if let Ok(mut q) = input_queue.lock() {
                    q.handle_resize(size.width, size.height);
                }
                needs_immediate_dispatch = true;
            }
            WindowEvent::Focused(focused) => {
                if let Ok(mut q) = input_queue.lock() {
                    q.handle_focus(focused);
                }
                needs_immediate_dispatch = true;
            }
            WindowEvent::ModifiersChanged(modifiers) => {
                if let Ok(mut q) = input_queue.lock() {
                    q.set_modifiers(modifiers.state());
                }
            }
            WindowEvent::CursorMoved { position, .. } => {
                let locked = {
                    if let Ok(q) = input_queue.lock() {
                        q.pointer_locked
                    } else {
                        false
                    }
                };
                if !locked {
                    if let Ok(mut q) = input_queue.lock() {
                        q.handle_cursor_moved(position.x, position.y);
                    }
                    needs_immediate_dispatch = true;
                }
            }
            WindowEvent::MouseInput { state, button, .. } => {
                if let Ok(mut q) = input_queue.lock() {
                    q.handle_mouse_input(state, button);
                }
                needs_immediate_dispatch = true;
            }
            WindowEvent::MouseWheel { delta, phase, .. } => {
                if let Ok(mut q) = input_queue.lock() {
                    q.handle_mouse_wheel(delta, phase);
                }
                needs_immediate_dispatch = true;
            }
            WindowEvent::KeyboardInput { event: key_event, is_synthetic, .. } => {
                if !is_synthetic {
                    if let Ok(mut q) = input_queue.lock() {
                        q.handle_keyboard_input(
                            key_event.state,
                            key_event.physical_key,
                            key_event.repeat,
                        );
                    }
                    needs_immediate_dispatch = true;
                }
            }
            WindowEvent::Touch(touch) => {
                if let Ok(mut q) = input_queue.lock() {
                    q.handle_touch(touch);
                }
                needs_immediate_dispatch = true;
            }
            WindowEvent::RedrawRequested => {
                gpu.window.request_redraw();
            }
            _ => {}
        }
    
        if needs_immediate_dispatch {
            if let Some(ref mut worker) = self.worker {
                let _guard = self.tokio_rt.enter();
                let _ = worker.js_runtime.execute_script(
                    "<input_dispatch>",
                    deno_core::ModuleCodeString::from_static(
                        "if (globalThis.__dispatchInputEvents) globalThis.__dispatchInputEvents();"
                    ),
                );
            }
        }
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
            self.tokio_rt.block_on(async {
                tokio::select! {
                    biased;
                    result = worker.js_runtime.run_event_loop(deno_core::PollEventLoopOptions {
                        wait_for_inspector: false,
                        pump_v8_message_loop: true,
                    }) => {
                        if let Err(e) = result {
                            log::error!("Event loop error: {:?}", e);
                        }
                    }
                    _ = tokio::time::sleep(std::time::Duration::from_millis(1)) => {}
                }
            });
            
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

#[deno_core::op2(async)]
pub async fn op_yield_to_runtime() -> Result<(), deno_error::JsErrorBox> {
    tokio::task::yield_now().await;
    Ok(())
}

fn run_xr_mode(script_path: &str) -> Result<(), AnyError> {
    info!("Starting XR mode...");
    
    let rt = tokio::runtime::Builder::new_current_thread()
        .enable_all()
        .build()
        .expect("Failed to create tokio runtime");
    
    rt.block_on(async {
        let (mut worker, main_module, _) = create_main_worker(script_path).await?;
        worker.execute_main_module(&main_module).await?;
        info!("JS main executed for XR");
        
        loop {
            if should_exit_xr() {
                info!("XR session ended, exiting...");
                break;
            }
            
            match worker.js_runtime.run_event_loop(deno_core::PollEventLoopOptions {
                wait_for_inspector: false,
                pump_v8_message_loop: true,
            }).await {
                Ok(_) => {},
                Err(e) => {
                    error!("Event loop error: {:?}", e);
                    break;
                }
            }
        }
        
        Ok::<(), AnyError>(())
    })
}

// ======================= Main =======================

#[cfg(not(target_os = "android"))]
pub fn main() -> Result<(), AnyError> {
    env_logger::init();
    
    // Parse command line args
    let args: Vec<String> = std::env::args().collect();
    let xr_mode = args.iter().any(|a| a == "--xr");
    let script_path = args.iter()
        .find(|a| !a.starts_with('-') && *a != &args[0])
        .map(|s| s.as_str())
        .unwrap_or("js/entry.js");
    
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
