// src/module_loader.rs - Module loading and web worker creation

use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;

use deno_core::{
    ModuleSpecifier, ModuleLoader, ModuleSource, ModuleSourceCode,
    ModuleType, RequestedModuleType, ResolutionKind, ModuleLoadResponse,
    ModuleLoadReferrer,
};
use deno_error::JsErrorBox;
use deno_fs::RealFs;
use deno_resolver::npm::{DenoInNpmPackageChecker, NpmResolver};
use deno_runtime::BootstrapOptions;
use deno_runtime::web_worker::{WebWorker, WebWorkerOptions, WebWorkerServiceOptions, WorkerThreadType};
use deno_runtime::ops::worker_host::{CreateWebWorkerCb, CreateWebWorkerArgs};
use import_map::ImportMap;

use crate::{snapshot_options_ext, gfx_host};
use deno_core::SharedArrayBufferStore;

use tokio::runtime::Handle;
use deno_runtime::deno_permissions::PermissionsContainer;
use deno_runtime::permissions::RuntimePermissionDescriptorParser;
// ======================= Import Map Module Loader =======================

#[derive(Clone)]
pub struct ImportMapModuleLoader {
    import_map: Option<Arc<ImportMap>>,
    script_dir: PathBuf,
    /// Path to the setup/shim file that should be auto-imported
    setup_module_path: Option<PathBuf>,
}

#[cfg(target_os = "windows")]
fn normalize_file_path(path: &str) -> String {
    // Remove leading slash before drive letter: /C:/... -> C:/...
    if path.len() > 2 && path.starts_with('/') && path.chars().nth(2) == Some(':') {
        path[1..].to_string()
    } else {
        path.to_string()
    }
}

#[cfg(not(target_os = "windows"))]
fn normalize_file_path(path: &str) -> String {
    path.to_string()
}

impl ImportMapModuleLoader {
    pub fn new(script_dir: PathBuf, import_map_path: Option<PathBuf>) -> Result<Self, anyhow::Error> {
        let import_map = if let Some(map_path) = import_map_path {
            let map_text = std::fs::read_to_string(&map_path)?;
            let base_url = ModuleSpecifier::from_file_path(&map_path)
                .map_err(|_| anyhow::anyhow!("Invalid import map path"))?;
            
            let result = import_map::parse_from_json(base_url, &map_text)?;
            
            for warning in result.diagnostics.iter() {
                log::warn!("Import map warning: {}", warning);
            }
            
            Some(Arc::new(result.import_map))
        } else {
            None
        };

        // Look for setup.js in the script directory
        let setup_path = std::env::current_dir()
            .map(|d| d.join("src/shims/setup.js"))
            .unwrap_or_else(|_| PathBuf::from("src/shims/setup.js"));
        let setup_module_path = if setup_path.exists() {
            log::info!("Found setup.js at {:?}, will auto-inject", setup_path);
            Some(setup_path)
        } else {
            log::info!("No setup.js found at {:?}", setup_path);
            None
        };

        Ok(Self {
            import_map,
            script_dir,
            setup_module_path,
        })
    }

    /// Create a loader with a custom setup file path
    pub fn with_setup_file(mut self, setup_path: PathBuf) -> Self {
        if setup_path.exists() {
            self.setup_module_path = Some(setup_path);
        }
        self
    }

    /// Check if this module is the setup module itself (to avoid circular injection)
    fn is_setup_module(&self, specifier: &ModuleSpecifier) -> bool {
        if let Some(ref setup_path) = self.setup_module_path {
            if let Ok(spec_path) = specifier.to_file_path() {
                // Canonicalize both paths for comparison
                let setup_canonical = setup_path.canonicalize().ok();
                let spec_canonical = spec_path.canonicalize().ok();
                
                if let (Some(s1), Some(s2)) = (setup_canonical, spec_canonical) {
                    return s1 == s2;
                }
                
                // Fallback to simple comparison
                return spec_path == *setup_path;
            }
        }
        false
    }

    /// Check if this is a module that should have setup.js injected
    fn should_inject_setup(&self, specifier: &ModuleSpecifier, code: &str) -> bool {
        // Don't inject if there's no setup module configured
        if self.setup_module_path.is_none() {
            log::info!("should_inject_setup: no setup module configured");
            return false;
        }

        // Don't inject into the setup module itself
        if self.is_setup_module(specifier) {
            log::info!("should_inject_setup: skipping setup module itself");
            return false;
        }

        // Don't inject into indexDBShim.js (the shim file that setup.js imports)
        let path = specifier.path();
        if path.contains("indexDBShim") || path.contains("indexedDBShim") {
            log::info!("should_inject_setup: skipping indexDB shim file");
            return false;
        }

        // Don't inject if the file already imports setup.js
        if code.contains("import \"./setup.js\"") 
            || code.contains("import './setup.js'")
            || code.contains("from \"./setup.js\"")
            || code.contains("from './setup.js'")
            || code.contains("import \"setup\"")
            || code.contains("import 'setup'")
            || code.contains("import \"../setup.js\"")
            || code.contains("import '../setup.js'")
            || code.contains("import \"../../setup.js\"")
            || code.contains("import '../../setup.js'")
            || code.contains("import \"../../../setup.js\"")
            || code.contains("import '../../../setup.js'")
            || code.contains("import \"../../../../setup.js\"")
            || code.contains("import '../../../../setup.js'")
        {
            log::info!("should_inject_setup: file already imports setup.js");
            return false;
        }

        // Skip node_modules - these are external libraries
        if path.contains("node_modules") {
            log::info!("should_inject_setup: skipping node_modules");
            return false;
        }

        // Skip three.js library files (they don't need browser shims typically)
        // But be careful - some three.js addons might need them
        if path.contains("/three/build/") || path.contains("/three/src/") {
            log::info!("should_inject_setup: skipping three.js core library file");
            return false;
        }

        // INJECT INTO ALL OTHER LOCAL JS FILES
        // This ensures any file using browser APIs gets the shims
        log::info!("should_inject_setup: WILL INJECT into {}", specifier);
        true
    }

    /// Get the import statement to prepend
    fn get_setup_import(&self, specifier: &ModuleSpecifier) -> Option<String> {
        let setup_path = self.setup_module_path.as_ref()?;
        
        if let Ok(module_path) = specifier.to_file_path() {
            if let Some(module_dir) = module_path.parent() {
                if let Some(rel_path) = pathdiff::diff_paths(setup_path, module_dir) {
                    // Convert Windows backslashes to forward slashes for JS imports
                    let rel_str = rel_path.to_string_lossy().replace('\\', "/");
                    
                    let import_path = if rel_str.starts_with('.') {
                        rel_str
                    } else {
                        format!("./{}", rel_str)
                    };
                    return Some(format!("import \"{}\";\n", import_path));
                }
            }
        }

        if let Ok(setup_url) = ModuleSpecifier::from_file_path(setup_path) {
            return Some(format!("import \"{}\";\n", setup_url));
        }

        None
    }
}



impl ModuleLoader for ImportMapModuleLoader {
    fn resolve(
        &self,
        specifier: &str,
        referrer: &str,
        _kind: ResolutionKind,
    ) -> Result<ModuleSpecifier, JsErrorBox> {
        let referrer_url = if referrer.is_empty() || referrer == "." {
            ModuleSpecifier::from_file_path(&self.script_dir.join("dummy.js"))
                .map_err(|_| JsErrorBox::generic("Invalid script dir"))?
        } else {
            ModuleSpecifier::parse(referrer)
                .or_else(|_| {
                    ModuleSpecifier::from_file_path(referrer)
                        .map_err(|_| deno_core::url::ParseError::EmptyHost)
                })
                .map_err(|e| JsErrorBox::generic(e.to_string()))?
        };

        // Try import map resolution first
        if let Some(ref import_map) = self.import_map {
            match import_map.resolve(specifier, &referrer_url) {
                Ok(resolved) => return Ok(resolved),
                Err(e) => {
                    let err_str = e.to_string();
                    if !err_str.contains("Relative import path") && 
                       !specifier.starts_with("./") && 
                       !specifier.starts_with("../") &&
                       !specifier.starts_with("/") {
                        // Unmapped bare specifier, fall through
                    } else {
                        return Err(JsErrorBox::generic(e.to_string()));
                    }
                }
            }
        }

        // Default resolution for relative/absolute paths
        let resolved = referrer_url.join(specifier)
            .map_err(|e| JsErrorBox::generic(e.to_string()))?;
        Ok(resolved)
    }

    fn load(
        &self,
        module_specifier: &ModuleSpecifier,
        _maybe_referrer: Option<&ModuleLoadReferrer>,
        _is_dyn_import: bool,
        _requested_module_type: RequestedModuleType,
    ) -> ModuleLoadResponse {
        let specifier = module_specifier.clone();
        
        log::info!("ModuleLoader: loading {}", specifier);
        
        let file_path = match specifier.to_file_path() {
            Ok(mut p) => {
                #[cfg(target_os = "windows")]
                {
                    // Fix path if it has leading slash before drive letter
                    let path_str = p.to_string_lossy().to_string();
                    if path_str.starts_with("\\\\?\\") {
                        p = PathBuf::from(path_str.trim_start_matches("\\\\?\\"));
                    }
                }
                p
            }
            Err(_) => {
                log::error!("ModuleLoader: cannot convert to file path: {}", specifier);
                return ModuleLoadResponse::Sync(Err(
                    JsErrorBox::generic(format!(
                        "Cannot load non-file URL: {}", specifier
                    ))
                ));
            }
        };
    
        log::info!("ModuleLoader: reading file {}", file_path.display());
        
        let code = match std::fs::read_to_string(&file_path) {
            Ok(c) => {
                log::info!("ModuleLoader: successfully read {} bytes from {}", c.len(), file_path.display());
                c
            }
            Err(e) => {
                log::error!("ModuleLoader: failed to read: {}", e);
                return ModuleLoadResponse::Sync(Err(
                    JsErrorBox::generic(format!(
                        "Failed to read {}: {}", file_path.display(), e
                    ))
                ));
            }
        };


        // Inject setup.js import if needed
        let final_code = if self.should_inject_setup(&specifier, &code) {
            if let Some(import_stmt) = self.get_setup_import(&specifier) {
                log::info!("ModuleLoader: INJECTING '{}' into {}", import_stmt.trim(), specifier);
                format!("{}{}", import_stmt, code)
            } else {
                log::warn!("ModuleLoader: should inject but get_setup_import returned None for {}", specifier);
                code
            }
        } else {
            log::info!("ModuleLoader: NOT injecting setup into {}", specifier);
            code
        };
    
        ModuleLoadResponse::Sync(Ok(ModuleSource::new(
            ModuleType::JavaScript,
            ModuleSourceCode::String(final_code.into()),
            &specifier,
            None,
        )))
    }
}

// ======================= Web Worker Creation =======================



pub fn create_web_worker_callback(
    module_loader: ImportMapModuleLoader,
    fs: Arc<RealFs>,
    shared_array_buffer_store: SharedArrayBufferStore,
    runtime_handle: Handle,
) -> Arc<CreateWebWorkerCb> {
    Arc::new(move |args: deno_runtime::ops::worker_host::CreateWebWorkerArgs| {
        let module_loader = module_loader.clone();
        let fs = fs.clone();
        let shared_array_buffer_store = shared_array_buffer_store.clone();
        let handle = runtime_handle.clone();

        // Enter the runtime context
        let _guard = handle.enter();

        let permission_desc_parser = Arc::new(
            RuntimePermissionDescriptorParser::new(sys_traits::impls::RealSys)
        );

        let service_options: WebWorkerServiceOptions<
            DenoInNpmPackageChecker,
            NpmResolver<sys_traits::impls::RealSys>,
            sys_traits::impls::RealSys,
        > = WebWorkerServiceOptions {
            deno_rt_native_addon_loader: None,
            module_loader: Rc::new(module_loader),
            permissions: PermissionsContainer::allow_all(permission_desc_parser),
            blob_store: Default::default(),
            broadcast_channel: Default::default(),
            feature_checker: Default::default(),
            node_services: Default::default(),
            npm_process_state_provider: Default::default(),
            root_cert_store_provider: Default::default(),
            shared_array_buffer_store: Some(shared_array_buffer_store),
            compiled_wasm_module_store: Default::default(),
            fs,
            maybe_inspector_server: None,
        };

        // For nested workers
        let nested_worker_cb: Arc<CreateWebWorkerCb> = Arc::new(|_| {
            panic!("Nested web workers are not supported")
        });

     
        let options = WebWorkerOptions {
            name: args.name.clone(),
            main_module: args.main_module.clone(),
            worker_id: args.worker_id,
            create_params: None,
            bootstrap: BootstrapOptions {
                location: Some(args.main_module.clone()),
                ..Default::default()
            },
            extensions: vec![
                crate::snapshot_options_ext::init(),
                // Uncomment if workers need GPU access:
                crate::gfx_host::init(),
            ],
            // ================================
            startup_snapshot: None,
            unsafely_ignore_certificate_errors: None,
            seed: None,
            format_js_error_fn: None,
            create_web_worker_cb: nested_worker_cb,
            close_on_idle: false,
            maybe_worker_metadata: None,
            stdio: Default::default(),
            trace_ops: None,
            enable_stack_trace_arg_in_ops: false,
            cache_storage_dir: None,
            enable_raw_imports: false,
            maybe_coverage_dir: None,
            worker_type: WorkerThreadType::Module,
        };

        WebWorker::bootstrap_from_options(service_options, options)
    })
}