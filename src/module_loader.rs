// src/module_loader.rs - Module loading and web worker creation

use std::path::PathBuf;
use std::rc::Rc;
use std::sync::Arc;

use deno_core::{
    ModuleSpecifier, ModuleLoader, ModuleSource, ModuleSourceCode,
    ModuleType, ResolutionKind, ModuleLoadResponse,
    ModuleLoadReferrer, ModuleLoadOptions,
};
use deno_error::JsErrorBox;
use deno_fs::RealFs;
use deno_resolver::npm::{DenoInNpmPackageChecker, NpmResolver};
use deno_runtime::BootstrapOptions;
use deno_runtime::web_worker::{WebWorker, WebWorkerOptions, WebWorkerServiceOptions, WorkerThreadType};
use deno_runtime::ops::worker_host::CreateWebWorkerCb;
use import_map::ImportMap;

use deno_core::{SharedArrayBufferStore, CompiledWasmModuleStore};

use tokio::runtime::Handle;
use deno_runtime::deno_permissions::PermissionsContainer;
use deno_runtime::permissions::RuntimePermissionDescriptorParser;

// ======================= Import Map Module Loader =======================

#[derive(Clone)]
pub struct ImportMapModuleLoader {
    import_map: Option<Arc<ImportMap>>,
    script_dir: PathBuf,
    setup_module_path: Option<PathBuf>,
}

#[cfg(target_os = "windows")]
fn normalize_file_path(path: &str) -> String {
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

   

        #[cfg(target_os = "android")]
        let setup_path = std::path::PathBuf::from("/data/local/tmp/combined_app/src/shims/setup.js");
        #[cfg(not(target_os = "android"))]
        let setup_path = std::env::current_dir()
            .map(|d| d.join("src/shims/setup.js"))
            .unwrap_or_else(|_| PathBuf::from("src/shims/setup.js"));

        let setup_module_path = setup_path.exists().then_some(setup_path);

        Ok(Self {
            import_map,
            script_dir,
            setup_module_path,
        })
    }

    pub fn with_setup_file(mut self, setup_path: PathBuf) -> Self {
        if setup_path.exists() {
            self.setup_module_path = Some(setup_path);
        }
        self
    }

    fn is_setup_module(&self, specifier: &ModuleSpecifier) -> bool {
        let Some(ref setup_path) = self.setup_module_path else { return false };
        let Ok(spec_path) = specifier.to_file_path() else { return false };
        
        if let (Some(s1), Some(s2)) = (setup_path.canonicalize().ok(), spec_path.canonicalize().ok()) {
            return s1 == s2;
        }
        spec_path == *setup_path
    }

    fn should_inject_setup(&self, specifier: &ModuleSpecifier, code: &str) -> bool {
        if self.setup_module_path.is_none() || self.is_setup_module(specifier) {
            return false;
        }

        let path = specifier.path();
        
        if path.contains("indexDBShim") || path.contains("indexedDBShim") {
            return false;
        }

        let setup_imports = [
            "import \"./setup.js\"", "import './setup.js'",
            "from \"./setup.js\"", "from './setup.js'",
            "import \"setup\"", "import 'setup'",
            "import \"../setup.js\"", "import '../setup.js'",
            "import \"../../setup.js\"", "import '../../setup.js'",
            "import \"../../../setup.js\"", "import '../../../setup.js'",
            "import \"../../../../setup.js\"", "import '../../../../setup.js'",
        ];
        if setup_imports.iter().any(|s| code.contains(s)) {
            return false;
        }

        if path.contains("node_modules") || path.contains("/three/build/") || path.contains("/three/src/") {
            return false;
        }

        true
    }

    fn get_setup_import(&self, specifier: &ModuleSpecifier) -> Option<String> {
        let setup_path = self.setup_module_path.as_ref()?;
        
        if let Ok(module_path) = specifier.to_file_path() {
            if let Some(module_dir) = module_path.parent() {
                if let Some(rel_path) = pathdiff::diff_paths(setup_path, module_dir) {
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

        ModuleSpecifier::from_file_path(setup_path)
            .ok()
            .map(|url| format!("import \"{}\";\n", url))
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

        if let Some(ref import_map) = self.import_map {
            match import_map.resolve(specifier, &referrer_url) {
                Ok(resolved) => return Ok(resolved),
                Err(e) => {
                    let err_str = e.to_string();
                    if err_str.contains("Relative import path") || 
                       specifier.starts_with("./") || 
                       specifier.starts_with("../") ||
                       specifier.starts_with("/") {
                        return Err(JsErrorBox::generic(e.to_string()));
                    }
                }
            }
        }

        referrer_url.join(specifier)
            .map_err(|e| JsErrorBox::generic(e.to_string()))
    }

    fn load(
        &self,
        module_specifier: &ModuleSpecifier,
        _maybe_referrer: Option<&ModuleLoadReferrer>,
        _options: ModuleLoadOptions,
    ) -> ModuleLoadResponse {
        let specifier = module_specifier.clone();
        
        let file_path = match specifier.to_file_path() {
            Ok(mut p) => {
                #[cfg(target_os = "windows")]
                {
                    let path_str = p.to_string_lossy().to_string();
                    if path_str.starts_with("\\\\?\\") {
                        p = PathBuf::from(path_str.trim_start_matches("\\\\?\\"));
                    }
                }
                p
            }
            Err(_) => {
                return ModuleLoadResponse::Sync(Err(
                    JsErrorBox::generic(format!("Cannot load non-file URL: {}", specifier))
                ));
            }
        };
    
        // Handle ?raw imports (Vite-style): export file content as a string
        let is_raw = specifier.query() == Some("raw");

        // Try appending .js if the file doesn't exist (Node/bundler convention)
        let file_path = if !file_path.exists() && file_path.extension().is_none() {
            let with_js = file_path.with_extension("js");
            if with_js.exists() { with_js } else { file_path }
        } else {
            file_path
        };

        let code = match std::fs::read_to_string(&file_path) {
            Ok(c) => c,
            Err(e) => {
                return ModuleLoadResponse::Sync(Err(
                    JsErrorBox::generic(format!("Failed to read {}: {}", file_path.display(), e))
                ));
            }
        };

        let final_code = if is_raw {
            // Wrap as default export string, escaping backticks and backslashes
            let escaped = code.replace('\\', "\\\\").replace('`', "\\`").replace("${", "\\${");
            format!("export default `{}`;", escaped)
        } else if self.should_inject_setup(&specifier, &code) {
            self.get_setup_import(&specifier)
                .map(|import| format!("{}{}", import, code))
                .unwrap_or(code)
        } else {
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
    compiled_wasm_module_store: CompiledWasmModuleStore,
    runtime_handle: Handle,
) -> Arc<CreateWebWorkerCb> {
    Arc::new(move |args| {
        let module_loader = module_loader.clone();
        let fs = fs.clone();
        let shared_array_buffer_store = shared_array_buffer_store.clone();
        let compiled_wasm_module_store = compiled_wasm_module_store.clone();
        let handle = runtime_handle.clone();

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
            compiled_wasm_module_store: Some(compiled_wasm_module_store),
            fs,
            bundle_provider: Default::default(),
            main_inspector_session_tx: Default::default(),
        };

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
                crate::gfx_host::init(),
            ],
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
            maybe_cpu_prof_config: None,
            worker_type: WorkerThreadType::Module,
        };

        WebWorker::bootstrap_from_options(service_options, options)
    })
}