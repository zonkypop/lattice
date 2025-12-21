# Running on Linux with chrome and WebGPU

google-chrome --enable-features=Vulkan,DefaultANGLEVulkan,UnsafeWebGPU --use-vulkan --enable-unsafe-webgpu
[your chromium browser] --enable-features=Vulkan,DefaultANGLEVulkan,UnsafeWebGPU --use-vulkan --enable-unsafe-webgpu

# Run in Desktop mode

cargo run --release

# Run in OpenXR mode

cargo run --release -- --xr
