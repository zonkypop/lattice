Lattice lets you run WebGPU/JS code natively - using Deno, WGPU, OpenXR and Vulkan!

### Run on Desktop

cargo run --release -- http://localhost:8000 (or some other URL)

### Run with WebXR/OpenXR

cargo run --release -- --xr http://localhost:8000 (or some other URL)


Lattice is a WIP, to do :
- WebRTC
- Steam Frame Builds
- General slop cleanup

Supported / tested platforms:
- Linux Desktop + OpenXR
- Windows Desktop + OpenXR
- Quest OpenXR
- MacOS Desktop

Audio via [web-audio-api-rs](https://github.com/orottier/web-audio-api-rs) / [cpal](https://github.com/rustaudio/cpal)

