# Running on Linux with chrome and WebGPU

google-chrome --enable-features=Vulkan,DefaultANGLEVulkan,UnsafeWebGPU --use-vulkan --enable-unsafe-webgpu
[your chromium browser] --enable-features=Vulkan,DefaultANGLEVulkan,UnsafeWebGPU --use-vulkan --enable-unsafe-webgpu

# Run in Desktop mode

cargo run --release

# Run in OpenXR mode

cargo run --release -- --xr

# Android

export ANDROID_HOME=~/android
export ANDROID_NDK_HOME=$ANDROID_HOME/ndk/28.2.13676358
export ANDROID_NDK_ROOT=$ANDROID_NDK_HOME
export CMAKE_ANDROID_NDK=$ANDROID_NDK_HOME

echo "Building..."
RUSTFLAGS="-C link-arg=$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/lib/clang/19/lib/linux/libclang_rt.builtins-aarch64-android.a" \
GN_ARGS="use_custom_libcxx=false" \
BINDGEN_EXTRA_CLANG_ARGS="--sysroot=$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/sysroot -I$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/sysroot/usr/include/c++/v1" \
V8_FROM_SOURCE=1 \
cargo apk build --release --lib
