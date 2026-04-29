#!/usr/bin/env bash
# build_tint.sh — Build Tint as a static library from Dawn source.
#
# Usage:
#   ./build_tint.sh                          # desktop (x86_64-linux)
#   ./build_tint.sh android                  # Android aarch64 (for Quest)
#
# Prerequisites:
#   - CMake 3.20+
#   - Python 3
#   - depot_tools on PATH (for gclient)
#   - For Android: ANDROID_NDK_HOME set
#
# Output:
#   tint-ffi/lib/<target>/libtint_ffi.a      # the static lib to link

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DAWN_DIR="${DAWN_DIR:-$SCRIPT_DIR/dawn}"
TARGET="${1:-desktop}"

echo "=== Building Tint static library (target: $TARGET) ==="

# ---- Clone Dawn if needed ----
if [ ! -d "$DAWN_DIR" ]; then
    echo "Cloning Dawn..."
    git clone https://dawn.googlesource.com/dawn "$DAWN_DIR"
    cd "$DAWN_DIR"

    # Bootstrap dependencies via gclient (needs depot_tools)
    if command -v gclient &>/dev/null; then
        cp scripts/standalone.gclient .gclient
        gclient sync --no-history
    else
        echo ""
        echo "WARNING: depot_tools/gclient not found."
        echo "Install depot_tools and add to PATH:"
        echo "  git clone https://chromium.googlesource.com/chromium/tools/depot_tools.git"
        echo "  export PATH=\$PWD/depot_tools:\$PATH"
        echo ""
        exit 1
    fi
else
    echo "Using existing Dawn at: $DAWN_DIR"
fi

cd "$DAWN_DIR"

# ---- Common CMake flags (minimize build scope) ----
COMMON_FLAGS=(
    -DCMAKE_BUILD_TYPE=Release
    -DDAWN_BUILD_SAMPLES=OFF
    -DDAWN_BUILD_TESTS=OFF
    -DDAWN_BUILD_BENCHMARKS=OFF
    -DDAWN_ENABLE_D3D11=OFF
    -DDAWN_ENABLE_D3D12=OFF
    -DDAWN_ENABLE_METAL=OFF
    -DDAWN_ENABLE_NULL=OFF
    -DDAWN_ENABLE_DESKTOP_GL=OFF
    -DDAWN_ENABLE_OPENGLES=OFF
    -DDAWN_ENABLE_VULKAN=OFF
    -DTINT_BUILD_SPV_READER=OFF
    -DTINT_BUILD_SPV_WRITER=ON
    -DTINT_BUILD_WGSL_READER=ON
    -DTINT_BUILD_WGSL_WRITER=OFF
    -DTINT_BUILD_GLSL_WRITER=OFF
    -DTINT_BUILD_GLSL_VALIDATOR=OFF
    -DTINT_BUILD_HLSL_WRITER=OFF
    -DTINT_BUILD_MSL_WRITER=OFF
    -DTINT_BUILD_CMD_TOOLS=OFF
    -DTINT_BUILD_TESTS=OFF
    -DTINT_BUILD_BENCHMARKS=OFF
)

if [ "$TARGET" = "android" ]; then
    # ---- Android cross-compile (aarch64) ----
    if [ -z "${ANDROID_NDK_HOME:-}" ]; then
        echo "ERROR: ANDROID_NDK_HOME not set"
        exit 1
    fi

    BUILD_DIR="out-android-arm64"
    TOOLCHAIN="${ANDROID_NDK_HOME}/build/cmake/android.toolchain.cmake"
    OUTPUT_DIR="$SCRIPT_DIR/lib/aarch64-linux-android"

    cmake -B "$BUILD_DIR" -G Ninja \
        "${COMMON_FLAGS[@]}" \
        -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN" \
        -DANDROID_ABI=arm64-v8a \
        -DANDROID_PLATFORM=android-32 \
        -DANDROID_STL=c++_static

    cmake --build "$BUILD_DIR" -j$(nproc) --target tint_lang_spirv_writer tint_lang_wgsl_reader tint_lang_core

else
    # ---- Desktop native ----
    BUILD_DIR="out-desktop"
    OUTPUT_DIR="$SCRIPT_DIR/lib/x86_64-unknown-linux-gnu"

    cmake -B "$BUILD_DIR" -G Ninja "${COMMON_FLAGS[@]}"
    cmake --build "$BUILD_DIR" -j$(nproc) --target tint_lang_spirv_writer tint_lang_wgsl_reader tint_lang_core
fi

# ---- Compile the FFI wrapper ----
echo "=== Compiling tint_ffi.cpp ==="
mkdir -p "$OUTPUT_DIR"

if [ "$TARGET" = "android" ]; then
    # Use NDK clang++
    NDK_CLANG="${ANDROID_NDK_HOME}/toolchains/llvm/prebuilt/linux-x86_64/bin/aarch64-linux-android32-clang++"
    "$NDK_CLANG" -c -O2 -std=c++20 -fPIC \
        -I"$DAWN_DIR" \
        -I"$DAWN_DIR/$BUILD_DIR/gen" \
        -I"$DAWN_DIR/include" \
        "$SCRIPT_DIR/tint_ffi.cpp" \
        -o "$OUTPUT_DIR/tint_ffi.o"
else
    c++ -c -O2 -std=c++20 -fPIC \
        -I"$DAWN_DIR" \
        -I"$DAWN_DIR/$BUILD_DIR/gen" \
        -I"$DAWN_DIR/include" \
        "$SCRIPT_DIR/tint_ffi.cpp" \
        -o "$OUTPUT_DIR/tint_ffi.o"
fi

# ---- Bundle everything into one archive ----
echo "=== Creating combined static library ==="

# Collect all tint .a files
TINT_LIBS=$(find "$DAWN_DIR/$BUILD_DIR" -name "libtint*.a" | sort)

# Create a thin archive from all tint libs + our wrapper
echo "Found tint libraries:"
echo "$TINT_LIBS"

# Use a MRI script to merge archives
MRI_SCRIPT=$(mktemp)
echo "CREATE $OUTPUT_DIR/libtint_ffi.a" > "$MRI_SCRIPT"
for lib in $TINT_LIBS; do
    echo "ADDLIB $lib" >> "$MRI_SCRIPT"
done
echo "ADDMOD $OUTPUT_DIR/tint_ffi.o" >> "$MRI_SCRIPT"
echo "SAVE" >> "$MRI_SCRIPT"
echo "END" >> "$MRI_SCRIPT"

ar -M < "$MRI_SCRIPT"
rm "$MRI_SCRIPT" "$OUTPUT_DIR/tint_ffi.o"
ranlib "$OUTPUT_DIR/libtint_ffi.a"

echo ""
echo "=== Done! ==="
echo "Static library: $OUTPUT_DIR/libtint_ffi.a"
echo "Size: $(du -h "$OUTPUT_DIR/libtint_ffi.a" | cut -f1)"
