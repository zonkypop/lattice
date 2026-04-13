## Building for Android (Quest 3) - Deno/V8 147.x

### The suffering is from V8 / Deno / rusty_v8 not providing prebuilt binaries for Android arm64.

---

## Prerequisites

### Android SDK/NDK Setup

```bash
export ANDROID_HOME=~/android
export ANDROID_NDK_HOME=$ANDROID_HOME/ndk/28.2.13676358
export ANDROID_NDK_ROOT=$ANDROID_NDK_HOME
export CMAKE_ANDROID_NDK=$ANDROID_NDK_HOME
```

### Rust Target

```bash
rustup target add aarch64-linux-android
cargo install cargo-apk
```

### Build Tools

```bash
sudo apt-get install -y ninja-build python3 clang lld cmake

# Latest gn (system version is too old)
cd /tmp
git clone https://gn.googlesource.com/gn
cd gn
python build/gen.py
ninja -C out
sudo cp out/gn /usr/local/bin/
```

### Cargo.toml

Ensure `[lib]` section has:
```toml
[lib]
path = "src/lib.rs"
crate-type = ["rlib", "cdylib"]
```

---

## Quick Start

All V8/Deno fixes are automated in `fix_v8_android.sh`. After `cargo fetch`:

```bash
# 1. Apply all V8 + deno_core patches
./fix_v8_android.sh

# 2. Build, install, and run
./run_quest.sh
```

First build takes 20-40 minutes (V8 compiles ~3600 C++ targets). Subsequent builds are incremental.

---

## Build Command

```bash
RUSTFLAGS="-C link-arg=$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/lib/clang/19/lib/linux/libclang_rt.builtins-aarch64-android.a" \
GN_ARGS="use_custom_libcxx=false" \
LIBCLANG_PATH="$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/lib" \
BINDGEN_EXTRA_CLANG_ARGS="--target=aarch64-linux-android23 --sysroot=$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64/sysroot" \
RUSTY_V8_SRC_BINDING_PATH="$PWD/src_binding_release_aarch64-linux-android.rs" \
V8_FROM_SOURCE=1 \
cargo apk build --release --lib
```

Key env vars explained:
- `RUSTFLAGS` - links clang builtins for Android aarch64
- `GN_ARGS` - disables V8's custom libc++ (use NDK's instead)
- `LIBCLANG_PATH` - points bindgen to NDK's clang 19
- `BINDGEN_EXTRA_CLANG_ARGS` - cross-compilation target/sysroot for non-V8 crates (libnghttp2 etc.)
- `RUSTY_V8_SRC_BINDING_PATH` - pre-built FFI bindings to skip V8's broken bindgen for Android
- `V8_FROM_SOURCE` - no prebuilt V8 for Android, must compile from source

---

## What fix_v8_android.sh Does

All fixes target `~/.cargo/registry/src/.../v8-147.1.0/`. These get wiped on `cargo update`.

### Fix 1: Create known-target-triples.txt
GN build needs this file listing valid Rust targets.

### Fix 2: Create missing .pydeps files
Android BUILD.gn references .pydeps files not included in the crate.

### Fix 3: Symlink NDK sysroot
V8 expects the NDK at `third_party/android_toolchain/ndk/...`.

### Fix 3b: Symlink host Linux sysroot
V8's build.rs forces `use_sysroot=true` for Android. Host build tools need a Debian sysroot.
Symlinks `/` as `build/linux/debian_bullseye_amd64-sysroot`.

### Fix 4: Download missing Rust crates
V8's `third_party/rust/` has BUILD.gn files but no source code (~181 crates).
Auto-downloads from crates.io based on version info in each BUILD.gn.

### Fix 5: Create metadata files
`.cargo-checksum.json` for downloaded vendor crates.

### Fix 6: V8 source patches

**Patch 6a: simdutf atomic functions**
`builtins-typed-array.cc` uses `simdutf::atomic_*` functions not available in this build.
Replace with non-atomic `simdutf::base64_to_binary_safe` / `simdutf::binary_to_base64`.

**Patch 6b: std::atomic_ref -> __atomic builtins (9 files)**
Android NDK's libc++ doesn't implement `std::atomic_ref` (C++20 library feature).
All usages replaced with GCC/Clang `__atomic` builtins across:
- `v8/src/base/atomicops.h` (central atomic operations header - complete rewrite)
- `v8/src/base/memcopy.h`
- `v8/src/heap/cppgc/heap-page.h`
- `v8/src/heap/cppgc/object-start-bitmap.h`
- `v8/src/heap/cppgc/heap-object-header.h`
- `v8/src/heap/sweeper.cc`
- `v8/src/objects/simd.cc`

### Fix 7: Patch allocation.h
Same `std::atomic_ref` issue but in `v8/include/cppgc/allocation.h` (missed by initial src/ search).

### Fix 8: Patch build.rs to skip bindgen
V8's bindgen step fails for Android cross-compilation (can't resolve NDK C headers).
Patches `build.rs` to honor `RUSTY_V8_SRC_BINDING_PATH` in `V8_FROM_SOURCE` mode,
skipping bindgen entirely when a pre-built binding file is provided.

### Fix 9: Download pre-built binding file
Downloads `src_binding_release_aarch64-unknown-linux-gnu.rs` from rusty_v8 GitHub releases.
aarch64 Linux and Android share the same ABI, so the bindings are compatible.

### Fix 10: Patch deno_core errno_location
`deno_core/uv_compat/tty.rs` only handles macOS and Linux.
Adds Android case using `__errno()` (Android's errno location function).

---

## Troubleshooting

- **"known-target-triples.txt" not found**: Run fix_v8_android.sh (Fix 1)
- **"debian_bullseye_amd64-sysroot" missing**: Run fix_v8_android.sh (Fix 3b)
- **std::atomic_ref errors**: Run fix_v8_android.sh (Fix 6b + 7)
- **LIBCLANG_PATH not set / FP_NAN undefined**: Ensure LIBCLANG_PATH points to NDK's clang 19 lib
- **uint32_t undefined in bindgen**: Use RUSTY_V8_SRC_BINDING_PATH instead of fixing bindgen
- **bits/libc-header-start.h not found**: Set BINDGEN_EXTRA_CLANG_ARGS with --target and --sysroot
- **errno_location not implemented**: Run fix_v8_android.sh (Fix 10)
- **All fixes wiped**: `cargo update` re-downloads crates. Re-run fix_v8_android.sh.
