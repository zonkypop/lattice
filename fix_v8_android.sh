#!/bin/bash
set -e

V8_RUST="$HOME/.cargo/registry/src/index.crates.io-1949cf8c6b5b557f/v8-147.1.0"
ANDROID_NDK_HOME="$HOME/android/ndk/28.2.13676358"

echo "=== Fixing V8 147.1.0 for Android build ==="
echo "V8_RUST=$V8_RUST"
echo ""

# -------------------------------------------------------
# Fix 1: Create known-target-triples.txt
# -------------------------------------------------------
echo ">>> Fix 1: Creating known-target-triples.txt"
mkdir -p "$V8_RUST/build/rust"
cat > "$V8_RUST/build/rust/known-target-triples.txt" << 'EOF'
aarch64-linux-android
x86_64-unknown-linux-gnu
EOF
echo "    Done."

# -------------------------------------------------------
# Fix 2: Create missing pydeps files
# -------------------------------------------------------
echo ">>> Fix 2: Creating missing pydeps files"
cd "$V8_RUST/build/android"
pydeps_count=0
grep -ro '"[^"]*\.pydeps"' BUILD.gn 2>/dev/null | tr -d '"' | sort -u | while read f; do
  mkdir -p "$(dirname "$f")"
  touch "$f"
done
# Also check for pydeps references in other BUILD.gn files
find "$V8_RUST/build" -name "BUILD.gn" -exec grep -ho '"[^"]*\.pydeps"' {} \; 2>/dev/null | tr -d '"' | sort -u | while read f; do
  if [ ! -f "$V8_RUST/build/$f" ] && [ ! -f "$V8_RUST/$f" ]; then
    dir=$(dirname "$f")
    mkdir -p "$V8_RUST/build/$dir" 2>/dev/null || true
    touch "$V8_RUST/build/$f" 2>/dev/null || true
  fi
done
echo "    Done."

# -------------------------------------------------------
# Fix 3: Symlink NDK sysroot
# -------------------------------------------------------
echo ">>> Fix 3: Symlinking NDK sysroot"
mkdir -p "$V8_RUST/third_party/android_toolchain/ndk/toolchains/llvm/prebuilt/"
ln -sf "$ANDROID_NDK_HOME/toolchains/llvm/prebuilt/linux-x86_64" \
  "$V8_RUST/third_party/android_toolchain/ndk/toolchains/llvm/prebuilt/linux-x86_64"
echo "    Done."

# -------------------------------------------------------
# Fix 3b: Symlink host Linux sysroot (for host build tools)
# -------------------------------------------------------
echo ">>> Fix 3b: Symlinking host Linux sysroot"
mkdir -p "$V8_RUST/build/linux"
ln -sf / "$V8_RUST/build/linux/debian_bullseye_amd64-sysroot"
echo "    Done."

# -------------------------------------------------------
# Fix 4: Download missing Rust crates to vendor dir
# -------------------------------------------------------
echo ">>> Fix 4: Downloading missing Rust crates"
VENDOR_DIR="$V8_RUST/third_party/rust/chromium_crates_io/vendor"
mkdir -p "$VENDOR_DIR"

download_crate() {
  local crate_dir=$1   # e.g., "serde"
  local version_dir=$2 # e.g., "v1"
  local build_gn="$V8_RUST/third_party/rust/$crate_dir/$version_dir/BUILD.gn"

  if [ ! -f "$build_gn" ]; then
    echo "    SKIP: No BUILD.gn for $crate_dir/$version_dir"
    return
  fi

  # Extract exact version from BUILD.gn
  local exact_version
  exact_version=$(grep 'cargo_pkg_version' "$build_gn" | head -1 | sed 's/.*= *"\(.*\)".*/\1/')

  if [ -z "$exact_version" ]; then
    echo "    SKIP: No version found for $crate_dir/$version_dir"
    return
  fi

  # The crate name on crates.io uses hyphens, but dir may use underscores
  # Check cargo_pkg_name in BUILD.gn first
  local pkg_name
  pkg_name=$(grep 'cargo_pkg_name' "$build_gn" | head -1 | sed 's/.*= *"\(.*\)".*/\1/')
  if [ -z "$pkg_name" ]; then
    # Fall back to converting underscores to hyphens
    pkg_name="${crate_dir//_/-}"
  fi

  # The vendor dir name format: {crate-name-with-hyphens}-v{epoch} or {crate_name}-v{epoch}
  # Check what the BUILD.gn references
  local vendor_name
  vendor_name=$(grep -o 'vendor/[^/]*' "$build_gn" | head -1 | sed 's|vendor/||')
  if [ -z "$vendor_name" ]; then
    vendor_name="${pkg_name}-${version_dir}"
  fi

  local target_dir="$VENDOR_DIR/$vendor_name"

  # Check if already downloaded
  if [ -f "$target_dir/Cargo.toml" ] || [ -f "$target_dir/src/lib.rs" ]; then
    return
  fi

  echo "    Downloading $pkg_name $exact_version -> $vendor_name"

  # Download from crates.io
  local tmp_crate="/tmp/v8_crate.crate"
  local tmp_extract="/tmp/v8_crate_extract"

  if ! curl -sL "https://static.crates.io/crates/$pkg_name/$pkg_name-$exact_version.crate" -o "$tmp_crate" 2>/dev/null; then
    # Try with underscores instead of hyphens
    local alt_name="${pkg_name//-/_}"
    if ! curl -sL "https://static.crates.io/crates/$alt_name/$alt_name-$exact_version.crate" -o "$tmp_crate" 2>/dev/null; then
      echo "    FAILED to download $pkg_name-$exact_version"
      return
    fi
    pkg_name="$alt_name"
  fi

  rm -rf "$tmp_extract" && mkdir -p "$tmp_extract"

  if ! tar xzf "$tmp_crate" -C "$tmp_extract" 2>/dev/null; then
    echo "    FAILED to extract $pkg_name-$exact_version"
    return
  fi

  # Find the extracted directory
  local extracted_dir
  extracted_dir=$(ls -d "$tmp_extract"/*/ 2>/dev/null | head -1)
  if [ -z "$extracted_dir" ]; then
    echo "    FAILED: no extracted dir for $pkg_name-$exact_version"
    return
  fi

  mkdir -p "$target_dir"
  cp -r "$extracted_dir"* "$target_dir/"
  rm -rf "$tmp_crate" "$tmp_extract"
}

# Iterate over all crate directories
for crate_path in "$V8_RUST"/third_party/rust/*/; do
  crate_name=$(basename "$crate_path")
  [ "$crate_name" = "chromium_crates_io" ] && continue

  for version_path in "$crate_path"v*/; do
    if [ -d "$version_path" ]; then
      version_name=$(basename "$version_path")

      # Check if source is missing
      src_files=$(find "$version_path" -name "*.rs" -o -name "Cargo.toml" 2>/dev/null | head -1)
      if [ -z "$src_files" ]; then
        download_crate "$crate_name" "$version_name"
      fi
    fi
  done
done

echo "    Done downloading crates."

# -------------------------------------------------------
# Fix 5: Create metadata files where needed
# -------------------------------------------------------
echo ">>> Fix 5: Creating metadata files"
for d in "$VENDOR_DIR"/*/; do
  if [ -d "$d" ]; then
    [ ! -f "$d/.cargo-checksum.json" ] && echo '{"files":{}}' > "$d/.cargo-checksum.json"
  fi
done
echo "    Done."

# -------------------------------------------------------
# Fix 6: V8 Source Patches
# -------------------------------------------------------
echo ">>> Fix 6: Applying V8 source patches"

# Patch 1: simdutf atomic functions
BUILTINS_FILE="$V8_RUST/v8/src/builtins/builtins-typed-array.cc"
if grep -q "simdutf::atomic_base64_to_binary_safe" "$BUILTINS_FILE" 2>/dev/null; then
  echo "    Patching simdutf atomic functions..."
  sed -i 's/simdutf::atomic_base64_to_binary_safe/simdutf::base64_to_binary_safe/g' "$BUILTINS_FILE"
  sed -i 's/simdutf::atomic_binary_to_base64/simdutf::binary_to_base64/g' "$BUILTINS_FILE"
  echo "    Done."
else
  echo "    simdutf patch not needed or already applied."
fi

# Patch 2: std::atomic_ref - replace with __atomic builtins throughout V8
# Android NDK's libc++ doesn't support std::atomic_ref (C++20 library feature)
echo "    Patching std::atomic_ref usages..."

# atomicops.h - complete rewrite using __atomic builtins
ATOMICOPS="$V8_RUST/v8/src/base/atomicops.h"
if grep -q "std::atomic_ref" "$ATOMICOPS" 2>/dev/null; then
  echo "      Patching atomicops.h..."
  sed -i 's/std::atomic_ref<T>(\*ptr)\.compare_exchange_strong(old_value, new_value,\s*std::memory_order_relaxed);/__atomic_compare_exchange_n(ptr, \&old_value, new_value, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED);/g' "$ATOMICOPS"
  sed -i 's/std::atomic_ref<T>(\*ptr)\.compare_exchange_strong(old_value, new_value,\s*std::memory_order_acq_rel,\s*std::memory_order_acquire);/__atomic_compare_exchange_n(ptr, \&old_value, new_value, false, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE);/g' "$ATOMICOPS"
  sed -i 's/std::atomic_ref<T>(\*ptr)\.compare_exchange_strong(old_value, new_value,\s*std::memory_order_release,\s*std::memory_order_relaxed);/__atomic_compare_exchange_n(ptr, \&old_value, new_value, false, __ATOMIC_RELEASE, __ATOMIC_RELAXED);/g' "$ATOMICOPS"
  sed -i 's/std::atomic_ref<T>(\*ptr)\.compare_exchange_strong(old_value, new_value,\s*std::memory_order_seq_cst,\s*std::memory_order_seq_cst);/__atomic_compare_exchange_n(ptr, \&old_value, new_value, false, __ATOMIC_SEQ_CST, __ATOMIC_SEQ_CST);/g' "$ATOMICOPS"
  sed -i 's/return std::atomic_ref<T>(\*ptr)\.exchange(new_value,\s*std::memory_order_relaxed);/return __atomic_exchange_n(ptr, new_value, __ATOMIC_RELAXED);/g' "$ATOMICOPS"
  sed -i 's/return std::atomic_ref<T>(\*ptr)\.exchange(new_value,\s*std::memory_order_seq_cst);/return __atomic_exchange_n(ptr, new_value, __ATOMIC_SEQ_CST);/g' "$ATOMICOPS"
  sed -i 's/return std::atomic_ref<T>(\*ptr)\.fetch_or(bits, std::memory_order_relaxed);/return __atomic_fetch_or(ptr, bits, __ATOMIC_RELAXED);/g' "$ATOMICOPS"
  sed -i 's/return increment + std::atomic_ref<T>(\*ptr)\.fetch_add(\s*increment, std::memory_order_relaxed);/return increment + __atomic_fetch_add(ptr, increment, __ATOMIC_RELAXED);/g' "$ATOMICOPS"
  sed -i 's/std::atomic_ref<T>(\*ptr)\.store(value, std::memory_order_relaxed);/__atomic_store_n(ptr, value, __ATOMIC_RELAXED);/g' "$ATOMICOPS"
  sed -i 's/std::atomic_ref<T>(\*ptr)\.store(value, std::memory_order_release);/__atomic_store_n(ptr, value, __ATOMIC_RELEASE);/g' "$ATOMICOPS"
  sed -i 's/std::atomic_ref<T>(\*ptr)\.store(value, std::memory_order_seq_cst);/__atomic_store_n(ptr, value, __ATOMIC_SEQ_CST);/g' "$ATOMICOPS"
  sed -i 's/return std::atomic_ref<T>(\*const_cast<T\*>(ptr))\s*\.load(std::memory_order_relaxed);/return __atomic_load_n(const_cast<T*>(ptr), __ATOMIC_RELAXED);/g' "$ATOMICOPS"
  sed -i 's/return std::atomic_ref<T>(\*const_cast<T\*>(ptr))\s*\.load(std::memory_order_acquire);/return __atomic_load_n(const_cast<T*>(ptr), __ATOMIC_ACQUIRE);/g' "$ATOMICOPS"
  sed -i 's/return std::atomic_ref<T>(\*const_cast<T\*>(ptr))\s*\.load(std::memory_order_seq_cst);/return __atomic_load_n(const_cast<T*>(ptr), __ATOMIC_SEQ_CST);/g' "$ATOMICOPS"
fi

# memcopy.h
MEMCOPY="$V8_RUST/v8/src/base/memcopy.h"
if grep -q "std::atomic_ref" "$MEMCOPY" 2>/dev/null; then
  echo "      Patching memcopy.h..."
  sed -i 's/std::atomic_ref<T>(destination\[i\])\.store(value, std::memory_order_relaxed);/__atomic_store_n(\&destination[i], value, __ATOMIC_RELAXED);/g' "$MEMCOPY"
fi

# heap-page.h
HEAP_PAGE="$V8_RUST/v8/src/heap/cppgc/heap-page.h"
if grep -q "std::atomic_ref" "$HEAP_PAGE" 2>/dev/null; then
  echo "      Patching heap-page.h..."
  sed -i 's/(void)std::atomic_ref<PageType>(const_cast<PageType\&>(type_))\s*\.load(std::memory_order_acquire);/(void)__atomic_load_n(const_cast<PageType*>(\&type_), __ATOMIC_ACQUIRE);/g' "$HEAP_PAGE"
  sed -i 's/std::atomic_ref<PageType>(type_)\.store(type_, std::memory_order_release);/__atomic_store_n(\&type_, type_, __ATOMIC_RELEASE);/g' "$HEAP_PAGE"
fi

# object-start-bitmap.h
OSB="$V8_RUST/v8/src/heap/cppgc/object-start-bitmap.h"
if grep -q "std::atomic_ref" "$OSB" 2>/dev/null; then
  echo "      Patching object-start-bitmap.h..."
  sed -i 's/std::atomic_ref<uint8_t>(object_start_bit_map_\[cell_index\])\s*\.store(value, std::memory_order_release);/__atomic_store_n(\&object_start_bit_map_[cell_index], value, __ATOMIC_RELEASE);/g' "$OSB"
  sed -i 's/return std::atomic_ref<uint8_t>(\s*const_cast<uint8_t\&>(object_start_bit_map_\[cell_index\]))\s*\.load(std::memory_order_acquire);/return __atomic_load_n(const_cast<uint8_t*>(\&object_start_bit_map_[cell_index]), __ATOMIC_ACQUIRE);/g' "$OSB"
fi

# heap-object-header.h
HOH="$V8_RUST/v8/src/heap/cppgc/heap-object-header.h"
if grep -q "std::atomic_ref" "$HOH" 2>/dev/null; then
  echo "      Patching heap-object-header.h..."
  sed -i 's/std::atomic_ref<uint16_t>(encoded_high_)\s*\.store(/__atomic_store_n(\&encoded_high_,/g' "$HOH"
  sed -i 's/std::memory_order_relaxed);/__ATOMIC_RELAXED);/g' "$HOH"
  # TryMarkAtomic
  sed -i 's/std::atomic_ref<uint16_t> atomic_encoded(encoded_low_);/\/\/ Using __atomic builtins instead of std::atomic_ref/g' "$HOH"
  sed -i 's/uint16_t old_value = atomic_encoded\.load(std::memory_order_relaxed);/uint16_t old_value = __atomic_load_n(\&encoded_low_, __ATOMIC_RELAXED);/g' "$HOH"
  sed -i 's/return atomic_encoded\.compare_exchange_strong(old_value, new_value,\s*std::memory_order_relaxed);/return __atomic_compare_exchange_n(\&encoded_low_, \&old_value, new_value, false, __ATOMIC_RELAXED, __ATOMIC_RELAXED);/g' "$HOH"
  # LoadEncoded / StoreEncoded
  sed -i 's/return std::atomic_ref(const_cast<uint16_t\&>(half))\.load(memory_order);/return __atomic_load_n(const_cast<uint16_t*>(\&half), static_cast<int>(memory_order));/g' "$HOH"
  sed -i 's/std::atomic_ref<uint16_t> atomic_encoded(half);/\/\/ Using __atomic builtins instead of std::atomic_ref/g' "$HOH"
  sed -i 's/uint16_t value = atomic_encoded\.load(std::memory_order_relaxed);/uint16_t value = __atomic_load_n(\&half, __ATOMIC_RELAXED);/g' "$HOH"
  sed -i 's/atomic_encoded\.store(value, memory_order);/__atomic_store_n(\&half, value, static_cast<int>(memory_order));/g' "$HOH"
fi

# sweeper.cc
SWEEPER="$V8_RUST/v8/src/heap/sweeper.cc"
if grep -q "std::atomic_ref" "$SWEEPER" 2>/dev/null; then
  echo "      Patching sweeper.cc..."
  sed -i 's/std::atomic_ref<Tagged_t>(\*current_addr)\s*\.store(kZapTagged, std::memory_order_relaxed);/__atomic_store_n(current_addr, kZapTagged, __ATOMIC_RELAXED);/g' "$SWEEPER"
fi

# simd.cc
SIMD_FILE="$V8_RUST/v8/src/objects/simd.cc"
if grep -q "std::atomic_ref" "$SIMD_FILE" 2>/dev/null; then
  echo "      Patching simd.cc..."
  sed -i 's/std::atomic_ref<uint8_t>(buffer\[index++\])\s*\.store(result\.value(), std::memory_order_relaxed);/__atomic_store_n(\&buffer[index++], result.value(), __ATOMIC_RELAXED);/g' "$SIMD_FILE"
fi

echo "    Done."

# -------------------------------------------------------
# Fix 7: Patch allocation.h (std::atomic_ref in include/ dir)
# -------------------------------------------------------
echo ">>> Fix 7: Patching allocation.h"
ALLOC="$V8_RUST/v8/include/cppgc/allocation.h"
if grep -q "std::atomic_ref" "$ALLOC" 2>/dev/null; then
  echo "      Patching allocation.h..."
  sed -i 's/std::atomic_ref<uint16_t>(\*reinterpret_cast<uint16_t\*>(payload))\.load(std::memory_order_acquire);/__atomic_load_n(reinterpret_cast<uint16_t*>(payload), __ATOMIC_ACQUIRE);/g' "$ALLOC"
  sed -i 's/std::atomic_ref<uint16_t>(\*reinterpret_cast<uint16_t\*>(payload))\.store(value, std::memory_order_release);/__atomic_store_n(reinterpret_cast<uint16_t*>(payload), value, __ATOMIC_RELEASE);/g' "$ALLOC"
fi
echo "    Done."

# -------------------------------------------------------
# Fix 8: Patch build.rs to skip bindgen when RUSTY_V8_SRC_BINDING_PATH is set
# -------------------------------------------------------
echo ">>> Fix 8: Patching build.rs to allow pre-built bindings"
BUILD_RS="$V8_RUST/build.rs"
if grep -q 'build_v8(is_asan);' "$BUILD_RS" && ! grep -q 'RUSTY_V8_SRC_BINDING_PATH' <(sed -n '/build_v8(is_asan)/,/return;/p' "$BUILD_RS"); then
  echo "      Patching build.rs..."
  sed -i '/build_v8(is_asan);/{
    n
    s/build_binding();/\/\/ Allow skipping bindgen by providing a pre-built binding file\n    if let Ok(binding) = env::var("RUSTY_V8_SRC_BINDING_PATH") {\n      println!("cargo:rustc-env=RUSTY_V8_SRC_BINDING_PATH={binding}");\n    } else {\n      build_binding();\n    }/
  }' "$BUILD_RS"
fi

# Also patch build.rs for Android clang resource dir
if ! grep -q 'target_os == "android"' "$BUILD_RS"; then
  echo "      Adding Android support to build.rs clang args..."
  sed -i 's/} else if target_os == "linux" {/} else if target_os == "linux" || target_os == "android" {/' "$BUILD_RS"
fi
echo "    Done."

# -------------------------------------------------------
# Fix 9: Download pre-built binding file
# -------------------------------------------------------
echo ">>> Fix 9: Downloading pre-built binding file for aarch64"
BINDING_FILE="$HOME/Desktop/DEV/research/src_binding_release_aarch64-linux-android.rs"
if [ ! -f "$BINDING_FILE" ]; then
  echo "      Downloading from rusty_v8 releases..."
  V8_VERSION=$(grep '^version' "$V8_RUST/Cargo.toml" | head -1 | sed 's/.*"\(.*\)".*/\1/')
  curl -sL -o "$BINDING_FILE" "https://github.com/denoland/rusty_v8/releases/download/v${V8_VERSION}/src_binding_release_aarch64-unknown-linux-gnu.rs"
  echo "      Downloaded $(wc -l < "$BINDING_FILE") lines."
else
  echo "      Already exists."
fi
echo "    Done."

# -------------------------------------------------------
# Fix 10: Patch deno_core errno_location for Android
# -------------------------------------------------------
echo ">>> Fix 10: Patching deno_core errno_location for Android"
DENO_CORE_TTY=$(find "$HOME/.cargo/registry/src" -path "*/deno_core-*/uv_compat/tty.rs" 2>/dev/null | head -1)
if [ -n "$DENO_CORE_TTY" ] && ! grep -q 'target_os = "android"' "$DENO_CORE_TTY" 2>/dev/null; then
  echo "      Patching $DENO_CORE_TTY..."
  sed -i 's/#\[cfg(not(any(target_os = "macos", target_os = "linux")))]/\
  #[cfg(target_os = "android")]\
  fn errno_location() -> *mut c_int {\
    unsafe extern "C" {\
      fn __errno() -> *mut c_int;\
    }\
    unsafe { __errno() }\
  }\
\
  #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "android")))]/' "$DENO_CORE_TTY"
fi
echo "    Done."

echo ""
echo "=== All fixes applied! ==="
echo "You can now try building with run_quest.sh"
