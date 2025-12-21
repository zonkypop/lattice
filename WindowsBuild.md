# Windows Build Guide

## Prerequisites

### 1. Install Rust

Download and install Rust from [rustup.rs](https://rustup.rs/)

After installation, verify:

```powershell
rustc --version
cargo --version
```

### 2. Install Visual Studio 2022

Download **Visual Studio 2022 Community** (NOT the 2026 preview):

- Direct link: https://c2rsetup.officeapps.live.com/c2r/downloadVS.aspx?sku=community&channel=Release&version=VS2022
- Or download **Build Tools 2022**: https://aka.ms/vs/17/release/vs_BuildTools.exe

During installation, select:

- **Desktop development with C++** workload
- Ensure these components are checked:
  - C++ AddressSanitizer
  - Windows 11 SDK
  - MSVC v142 - VS 2019 C++ x64/x86 build tools
  - C++/CLI support for v143 build tools

### 3. Install CMake 3.28.x

**Important:** CMake 4.x will NOT work. You must use version 3.28.x.

```powershell
winget install --id Kitware.CMake --version 3.28.1 --force
```

After installation, close and reopen PowerShell, then verify:

```powershell
cmake --version
```

Should show: `cmake version 3.28.1`

If you see version 4.x:

1. Open Windows Settings → Add or remove programs
2. Search for "CMake" and uninstall any 4.x versions
3. Reinstall CMake 3.28.1 using the command above
4. Restart PowerShell

### 4. Install NASM

```powershell
winget install NASM.NASM
```

Close and reopen PowerShell after installation.

### 5. Add Windows-sys Feature (if needed)

If you encounter errors about missing `Win32_System_SystemInformation`, add this to your `Cargo.toml`:

```toml
[dependencies.windows-sys]
version = "0.59"
features = ["Win32_System_SystemInformation"]
```

## Build Instructions

### 1. Open PowerShell in your project directory

### 2. Set required environment variables

```powershell
$env:AWS_LC_SYS_PREBUILT_NASM = "1"
$env:CFLAGS = "/std:c11"
```

### 3. Configure Rust toolchain

```powershell
rustup target add x86_64-pc-windows-msvc
rustup default stable-x86_64-pc-windows-msvc
```

### 4. Clean previous build artifacts

```powershell
cargo clean
```

### 5. Build the project

```powershell
cargo build --release
```

**Note:** Use `--release` mode. Debug builds may encounter PDB file limit errors.

## Running the Application

After successful compilation:

```powershell
.\target\release\combined_app.exe
```

Or use cargo:

```powershell
cargo run --release
```

## Troubleshooting

### CMake Version Error

**Error:** `Compatibility with CMake < 3.5 has been removed from CMake`
**Solution:** You're using CMake 4.x. Uninstall it and install CMake 3.28.1 (see Prerequisites #3)

### NASM Not Found

**Error:** `Missing dependency: nasm`
**Solution:**

1. Ensure NASM is installed: `nasm -v`
2. If not found, set: `$env:AWS_LC_SYS_PREBUILT_NASM = "1"`

### C11 Atomics Error

**Error:** `C atomics require C11 or later`
**Solution:** Set the CFLAGS: `$env:CFLAGS = "/std:c11"`

### Visual Studio 2026 Detected

**Error:** References to VS version 18 in build output
**Solution:** Uninstall Visual Studio 2026 preview. Only use Visual Studio 2022 (version 17.x)

### PDB Linker Error

**Error:** `LINK : fatal error LNK1318: Unexpected PDB error`
**Solution:** Run `cargo clean` and always use `cargo build --release`

### Environment Variables Reset

If you close PowerShell, you'll need to set the environment variables again:

```powershell
$env:AWS_LC_SYS_PREBUILT_NASM = "1"
$env:CFLAGS = "/std:c11"
```

## Quick Build Script

```powershell
# Set required environment variables
$env:AWS_LC_SYS_PREBUILT_NASM = "1"
$env:CFLAGS = "/std:c11"

# Clean and build
cargo clean
cargo build --release

```


Uncomment .cargo/config.toml