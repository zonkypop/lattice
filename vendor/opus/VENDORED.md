## Vendored libopus

Opus audio codec C source (from audiopus_sys 0.1.8), compiled via the `cc` crate in `build.rs`.

The `audiopus` / `audiopus_sys` crates build libopus using autotools (`autogen.sh` + `configure` + `make`). This breaks cross-compilation for Android (aarch64) because autotools ignores cargo-ndk's `CC` environment variables and builds for the host platform instead of the target.

By vendoring the source and using the `cc` crate directly, the build respects Cargo's cross-compilation toolchain automatically — works for Linux, macOS, Windows, and Android without extra setup.

### License

Opus is BSD-licensed. See `COPYING` and `LICENSE_PLEASE_READ.txt`.
