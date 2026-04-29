#ifndef TINT_FFI_H
#define TINT_FFI_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef struct {
    uint32_t* data;   // SPIR-V words (caller must free via tint_free_result)
    size_t    len;     // number of u32 words
    char*     error;   // error message if compilation failed (null on success)
} TintResult;

/// Compile WGSL source code to SPIR-V binary for a specific entry point.
/// @param wgsl_source   WGSL source code (not null-terminated required)
/// @param source_len    length of source in bytes
/// @param entry_point   entry point function name (null-terminated C string)
/// The caller MUST call tint_free_result() to release the memory.
TintResult tint_wgsl_to_spirv(const char* wgsl_source, size_t source_len,
                               const char* entry_point);

/// Free memory allocated by tint_wgsl_to_spirv().
void tint_free_result(TintResult result);

#ifdef __cplusplus
}
#endif

#endif // TINT_FFI_H
