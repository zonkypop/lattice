// tint_ffi.cpp — Thin C wrapper around Tint's WGSL→SPIR-V pipeline.
//
// Built against Dawn/Tint (2024+ IR-based API).
// Path: WGSL source → IR Module → SPIR-V binary (per entry point)

#include "tint_ffi.h"

#include <cstring>
#include <string>

#include "src/tint/lang/wgsl/reader/reader.h"
#include "src/tint/lang/spirv/writer/writer.h"

static TintResult make_error(const std::string& msg) {
    TintResult r;
    r.data = nullptr;
    r.len = 0;
    r.error = static_cast<char*>(malloc(msg.size() + 1));
    if (r.error) {
        memcpy(r.error, msg.c_str(), msg.size() + 1);
    }
    return r;
}

extern "C" {

TintResult tint_wgsl_to_spirv(const char* wgsl_source, size_t source_len,
                               const char* entry_point) {
    std::string source(wgsl_source, source_len);
    tint::Source::File file("shader.wgsl", source);

    // WGSL → IR
    auto ir_result = tint::wgsl::reader::WgslToIR(&file);
    if (ir_result != tint::Success) {
        return make_error("WGSL parse error: " + ir_result.Failure().reason);
    }

    // IR → SPIR-V (for the specified entry point)
    tint::spirv::writer::Options spirv_options;
    if (entry_point && entry_point[0] != '\0') {
        spirv_options.entry_point_name = std::string(entry_point);
    }

    auto spirv_result = tint::spirv::writer::Generate(ir_result.Get(), spirv_options);
    if (spirv_result != tint::Success) {
        return make_error("SPIR-V generation error: " + spirv_result.Failure().reason);
    }

    // Copy output
    auto& spirv = spirv_result.Get().spirv;

    TintResult r;
    r.error = nullptr;
    r.len = spirv.size();
    r.data = static_cast<uint32_t*>(malloc(r.len * sizeof(uint32_t)));
    if (!r.data) {
        return make_error("Failed to allocate SPIR-V output buffer");
    }
    memcpy(r.data, spirv.data(), r.len * sizeof(uint32_t));

    return r;
}

void tint_free_result(TintResult result) {
    if (result.data) free(result.data);
    if (result.error) free(result.error);
}

} // extern "C"
