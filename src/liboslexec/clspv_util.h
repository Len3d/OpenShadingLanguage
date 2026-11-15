// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#pragma once

#include <string>
#include <vector>

#include <OSL/oslconfig.h>

OSL_NAMESPACE_BEGIN

namespace pvt {

/// True if this build was configured against a clspv build tree (CLSPV_ROOT),
/// i.e. if OpenCL C can be lowered to Vulkan-compatible SPIR-V in-process.
bool clspv_available();

/// Lower `opencl_source` to Vulkan-flavor SPIR-V with clspv's own passes.
///
/// This is the SPIR-V counterpart of LLVM_Util::ptx_compile_group(): the caller
/// hands in a kernel source and gets back the binary words. clspv is used rather
/// than an LLVM backend because LLVM's SPIRV target can only emit the OpenCL
/// Kernel flavor (OpCapability Kernel, Physical64 OpenCL), which Vulkan rejects;
/// clspv emits OpCapability Shader / Logical GLSL450.
///
/// Returns false on failure, with the clspv build log in `out_log`.
OSL_DLL_EXPORT bool clspv_compile(
    const std::string& opencl_source, const std::string& options,
                   std::vector<uint32_t>& out_spirv, std::string& out_log);

}  // namespace pvt
OSL_NAMESPACE_END
