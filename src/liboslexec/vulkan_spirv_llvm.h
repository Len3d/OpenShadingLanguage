// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#pragma once

#include <string>
#include <vector>

#include <OSL/oslconfig.h>

OSL_NAMESPACE_BEGIN

namespace pvt {

/// Compile the legalised device IR with LLVM's own SPIR-V backend instead of
/// clspv.
///
/// LLVM registers `spirv`/`spirv32`/`spirv64` as separate targets, and its
/// SPIR-V backend emits two very different flavors depending on the OS
/// component of the triple:
///
///   spirv64-unknown-unknown  -> OpCapability Kernel + Physical64 OpenCL
///                               (the "OpenCL" flavor Vulkan rejects)
///   spirv64-unknown-vulkan*  -> OpCapability Shader + Logical GLSL450
///                               (the Vulkan flavor)
///
/// This entry point retargets the module to the Vulkan flavor, gives the
/// entry points the attributes the backend insists on in that environment
/// (`hlsl.shader=compute`, and `hlsl.numthreads` for the local size), and
/// emits the module's words.
///
/// Returns false with the reason in `out_log`.
OSL_DLL_EXPORT bool vulkan_compile_ir_via_llvm_vulkan(
    const std::string& ir_text, std::vector<uint32_t>& out_spirv,
    std::string& out_log);

}  // namespace pvt

OSL_NAMESPACE_END
