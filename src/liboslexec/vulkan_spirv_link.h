// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#pragma once

#include <string>
#include <vector>

#include <OSL/oslconfig.h>

OSL_NAMESPACE_BEGIN

namespace pvt {

/// Link two or more SPIR-V modules into one.
///
/// When the materials are compiled by LLVM's own SPIR-V backend and the render
/// kernel keeps going through clspv, the two halves end up in separate
/// modules, and SPIR-V has no cross-module calls: the entry points must be
/// resolved by a linker before the driver sees them.  Both compilers emit
/// SPIR-V 1.4 with Logical/GLSL450, so the modules are link-compatible; the
/// matching is done through OpDecorate LinkageAttributes (Export on the
/// material side, Import on the side that calls them).
///
/// `create_library` keeps the exported symbols (a library) instead of dropping
/// them (an executable).  Pass false for the final, runnable module.
///
/// The merged module is validated against the Vulkan 1.3 environment before it
/// is handed back, so a failure here is a real "the driver would have rejected
/// this" rather than a linker quirk.  Returns false with the reason in
/// `out_log`.
OSL_DLL_EXPORT bool vulkan_link_spirv_modules(
    const std::vector<std::vector<uint32_t>>& modules,
    std::vector<uint32_t>& out_spirv, std::string& out_log,
    bool create_library = false);

}  // namespace pvt

OSL_NAMESPACE_END
