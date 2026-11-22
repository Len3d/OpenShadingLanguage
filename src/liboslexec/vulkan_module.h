// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#pragma once

// Public-ish surface of the Vulkan batch module assembly (vulkan_module.cpp).
//
// This header is meant to be included by a renderer (testrender's Vulkan path)
// as well as by liboslexec itself, so it deliberately mentions no LLVM types in
// the functions a renderer needs: it passes IR around as text.

#include <string>
#include <vector>

#include <OSL/oslconfig.h>

namespace llvm {
class Module;
}

OSL_NAMESPACE_BEGIN

class ShaderGroup;

namespace pvt {

/// Legalised IR text kept by a group compiled with the Vulkan backend enabled
/// (OSL_USE_VULKAN=1).
OSL_DLL_EXPORT std::string vulkan_group_ir (const ShaderGroup& group);

/// Names of the functions a caller has to invoke for that group (one per
/// material layer, in order).
OSL_DLL_EXPORT std::vector<std::string> vulkan_group_entry_names (const ShaderGroup& group);

/// Which functions a caller has to invoke for this module (a non-trivial
/// osl_init_group_*, or the osl_layer_* functions when that one was folded to
/// an empty body).  Run on the legalised group module so the whole-frame
/// assembly knows what to call.
bool vulkan_collect_entry_names (llvm::Module& M, std::vector<std::string>& out,
                                 std::string& out_log);

/// Link the device frame renderer with every group of the scene and synthesise
/// `osl_vk_shade_dispatch` plus the compute kernel.  Returns the module as IR
/// text, ready for clspv `-x ir`.
OSL_DLL_EXPORT bool vulkan_build_render_module (
    const std::vector<std::string>& group_ir,
                                 const std::vector<std::vector<std::string>>& entries,
                                 std::string& out_ir, std::string& out_log);

}  // namespace pvt

OSL_NAMESPACE_END
