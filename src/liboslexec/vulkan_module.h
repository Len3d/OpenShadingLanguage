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

/// Legalised IR text kept by a group compiled with OSL_VULKAN_BATCH=1.
OSL_DLL_EXPORT std::string vulkan_group_ir (const ShaderGroup& group);

/// Names of the functions a caller has to invoke for that group (one per
/// material layer, in order).
OSL_DLL_EXPORT std::vector<std::string> vulkan_group_entry_names (const ShaderGroup& group);

/// Same target selection as vulkan_add_kernel_entry, exposed so the batch path
/// can record entry names without building a kernel.
bool vulkan_collect_entry_names (llvm::Module& M, std::vector<std::string>& out,
                                 std::string& out_log);

/// Link the device frame renderer with every group of the scene and synthesise
/// `osl_vk_shade_dispatch` plus the compute kernel.  Returns the module as IR
/// text, ready for clspv `-x ir`.
/// When `out_group_ir` is non-null the function runs in *split* mode: the
/// groups are not linked into the kernel module, the dispatcher only declares
/// the group entry points, and each group's IR text is returned in
/// `out_group_ir` for a separate compile (Vulkan flavor SPIR-V backend).  The
/// two halves are then merged with spirv-link instead of being joined here.
OSL_DLL_EXPORT bool vulkan_build_render_module (
    const std::vector<std::string>& group_ir,
    const std::vector<std::vector<std::string>>& entries, std::string& out_ir,
    std::string& out_log, std::vector<std::string>* out_group_ir = nullptr,
    std::string* out_entry_ir = nullptr);

}  // namespace pvt

OSL_NAMESPACE_END
