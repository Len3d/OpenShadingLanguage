// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#pragma once

#include <string>
#include <vector>

#include <llvm/ADT/StringRef.h>

#include <OSL/oslconfig.h>

namespace llvm {
class Module;
}

OSL_NAMESPACE_BEGIN

namespace pvt {

/// Address space number that clspv calls "Generic" (OpenCL generic address
/// space).  See clspv's include/clspv/AddressSpace.h:
///   Private = 0, Global = 1, Constant = 2, Local = 3, Generic = 4, ...
/// OSL builds its IR with LLVM's default pointer type (`ptr`, i.e. address
/// space 0), which clspv interprets as *Private* -- a pointer that cannot
/// point into a storage buffer.  Everything downstream (InferAddressSpaces,
/// LowerAddrSpaceCast, SPIRVProducer) then produces either wrong code or a
/// crash, so the generic pointers have to be rewritten before clspv sees the
/// module.
static const int kVulkanGenericAddressSpace = 4;

/// Switch @p M over to the spirv64 triple and data layout that both the device
/// runtime bitcode and clspv are built for.
///
/// This has to happen *before* the device runtime bitcode is linked in: the
/// bitcode carries the spirv64 layout, and linking two modules with different
/// layouts leaves every global constant with a type whose layout does not match
/// the module it now lives in -- LLVM warns about the mismatch, and the module
/// fails verification right afterwards.
void vulkan_set_module_target (llvm::Module& M);

/// Rewrite a module so that clspv's IR front end (`clspv -x ir`) can consume it.
///
/// The transformation is deliberately mechanical, so that it can run on any
/// module OSL produces, without knowing which ops are in it:
///
///  - every pointer type in address space 0 becomes address space 4 (generic),
///    including function signatures, allocas, globals and intrinsic overloads;
///  - `inttoptr` of a baked host address (OSL materialises a few of those as
///    constant pointer initialisers, which can never be replayed on a device)
///    becomes a null pointer;
///  - the module gets the spirv64 triple and data layout that clspv expects.
///
/// After this pass the module only uses generic/global pointers, which is
/// exactly the shape clspv's own pipeline is written for (it runs LLVM's
/// InferAddressSpacesPass with generic == 4 to push pointers back down to
/// Private/Global).
///
/// Returns false if the module cannot be rewritten (an instruction the
/// rewriter does not understand), and fills @p out_log with the details.
bool vulkan_legalize_module (llvm::Module& M, std::string& out_log);

/// Same as above, keeping the set of instruction opcodes that were rewritten
/// and the ones that were dropped, for diagnostics.
std::string vulkan_legalize_summary ();

/// Link the embedded Vulkan device runtime (vulkan_shadeops.cpp, compiled to
/// bitcode at build time) into @p M.
///
/// A compiled group only *declares* the shadeops it needs -- the three for the
/// render-uv groups are osl_allocate_weighted_closure_component,
/// osl_init_texture_options and osl_texture -- because on the host they live in
/// liboslexec's own code.  For SPIR-V they have to be part of the module, so
/// this links the device implementations in, replacing the declarations.
///
/// Returns false (with @p out_log filled) when the embedded bitcode could not
/// be parsed or the link rejected a type mismatch.
bool vulkan_link_shadeops (llvm::Module& M, std::string& out_log);

/// Which functions a caller has to invoke for this module (a non-trivial
/// osl_init_group_*, or the osl_layer_* functions when that one was folded
/// to an empty body).  Used by the batch path, which keeps the IR instead of
/// building a kernel right away.
bool vulkan_collect_entry_names (llvm::Module& M, std::vector<std::string>& out,
                                 std::string& out_log);

}  // namespace pvt

OSL_NAMESPACE_END
