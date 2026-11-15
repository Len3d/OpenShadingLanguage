// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

// Host-side proof that the self-contained device ABI in vulkan_types.h still
// matches the layout OSL's compiled shader code indexes into.  There is no
// code in here: every entry is a static_assert, so a build that links this
// translation unit is the evidence that the device struct is correct.
//
// vulkan_types.h cannot include OSL headers (the device runtime is compiled by
// clang for spirv64, which has no C++ standard library), so this file is the
// only place where the two layouts meet.

#include "vulkan_types.h"

#include <OSL/oslexec.h>
#include <OSL/shaderglobals.h>

#include <cstddef>

namespace {

using osl_vk_abi::ShaderGlobals;
using HostSG = OSL::ShaderGlobals;

// Overall size and alignment: the device shader globals must be the same
// object size the host buffers, otherwise every field past the first
// difference is shifted.
static_assert(sizeof(ShaderGlobals) == sizeof(HostSG),
              "device ShaderGlobals size differs from OSL::ShaderGlobals");
static_assert(alignof(ShaderGlobals) == alignof(HostSG),
              "device ShaderGlobals alignment differs from OSL::ShaderGlobals");

// Every field the compiled group code touches by offset.  Keep this list in
// sync with the struct; missing an entry is caught by the size assert above
// only if the mistake changes the total size, which is why the fields that
// matter are checked individually.
#define OSL_VK_OFFSET(field)                                                  \
    static_assert(                                                            \
        offsetof(ShaderGlobals, field) == offsetof(HostSG, field),            \
        "offset of " #field " differs from the host ShaderGlobals")

OSL_VK_OFFSET(P);
OSL_VK_OFFSET(dPdx);
OSL_VK_OFFSET(dPdy);
OSL_VK_OFFSET(dPdz);
OSL_VK_OFFSET(I);
OSL_VK_OFFSET(dIdx);
OSL_VK_OFFSET(dIdy);
OSL_VK_OFFSET(N);
OSL_VK_OFFSET(Ng);
OSL_VK_OFFSET(u);
OSL_VK_OFFSET(dudx);
OSL_VK_OFFSET(dudy);
OSL_VK_OFFSET(v);
OSL_VK_OFFSET(dvdx);
OSL_VK_OFFSET(dvdy);
OSL_VK_OFFSET(dPdu);
OSL_VK_OFFSET(dPdv);
OSL_VK_OFFSET(time);
OSL_VK_OFFSET(dtime);
OSL_VK_OFFSET(dPdtime);
OSL_VK_OFFSET(Ps);
OSL_VK_OFFSET(dPsdx);
OSL_VK_OFFSET(dPsdy);
OSL_VK_OFFSET(renderstate);
OSL_VK_OFFSET(tracedata);
OSL_VK_OFFSET(objdata);
OSL_VK_OFFSET(context);
OSL_VK_OFFSET(shadingStateUniform);
OSL_VK_OFFSET(thread_index);
OSL_VK_OFFSET(shade_index);
OSL_VK_OFFSET(renderer);
OSL_VK_OFFSET(object2common);
OSL_VK_OFFSET(shader2common);
OSL_VK_OFFSET(Ci);
OSL_VK_OFFSET(surfacearea);
OSL_VK_OFFSET(raytype);
OSL_VK_OFFSET(flipHandedness);
OSL_VK_OFFSET(backfacing);

#undef OSL_VK_OFFSET

// The group IR stores the output closure straight into `sg + offsetof(Ci)`.
// Pinning the value documents the assumption the device runtime relies on.
static_assert(offsetof(ShaderGlobals, Ci) == 288,
              "Ci is no longer at byte 288 of ShaderGlobals");

// Sizes quoted in the design notes and used by the host uploader.  Pinning
// them here turns "computed by hand" into "checked at every build".
static_assert(sizeof(ShaderGlobals) == 312,
              "ShaderGlobals is no longer 312 bytes");
static_assert(osl_vk_abi::kShadeBlockStride == 640,
              "ShadeBlock stride is no longer 640 bytes");
static_assert(osl_vk_abi::kGroupDataOffset == 576,
              "the group-data block must start right after the closure pool");

// The closure pool must start 16-byte aligned: the device allocator no longer
// rounds its cursor (that would need ptrtoint/inttoptr, which SPIR-V forbids
// for logical storage classes), so it relies on the base being aligned and on
// every allocation size being rounded to 16.
static_assert(offsetof(osl_vk_abi::ShadeBlock, closure_pool) == 320,
              "closure_pool must start at byte 320 (16-aligned after sg)");
static_assert(offsetof(osl_vk_abi::ShadeBlock, closure_pool) % 16 == 0,
              "closure_pool is not 16-byte aligned");
static_assert(osl_vk_abi::kShadeBlockStride % 16 == 0,
              "ShadeBlock stride is not a multiple of 16");

}  // namespace
