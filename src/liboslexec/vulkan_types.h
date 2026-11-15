// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#pragma once

// Device-side ABI for the Vulkan flavor of the SPIR-V backend.
//
// This header is deliberately SELF-CONTAINED: it includes nothing at all, so
// that the exact same declarations can be compiled twice --
//   * by MSVC, together with the real OSL headers, where
//     vulkan_abi_check.cpp static_asserts every offset against
//     OSL::ShaderGlobals;
//   * by `clang -target spirv64 -emit-llvm`, as part of the device runtime
//     (clspv's target has no C++ standard library, so pulling in
//     <OSL/oslcfg.h> from a device translation unit fails at <cstring>).
//
// Keeping one declaration instead of two copies means the only thing that has
// to be *proven* is that it matches OSL's host layout -- which the static
// asserts do at every build.
//
// -----------------------------------------------------------------------------
// Buffer layout (one storage buffer, std430, per dispatch)
// -----------------------------------------------------------------------------
//
//   byte 0                                  : ShadeBlock[0]
//   byte kShadeBlockStride                  : ShadeBlock[1]
//   ...
//
//   struct ShadeBlock {
//       ShaderGlobals sg;             // offsets identical to OSL::ShaderGlobals
//       char          closure_pool[kClosurePoolSize];   // bump allocator
//   };
//
// One ShadeBlock per invocation: `sg` is filled in by the host before dispatch
// (u/v/N/P/... plus surfacearea), `sg.renderstate` points at `closure_pool`,
// `sg.Ci` starts out null and the shader stores its resulting closure chain
// pointer straight into it.  Both of those are exactly what the CUDA/OptiX
// flavor does (see src/testrender/cuda/optix_raytracer.cu:sg.renderstate =
// &closure_pool[0]) -- and `sg.Ci` being a plain pointer field, not a pointer
// to storage, is what the compiled layer code assumes: the group IR contains
// `store ptr %closure, ptr %sg_plus_288`, and 288 is the offset of `Ci`.
//
// `groupdata` / `userdata` / `output` are all passed as NULL, again matching
// the OptiX flavor.  For the render-uv groups the optimizer already proves
// those arguments dead (`nocapture readnone` in the emitted signature), so
// there is nothing to point them at; a group that really reads groupdata will
// need a real block here later.

namespace osl_vk_abi {

struct Vec3 {
    float x, y, z;
};

/// Device flavor of testrender's BVHNode (src/testrender/bvh.h), byte for
/// byte: six floats of bounds (x,y,z pairs as lo/hi) then the child index and
/// the primitive count.  A node with nprims != 0 is a leaf whose triangles are
/// `bvh_indices[child .. child+nprims)`; otherwise `child` is the index of the
/// first of two consecutive interior children.
struct BvhNodeVK {
    float bounds[6];
    unsigned child;
    unsigned nprims;
};

/// Device flavor of OSL::ShaderGlobals.
///
/// Field-for-field identical to src/include/OSL/shaderglobals.h, except that
/// `Ci` is carried as `void*` rather than `ClosureColor*` (same size and
/// alignment) so that this header stays free of OSL types.
struct ShaderGlobals {
    Vec3 P, dPdx, dPdy;
    Vec3 dPdz;

    Vec3 I, dIdx, dIdy;

    Vec3 N;

    Vec3 Ng;

    float u, dudx, dudy;
    float v, dvdx, dvdy;

    Vec3 dPdu, dPdv;

    float time;
    float dtime;
    Vec3 dPdtime;

    Vec3 Ps, dPsdx, dPsdy;

    void* renderstate;  ///< -> closure_pool of this block
    void* tracedata;
    void* objdata;

    void* context;              ///< stand-in for the host ShadingContext
    void* shadingStateUniform;

    int thread_index;
    int shade_index;

    void* renderer;

    void* object2common;  ///< TransformationPtr on the host side
    void* shader2common;  ///< TransformationPtr on the host side

    void* Ci;  ///< ClosureColor* on the host side; written by the shader

    float surfacearea;
    int raytype;
    int flipHandedness;
    int backfacing;
};

/// Bytes reserved per invocation for the closure bump allocator.
///
/// A single `emission()` or `diffuse(N)` closure is a ClosureComponent plus a
/// small parameter array; 256 B leaves room for a handful of them (closure
/// add/mul chains included) and keeps the per-invocation footprint small
/// enough that a full frame's blocks still fit in one buffer.
static const int kClosurePoolSize = 256;

/// Bytes reserved per invocation for the group's "group data" block.
///
/// OSL materialises a layer's parameter block in memory whenever the block
/// cannot be SROA-promoted to SSA values -- which happens as soon as a layer
/// body has more than straight-line code, e.g. checkerboard's
/// `if (int(cs) ^ int(ct))`.  That block is the second argument of
/// `osl_layer_...` / `osl_init_group_...`, and its first word is an
/// "already initialised" flag that `osl_init_group_*` clears.  Any shader
/// compiled that way must be handed real memory: with a null pointer the flag
/// reads whatever sits at buffer offset 0 and the parameter stores land on the
/// frame header, so the layer either skips its initialisation or feeds a
/// garbage weight to the closure allocator (which then returns null and the
/// material renders black, with no diagnostic at all).
static const int kGroupDataSize = 64;

/// Invocations per workgroup.  Must also be declared to clspv through
/// `__attribute__((reqd_work_group_size(N,1,1)))` on the kernel, otherwise the
/// SPIR-V has no workgroup size and Vulkan rejects the pipeline.
static const int kWorkgroupSize = 64;

struct ShadeBlock {
    ShaderGlobals sg;
    /// Padding to put the pool on a 16-byte boundary.
    ///
    /// The pool has to *start* aligned, because the allocator no longer rounds
    /// the cursor up: rounding a pointer means going through an integer
    /// (`ptrtoint`/`inttoptr`), and SPIR-V has no integer-to-pointer conversion
    /// for logical storage classes -- it only exists for physical (buffer
    /// device address) pointers.  Instead every allocation size is rounded up
    /// to 16 and the base is 16-aligned, so the cursor never needs fixing up.
    /// sizeof(ShaderGlobals) == 312, so 8 bytes of padding land the pool at 320.
    char pad[8];
    char closure_pool[kClosurePoolSize];
    /// The layer parameter block the shader is handed as its `groupdata`
    /// argument.  It sits *after* the pool on purpose: the bump allocator only
    /// ever grows upward from `closure_pool`, so it can never overlap this.
    char groupdata[kGroupDataSize];
};

/// Offset of the group-data block inside a ShadeBlock; the module builder hands
/// exactly this address to every group entry point.  Spelled out instead of
/// `offsetof` because this header includes nothing at all.
static const int kGroupDataOffset =
    (int)sizeof(ShaderGlobals) + 8 + kClosurePoolSize;

/// Distance between consecutive ShadeBlocks, rounded up so every block starts
/// on a 16-byte boundary (storage-buffer friendly, and keeps `sg` -- which
/// contains Vec3 arrays -- naturally aligned).
static const int kShadeBlockStride = ((int)sizeof(ShadeBlock) + 15) & ~15;

// -----------------------------------------------------------------------------
// Whole-frame render layout (the Vulkan equivalent of cuda/optix_raytracer.cu)
// -----------------------------------------------------------------------------
//
// The CUDA flavor compiles SimpleRaytracer itself into the module and gets the
// shader through an OptiX direct-call table.  Vulkan cannot call through a
// function pointer, so the frame kernel calls `osl_vk_shade_dispatch(sg, id)`
// instead, a switch over every material of the scene that the module builder
// synthesises once all the groups are linked into one module.
//
// Everything the kernel needs lives in the same one storage buffer:
//
//   byte 0                 : FrameParams
//   kTextureTableOffset    : texture table (unchanged)
//   kTextureDataOffset     : texture pixels, mip chains back to back
//   params.tri_offset      : TriangleIndices per primitive
//   params.vert_offset     : Vec3 per vertex
//   params.uv_offset       : Vec2 per vertex
//   params.tri_uv_offset   : TriangleIndices into uvs (the .oso "uv" corners)
//   params.tri_shader_off  : int per primitive -> material id
//   params.out_offset      : float4 per pixel, row major
//   params.arena_offset    : ShadeBlock per invocation (stride arena_stride)
//
// Offsets are byte offsets into the buffer; the kernel derives every pointer
// from `gep(base, offset)`, which is the only form clspv can attribute to the
// storage buffer resource.
struct TriangleIndices {
    int a, b, c;
};

struct FrameParams {
    int width, height, aa, max_bounces;
    float eye[3];
    float dir[3];
    float up[3];
    float fov;
    int num_triangles, num_vertices;
    int bg_shader_id;  ///< -1 = no background shader
    int pad0;
    float bg_color[3];
    int pad1;
    long long tri_offset;
    long long vert_offset;
    long long uv_offset;
    long long tri_uv_offset;
    long long normal_offset;   ///< Vec3 per vertex
    long long tri_n_offset;    ///< TriangleIndices into normals (-1 = none)
    long long tri_shader_offset;
    long long out_offset;
    long long arena_offset;
    long long arena_stride;
    /// Light-emitting primitives: triangle ids, one int each, exactly the list
    /// testrender keeps in SimpleRaytracer::m_lightprims.  Explicit light
    /// sampling (NEE) walks this list the same way the CPU tracer does.
    long long lightprim_offset;
    int num_lightprims;
    int pad_light;
    /// One float per primitive: the surface area of the mesh that primitive
    /// belongs to.  testrender puts this into sg->surfacearea
    /// (simpleraytracer.cpp:945) and emitter.osl divides the light power by it,
    /// so leaving it at zero turns every emitter into 0 radiance.
    long long area_offset;
    /// Which aa sample this dispatch covers, and how many there are in total.
    /// The host may split the frame into one dispatch per sample; with
    /// sample_count == 0 the kernel falls back to doing aa*aa samples itself
    /// (the per-point and probe paths use that).
    int sample_index;
    int sample_count;
    /// How many samples one dispatch shades (0 or 1 = one).  Batching trims the
    /// per-dispatch round trip, but a thread must not shade too many: reusing a
    /// single ShadeBlock across samples is what produced the black striping,
    /// so the host keeps the batch small and verified.
    int sample_batch;
    /// The CPU BVH, uploaded as its own two flat arrays (bvh.h): nodes are
    /// `BvhNodeVK` (32 B, same layout as the CPU's BVHNode) and the triangle
    /// ids the leaves point at.  Zero nodes means "no BVH": the kernel then
    /// falls back to the linear sweep, which is what the old code did.
    long long bvh_offset;
    long long bvh_index_offset;
    int num_bvh_nodes;
    int num_bvh_indices;
    /// Texture filter variant, set from the host so the two paths can be
    /// compared without rebuilding: 0 = mip level estimate plus a trilinear
    /// blend, 1 = mip 0 with bilinear only, 2 = nearest mip level with
    /// bilinear, 3 = finest level, no mip chain at all.
    int tex_mode;
    /// Russian roulette depth.  testrender's CompositeBSDF::prepare()
    /// (shading.h:356-386) only normalises the closure probabilities when
    /// `absorb` is false, and the tracer passes `absorb = (b >= rr_depth)`
    /// (simpleraytracer.cpp:1092).  Past that depth a closure whose sampled
    /// probability is < 1 therefore leaves part of the sample space unselected,
    /// and `sample()` returns an empty sample there -- that *is* the CPU's
    /// roulette.  Without it the device keeps every deep path alive at full
    /// weight, which is exactly what makes its firefly tail longer than the
    /// CPU's.
    int rr_depth;
    /// The camera basis, uploaded straight from testrender's own Camera
    /// (raytracer.h:106-160 Camera::finalize) instead of being recomputed on the
    /// device.  finalize() uses OIIO::fast_tan, a polynomial approximation of
    /// tan, so a device-side tanf() would aim every camera ray a hair off the
    /// CPU's -- enough to move a sample onto a different triangle near an edge,
    /// which is exactly the kind of difference that survives as a stray firefly.
    float cx[3];
    float cy[3];
    float invw, invh;
    /// Constants of the device-side mip selector, which now follows OIIO's
    /// ellipse + log2 rule (see osl_texture in vulkan_shadeops.cpp):
    ///   footprint (texels) = major axis of the (s,t)-vs-(x,y) Jacobian ellipse
    ///                        * tex_foot_scale
    ///   level              = log2(footprint) + tex_lod_bias
    /// OIIO multiplies the axis by a filter-width factor and folds the
    /// sblur/tblur terms in, so both constants are exposed as host overrides
    /// (OSL_VULKAN_TEX_FOOT_SCALE / OSL_VULKAN_TEX_LOD_BIAS) to be calibrated
    /// against the CPU path without rebuilding the device shaders.
    float tex_foot_scale;
    float tex_lod_bias;
};

/// Where FrameParams sits: the first bytes of the buffer, so the kernel can
/// find it from the raw argument without a second descriptor.
static const long long kFrameParamsOffset = 0;

}  // namespace osl_vk_abi
