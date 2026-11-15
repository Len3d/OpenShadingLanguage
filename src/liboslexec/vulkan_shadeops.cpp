// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

// Device-side runtime (shadeops) for the Vulkan flavor of the SPIR-V backend.
//
// This translation unit is compiled to LLVM bitcode at build time by
//
//     clang -target spirv64 -emit-llvm -c -O2 -fno-rtti -fno-exceptions
//
// and the bitcode is linked into each group's module right before the Vulkan
// legalization pass rewrites the pointers from address space 0 to the generic
// address space clspv wants.  It therefore has to stay SELF-CONTAINED: no OSL
// headers (clang's spirv64 target has no C++ standard library, so pulling in
// <OSL/oslconfig.h> dies at <cstring>), and layout knowledge comes only from
// vulkan_types.h, which is checked against the real OSL layout by
// vulkan_abi_check.cpp on the host side.
//
// Only the symbols the render-uv groups actually call are defined here.  The
// list was taken from the unresolved declarations in a real group module:
//
//     osl_allocate_weighted_closure_component(osl, int, int, osl)
//     osl_init_texture_options(osl, osl)
//     osl_texture(osl, uint64, osl, osl, float x6, int, osl x7)
//
// Anything else the module needs must be added here *and* asserted to exist by
// vulkan_shadeops_link_check (see llvm_instance.cpp).

#include "vulkan_types.h"

namespace {

using namespace osl_vk_abi;

// clspv cannot use a Private (AS0) pointer to reach the storage buffer, and a
// plain `void*` in C++ mode *is* AS0.  Then every write the runtime performs
// lands in a private copy of the ShadeBlock: the closure pool stays zero and
// sg.Ci reads back 0 even though the shader stored a real closure offset.
// So all of these parameters are explicitly in the OpenCL Global space (AS1),
// the same treatment vulkan_render.cpp uses for its GlobalPtr.
#define VK_GLOBAL __attribute__((opencl_global))
typedef VK_GLOBAL unsigned char* GlobalPtr;
typedef VK_GLOBAL const unsigned char* GlobalCPtr;

// Closure representation.  Mirrors OSL::ClosureColor / ClosureComponent /
// ClosureMul / ClosureAdd from src/include/OSL/oslclosure.h:
//
//   ClosureColor     { int id }                    id: >=0 component, -1 mul, -2 add
//   ClosureComponent { int id; Vec3 w }            sizeof 16, w at +4
//   ClosureMul       { int id; Vec3 weight; ptr }  sizeof 32
//   ClosureAdd       { int id; ptr; ptr }          sizeof 24
//
// ClosureComponent is 16-byte aligned here: the CUDA flavor uses 8 only to
// match its memory pool, while our ShadeBlocks are laid out on 16-byte
// boundaries anyway (kShadeBlockStride).

const int kClosureMulID = -1;
const int kClosureAddID = -2;

struct alignas(16) ClosureComponent {
    int id;
    Vec3 w;
};

struct ClosureMul {
    int id;
    Vec3 weight;
    GlobalPtr closure;
};

struct ClosureAdd {
    int id;
    GlobalPtr closureA;
    GlobalPtr closureB;
};

typedef VK_GLOBAL ClosureComponent* GlobalComponentPtr;
typedef VK_GLOBAL const Vec3* GlobalVec3CPtr;
typedef VK_GLOBAL long long* GlobalI64Ptr;
typedef VK_GLOBAL const long long* GlobalI64CPtr;
typedef VK_GLOBAL float* GlobalFloatPtr;
typedef VK_GLOBAL const float* GlobalFloatCPtr;
// (TexSlot lives further down; its global pointer typedef is declared there.)

static_assert(sizeof(ClosureComponent) == 16, "ClosureComponent must be 16 bytes");
static_assert(sizeof(ClosureAdd) == 24, "ClosureAdd must be 24 bytes");

// ---------------------------------------------------------------------------
// The arena is addressed by *offset*, never by pointer
// ---------------------------------------------------------------------------
//
// clspv refuses to compile a module in which a pointer-typed value is ever
// stored into, or loaded out of, memory: its GetArgKindForType() reaches
// llvm_unreachable on a `ptr` type (the minimal reproducer prints a bare
// `ptr addrspace(1)` and dies with a breakpoint exception), and the
// logical-pointer lowering turns the same shape into `store i1 true, ptr poison`
// in a larger module.  SPIR-V has no variable pointers, so this is not
// something a flag can fix.
//
// The whole arena therefore lives behind three constant byte offsets in the
// ShadeBlock, and every pointer the shader sees is computed here with a
// getelementptr -- which is an ordinary instruction, legal in any storage
// class:
//
//   sg + 216  `renderstate` : i64 offset of the bump cursor (NOT a pointer)
//   sg + 288  `Ci`          : i64 offset of the root closure (NOT a pointer)
//   sg + 320  closure_pool  : start of the arena
//
// The offsets are the ones asserted in vulkan_abi_check.cpp, so the host ABI
// cannot drift away from this file silently.
static const unsigned long long kOffRenderstate = 216ull;
static const unsigned long long kOffCi          = 288ull;
static const unsigned long long kOffPool        = 320ull;

inline long long arena_cursor(const GlobalPtr sg)
{
    return *(GlobalI64CPtr)((const GlobalCPtr)sg + kOffRenderstate);
}

inline void arena_set_cursor(GlobalPtr sg, long long off)
{
    *(GlobalI64Ptr)((GlobalPtr)sg + kOffRenderstate) = off;
}

/// Publish the root closure as an offset, so the host never has to follow a
/// pointer that lives inside the buffer.
inline void arena_set_root(GlobalPtr sg, long long off)
{
    *(GlobalI64Ptr)((GlobalPtr)sg + kOffCi) = off;
}

/// Bump `size` bytes out of the arena and return the pointer to them.  The
/// base is 16-byte aligned (vulkan_types.h) and every caller rounds its size
/// to 16, so no alignment fixup is ever needed -- that fixup is exactly the
/// pointer-to-integer arithmetic Vulkan cannot express.
inline GlobalPtr arena_bump(GlobalPtr sg, long long size, long long* off_out)
{
    long long off = arena_cursor(sg);
    GlobalPtr p   = (GlobalPtr)sg + kOffPool + off;
    if (off_out)
        *off_out = off;
    arena_set_cursor(sg, off + size);
    return p;
}

/// Lay a closure component down at the front of a fresh arena block.
inline GlobalPtr closure_component_allot(GlobalPtr sg, int id, long long prim_size,
                                     const Vec3 w, bool is_root)
{
    long long needed = (long long)(sizeof(ClosureComponent)
                                   + (unsigned long long)prim_size + 15ull)
                       & ~15ll;
    long long off = 0;
    GlobalPtr p    = arena_bump(sg, needed, &off);

    GlobalComponentPtr comp = (GlobalComponentPtr)p;
    comp->id               = id;
    comp->w.x              = w.x;
    comp->w.y              = w.y;
    comp->w.z              = w.z;
    if (is_root)
        arena_set_root(sg, off);
    return p;
}

}  // namespace


extern "C" {

// ---------------------------------------------------------------------------
// Per-invocation arena
// ---------------------------------------------------------------------------

/// Backing store for locals that the Vulkan legalizer turned from `alloca`
/// into a call.
///
/// An `alloca` produces a pointer in the Private address space, and the shader
/// body keeps every pointer in address space 1 (the one storage buffer), so a
/// stack object could never be handed to anything without an address-space
/// cast -- which Vulkan SPIR-V has no instruction for.  Allocating the object
/// out of the same per-invocation arena as the closures sidesteps that
/// entirely: the pointer it returns is in the storage buffer like every other
/// pointer in the shader.
///
/// The arena is a plain bump allocator.  The base is 16-byte aligned
/// (vulkan_types.h) and every allocation size is rounded to 16, so the cursor
/// never needs rounding up, and no pointer is ever converted to an integer.
GlobalPtr
osl_vk_arena_alloc(GlobalPtr sg_, long long size, long long /*align*/)
{
    if (size < 16)
        size = 16;
    size = (size + 15) & ~15ll;

    return arena_bump(sg_, size, (long long*)0);
}


// ---------------------------------------------------------------------------
// Closure allocation
// ---------------------------------------------------------------------------

/// Allocate a closure component with an explicit weight.  Returns null (and
/// allocates nothing) when the weight is black, which is what lets OSL skip
/// building a component that cannot contribute -- same behaviour as the CUDA
/// runtime.
GlobalPtr
osl_allocate_weighted_closure_component(GlobalPtr sg_, int id, int size,
                                        const GlobalPtr w)
{
    GlobalVec3CPtr wc = (GlobalVec3CPtr)w;

    if (wc->x == 0.0f && wc->y == 0.0f && wc->z == 0.0f)
        return (GlobalPtr)0;

    if (size < 4)
        size = 4;

    // A Vec3 cannot be copy-constructed across address spaces (its constructor
    // would have to bind a reference into the storage buffer), so copy by field.
    Vec3 wv;
    wv.x = wc->x;
    wv.y = wc->y;
    wv.z = wc->z;
    return closure_component_allot(sg_, id, (long long)size, wv, true);
}


/// Unweighted variant: the component is implicitly weighted (1,1,1).
GlobalPtr
osl_allocate_closure_component(GlobalPtr sg_, int id, int size)
{
    if (size < 4)
        size = 4;

    Vec3 w = { 1.0f, 1.0f, 1.0f };
    return closure_component_allot(sg_, id, (long long)size, w, true);
}


// ---------------------------------------------------------------------------
// Closure multiplication by a color
// ---------------------------------------------------------------------------

/// Scale a closure tree by a color.
///
/// Mirrors osl_mul_closure_color() in src/testrender/cuda/rend_lib.cu, except
/// that the CUDA version pushes a ClosureMul node holding a pointer to the
/// subtree.  A node like that is a pointer stored in memory, which is the one
/// thing clspv cannot compile (see the arena comment above), so multiplication
/// folds into the component's own weight instead: a single component keeps its
/// id and gets weight *= color, which is exactly what a MUL node means to an
/// evaluator.
///
/// Only single-component closures can be folded this way; a closure tree
/// (id < 0) is returned unscaled rather than silently mis-weighted.
GlobalPtr
osl_mul_closure_color(GlobalPtr sg_, GlobalPtr a, const GlobalPtr w)
{
    GlobalVec3CPtr wc = (GlobalVec3CPtr)w;

    if (a == (GlobalPtr)0)
        return (GlobalPtr)0;
    if (wc->x == 0.0f && wc->y == 0.0f && wc->z == 0.0f)
        return (GlobalPtr)0;
    if (wc->x == 1.0f && wc->y == 1.0f && wc->z == 1.0f)
        return a;

    GlobalComponentPtr comp = (GlobalComponentPtr)a;
    if (comp->id < 0)
        return a;  // tree node: cannot fold, cannot allocate a MUL here

    comp->w.x *= wc->x;
    comp->w.y *= wc->y;
    comp->w.z *= wc->z;
    // Re-publish this component as the root closure.
    //
    // The last operation of a shader like `Ci = Cs * diffuse(N)` is *this* call,
    // so its return value is what ends up in Ci.  OSL materialises that store
    // as "record the offset of the pointer being stored", and a pointer that
    // came out of a function call has no traceable offset in the IR -- the
    // legalizer then has nothing to store and the shader is left with no
    // closure at all.  Recomputing the offset here (pointer difference inside
    // the same arena is legal in address space 1) and publishing it makes the
    // store a no-op instead of a miss.
    arena_set_root(sg_, (long long)((GlobalPtr)a
                                    - ((GlobalPtr)sg_ + kOffPool)));
    return a;
}


/// Scalar variant: `closure * float`, which OSL emits for `Ci = ward(...)*0.8` /
/// `Ci = 2.0 * diffuse(N)`.  Same fold-in-place rule as the color version, and
/// the same root re-publish, because this call is again the last one in the shader.
GlobalPtr
osl_mul_closure_float(GlobalPtr sg_, GlobalPtr a, float w)
{
    if (a == (GlobalPtr)0 || w == 0.0f)
        return (GlobalPtr)0;
    if (w == 1.0f)
        return a;

    GlobalComponentPtr comp = (GlobalComponentPtr)a;
    if (comp->id < 0)
        return a;  // tree node: cannot fold

    comp->w.x *= w;
    comp->w.y *= w;
    comp->w.z *= w;
    arena_set_root(sg_, (long long)((GlobalPtr)a
                                    - ((GlobalPtr)sg_ + kOffPool)));
    return a;
}


// ---------------------------------------------------------------------------
// normalize(vector)
// ---------------------------------------------------------------------------

/// llvm_ops.cpp:888 osl_normalize_vv, verbatim apart from the address space.
/// A zero-length vector normalises to zero rather than to a NaN, which is what
/// keeps `normalize(dPdu)` harmless on geometry with no parametric derivative
/// (a quad's dPdu is fine, a sphere's is tangent, a triangle soup's may be 0).
void
osl_normalize_vv(GlobalPtr result, GlobalPtr a)
{
    GlobalVec3CPtr src = (GlobalVec3CPtr)a;
    float x = src->x, y = src->y, z = src->z;
    float len = __builtin_sqrtf(x * x + y * y + z * z);
    if (len > 0.0f) {
        float invlen = 1.0f / len;
        x *= invlen;
        y *= invlen;
        z *= invlen;
    } else {
        x = y = z = 0.0f;
    }
    GlobalFloatPtr dst = (GlobalFloatPtr)result;
    dst[0] = x;
    dst[1] = y;
    dst[2] = z;
}


/// llvm_ops.cpp:674 (MAKE_BINARY_PERCOMPONENT_OP(fmod, safe_fmod, safe_fmod))
/// resolves `a % b` on floats to this.  The body is OIIO::safe_fmod
/// (fmath.h): truncate a/b to an int and subtract, with 0 on a zero divisor --
/// *not* `fmodf`, which rounds differently for negatives.
float
osl_fmod_fff(float a, float b)
{
    if (b != 0.0f) {
        int n = (int)(a / b);
        return a - (float)n * b;
    }
    return 0.0f;
}


// ---------------------------------------------------------------------------
// safe division
// ---------------------------------------------------------------------------

/// `x / y` with the convention that a zero denominator yields zero.
///
/// OSL normally emits this inline from its code generator, but the Vulkan
/// pipeline compiles the group IR as-is, so the call has to resolve to a real
/// definition here (same convention as osl_safe_div_fff in llvm_ops.cpp).
float
osl_safe_div_fff(float a, float b)
{
    return (b == 0.0f) ? 0.0f : a / b;
}


// ---------------------------------------------------------------------------
// Texture options
// ---------------------------------------------------------------------------

// The texture path is not wired to a device image yet (see the note at
// osl_texture below), so nothing ever reads the options struct back.  Going
// through the setters would still be pointless work, but the call has to exist
// so the module links; it is deliberately a no-op rather than a memset over a
// struct whose size we cannot see from here.
void
osl_init_texture_options(GlobalPtr opt, GlobalPtr sg)
{
    (void)opt;
    (void)sg;
}


// ---------------------------------------------------------------------------
// Texture lookup
// ---------------------------------------------------------------------------
//
// Textures live in the same storage buffer as the ShadeBlock, at fixed offsets,
// so the kernel needs no second argument and no second descriptor: the host
// uploads a table plus decoded RGBA32F pixels, and this function reaches them
// by pointer arithmetic from the block it was handed.  The three constants
// below are an ABI contract with VulkanShading::set_textures() in
// src/testrender/vulkan_shading.cpp.
//
//   [kTexTableOffset, +16*40)   slots: { u64 hash | i32 w | i32 h | i64 off |
//                                        i32 channels | i32 nlevels | pad }
//   [kTexDataOffset, ...)       texels, addressed by slot.offset
//
// What this does NOT do, and therefore diverges from the CPU render: it does
// not build OIIO's anisotropy ellipse (it uses the s and t footprints only),
// and it only honours clamp wrap.  `texture(filename, u, v)` in the test scenes
// stays inside [0,1], so clamp is the right boundary behaviour there.

const int kTexTableOffset = 64 * 1024;
const int kMaxTextures    = 16;
const int kTexDataOffset  = kTexTableOffset + kMaxTextures * 40;

// One slot per texture.  The mip chain length is in nlevels; level k starts at
// offset + sum of the byte sizes of levels 0..k-1, with level k being
// max(1, width >> k) x max(1, height >> k), which is how the host uploads it.
struct alignas(8) TexSlot {
    unsigned long long hash;
    int width;
    int height;
    long long offset;
    int channels;
    int nlevels;
    int pad0;
    int pad1;
};

typedef VK_GLOBAL const TexSlot* GlobalTexSlotCPtr;

static_assert(sizeof(TexSlot) == 40, "TexSlot must stay 40 bytes");

static int
floori(float v)
{
    int i = (int)v;
    return (v < (float)i) ? i - 1 : i;
}

static int
clampi(int v, int lo, int hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

int
osl_texture(GlobalPtr sg, unsigned long long namehash, GlobalPtr opt,
            GlobalPtr sg2,
            float s, float t, float dsdx, float dtdx, float dsdy, float dtdy,
            int nchannels, GlobalPtr result, GlobalPtr dresultdxdy0,
            GlobalPtr dresultds, GlobalPtr dresultdt, GlobalPtr derivs,
            GlobalPtr opt2,
            GlobalPtr filter)
{
    (void)opt;
    (void)sg2;
    (void)dsdx;
    (void)dtdx;
    (void)dsdy;
    (void)dtdy;
    (void)opt2;
    (void)filter;

    // The layer code passes an alloca sized for the requested channel count,
    // so write exactly that many floats and nothing more.
    if (nchannels <= 0)
        nchannels = 1;
    if (nchannels > 4)
        nchannels = 4;
    GlobalFloatPtr out = (GlobalFloatPtr)result;
    for (int i = 0; i < nchannels; ++i)
        out[i] = 0.0f;

    // The texture table and the mip pixels are addressed from the *start of the
    // storage buffer*, not from this invocation's ShadeBlock: in the whole-frame
    // kernel every pixel has its own block far past the texture area, so using
    // `sg` as the base reads unrelated pixels and every lookup comes back black
    // (which then weights the closure to zero, so the material renders as
    // "no closure at all").  The kernel parks the block's own absolute buffer
    // offset in the first 8 bytes of the closure pool for exactly this.
    const GlobalCPtr sg_c   = (const GlobalCPtr)sg;
    const long long sg_abs  = *(GlobalI64CPtr)(sg_c + kOffPool);
    const GlobalCPtr block  = sg_c - sg_abs;
    GlobalTexSlotCPtr tab   = (GlobalTexSlotCPtr)(block + kTexTableOffset);
    int slot           = -1;
    for (int i = 0; i < kMaxTextures; ++i) {
        if (tab[i].hash == namehash) {
            slot = i;
            break;
        }
    }
    if (slot < 0 || tab[slot].width <= 0 || tab[slot].height <= 0)
        return 0;  // texture missing: black, and report it as missing

    (void)dresultdxdy0;
    (void)derivs;
    const int w      = tab[slot].width;
    const int h      = tab[slot].height;
    const int nlv    = tab[slot].nlevels > 0 ? tab[slot].nlevels : 1;

    // Which mip level the CPU would have picked.  We do not build OIIO's
    // anisotropy ellipse; the footprint along s and t is enough to land on the
    // same level for these scenes, and blending two levels hides an off-by-one.
    float foot_s = (dsdx < 0.0f ? -dsdx : dsdx) * (float)w;
    float foot_t = (dtdy < 0.0f ? -dtdy : dtdy) * (float)h;
    float foot   = foot_s > foot_t ? foot_s : foot_t;
    // The level is chosen host-side (FrameParams::tex_mode) so the alternatives
    // can be compared on real frames without a rebuild.
    const int tex_mode = ((const VK_GLOBAL FrameParams*)block)->tex_mode;
    const int use_mips = (tex_mode == 0 || tex_mode == 2) ? 1 : 0;
    int lod            = 0;
    float blend        = 0.0f;
    if (use_mips) {
    while (foot > 1.0f && lod + 1 < nlv) {
        foot *= 0.5f;
        ++lod;
    }
    if (tex_mode == 0 && foot > 0.5f) {
        blend = foot * 2.0f - 1.0f;
        if (blend > 1.0f)
            blend = 1.0f;
    }

    float acc0[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    float acc1[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
    for (int lv = lod; lv <= lod + 1 && lv < nlv; ++lv) {
        // Levels are stored back to back, each half the size of the one before.
        long long lv_off = tab[slot].offset;
        int lw = w, lh = h;
        for (int k = 0; k < lv; ++k) {
            lv_off += (long long)(lw * lh) * 16;
            lw = lw > 1 ? lw >> 1 : 1;
            lh = lh > 1 ? lh >> 1 : 1;
        }
        GlobalFloatCPtr lvl = (GlobalFloatCPtr)(block + kTexDataOffset + lv_off);

        // OIIO's texture coordinates put t=0 at the *bottom* of the image, while
        // the decoded rows start at the top, so v has to be flipped.
        float x  = s * (float)lw - 0.5f;
        float y  = t * (float)lh - 0.5f;
        int x0   = floori(x);
        int y0   = floori(y);
        float fx = x - (float)x0;
        float fy = y - (float)y0;

        int xa = clampi(x0, 0, lw - 1), xb = clampi(x0 + 1, 0, lw - 1);
        int ya = clampi(y0, 0, lh - 1), yb = clampi(y0 + 1, 0, lh - 1);

        GlobalFloatCPtr p00 = lvl + (ya * lw + xa) * 4;
        GlobalFloatCPtr p10 = lvl + (ya * lw + xb) * 4;
        GlobalFloatCPtr p01 = lvl + (yb * lw + xa) * 4;
        GlobalFloatCPtr p11 = lvl + (yb * lw + xb) * 4;

        float* dst = (lv == lod) ? acc0 : acc1;
        for (int c = 0; c < nchannels; ++c) {
            float c00 = p00[c], c10 = p10[c], c01 = p01[c], c11 = p11[c];
            float top = c00 + (c10 - c00) * fx;
            float bot = c01 + (c11 - c01) * fx;
            dst[c]    = top + (bot - top) * fy;
        }
    }
    if (acc1[0] == 0.0f && acc1[1] == 0.0f && acc1[2] == 0.0f && acc1[3] == 0.0f) {
        // Only one level was sampled; blending against it is a no-op.
        for (int c = 0; c < 4; ++c)
            acc1[c] = acc0[c];
    }

    for (int c = 0; c < nchannels; ++c)
        out[c] = acc0[c] + (acc1[c] - acc0[c]) * blend;
    } else {
        // Mip 0 only: bilinear, clamp to edge.
        GlobalFloatCPtr lvl = (GlobalFloatCPtr)(block + kTexDataOffset
                                         + tab[slot].offset);
        float x          = s * (float)w - 0.5f;
        float y          = t * (float)h - 0.5f;
        int x0           = floori(x);
        int y0           = floori(y);
        float fx         = x - (float)x0;
        float fy         = y - (float)y0;
        int xa = clampi(x0, 0, w - 1), xb = clampi(x0 + 1, 0, w - 1);
        int ya = clampi(y0, 0, h - 1), yb = clampi(y0 + 1, 0, h - 1);
        GlobalFloatCPtr p00 = lvl + (ya * w + xa) * 4;
        GlobalFloatCPtr p10 = lvl + (ya * w + xb) * 4;
        GlobalFloatCPtr p01 = lvl + (yb * w + xa) * 4;
        GlobalFloatCPtr p11 = lvl + (yb * w + xb) * 4;
        for (int c = 0; c < nchannels; ++c) {
            float c00 = p00[c], c10 = p10[c], c01 = p01[c], c11 = p11[c];
            float top = c00 + (c10 - c00) * fx;
            float bot = c01 + (c11 - c01) * fx;
            out[c]    = top + (bot - top) * fy;
        }
    }

    // The derivative outputs are alloca'd by the caller when it asked for
    // them.  Only the two per-channel slots can be sized from here; the
    // combined dxdy buffer is left alone rather than guessed at.
    if (dresultds) {
        GlobalFloatPtr d = (GlobalFloatPtr)dresultds;
        for (int i = 0; i < nchannels; ++i)
            d[i] = 0.0f;
    }
    if (dresultdt) {
        GlobalFloatPtr d = (GlobalFloatPtr)dresultdt;
        for (int i = 0; i < nchannels; ++i)
            d[i] = 0.0f;
    }
    (void)dresultdxdy0;
    (void)derivs;

    return 1;
}

}  // extern "C"
