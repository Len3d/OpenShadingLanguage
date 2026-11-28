// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

// Whole-frame render kernel for the Vulkan flavor of the SPIR-V backend.
//
// This is the counterpart of src/testrender/cuda/optix_raytracer.cu.  The CUDA
// flavor compiles testrender's own SimpleRaytracer into the module and reaches
// the shader through an OptiX direct-call table; Vulkan has neither, so this
// file carries the renderer (camera ray generation, pixel sampling, triangle
// intersection, path throughput and accumulation) *and* calls the shader
// through `osl_vk_shade_dispatch()`, which the module builder synthesises as a
// switch over every material once all groups are linked together.
//
// Deliberately self contained, like vulkan_shadeops.cpp: `clang -target
// spirv64` has no C++ standard library, so no <cmath>/<cstring>/OSL header may
// appear here.  The one thing that must stay in sync with the host is the
// layout in vulkan_types.h.

#include "vulkan_types.h"

// Address space 1 (shader storage buffer): what clspv turns into a storage
// buffer descriptor.  A plain `void*` would be Private, i.e. unreachable from
// the kernel argument.
#define VK_GLOBAL __attribute__((opencl_global))

typedef VK_GLOBAL unsigned char* GlobalPtr;
typedef VK_GLOBAL float* GlobalFloat;
typedef VK_GLOBAL int* GlobalInt;
typedef VK_GLOBAL long long* GlobalI64;
typedef VK_GLOBAL osl_vk_abi::Vec3* GlobalVec3;
typedef VK_GLOBAL osl_vk_abi::TriangleIndices* GlobalTri;

// clspv's DefineOpenCLWorkItemBuiltinsPass looks for exactly this mangled
// name, so the declaration has to stay C++-style (no extern "C").
//
// The return type must be the pointer-width size_t (i64 under
// `--arch=spirv64`), *not* `unsigned int`: clspv replaces this call with an
// i64-valued computation and then rewrites the uses in place.  If the module
// declares an i32 return, that rewrite hands i64 values to i32 operands and the
// module fails verification ("Type too small for ZExt"), i.e. clspv aborts.
unsigned long long get_global_id(unsigned int dim);

// Synthesised by the module builder: runs material `shader_id` on `sg`.
extern "C" void osl_vk_shade_dispatch(GlobalPtr sg, int shader_id);

// One restricted dispatcher per material: each can only reach its own group
// (plus the light group), which is what keeps a per-material shading kernel
// from carrying every shader.  The module builder fills these in.
extern "C" void osl_vk_shade_dispatch_0(GlobalPtr sg, int shader_id);
extern "C" void osl_vk_shade_dispatch_1(GlobalPtr sg, int shader_id);
extern "C" void osl_vk_shade_dispatch_2(GlobalPtr sg, int shader_id);
extern "C" void osl_vk_shade_dispatch_3(GlobalPtr sg, int shader_id);
extern "C" void osl_vk_shade_dispatch_4(GlobalPtr sg, int shader_id);
extern "C" void osl_vk_shade_dispatch_5(GlobalPtr sg, int shader_id);
extern "C" void osl_vk_shade_dispatch_6(GlobalPtr sg, int shader_id);
extern "C" void osl_vk_shade_dispatch_7(GlobalPtr sg, int shader_id);
extern "C" void osl_vk_shade_dispatch_8(GlobalPtr sg, int shader_id);
extern "C" void osl_vk_shade_dispatch_9(GlobalPtr sg, int shader_id);
extern "C" void osl_vk_shade_dispatch_10(GlobalPtr sg, int shader_id);
extern "C" void osl_vk_shade_dispatch_11(GlobalPtr sg, int shader_id);
extern "C" void osl_vk_shade_dispatch_12(GlobalPtr sg, int shader_id);
extern "C" void osl_vk_shade_dispatch_13(GlobalPtr sg, int shader_id);
extern "C" void osl_vk_shade_dispatch_14(GlobalPtr sg, int shader_id);
extern "C" void osl_vk_shade_dispatch_15(GlobalPtr sg, int shader_id);

/// Material slots the device runtime can specialize a shading kernel for (0..15,
/// see the instantiations at the end of this file).  Scenes with more shading
/// groups than that need a fix in the host dispatch, not here.  The testsuite's
/// widest scene (render-mx-furnace-*, one group per sphere) needs 12.

/// Route a shader id through the dispatcher that belongs to kernel MAT.  With
/// `if constexpr` each instantiation references exactly one of them, so the
/// other groups become unreachable in that kernel and clspv drops their code.
template <int MAT>
inline void wave_shade_dispatch(GlobalPtr sg, int shader_id)
{
    if constexpr (MAT == 0)
        osl_vk_shade_dispatch_0(sg, shader_id);
    else if constexpr (MAT == 1)
        osl_vk_shade_dispatch_1(sg, shader_id);
    else if constexpr (MAT == 2)
        osl_vk_shade_dispatch_2(sg, shader_id);
    else if constexpr (MAT == 3)
        osl_vk_shade_dispatch_3(sg, shader_id);
    else if constexpr (MAT == 4)
        osl_vk_shade_dispatch_4(sg, shader_id);
    else if constexpr (MAT == 5)
        osl_vk_shade_dispatch_5(sg, shader_id);
    else if constexpr (MAT == 6)
        osl_vk_shade_dispatch_6(sg, shader_id);
    else if constexpr (MAT == 7)
        osl_vk_shade_dispatch_7(sg, shader_id);
    else if constexpr (MAT == 8)
        osl_vk_shade_dispatch_8(sg, shader_id);
    else if constexpr (MAT == 9)
        osl_vk_shade_dispatch_9(sg, shader_id);
    else if constexpr (MAT == 10)
        osl_vk_shade_dispatch_10(sg, shader_id);
    else if constexpr (MAT == 11)
        osl_vk_shade_dispatch_11(sg, shader_id);
    else if constexpr (MAT == 12)
        osl_vk_shade_dispatch_12(sg, shader_id);
    else if constexpr (MAT == 13)
        osl_vk_shade_dispatch_13(sg, shader_id);
    else if constexpr (MAT == 14)
        osl_vk_shade_dispatch_14(sg, shader_id);
    else
        osl_vk_shade_dispatch_15(sg, shader_id);
}

using namespace osl_vk_abi;

// ---------------------------------------------------------------------------
// Small vector helpers (Vec3 is the ABI type, so no operator overloading on the
// shared struct -- it has to stay POD for both host and device).
// ---------------------------------------------------------------------------

inline Vec3 mk(float x, float y, float z)
{
    Vec3 v;
    v.x = x;
    v.y = y;
    v.z = z;
    return v;
}

inline Vec3 add(const Vec3& a, const Vec3& b) { return mk(a.x + b.x, a.y + b.y, a.z + b.z); }
inline Vec3 sub(const Vec3& a, const Vec3& b) { return mk(a.x - b.x, a.y - b.y, a.z - b.z); }
inline Vec3 mul(const Vec3& a, float s) { return mk(a.x * s, a.y * s, a.z * s); }
inline Vec3 mulv(const Vec3& a, const Vec3& b) { return mk(a.x * b.x, a.y * b.y, a.z * b.z); }
inline float dot(const Vec3& a, const Vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline Vec3 cross(const Vec3& a, const Vec3& b)
{
    return mk(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}
inline float length(const Vec3& a) { return __builtin_sqrtf(dot(a, a)); }
inline Vec3 normalize(const Vec3& a)
{
    // Divide, do not multiply by the reciprocal: Imath's normalize()
    // (ImathVec.h:2083) divides each component by the length and says
    // explicitly why -- `1/l` can overflow, and even when it does not, the
    // reciprocal rounds first and the product rounds again, so the result
    // differs from the CPU's by an ulp on a large fraction of vectors.  An ulp
    // in the camera-ray direction decides whether a sample that grazes a
    // silhouette hits the surface, and in render-raytypes that flips whole
    // pixels by an order of magnitude (the reference is a 4-sample frame, so a
    // single sample carries a quarter of the pixel).
    float l = length(a);
    return l > 0.0f ? mk(a.x / l, a.y / l, a.z / l) : a;
}
/// Two vectors spanning the plane perpendicular to `d` -- the same construction
/// OSL's `ortho()` uses, so the texture derivatives line up with the CPU ones.
inline void ortho(const Vec3& d, Vec3& a, Vec3& b)
{
    float len2 = dot(d, d);
    if (len2 > 0.0f) {
        float inv  = 1.0f / len2;
        float ayz  = -d.y - d.z;
        float azz  = d.y - d.z;
        float axx  = d.x + ayz * (d.y - d.z) * inv;
        float ayy  = ayz + d.x * d.y * inv + d.z;
        float azzz = d.x * d.z * inv + d.y + azz;
        a = normalize(mk(axx, ayy, azzz));
        b = normalize(cross(d, a));
    } else {
        a = mk(0.0f, 1.0f, 0.0f);
        b = mk(0.0f, 0.0f, 1.0f);
    }
}

// ---------------------------------------------------------------------------
// Sampler: the same Owen-scrambled Sobol / hash construction testrender uses on
// the CPU (src/testrender/sampling.h), copied so that a pixel's samples land in
// the same places the CPU path puts them -- otherwise AA jitter alone would make
// the two images differ.
// ---------------------------------------------------------------------------

// sampling.h:256 -- hash-prospector.  Same constants, not merely "a" hash: the
// CPU and CUDA paths both use these, and sample placement has to match them.
inline unsigned int vk_hash(unsigned int s)
{
    s = s ^ (s >> 16);
    s = s * 0x21f0aaadu;
    s = s ^ (s >> 15);
    s = s * 0xd35a2d97u;
    s = s ^ (s >> 15);
    return s;
}

inline unsigned int reversebits(unsigned int x)
{
    x = ((x & 0xaaaaaaaau) >> 1) | ((x & 0x55555555u) << 1);
    x = ((x & 0xccccccccu) >> 2) | ((x & 0x33333333u) << 2);
    x = ((x & 0xf0f0f0f0u) >> 4) | ((x & 0x0f0f0f0fu) << 4);
    x = ((x & 0xff00ff00u) >> 8) | ((x & 0x00ff00ffu) << 8);
    return (x >> 16) | (x << 16);
}

// sampling.h:281 -- hashes the n-th bit of `p` and returns its input reversed,
// which is why the callers below reverse the *result* before taking the top
// 24 bits.
inline unsigned int vk_owen_scramble(unsigned int p, unsigned int s)
{
    p ^= p * 0x3d20adeau;
    p += s;
    p *= (s >> 16) | 1u;
    p ^= p * 0x05526c56u;
    p ^= p * 0x53a22864u;
    return reversebits(p);
}

// sampling.h's zmatrix (the third Sobol dimension), at namespace scope and in
// the `constant` address space.
//
// It used to live inside Sampler::get() as a function-local `static const`.
// clang then emits it as a *local* array with a constant initializer, and clspv
// hoists that into a module-scope variable of Function storage class carrying
// the initializer -- which SPIR-V forbids ("Variables cannot have function
// storage class outside of a function"), so the module fails validation and
// Intel's compiler aborts on it.  At namespace scope in address space 0 the
// same thing happens (this translation unit is not compiled as OpenCL C, so a
// plain global lands in the private space); `address_space(2)` makes it a
// constant global, which clspv emits as an OpVariable of Constant storage
// class -- legal at module scope.
static const unsigned int kSobolZMatrix[24]
    __attribute__((address_space(2)))
    = { 0x000001u, 0x000003u, 0x000006u, 0x000009u, 0x000017u, 0x00003au,
        0x000071u, 0x0000a3u, 0x000116u, 0x000339u, 0x000677u, 0x0009aau,
        0x001601u, 0x003903u, 0x007706u, 0x00aa09u, 0x010117u, 0x03033au,
        0x060671u, 0x0909a3u, 0x171616u, 0x3a3939u, 0x717777u, 0xa3aaaau };

struct Sampler {
    unsigned int seed, index;

    inline void init(int px, int py, int si)
    {
        seed  = (((unsigned int)px & 2047u) << 22) | (((unsigned int)py & 2047u) << 11);
        index = reversebits((unsigned int)si);
    }

    /// Owen-scrambled 24-bit index, biased so the sample stays inside [0,1).
    /// Mirrors sampling.h:222-251 step for step: scramble the index with
    /// owen_scramble (not a bare XOR), run the Sobol matrix product, then
    /// scramble *and reverse* each coordinate before taking the top 24 bits.
    /// The reversal is what puts the value in [0,1); dropping it leaves every
    /// sample coordinate down near 2^-10, i.e. one fixed point per pixel and no
    /// antialiasing at all.
    inline Vec3 get()
    {
        seed += 4;  // advance depth for the next call
        // The Sobol matrix product is the same construction as sampling.h's
        // zmatrix/ymatrix loop; see there for the reference.
        unsigned int si = vk_owen_scramble(index, vk_hash(seed - 4)) & 0xFFFFFFu;
        unsigned int rx = si, ry = 0, rz = 0, ymatrix = 1;
        for (int c = 0; c < 24; ++c) {
            unsigned int bit = (si >> c) & 1u;
            ry ^= bit * ymatrix;
            rz ^= bit * kSobolZMatrix[c];
            ymatrix ^= ymatrix << 1;
        }
        // Scramble the results and scale by 2^-24 to guarantee equally spaced
        // values in [0,1).
        return mk((float)(vk_owen_scramble(rx, vk_hash(seed - 3)) >> 8)
                      * 5.96046448e-8f,
                  (float)(vk_owen_scramble(ry, vk_hash(seed - 2)) >> 8)
                      * 5.96046448e-8f,
                  (float)(vk_owen_scramble(rz, vk_hash(seed - 1)) >> 8)
                      * 5.96046448e-8f);
    }
};

// ---------------------------------------------------------------------------
// Camera
// ---------------------------------------------------------------------------

struct Camera {
    Vec3 eye, dir, up, cx, cy;
    float fov;
    int xres, yres;
    float invw, invh;

    inline void setup(VK_GLOBAL const FrameParams* p)
    {
        eye  = mk(p->eye[0], p->eye[1], p->eye[2]);
        dir  = normalize(mk(p->dir[0], p->dir[1], p->dir[2]));
        up   = mk(p->up[0], p->up[1], p->up[2]);
        fov  = p->fov;
        xres = p->width;
        yres = p->height;
        invw = p->invw > 0.0f ? p->invw : 1.0f / (float)xres;
        invh = p->invh > 0.0f ? p->invh : 1.0f / (float)yres;
        // The host fills these in from testrender's finalized Camera, i.e. with
        // OIIO::fast_tan.  Recomputing them here with tanf would aim the rays
        // slightly differently from the CPU (see the FrameParams comment).
        if (p->cx[0] != 0.0f || p->cx[1] != 0.0f || p->cx[2] != 0.0f) {
            cx = mk(p->cx[0], p->cx[1], p->cx[2]);
            cy = mk(p->cy[0], p->cy[1], p->cy[2]);
        } else {
            float k    = __builtin_tanf(fov * 3.14159265358979f / 360.0f);
            Vec3 right = normalize(cross(dir, up));
            cx         = mul(right, (float)xres * k / (float)yres);
            cy         = mul(normalize(cross(cx, dir)), k);
        }
    }
};

struct Ray {
    Vec3 origin, direction;
    float radius, spread;
};

// ---------------------------------------------------------------------------
// Scene: brute force over the triangle list in the buffer.
// ---------------------------------------------------------------------------

struct Hit {
    float t;
    float u, v;  ///< barycentrics of b (index 1) and c (index 2)
    int id;
};

inline float inf() { return 3.402823466e+38f; }

// Loading a struct straight out of the storage buffer (`Vec3 v = ptr[i]`) makes
// clang bind the implicit copy constructor to a generic address-space reference,
// which it then rejects.  Reading the fields one at a time is the whole fix --
// and it keeps every access a plain `gep` off the kernel argument, which is what
// clspv can attribute to the buffer.
typedef VK_GLOBAL const Vec3* GlobalCVec3;
typedef VK_GLOBAL const TriangleIndices* GlobalCTri;
typedef VK_GLOBAL const unsigned char* GlobalCPtr;
typedef VK_GLOBAL const float* GlobalFloatCPtr;

inline Vec3 load_vec(GlobalCVec3 p)
{
    Vec3 v;
    v.x = p->x;
    v.y = p->y;
    v.z = p->z;
    return v;
}

inline void store_vec(VK_GLOBAL Vec3* p, const Vec3& v)
{
    p->x = v.x;
    p->y = v.y;
    p->z = v.z;
}

inline TriangleIndices load_tri(GlobalCTri p)
{
    TriangleIndices t;
    t.a = p->a;
    t.b = p->b;
    t.c = p->c;
    return t;
}

/// Byte-addressed reads for the traversal.
///
/// `p[i]` on a *typed* pointer with a runtime index is lowered by clspv to
/// `OpPtrAccessChain`, and the Intel compiler crashes on that instruction
/// (measured: the whole module fails to build a pipeline as soon as the
/// intersection pass contains five of them, while the old megakernel module --
/// which had none -- built fine).  Doing the arithmetic in bytes keeps every
/// access a plain `OpAccessChain` off the kernel argument, which is the form
/// the rest of this file already uses.
inline TriangleIndices tri_at(GlobalCPtr tris, int id)
{
    GlobalCPtr p = tris + (unsigned long long)id * sizeof(TriangleIndices);
    TriangleIndices t;
    t.a = *(VK_GLOBAL const int*)p;
    t.b = *(VK_GLOBAL const int*)(p + 4);
    t.c = *(VK_GLOBAL const int*)(p + 8);
    return t;
}

inline Vec3 vec_at(GlobalCPtr verts, int id)
{
    GlobalCPtr p = verts + (unsigned long long)id * sizeof(Vec3);
    Vec3 v;
    v.x = *(VK_GLOBAL const float*)p;
    v.y = *(VK_GLOBAL const float*)(p + 4);
    v.z = *(VK_GLOBAL const float*)(p + 8);
    return v;
}

/// Möller-Trumbore, matching the CPU's triangle test closely enough that a
/// shadow ray arrives at the same primitive.
inline bool intersect_tri(const Ray& r, const Vec3& va, const Vec3& vb,
                          const Vec3& vc, float& t, float& u, float& v)
{
    const Vec3 e1 = sub(vb, va), e2 = sub(vc, va);
    const Vec3 pv = cross(r.direction, e2);
    const float det = dot(e1, pv);
    if (__builtin_fabsf(det) < 1e-12f)
        return false;
    const float inv = 1.0f / det;
    const Vec3 tv   = sub(r.origin, va);
    u               = dot(tv, pv) * inv;
    if (u < 0.0f || u > 1.0f)
        return false;
    const Vec3 qv = cross(tv, e1);
    v             = dot(r.direction, qv) * inv;
    if (v < 0.0f || u + v > 1.0f)
        return false;
    t = dot(e2, qv) * inv;
    return t > 1e-6f;
}

struct SceneView {
    GlobalTri tris;
    GlobalVec3 verts;
    GlobalVec3 normals;
    /// Per-primitive per-corner indices into the normal table (12 B stride).
    /// It has to be the triangle type, not `int`: indexing an `int` pointer by
    /// a primitive id walks 4 bytes per corner group instead of 12 and blends
    /// three unrelated normals into the shading normal.
    GlobalCTri tri_n;
    GlobalInt lightprims;
    VK_GLOBAL const float* uvs;  ///< two floats per uv corner (x, y)
    GlobalCTri tri_uv;
    VK_GLOBAL const BvhNodeVK* nodes;   ///< the CPU BVH, uploaded verbatim
    VK_GLOBAL const unsigned* bvh_idx;  ///< triangle ids the leaves point at
    GlobalPtr stack;                    ///< traversal stack, 64 slots per pixel
    int ntris;
    int nlightprims;
    int num_bvh_nodes;
};

// --- helpers for the sheared watertight traversal (bvh.cpp:220-262) ---------

inline float compv(const Vec3& v, int i)
{
    return i == 0 ? v.x : (i == 1 ? v.y : v.z);
}

inline float bminf(float a, float b) { return b < a ? b : a; }
inline float bmaxf(float a, float b) { return b > a ? b : a; }

inline bool box_intersect_vk(const Vec3& org, const Vec3& rdir, float tmax,
                             VK_GLOBAL const float* bounds, float* dist)
{
    const float tx1 = (bounds[0] - org.x) * rdir.x;
    const float tx2 = (bounds[1] - org.x) * rdir.x;
    const float ty1 = (bounds[2] - org.y) * rdir.y;
    const float ty2 = (bounds[3] - org.y) * rdir.y;
    const float tz1 = (bounds[4] - org.z) * rdir.z;
    const float tz2 = (bounds[5] - org.z) * rdir.z;
    float t0        = bminf(tx1, tx2);
    tmax            = bminf(tmax, bmaxf(tx1, tx2));
    t0              = bmaxf(t0, bminf(ty1, ty2));
    tmax            = bminf(tmax, bmaxf(ty1, ty2));
    t0              = bmaxf(t0, bminf(tz1, tz2));
    tmax            = bminf(tmax, bmaxf(tz1, tz2));
    *dist           = t0;
    t0              = bmaxf(0.0f, t0);
    return t0 <= tmax;
}

inline unsigned signmask_vk(float a)
{
    return __builtin_bit_cast(unsigned, a) & 0x80000000u;
}

inline float xorf_vk(float a, unsigned b)
{
    return __builtin_bit_cast(float, __builtin_bit_cast(unsigned, a) ^ b);
}

// --- the BVH traversal stack ------------------------------------------------
//
// Slot layout is {int node, float dist}, 8 bytes, at most
// `kTraversalStackDepth` deep.  Addressed by byte offset so the codegen stays
// in the same shape as the rest of the file (an `OpAccessChain` off the buffer
// argument) instead of a typed `p[i]`, which clspv turns into
// `OpPtrAccessChain` -- fine on a storage buffer, fatal on a private alloca.

inline int stack_node(GlobalPtr st, int i)
{
    return *(VK_GLOBAL const int*)(st + (unsigned long long)i * 8ULL);
}

inline float stack_dist(GlobalPtr st, int i)
{
    return *(VK_GLOBAL const float*)(st + (unsigned long long)i * 8ULL + 4ULL);
}

inline void stack_node_set(GlobalPtr st, int i, int node)
{
    *(VK_GLOBAL int*)(st + (unsigned long long)i * 8ULL) = node;
}

inline void stack_dist_set(GlobalPtr st, int i, float dist)
{
    *(VK_GLOBAL float*)(st + (unsigned long long)i * 8ULL + 4ULL) = dist;
}

// Two-speed intersect: `skip_b` is the light primitive a shadow ray is aimed
// at, so the shadow test has to ignore it as well as the surface it left (this
// is what Scene::intersect(r, tmax, skipID1, skipID2) does on the CPU).
//
// The traversal is a direct port of bvh.cpp:264-350: same stack discipline
// (nearer child pushed last so it pops first), same sheared coordinate frame
// (kz/kx/ky + shearDir), same watertight U/V/W test and the same
// xorf(T,mask) sign trick, and the same skipID1/skipID2 semantics -- so a ray
// reports the very triangle the CPU would report.
inline Hit intersect_scene(const SceneView& s, const Ray& r, float tmax,
                           int skip_id, int skip_id_b = -1)
{
    Hit h;
    h.t  = inf();
    h.u  = 0.0f;
    h.v  = 0.0f;
    h.id = -1;

    if (s.num_bvh_nodes <= 0) {
        // No BVH uploaded: linear sweep (the pre-BVH behaviour).
        for (int i = 0; i < s.ntris; ++i) {
            if (i == skip_id || i == skip_id_b)
                continue;
            TriangleIndices ti = tri_at((GlobalCPtr)s.tris, i);
            float t, u, v;
            if (!intersect_tri(r, vec_at((GlobalCPtr)s.verts, ti.a),
                               vec_at((GlobalCPtr)s.verts, ti.b),
                               vec_at((GlobalCPtr)s.verts, ti.c), t, u, v))
                continue;
            if (t < tmax && t < h.t) {
                h.t  = t;
                h.u  = u;
                h.v  = v;
                h.id = i;
            }
        }
        (void)s.normals;
        (void)s.tri_n;
        return h;
    }

    // The traversal stack lives in the storage buffer, not in a local array --
    // see the comment on the accessors above.
    GlobalPtr stack = s.stack;

    const Vec3 org = r.origin;
    const Vec3 dir = r.direction;
    const Vec3 rdir = mk(1.0f / dir.x, 1.0f / dir.y, 1.0f / dir.z);
    int kz = 0;
    if (__builtin_fabsf(dir.y) > __builtin_fabsf(compv(dir, kz)))
        kz = 1;
    if (__builtin_fabsf(dir.z) > __builtin_fabsf(compv(dir, kz)))
        kz = 2;
    const int kx = kz == 2 ? 0 : kz + 1;
    const int ky = kx == 2 ? 0 : kx + 1;
    const Vec3 shearDir
        = mk(compv(dir, kx) / compv(dir, kz), compv(dir, ky) / compv(dir, kz),
             compv(rdir, kz));

    float best             = tmax;
    int stackPtr           = 1;
    stack_node_set(stack, 0, 0);
    stack_dist_set(stack, 0, best);
    while (stackPtr != 0) {
        --stackPtr;
        if (best < stack_dist(stack, stackPtr))
            continue;
        const int ni            = stack_node(stack, stackPtr);
        VK_GLOBAL const BvhNodeVK* node
            = (VK_GLOBAL const BvhNodeVK*)((GlobalCPtr)s.nodes
                                           + (unsigned long long)ni
                                                 * sizeof(BvhNodeVK));
        const unsigned nprims   = node->nprims;
        if (nprims) {
            const unsigned base = node->child;
            for (unsigned i = 0; i < nprims; ++i) {
                const int id = (int)*(VK_GLOBAL const unsigned*)((GlobalCPtr)s.bvh_idx
                                    + (unsigned long long)(base + i) * 4ULL);
                if (id == skip_id || id == skip_id_b)
                    continue;  // the surface we just left, or the lit triangle
                TriangleIndices ti = tri_at((GlobalCPtr)s.tris, id);
                const Vec3 A = sub(vec_at((GlobalCPtr)s.verts, ti.a), org);
                const Vec3 B = sub(vec_at((GlobalCPtr)s.verts, ti.b), org);
                const Vec3 C = sub(vec_at((GlobalCPtr)s.verts, ti.c), org);
                const float Ax = compv(A, kx) - shearDir.x * compv(A, kz);
                const float Ay = compv(A, ky) - shearDir.y * compv(A, kz);
                const float Bx = compv(B, kx) - shearDir.x * compv(B, kz);
                const float By = compv(B, ky) - shearDir.y * compv(B, kz);
                const float Cx = compv(C, kx) - shearDir.x * compv(C, kz);
                const float Cy = compv(C, ky) - shearDir.y * compv(C, kz);
                const float U  = Cx * By - Cy * Bx;
                const float V  = Ax * Cy - Ay * Cx;
                const float W  = Bx * Ay - By * Ax;
                if ((U < 0.0f || V < 0.0f || W < 0.0f)
                    && (U > 0.0f || V > 0.0f || W > 0.0f))
                    continue;
                const float det = U + V + W;
                if (det == 0.0f)
                    continue;
                const float T = shearDir.z
                               * (U * compv(A, kz) + V * compv(B, kz)
                                  + W * compv(C, kz));
                const unsigned mask = signmask_vk(det);
                if (xorf_vk(T, mask) < 0.0f)
                    continue;
                if (xorf_vk(T, mask) > best * xorf_vk(det, mask))
                    continue;
                const float rcpDet = 1.0f / det;
                best               = T * rcpDet;
                h.t                = best;
                h.u                = V * rcpDet;
                h.v                = W * rcpDet;
                h.id               = id;
            }
        } else {
            int c1 = (int)node->child;
            int c2 = c1 + 1;
            float d1 = 0.0f, d2 = 0.0f;
            bool hit1 = box_intersect_vk(org, rdir, best, s.nodes[c1].bounds,
                                         &d1);
            bool hit2 = box_intersect_vk(org, rdir, best, s.nodes[c2].bounds,
                                         &d2);
            if (d1 > d2) {
                const bool th = hit1;
                hit1          = hit2;
                hit2          = th;
                const float td = d1;
                d1             = d2;
                d2             = td;
                const int tc   = c1;
                c1             = c2;
                c2             = tc;
            }
            stack_node_set(stack, stackPtr, c2);
            stack_dist_set(stack, stackPtr, d2);
            stackPtr += hit2;
            stack_node_set(stack, stackPtr, c1);
            stack_dist_set(stack, stackPtr, d1);
            stackPtr += hit1;
        }
    }
    if (h.id < 0)
        h.t = inf();
    (void)s.normals;
    (void)s.tri_n;
    return h;
}


/// MIS::power_heuristic<MIS::EVAL_WEIGHT>, copied from sampling.h: one side of
/// the product is an evaluation (the BSDF), the other a sampled weight (the
/// light direction).
inline float power_heuristic_eval_weight(float sampled_pdf, float other_pdf)
{
    float r, mis;
    if (sampled_pdf > other_pdf) {
        r   = other_pdf / sampled_pdf;
        mis = 1.0f / (1.0f + r * r);
    } else if (sampled_pdf < other_pdf) {
        r   = sampled_pdf / other_pdf;
        mis = 1.0f - 1.0f / (1.0f + r * r);
    } else {
        r   = 1.0f;
        mis = 0.5f;
    }
    return mis * ((other_pdf > sampled_pdf) ? (1.0f / r) : r);
}

/// MIS::power_heuristic<MIS::WEIGHT_EVAL>: the same balance weight without the
/// eval/weight ratio correction (sampling.h:155-158).  The self-emission of a
/// light primitive uses this one (simpleraytracer.cpp:1083), and it matters:
/// a camera ray carries bsdf_pdf = inf, and the EVAL_WEIGHT form would return
/// inf*0 = 0 and erase every visible light.
inline float power_heuristic_weight_eval(float sampled_pdf, float other_pdf)
{
    if (sampled_pdf > other_pdf)
        return 1.0f / (1.0f + (other_pdf / sampled_pdf) * (other_pdf / sampled_pdf));
    if (sampled_pdf < other_pdf)
        return 1.0f - 1.0f / (1.0f + (sampled_pdf / other_pdf) * (sampled_pdf / other_pdf));
    return 0.5f;
}

/// OIIO::fast_rint (fmath.h:1670): round to nearest, ties to even -- it is
/// std::rint on the build testrender uses.  A `x + 0.5` emulation would round
/// ties the other way, and `fast_sincos` does hit ties: phi == pi/2 gives
/// exactly x == 0.5, where 1 instead of 0 flips the quadrant and the sample.
inline float vk_fast_rint(float x) { return __builtin_rintf(x); }

inline float vk_madd(float a, float b, float c) { return __builtin_fmaf(a, b, c); }

inline float vk_clampf(float v, float lo, float hi)
{
    return v < lo ? lo : (v > hi ? hi : v);
}

/// OIIO::fast_sincos (fmath.h:1747), copied verbatim -- including the four-term
/// argument reduction and the two minimax polynomials, because the CPU's cosine
/// sampler feeds its result straight into the sampled direction.  Two separate
/// sin()/cos() calls would not match this, and they would also get fused into
/// `llvm.sincos.f32`, which clspv cannot lower.
inline void fast_sincos_vk(float x, float& sine, float& cosine)
{
    int q      = (int)vk_fast_rint(x * 0.318309886f);  // 1/pi
    float qf   = (float)q;
    x          = vk_madd(qf, -0.78515625f * 4, x);
    x          = vk_madd(qf, -0.00024187564849853515625f * 4, x);
    x          = vk_madd(qf, -3.7747668102383613586e-08f * 4, x);
    x          = vk_madd(qf, -1.2816720341285448015e-12f * 4, x);
    x          = 1.57079632679f - (1.57079632679f - x);  // crush denormals
    float s    = x * x;
    if ((q & 1) != 0)
        x = -x;
    float su = 2.6083159809786593541503e-06f;
    su       = vk_madd(su, s, -0.0001981069071916863322258f);
    su       = vk_madd(su, s, +0.00833307858556509017944336f);
    su       = vk_madd(su, s, -0.166666597127914428710938f);
    su       = vk_madd(s, su * x, x);
    float cu = -2.71811842367242206819355e-07f;
    cu       = vk_madd(cu, s, +2.47990446951007470488548e-05f);
    cu       = vk_madd(cu, s, -0.00138888787478208541870117f);
    cu       = vk_madd(cu, s, +0.0416666641831398010253906f);
    cu       = vk_madd(cu, s, -0.5f);
    cu       = vk_madd(cu, s, +1.0f);
    if ((q & 1) != 0)
        cu = -cu;
    sine   = vk_clampf(su, -1.0f, 1.0f);
    cosine = vk_clampf(cu, -1.0f, 1.0f);
}

/// Sampling::to_unit_disk (sampling.h:70-90): the concentric disk map.  Note the
/// argument order of fast_sincos -- it writes the *sine* into its first output,
/// so `x` below ends up holding sin(phi) and `y` cos(phi), exactly as the CPU's
/// version does.
inline void to_unit_disk_vk(float& x, float& y)
{
    const float PI_OVER_4 = 0.785398163f;
    const float PI_OVER_2 = 1.570796327f;
    float phi, r;
    float a = 2.0f * x - 1.0f;
    float b = 2.0f * y - 1.0f;
    if (a * a > b * b) {  // use squares instead of absolute values
        r   = a;
        phi = PI_OVER_4 * (b / a);
    } else if (b != 0.0f) {
        r   = b;
        phi = PI_OVER_2 - PI_OVER_4 * (a / b);
    } else {
        r   = 0.0f;
        phi = 0.0f;
    }
    float sn, cs;
    fast_sincos_vk(phi, sn, cs);
    x = sn * r;
    y = cs * r;
}

/// TangentFrame::from_normal (sampling.h:17-26), the Pixar orthonormal basis.
/// The basis matters as much as the disk mapping: the same (x,y) put through a
/// different basis is a different direction.
inline void pixar_frame(const Vec3& n, Vec3& u, Vec3& v)
{
    // The CPU uses copysignf(1.0f, n.z), i.e. it reads the *sign bit* of n.z.
    // A face normal whose z is a negative zero (the right wall of the cornell
    // scene gives exactly that) therefore selects sign == -1 and a basis that
    // is mirrored in u and v relative to `n.z >= 0 ? 1 : -1` (where -0.0
    // compares equal to 0.0 and would pick +1).  Same numbers, different
    // basis -- every cosine sample lands mirror-imaged, which is invisible on
    // the first hit's colour but shows up as a completely different second
    // bounce.  copysignf keeps the two implementations bit-identical.
    const float sign = __builtin_copysignf(1.0f, n.z);
    const float a    = -1.0f / (sign + n.z);
    const float b    = n.x * n.y * a;
    u = mk(1.0f + sign * n.x * n.x * a, sign * b, -sign * n.x);
    v = mk(b, sign + n.y * n.y * a, -n.y);
}

/// Sampling::sample_cosine_hemisphere (sampling.h:92-100), device side.
inline Vec3 sample_cosine_hemisphere(const Vec3& n, float rndx, float rndy,
                                     float& pdf)
{
    to_unit_disk_vk(rndx, rndy);
    float cos_theta = __builtin_sqrtf(1.0f - rndx * rndx - rndy * rndy);
    if (cos_theta < 0.0f)
        cos_theta = 0.0f;
    Vec3 u, v;
    pixar_frame(n, u, v);
    pdf = cos_theta * 0.31830988618f;
    return add(add(mul(u, rndx), mul(v, rndy)), mul(n, cos_theta));
}

/// TangentFrame::from_normal_and_tangent (sampling.h:30-42). Falls back to the
/// Pixar basis when T is zero or colinear with N, exactly like the CPU.
inline void frame_from_nt(const Vec3& n, const Vec3& t, Vec3& u, Vec3& v,
                          Vec3& w)
{
    Vec3 x      = sub(t, mul(n, dot(n, t)));
    float xlen2 = dot(x, x);
    if (xlen2 > 0.0f) {
        x = mul(x, 1.0f / __builtin_sqrtf(xlen2));
        u = x;
        v = cross(n, x);
        w = n;
    } else {
        pixar_frame(n, u, v);
        w = n;
    }
}

// Fresnel helpers live further down next to the delta closures, but the
// microfacet lobes need them too.
inline float fresnel_dielectric_vk(float cosi, float eta);
inline float fresnel_refraction_vk(const Vec3& I, const Vec3& N, float eta,
                                   Vec3& T);

/// The Ward anisotropic specular BRDF, device side.  Two functions that mirror
/// shading.cpp:412-494 field by field: WardParams is { Vec3 N; Vec3 T;
/// float ax; float ay; }, and both eval and sample hand back the same
/// (weight, pdf) pair the CPU produces.
struct WardVK {
    Vec3 N, T;
    float ax, ay;
};

inline void ward_eval_vk(const WardVK& p, const Vec3& wo, const Vec3& wi,
                         Vec3& weight, float& pdf)
{
    weight = mk(0.0f, 0.0f, 0.0f);
    pdf    = 0.0f;
    float cosNO = dot(p.N, wo);
    float cosNI = dot(p.N, wi);
    if (cosNI > 0.0f && cosNO > 0.0f) {
        Vec3 H = normalize(add(wi, wo));
        Vec3 u, v, w;
        frame_from_nt(p.N, p.T, u, v, w);
        float dotx = dot(u, H) / p.ax;
        float doty = dot(v, H) / p.ay;
        float dotn = dot(w, H);
        float oh   = dot(H, wi);
        float e = __builtin_expf(-(dotx * dotx + doty * doty) / (dotn * dotn));
        float c = 12.566370614f * p.ax * p.ay;  // 4*pi*ax*ay
        float k = oh * dotn * dotn * dotn;
        pdf     = e / (c * k);
        float s = __builtin_sqrtf(cosNI / cosNO);
        float val = k * s;
        weight  = mk(val, val, val);
    }
}

inline void ward_sample_vk(const WardVK& p, const Vec3& wo, float rx, float ry,
                           Vec3& wi, Vec3& weight, float& pdf)
{
    wi     = mk(0.0f, 0.0f, 0.0f);
    weight = mk(0.0f, 0.0f, 0.0f);
    pdf    = 0.0f;
    float cosNO = dot(p.N, wo);
    if (cosNO > 0.0f) {
        Vec3 u, v, w;
        frame_from_nt(p.N, p.T, u, v, w);
        float phi = 6.2831855f * rx;  // 2*pi
        float sp, cp;
        fast_sincos_vk(phi, sp, cp);
        float cosPhi = p.ax * cp;
        float sinPhi = p.ay * sp;
        float kk     = 1.0f / __builtin_sqrtf(cosPhi * cosPhi + sinPhi * sinPhi);
        cosPhi *= kk;
        sinPhi *= kk;
        float thetaDenom = (cosPhi * cosPhi) / (p.ax * p.ax)
                           + (sinPhi * sinPhi) / (p.ay * p.ay);
        float tanTheta2 = -__builtin_logf(1.0f - ry) / thetaDenom;
        float cosTheta  = 1.0f / __builtin_sqrtf(1.0f + tanTheta2);
        float sinTheta  = cosTheta * __builtin_sqrtf(tanTheta2);
        float hx = sinTheta * cosPhi, hy = sinTheta * sinPhi, hz = cosTheta;
        float dotx = hx / p.ax;
        float doty = hy / p.ay;
        float dotn = hz;
        Vec3 h = add(add(mul(u, hx), mul(v, hy)), mul(w, hz));
        float oh = dot(h, wo);
        wi       = sub(mul(h, 2.0f * oh), wo);
        float cosNI = dot(p.N, wi);
        if (cosNI > 0.0f) {
            float e = __builtin_expf(-(dotx * dotx + doty * doty) / (dotn * dotn));
            float c = 12.566370614f * p.ax * p.ay;
            float k = oh * dotn * dotn * dotn;
            pdf     = e / (c * k);
            float s = __builtin_sqrtf(cosNI / cosNO);
            float val = k * s;
            weight  = mk(val, val, val);
        }
    }
}

// ---------------------------------------------------------------------------
// Microfacet (Beckmann), device side
// ---------------------------------------------------------------------------
//
// shading.cpp:469-796 ported statement for statement: the visible-normal
// sampling of d'Eon & Heitz, the Beckmann slope distribution, the anisotropic
// masking-shadowing terms, and the two lobes `refract` selects between.  The
// only substitution is erf/erfinv, which have no SPIR-V builtin: OIIO's
// fmath.h fast_erf / fast_ierf polynomials are copied below, with the exact
// logf where fmath.h uses fast_log.  Those agree to ~1e-7, i.e. below the
// precision the float arithmetic around them keeps.

// The CPU's microfacet sampling is not libm: shading.cpp:533-561 calls
// OIIO::fast_erf / fast_ierf / fast_exp, whose fast_log2 / fast_exp2 are
// polynomial approximations (fmath.h:1830 and :1680, "2.7 ulp average, 232 ulp
// max").  An exact __builtin_logf here differs from the CPU's fast_log by up
// to ~1e-5 relative, which is enough to send the reflected ray of a near
// specular lobe through a different environment texel -- the pixels where the
// two renderers disagree are exactly those.  These three helpers are the same
// polynomials as fmath.h, so the sampled micronormal matches the CPU bit for
// bit; `#pragma clang fp contract(off)` keeps clang from fusing the madd pair
// into an fma, which the CPU build's mul+add does not do.
inline unsigned int mf_bitcast_u32(float f)
{
    return __builtin_bit_cast(unsigned int, f);
}

inline float mf_bitcast_f32(unsigned int u)
{
    return __builtin_bit_cast(float, u);
}

inline float mf_fast_log2(float xval)
{
#pragma clang fp contract(off)
    // clamp to avoid special cases, like fmath.h:1830
    const float FLT_MIN_ = 1.17549435e-38f;
    const float FLT_MAX_ = 3.40282347e+38f;
    float x              = xval < FLT_MAX_ ? xval : FLT_MAX_;
    x                    = x > FLT_MIN_ ? x : FLT_MIN_;
    const unsigned int bits = mf_bitcast_u32(x);
    const int exponent      = (int)(bits >> 23) - 127;
    const float f
        = mf_bitcast_f32((bits & 0x007FFFFFu) | 0x3f800000u) - 1.0f;
    const float f2 = f * f;
    const float f4 = f2 * f2;
    float hi       = f * -0.00931049621349f + 0.05206469089414f;
    float lo       = f * 0.47868480909345f + -0.72116591947498f;
    hi             = f * hi + -0.13753123777116f;
    hi             = f * hi + 0.24187369696082f;
    hi             = f * hi + -0.34730547155299f;
    lo             = f * lo + 1.442689881667200f;
    return ((f4 * hi) + (f * lo)) + (float)exponent;
}

inline float mf_fast_log(float x)
{
#pragma clang fp contract(off)
    return mf_fast_log2(x) * 0.6931471805599453f;  // M_LN2
}

/// fmath.h:1680 -- fast_exp2(float): range reduction, a 5th degree polynomial,
/// then the exponent is added back through the bit pattern.
inline float mf_fast_exp2(float xval)
{
#pragma clang fp contract(off)
    float x = xval < 126.0f ? xval : 126.0f;
    x       = x > -126.0f ? x : -126.0f;
    const int m = (int)x;
    x -= (float)m;
    x = 1.0f - (1.0f - x);  // crush denormals
    float r = 1.33336498402e-3f;
    r       = x * r + 9.810352697968e-3f;
    r       = x * r + 5.551834031939e-2f;
    r       = x * r + 0.2401793301105f;
    r       = x * r + 0.693144857883f;
    r       = x * r + 1.0f;
    return mf_bitcast_f32(mf_bitcast_u32(r) + ((unsigned int)m << 23));
}

inline float mf_fast_exp(float x)
{
#pragma clang fp contract(off)
    return mf_fast_exp2(x * 1.4426950408889634f);  // 1 / M_LN2
}

inline float mf_fast_erf(float x)
{
    // OIIO/fmath.h:2153 -- Abramowitz & Stegun 7.1.28.
    const float a1 = 0.0705230784f;
    const float a2 = 0.0422820123f;
    const float a3 = 0.0092705272f;
    const float a4 = 0.0001520143f;
    const float a5 = 0.0002765672f;
    const float a6 = 0.0000430638f;
    const float a  = __builtin_fabsf(x);
    const float b  = 1.0f - (1.0f - a);  // crush denormals
    const float r  = ((((((a6 * b + a5) * b + a4) * b + a3) * b + a2) * b + a1)
                      * b)
                     + 1.0f;
    const float s  = r * r;
    const float t  = s * s;
    const float u  = t * t;
    const float v  = u * u;
    return __builtin_copysignf(1.0f - 1.0f / v, x);
}

inline float mf_fast_ierf(float x)
{
    // OIIO/fmath.h:2189 -- Mike Giles' erfinv approximation.
    float a = __builtin_fabsf(x);
    if (a > 0.99999994f)
        a = 0.99999994f;
    float w = -mf_fast_log((1.0f - a) * (1.0f + a)), p;
    if (w < 5.0f) {
        w = w - 2.5f;
        p = 2.81022636e-08f;
        p = p * w + 3.43273939e-07f;
        p = p * w + -3.5233877e-06f;
        p = p * w + -4.39150654e-06f;
        p = p * w + 0.00021858087f;
        p = p * w + -0.00125372503f;
        p = p * w + -0.00417768164f;
        p = p * w + 0.246640727f;
        p = p * w + 1.50140941f;
    } else {
        w = __builtin_sqrtf(w) - 3.0f;
        p = -0.000200214257f;
        p = p * w + 0.000100950558f;
        p = p * w + 0.00134934322f;
        p = p * w + -0.00367342844f;
        p = p * w + 0.00573950773f;
        p = p * w + -0.0076224613f;
        p = p * w + 0.00943887047f;
        p = p * w + 1.00167406f;
        p = p * w + 2.83297682f;
    }
    return p * x;
}

/// shading.cpp:507-565 -- the Beckmann distribution's three entry points.
inline float beckmann_F(float tan_m2)
{
    return 0.31830988618f * mf_fast_exp(-tan_m2);
}

inline float beckmann_Lambda(float a2)
{
    const float a = __builtin_sqrtf(a2);
    return a < 1.6f ? (1.0f - 1.259f * a + 0.396f * a2)
                          / (3.535f * a + 2.181f * a2)
                    : 0.0f;
}

/// Wenzel Jakob's improved visible-normal Beckmann slope sample; the single
/// Newton step is what the CPU does too (shading.cpp:538-558).
inline void beckmann_sample_slope(float cos_theta, float randu, float randv,
                                  float& sx, float& sy)
{
#pragma clang fp contract(off)
    const float SQRT_PI_INV = 0.5641895835f;  // 1/sqrt(pi)
    float ct                = cos_theta < 1e-6f ? 1e-6f : cos_theta;
    float tanThetaI         = __builtin_sqrtf(1.0f - ct * ct) / ct;
    float cotThetaI         = 1.0f / tanThetaI;

    float c       = mf_fast_erf(cotThetaI);
    float K       = tanThetaI * SQRT_PI_INV;
    float yApprox = randu * (1.0f + c + K * (1.0f - c * c));
    float yExact  = randu * (1.0f + c + K * mf_fast_exp(-cotThetaI * cotThetaI));
    float b = K > 0 ? (0.5f - __builtin_sqrtf(K * (K - yApprox + 1.0f) + 0.25f))
                              / K
                    : yApprox - 1.0f;

    float invErf = mf_fast_ierf(b);
    float value  = 1.0f + b + K * mf_fast_exp(-invErf * invErf) - yExact;

    if (__builtin_fabsf(value) > 1e-6f) {
        b -= value / (1.0f - invErf * tanThetaI);  // newton step 1
        invErf = mf_fast_ierf(b);
        value  = 1.0f + b + K * mf_fast_exp(-invErf * invErf) - yExact;
        b -= value / (1.0f - invErf * tanThetaI);  // newton step 2
        sx = mf_fast_ierf(b);
    } else {
        sx = invErf;
    }
    sy = mf_fast_ierf(2.0f * randv - 1.0f);
}

/// fmath.h:1610 -- fast_sincos(float).  The CPU's Background::map goes through
/// it, and with it the *direction* of every importance-sampled background ray.
/// A near-specular lobe's NEE weight is extremely sensitive to that direction
/// (even a ~1e-6 rad shift moves it across the Beckmann peak), so an exact
/// sinf/cosf here leaves a visible difference on glossy highlights.  Same
/// polynomials and the same SLEEF argument reduction as fmath.h.
inline void mf_fast_sincos(float x, float& sine, float& cosine)
{
#pragma clang fp contract(off)
    // fast_rint: SSE4.1 rint, i.e. round to nearest even
    const int q = (int)__builtin_rintf(x * 0.31830988618379067154f);  // M_1_PI
    const float qf = (float)q;
    x              = qf * (-0.78515625f * 4.0f) + x;
    x              = qf * (-0.00024187564849853515625f * 4.0f) + x;
    x              = qf * (-3.7747668102383613586e-08f * 4.0f) + x;
    x              = qf * (-1.2816720341285448015e-12f * 4.0f) + x;
    const float MP = 1.57079632679489661923f;  // M_PI_2
    x              = MP - (MP - x);            // crush denormals
    const float s  = x * x;
    if ((q & 1) != 0)
        x = -x;
    float su = 2.6083159809786593541503e-06f;
    su       = su * s + -0.0001981069071916863322258f;
    su       = su * s + 0.00833307858556509017944336f;
    su       = su * s + -0.166666597127914428710938f;
    su       = s * (su * x) + x;
    float cu = -2.71811842367242206819355e-07f;
    cu       = cu * s + 2.47990446951007470488548e-05f;
    cu       = cu * s + -0.00138888787478208541870117f;
    cu       = cu * s + 0.0416666641831398010253906f;
    cu       = cu * s + -0.5f;
    cu       = cu * s + 1.0f;
    if ((q & 1) != 0)
        cu = -cu;
    sine   = su < -1.0f ? -1.0f : (su > 1.0f ? 1.0f : su);
    cosine = cu < -1.0f ? -1.0f : (cu > 1.0f ? 1.0f : cu);
}

/// fmath.h:1775 -- fast_atan2(float, float), scalar variant (the build does not
/// define OIIO_FMATH_SIMD_FRIENDLY, which is what selects the argument-swap
/// branch of the reduction; the two differ in the last bit of the ratio).
///
/// The CPU's Background::eval maps every escaped ray to the importance table
/// through this function.  A libm atan2f differs from it in the last bits, and
/// `u * res` then moves the texel index across a table boundary for a small
/// fraction of lookups, which swaps one environment radiance for a different
/// one -- the "fireflies relocate" effect the reference scene's header warns
/// about.  Same polynomial, same operations, same order.
inline float mf_fast_atan2(float y, float x)
{
#pragma clang fp contract(off)
    const float a               = __builtin_fabsf(x);
    const float b               = __builtin_fabsf(y);
    const bool b_is_gt_a        = b > a;
    const float k  = (b == 0.0f) ? 0.0f
                     : ((a == b) ? 1.0f : (b_is_gt_a ? a / b : b / a));
    const float s  = 1.0f - (1.0f - k);  // crush denormals
    const float t  = s * s;
    const float t1 = 0.430165678f * t + 1.0f;
    const float t2 = (0.0579354987f * t + 0.763007998f) * t + 1.0f;
    float r        = s * t1 / t2;
    if (b_is_gt_a)
        r = 1.570796326794896557998982f - r;  // account for arg reduction
    if (mf_bitcast_u32(x) & 0x80000000u)      // test the sign bit of x
        r = 3.14159265358979323846f - r;
    return __builtin_copysignf(r, y);
}

/// A flattened closure component: the branch weight already folded in (MUL
/// multiplies it, ADD splits it, a component is a leaf -- the same walk
/// testrender's process_bsdf_closure does at shading.cpp:1480-1690).
///
/// `N`/`U`/`eta`/`xalpha`/`yalpha`/`refract` are the `MicrofacetParams` fields
/// (`{ ustringhash dist; Vec3 N; Vec3 U; float xalpha; float yalpha;
/// float eta; int refract; }`, read on demand out of the closure pool), so the
/// microfacet lobes below take a component directly instead of a converted
/// copy: the conversion was pure overhead when inlined at every call site.
struct ClosureEntryVK {
    int id;
    Vec3 w;      ///< component weight, MUL folds included
    Vec3 N;      ///< reflection / refraction plane normal
    float eta;   ///< reflection / refraction index of refraction
    Vec3 U;      ///< microfacet tangent
    float xalpha, yalpha;  ///< microfacet roughness
    int refract;           ///< microfacet lobe: 0 reflection, 1 transmission
    float exponent;        ///< phong falloff exponent (PhongParams.exponent)
    float sigma;           ///< oren_nayar roughness (OrenNayarParams.sigma)
    Vec3 ior;              ///< conductor: real part of the complex IOR
    Vec3 ext;              ///< conductor: extinction (the imaginary part)
    Vec3 refl_tint;        ///< dielectric: reflection tint (Cr)
    Vec3 refr_tint;        ///< dielectric: transmission tint (Ct); non-black
                           ///< means the lobe also refracts, which this kernel
                           ///< does not model yet
    long long off;  ///< arena offset of the node, for parameters read on demand
};

inline float mf_Lambda(const ClosureEntryVK& p, const Vec3& w)
{
    float cosTheta2 = w.z * w.z;
    float cp2       = (w.x * p.xalpha) * (w.x * p.xalpha);
    float sp2       = (w.y * p.yalpha) * (w.y * p.yalpha);
    return beckmann_Lambda(cosTheta2 / (cp2 + sp2));
}

inline float mf_G1(float lv) { return 1.0f / (lv + 1.0f); }
inline float mf_G2(float li, float lo) { return 1.0f / (li + lo + 1.0f); }

inline float mf_D(const ClosureEntryVK& p, const Vec3& h)
{
    float cosThetaM = h.z;
    if (cosThetaM > 0.0f) {
        float cp2        = (h.x / p.xalpha) * (h.x / p.xalpha);
        float sp2        = (h.y / p.yalpha) * (h.y / p.yalpha);
        float cosThetaM2 = cosThetaM * cosThetaM;
        float cosThetaM4 = cosThetaM2 * cosThetaM2;
        float tanThetaM2 = (cp2 + sp2) / cosThetaM2;
        return beckmann_F(tanThetaM2) / (p.xalpha * p.yalpha * cosThetaM4);
    }
    return 0.0f;
}

inline Vec3 mf_sample_micronormal(const ClosureEntryVK& p, const Vec3& wo,
                                  float randu, float randv)
{
    Vec3 swo = mk(wo.x * p.xalpha, wo.y * p.yalpha, wo.z);
    swo      = normalize(swo);

    float cos_theta = swo.z > 0.0f ? swo.z : 0.0f;
    float cos_phi   = 1.0f;
    float sin_phi   = 0.0f;
    if (cos_theta < 0.99999f) {
        float invnorm = 1.0f
                        / __builtin_sqrtf(swo.x * swo.x + swo.y * swo.y);
        cos_phi = swo.x * invnorm;
        sin_phi = swo.y * invnorm;
    }

    float slope_x, slope_y;
    beckmann_sample_slope(cos_theta, randu, randv, slope_x, slope_y);

    float rx = cos_phi * slope_x - sin_phi * slope_y;
    float ry = sin_phi * slope_x + cos_phi * slope_y;
    rx *= p.xalpha;
    ry *= p.yalpha;

    float mlen = __builtin_sqrtf(rx * rx + ry * ry + 1.0f);
    return mk(-rx / mlen, -ry / mlen, 1.0f / mlen);
}

/// Microfacet::get_albedo (shading.cpp:576).  `refract==2` carries both lobes,
/// so its albedo is the whole unit; otherwise it is the Fresnel term of the
/// lobe: reflection gets fr, transmission 1-fr.
inline float mf_albedo(const ClosureEntryVK& p, const Vec3& wo)
{
    if (p.refract == 2)
        return 1.0f;
    float fr = fresnel_dielectric_vk(dot(p.N, wo), p.eta);
    return p.refract ? 1.0f - fr : fr;
}

/// Microfacet::eval (shading.cpp:585) -- the (weight, pdf) the NEE rays want.
///
/// The two lobes share everything except the half-vector and the Fresnel
/// orientation, and they are mutually exclusive (reflection wants `wil.z > 0`,
/// transmission `wil.z < 0`), so they are evaluated as one lobe body with the
/// half-vector and the Fresnel term selected, not as two branches that each
/// carry their own Beckmann evaluation.  clspv copies a branch body once per
/// call site, so the duplicated lobes were the bulk of the microfacet kernel.
inline void mf_eval_vk(const ClosureEntryVK& p, const Vec3& wo, const Vec3& wi,
                       float& w, float& pdf)
{
    w   = 0.0f;
    pdf = 0.0f;
    Vec3 u, v, nn;
    frame_from_nt(p.N, p.U, u, v, nn);
    Vec3 wol = mk(dot(u, wo), dot(v, wo), dot(nn, wo));
    Vec3 wil = mk(dot(u, wi), dot(v, wi), dot(nn, wi));

    // Reflection is possible for the mirror lobe and inside the combined one,
    // transmission for the transmitting lobe and inside the combined one.
    bool rf = (p.refract != 1) && (wol.z > 0.0f) && (wil.z > 0.0f);
    bool tr = (p.refract != 0) && (wol.z > 0.0f) && (wil.z < 0.0f);
    if (!rf && !tr)
        return;

    // Reflection half-vector, and the refraction one (eq. 16).  `eta < 1`
    // flips ht so its z stays positive, exactly like the CPU.
    Vec3 hrefl = normalize(add(wil, wol));
    Vec3 htraw = sub(mk(0.0f, 0.0f, 0.0f), add(mul(wil, p.eta), wol));
    if (p.eta < 1.0f)
        htraw = mul(htraw, -1.0f);
    Vec3 h = tr ? normalize(htraw) : hrefl;

    float cosHO = dot(h, wol);
    float cosHI = dot(h, wil);
    float Fr    = fresnel_dielectric_vk(cosHO, p.eta);
    float D     = mf_D(p, h);
    float Lo    = mf_Lambda(p, wol);
    float Li    = mf_Lambda(p, wil);
    float G2    = mf_G2(Li, Lo);
    float G1    = mf_G1(Lo);
    float out   = G2 / G1;
    float use   = tr ? (1.0f - Fr) : Fr;

    float invHt2 = 1.0f / dot(htraw, htraw);
    float pdfr   = (G1 * D * 0.25f) / wol.z;
    float pdft   = __builtin_fabsf(cosHI * cosHO) * (p.eta * p.eta) * (G1 * D)
          * invHt2 / wol.z;
    pdf = tr ? pdft : pdfr;
    // `refract == 2` carries both lobes and puts the Fresnel split in the pdf,
    // so the weight is the bare G2/G1; a single lobe keeps it in the weight.
    w = (p.refract == 2) ? out : out * use;
    if (p.refract == 2)
        pdf = pdf * use;

    // Transmission is only a lobe when the refraction half-vector faces out and
    // there is transmitted energy left (total internal reflection otherwise).
    if (tr && !(use > 0.0f && h.z > 0.0f)) {
        w   = 0.0f;
        pdf = 0.0f;
    }
}

/// Microfacet::sample (shading.cpp:652).  `refract==2` picks the lobe with the
/// third random number, weighted by the Fresnel term, exactly like the CPU.
/// Both lobes are produced unconditionally and one is selected: a branch here
/// means two complete Beckmann bodies per call site once clspv inlines.
inline void mf_sample_vk(const ClosureEntryVK& p, const Vec3& wo, float rx,
                         float ry, float rz, Vec3& wi, float& w, float& pdf)
{
    wi  = mk(0.0f, 0.0f, 0.0f);
    w   = 0.0f;
    pdf = 0.0f;
    Vec3 u, v, nn;
    frame_from_nt(p.N, p.U, u, v, nn);
    Vec3 wol    = mk(dot(u, wo), dot(v, wo), dot(nn, wo));
    float cosNO = wol.z;
    if (!(cosNO > 0.0f))
        return;

    Vec3 m      = mf_sample_micronormal(p, wol, rx, ry);
    float cosMO = dot(m, wol);
    float F     = fresnel_dielectric_vk(cosMO, p.eta);

    bool rf = (p.refract == 0) || (p.refract == 2 && rz < F);

    // Reflection direction and refraction direction, both from the same
    // micronormal; the CPU picks one and only the weight/pdf differ afterwards.
    Vec3 wilr     = sub(mul(m, 2.0f * cosMO), wol);
    Vec3 M        = add(add(mul(u, m.x), mul(v, m.y)), mul(nn, m.z));
    Vec3 wdir;
    float Ft      = fresnel_refraction_vk(mul(wo, -1.0f), M, p.eta, wdir);
    Vec3 wilt     = mk(dot(u, wdir), dot(v, wdir), dot(nn, wdir));

    Vec3 wil  = rf ? wilr : wilt;
    wi        = rf ? add(add(mul(u, wil.x), mul(v, wil.y)), mul(nn, wil.z))
                   : wdir;
    float cosHO = cosMO;
    float cosHI = dot(m, wil);

    float D   = mf_D(p, m);
    float Lo  = mf_Lambda(p, wol);
    float Li  = mf_Lambda(p, wil);
    float G2  = mf_G2(Li, Lo);
    float G1  = mf_G1(Lo);
    float out = G2 / G1;

    Vec3 ht      = sub(mk(0.0f, 0.0f, 0.0f), add(mul(wil, p.eta), wol));
    float invHt2 = 1.0f / dot(ht, ht);
    float pdfr   = (G1 * D * 0.25f) / cosNO;
    float pdft   = __builtin_fabsf(cosHI * cosHO) * (p.eta * p.eta) * (G1 * D)
          * invHt2 / __builtin_fabsf(wol.z);
    pdf = rf ? pdfr : pdft;
    float fru = rf ? F : Ft;
    if (p.refract == 2) {
        pdf = pdf * fru;
        w   = out;
    } else {
        w = fru * out;
    }
}

/// The Phong specular lobe (testrender/shading.cpp:320-361), device side:
/// the same (weight, pdf) pair the CPU produces.  `PhongParams` is
/// `{ Vec3 N; float exponent }`, the lobe is built around the mirror
/// direction, its weight is `N.wi * (exponent+2)/(exponent+1)` and its
/// solid-angle pdf the cosine-power falloff
/// `(exponent+1)/(2*pi) * cosRI^exponent`.
///
/// `OIIO::fast_safe_pow` on the domain this lobe uses.  It is not libm: fmath.h
/// answers `sign * fast_exp2(y*fast_log2(|x|))`, and the microfacet section
/// above already carries those two polynomials (they are what makes the CPU
/// and the device pick the same texel out of a near-specular lobe).  Reusing
/// them keeps the exponent-10000 plate bit-identical instead of a few hundred
/// ulp off, which at that exponent is a visibly different falloff.
inline float phong_pow_vk(float x, float y)
{
#pragma clang fp contract(off)
    if (y == 0.0f)
        return 1.0f;  // x^0 = 1
    if (x == 0.0f)
        return 0.0f;  // 0^y = 0
    if (y == 1.0f)
        return x;
    if (y == 2.0f)
        return x * x;  // fmath.h clamps here too; not reachable in this scene
    return mf_fast_exp2(y * mf_fast_log2(__builtin_fabsf(x)));
}

inline void phong_eval_vk(const ClosureEntryVK& p, const Vec3& wo,
                          const Vec3& wi, float& weight, float& pdf)
{
#pragma clang fp contract(off)
    weight      = 0.0f;
    pdf         = 0.0f;
    float cosNI = dot(p.N, wi);
    float cosNO = dot(p.N, wo);
    if (cosNI > 0.0f && cosNO > 0.0f) {
        // reflect the view vector
        Vec3 R      = sub(mul(p.N, 2.0f * cosNO), wo);
        float cosRI = dot(R, wi);
        if (cosRI > 0.0f) {
            // (exponent+1) * (1/(2*pi)) * cosRI^exponent
            pdf    = (p.exponent + 1.0f) * 0.15915494f
                     * phong_pow_vk(cosRI, p.exponent);
            weight = cosNI * (p.exponent + 2.0f) / (p.exponent + 1.0f);
        }
    }
}

inline void phong_sample_vk(const ClosureEntryVK& p, const Vec3& wo, float rx,
                            float ry, Vec3& wi, float& weight, float& pdf)
{
#pragma clang fp contract(off)
    wi     = mk(0.0f, 0.0f, 0.0f);
    weight = 0.0f;
    pdf    = 0.0f;
    float cosNO = dot(p.N, wo);
    if (cosNO > 0.0f) {
        Vec3 R    = sub(mul(p.N, 2.0f * cosNO), wo);
        float phi = 6.2831855f * rx;  // 2*pi
        float sp, cp;
        fast_sincos_vk(phi, sp, cp);
        // cosTheta = ry^(1/(exponent+1)), drawn in the frame around R.
        float cosTheta  = phong_pow_vk(ry, 1.0f / (p.exponent + 1.0f));
        float sinTheta2 = 1.0f - cosTheta * cosTheta;
        float sinTheta = sinTheta2 > 0.0f ? __builtin_sqrtf(sinTheta2) : 0.0f;
        Vec3 u, v;
        pixar_frame(R, u, v);
        wi = add(add(mul(u, cp * sinTheta), mul(v, sp * sinTheta)),
                 mul(R, cosTheta));
        phong_eval_vk(p, wo, wi, weight, pdf);
    }
}

/// `BsdfGlobals::visible_normal` (BSDL/bsdf_impl.h:48-78), device side: the
/// shading normal, or -- if it faces away from the viewer -- a normal in the
/// plane of `wo` and `N` that does face it.  BSDL's lobes are built on this, so
/// the Oren-Nayar lobe has to use it too; a plain flip would put the lobe on a
/// different axis than the CPU's at grazing angles on a curved surface.
///
/// The degenerate case (N parallel to wo) is the one place this differs from
/// the CPU: it falls back to the flip, since the CPU's own fallback needs the
/// geometric normal, which the closure pool does not carry.
inline Vec3 visible_normal_vk (const Vec3& wo, const Vec3& N)
{
    if (dot (wo, N) > 0.0f)
        return N;
    Vec3 V   = cross (wo, N);
    float mx = __builtin_fabsf (V.x), my = __builtin_fabsf (V.y),
          mz = __builtin_fabsf (V.z);
    float m = mx > my ? mx : my;
    m       = m > mz ? m : mz;
    if (m < 1e-4f)
        return mul (N, -1.0f);
    V = normalize (V);
    return normalize (add (cross (V, wo), mul (wo, 1e-4f)));
}

/// The Oren-Nayar diffuse lobe (`oren_nayar(N, sigma)`), device side.
///
/// testrender has no implementation of this closure of its own: shading.cpp:
/// 1498-1503 converts it into BSDL's `oren_nayar_diffuse_bsdf` with the lobe
/// albedo pinned to white (the closure weight carries the colour) and
/// `energy_compensation` off.  That leaves the "tiny improvement of Oren-Nayar"
/// QON model, which is the classic model written in a numerically nicer form
/// (bsdf_oren_nayar_diffuse_impl.h:56-69):
///
///     s = cosIO - cosNI*cosNO
///     A = 1 - 0.5*sigma^2/(sigma^2+0.33)     B = 0.45*sigma^2/(sigma^2+0.09)
///     f = (A + B * max(s,0)/max(cosNI,cosNO)) / pi
///
/// The lobe is sampled with the cosine hemisphere, so its direction pdf stays
/// `cosNI/pi` and the (weight, pdf) pair it reports has the same shape as
/// Lambert's -- `weight` is `f*pi`, i.e. dimensionless, and the albedo arrives
/// with the closure weight.  `sigma` is not clamped by the CPU either (it only
/// clamps `MxOrenNayarDiffuse::Data::roughness` when called through the
/// MaterialX closure, which is a different path), so a sigma outside [0,1]
/// keeps the same formula on both sides.
inline void on_eval_vk (const ClosureEntryVK& p, const Vec3& wo, const Vec3& wi,
                        float& weight, float& pdf)
{
#pragma clang fp contract(off)
    weight      = 0.0f;
    pdf         = 0.0f;
    Vec3 N      = visible_normal_vk (wo, p.N);
    float cosNI = dot (N, wi);
    float cosNO = dot (N, wo);
    if (cosNI > 1.0f)
        cosNI = 1.0f;
    if (cosNO > 1.0f)
        cosNO = 1.0f;
    if (!(cosNI > 0.0f) || !(cosNO > 0.0f))
        return;
    float cosIO = dot (wo, wi);
    if (cosIO > 1.0f)
        cosIO = 1.0f;
    if (cosIO < -1.0f)
        cosIO = -1.0f;
    const float s = cosIO - cosNI * cosNO;
    pdf           = cosNI * 0.31830988618f;
    const float s2 = p.sigma * p.sigma;
    const float A  = 1.0f - 0.5f * s2 / (s2 + 0.33f);
    const float B  = 0.45f * s2 / (s2 + 0.09f);
    float stinv    = 0.0f;
    if (s > 0.0f) {
        const float mx = cosNI > cosNO ? cosNI : cosNO;
        stinv          = s / mx;
    }
    weight = A + B * stinv;
}

/// BSDL's cosine-hemisphere mapper (tools.h:261-278), device side: Clarberg's
/// quadrant mapping with the two minimax polynomials.  It is *not* the same
/// function as testrender's `Sampling::sample_cosine_hemisphere` (sampling.h:
/// 93-101): for a given (randu, randv) the two land on different directions
/// (the CPU's own diffuse lobe uses testrender's, BSDL's lobes use this one),
/// so an Oren-Nayar path sampled with the wrong one diverges from the CPU on
/// every bounce and shows up as speckle across the whole sphere even though
/// both BRDFs agree exactly.
inline float fast_cos_quadrant_vk (float x)
{
    // minimax approximation of cos(x*pi/4) on [-2,2]
    const float c1 = 0.99998736f, c2 = -0.30837047f, c3 = 0.01578646f,
                c4 = -0.00029826362f;
    float x2 = x * x;
    float cp = c3 + c4 * x2;
    cp       = c2 + cp * x2;
    return c1 + cp * x2;
}

inline float fast_sin_quadrant_vk (float x)
{
    // minimax approximation of sin(x*pi/4) on [-2,2]
    const float s1 = 0.7853975892066955566406250000000000f,
                s2 = -0.0807407423853874206542968750000000f,
                s3 = 0.0024843954015523195266723632812500f,
                s4 = -0.0000341485538228880614042282104492f;
    float x2 = x * x;
    float sp = s3 + s4 * x2;
    sp       = s2 + sp * x2;
    sp       = s1 + sp * x2;
    return sp * x;
}

inline Vec3 cos_hemisphere_local_vk (float randu, float randv)
{
    // stretch unit square + get quadrant
    const float a = 2.0f * randu - 1.0f, qa = __builtin_fabsf (a);
    const float b = 2.0f * randv - 1.0f, qb = __builtin_fabsf (b);
    // map to radius / angle
    const float rad = qa > qb ? qa : qb;
    const float phi = qa > qb ? (qb / qa)
                              : ((qa == qb) ? 1.0f : 2.0f - qa / qb);
    // map to the disk, then flip back into the original quadrant
    const float x = __builtin_copysignf (rad * fast_cos_quadrant_vk (phi), a);
    const float y = __builtin_copysignf (rad * fast_sin_quadrant_vk (phi), b);
    // cosine-weighted hemisphere
    float cz      = 1.0f - rad * rad;
    cz            = cz > 0.0f ? __builtin_sqrtf (cz) : 0.0f;
    return mk (x, y, cz);
}

/// `OrenNayarDiffuseLobe::sample_impl`: the cosine hemisphere about the lobe's
/// visible normal, then `eval` at the drawn direction -- which is why the
/// sampled weight carries `f` and not just the albedo.  Same shape as
/// `phong_sample_vk`, so the caller can reuse the Ward/Phong pipeline.
inline void on_sample_vk (const ClosureEntryVK& p, const Vec3& wo, float rx,
                          float ry, Vec3& wi, float& weight, float& pdf)
{
    Vec3 N = visible_normal_vk (wo, p.N);
    Vec3 u, v;
    pixar_frame (N, u, v);
    Vec3 l = cos_hemisphere_local_vk (rx, ry);
    wi     = add (add (mul (u, l.x), mul (v, l.y)), mul (N, l.z));
    on_eval_vk (p, wo, wi, weight, pdf);
}

// ---------------------------------------------------------------------------
// MaterialX conductor (`conductor_bsdf`), device side
// ---------------------------------------------------------------------------
//
// testrender turns this closure into BSDL's `mtx::ConductorLobe`
// (shading.cpp:105): a GGX microfacet distribution whose Fresnel term comes
// from the *complex* refractive index (`ConductorFresnel`, bsdf_conductor_impl.h
// :19-76), evaluated and sampled with Turquin's multiple-scattering
// compensation (microfacet_tools_impl.h:341-391).
//
// Nothing of that is Beckmann, so it cannot ride on the `mf_*` path above: the
// distribution, the Fresnel term and the energy compensation all differ.  The
// lobe's own weight carries the Fresnel factor (unlike Oren-Nayar, where the
// weight is a scalar), which is why the albedo handed to `closure_entry_albedo`
// stays the plain closure weight and the colour arrives with the sampled
// weight.
static const int MX_CONDUCTOR_ID_VK = 18;
/// `dielectric_bsdf`'s closure id (shading.h:46, MX_DIELECTRIC_ID).  Only the
/// reflection-only flavor is modelled on the device: with a non-black
/// transmission tint the lobe also refracts, and a transmitted ray leaves the
/// surface side the wavefront sampler assumes, so those components still fall
/// back to what they did before (see the sample chain's last branch).
static const int MX_DIELECTRIC_ID_VK = 17;

/// `layer`'s closure id (shading.h:47, MX_LAYER_ID).  A layer is not a lobe of
/// its own: `layer(top, base)` puts `base` *under* `top`, with a weight of
/// `1 - opacity(top)`.  The host walker descends into both halves
/// (testrender/shading.cpp:1314-1325) and so must this one -- a layer left as a
/// leaf hides every closure inside it behind an id no family implements.
static const int MX_LAYER_ID_VK = 27;

/// `GGXDist` (microfacet_tools_decl.h:13-47): the anisotropic GGX slope
/// distribution, built from MaterialX's raw x/y roughness the way
/// `ConductorLobe` builds it (bsdf_conductor_impl.h:89-107) -- re-derive the
/// roughness/anisotropy parameterisation, regularise the roughness against the
/// path roughness, then square it into the two alphas.
struct GgxVK {
    float ax, ay;
    float roughness;  ///< the regularised roughness the energy table is indexed by
};

inline GgxVK ggx_make (float rx_in, float ry_in, float path_roughness)
{
#pragma clang fp contract(off)
    const float rx = vk_clampf (rx_in, 1e-4f, 2.0f);  // BSDL::EPSILON
    const float ry = vk_clampf (ry_in, 1e-4f, 2.0f);
    const float a1 = rx > ry ? rx : ry;
    const float a2 = rx > ry ? ry : rx;
    const float b     = a2 / a1;
    const float aniso = (1.0f - b) / (1.0f + b);
    // BsdfGlobals::regularize_roughness (bsdf_impl.h:143): the path roughness
    // widens the lobe of a ray that already travelled through a rough bounce.
    float rough = __builtin_sqrtf (a1 / (1.0f + aniso));
    rough       = 1.0f - (1.0f - rough) * (1.0f - path_roughness);
    const float an = rx < ry ? -aniso : aniso;
    const float r2 = rough * rough;
    GgxVK d;
    d.ax        = bmaxf (r2 * (1.0f + an), 1e-5f);  // GGXDist's ALPHA_MIN
    d.ay        = bmaxf (r2 * (1.0f - an), 1e-5f);
    d.roughness = rough;
    return d;
}

inline float ggx_D_vk (const GgxVK& d, const Vec3& h)
{
#pragma clang fp contract(off)
    const float cx  = h.x / d.ax;
    const float cy  = h.y / d.ay;
    const float c2  = cx * cx + cy * cy;
    const float cz2 = h.z * h.z;
    const float t   = cz2 + c2;
    return 1.0f / (3.14159274f * d.ax * d.ay * t * t);
}

inline float ggx_G1_vk (const GgxVK& d, Vec3 w)
{
#pragma clang fp contract(off)
    w = mk (w.x * d.ax, w.y * d.ay, w.z);
    return 2.0f * w.z / (w.z + length (w));
}

inline float ggx_G2_G1_vk (const GgxVK& d, Vec3 wi, Vec3 wo)
{
#pragma clang fp contract(off)
    wi            = mk (wi.x * d.ax, wi.y * d.ay, wi.z);
    wo            = mk (wo.x * d.ax, wo.y * d.ay, wo.z);
    const float nl = length (wi);
    const float nv = length (wo);
    return wi.z * (wo.z + nv) / (wo.z * nl + wi.z * nv);
}

/// `GGXDist::D_refl_D` -- the bounded-VNDF factor of Eto & Tokuyoshi, which is
/// also the pdf of `sample_for_refl` (microfacet_tools_impl.h:97-111).
inline float ggx_D_refl_D_vk (const GgxVK& d, const Vec3& wo, const Vec3& m)
{
#pragma clang fp contract(off)
    (void)m;
    const float axl = wo.x * d.ax, ayl = wo.y * d.ay;
    const float len2 = axl * axl + ayl * ayl;
    const float t    = __builtin_sqrtf (len2 + wo.z * wo.z);
    const float a    = vk_clampf (d.ax < d.ay ? d.ax : d.ay, 0.0f, 1.0f);
    const float s    = 1.0f + __builtin_sqrtf (wo.x * wo.x + wo.y * wo.y);
    const float a2 = a * a, s2 = s * s;
    const float k = (1.0f - a2) * s2 / (s2 + a2 * wo.z * wo.z);
    return 2.0f * wo.z / (k * wo.z + t);
}

inline Vec3 ggx_sample_for_refl_vk (const GgxVK& d, const Vec3& wo, float randu,
                                    float randv)
{
#pragma clang fp contract(off)
    const Vec3 i_std = normalize (mk (wo.x * d.ax, wo.y * d.ay, wo.z));
    const float phi  = 6.28318548f * randu;
    const float a    = vk_clampf (d.ax < d.ay ? d.ax : d.ay, 0.0f, 1.0f);
    const float s    = 1.0f + __builtin_sqrtf (wo.x * wo.x + wo.y * wo.y);
    const float a2 = a * a, s2 = s * s;
    const float k = (1.0f - a2) * s2 / (s2 + a2 * wo.z * wo.z);
    const float b = k * i_std.z;
    const float z = (1.0f - randv) * (1.0f + b) - b;
    const float sinTheta
        = __builtin_sqrtf (vk_clampf (1.0f - z * z, 0.0f, 1.0f));
    // BSDLConfig::Fast::cosf/sinf are std::cos/std::sin (double) on the CPU;
    // the float forms differ by ~1 ulp, which only matters as a mip-level
    // wobble in the reflected environment lookup.
    const Vec3 o_std = mk (sinTheta * __builtin_cosf (phi),
                           sinTheta * __builtin_sinf (phi), z);
    const Vec3 m_std = add (i_std, o_std);
    return normalize (mk (m_std.x * d.ax, m_std.y * d.ay, m_std.z));
}

/// `ConductorFresnel::eval` (bsdf_conductor_impl.h:19-42), one channel: the
/// Fresnel term of a conductor with complex index `n - i*k`, for the angle
/// between the view direction and the microfacet normal.
inline float conductor_fresnel_ch (float cos_theta, float n, float k)
{
#pragma clang fp contract(off)
    const float FLT_MIN_ = 1.17549435e-38f;
    cos_theta            = vk_clampf (cos_theta, 0.0f, 1.0f);
    const float ct2 = cos_theta * cos_theta;
    const float st2 = 1.0f - ct2;
    const float n2 = n * n, k2 = k * k;
    const float t0     = n2 - k2 - st2;
    const float a2b2   = __builtin_sqrtf (t0 * t0 + 4.0f * n2 * k2);
    const float t1     = a2b2 + ct2;
    const float a      = __builtin_sqrtf (0.5f * (a2b2 + t0));
    const float t2     = (2.0f * cos_theta) * a;
    float den          = t1 + t2;
    den                = den > FLT_MIN_ ? den : FLT_MIN_;
    const float rs     = (t1 - t2) / den;
    const float t3     = ct2 * a2b2 + st2 * st2;
    const float t4     = t2 * st2;
    float den2         = t3 + t4;
    den2               = den2 > FLT_MIN_ ? den2 : FLT_MIN_;
    const float rp     = rs * (t3 - t4) / den2;
    return vk_clampf (0.5f * (rp + rs), 0.0f, 2.0f);
}

inline Vec3 conductor_fresnel_vk (float cos_theta, const Vec3& ior,
                                  const Vec3& ext)
{
    return mk (conductor_fresnel_ch (cos_theta, ior.x, ext.x),
               conductor_fresnel_ch (cos_theta, ior.y, ext.y),
               conductor_fresnel_ch (cos_theta, ior.z, ext.z));
}

/// `ConductorFresnel::avg` (bsdf_conductor_impl.h:62-76): the cosine-weighted
/// average Fresnel, a two-variable fit of (ior, extinction).
inline Vec3 conductor_avg_vk (const Vec3& ior, const Vec3& ext)
{
#pragma clang fp contract(off)
    const float a = -0.32775145f, b = 0.18346033f, c = 0.61146583f,
                dd = -0.07785134f;
    float r[3];
    r[0] = a + b * ior.x + c * ext.x + dd * ior.x * ext.x;
    r[1] = a + b * ior.y + c * ext.y + dd * ior.y * ext.y;
    r[2] = a + b * ior.z + c * ext.z + dd * ior.z * ext.z;
    return mk (vk_clampf (r[0] / (1.0f + r[0]), 0.0f, 1.0f),
               vk_clampf (r[1] / (1.0f + r[1]), 0.0f, 1.0f),
               vk_clampf (r[2] / (1.0f + r[2]), 0.0f, 1.0f));
}

/// `TabulatedEnergyCurve<MiniMicrofacetGGX>::Emiss_eval` (microfacet_tools_impl.h
/// :113-210): linear in roughness between the 16 rows, and linear in cosine
/// over the table's own non-uniform cosine spacing (`get_cosine(i)` = (i/15)^2,
/// floored at 1e-6).
inline float mx_emiss_vk (VK_GLOBAL const FrameParams* fp, float roughness, float c)
{
#pragma clang fp contract(off)
    const int Nc = 16;
    float rf     = roughness * 15.0f;
    int ra       = (int)rf;
    ra           = ra < 0 ? 0 : (ra > 15 ? 15 : ra);
    int rb       = ra + 1 < 15 ? ra + 1 : 15;
    rf -= (float)ra;
    if (c < 0.0f)
        c = 0.0f;
    float cos0 = 1e-6f;
    if (c <= cos0)
        return fp->mx_energy[ra * Nc] * (1.0f - rf)
               + fp->mx_energy[rb * Nc] * rf;
    for (int i = 1; i < Nc; ++i) {
        const float q   = (float)i * (1.0f / 15.0f);
        const float cos1 = bmaxf (q * q, 1e-6f);
        if (c < cos1) {
            const float t = (c - cos0) / (cos1 - cos0);
            const float ea = fp->mx_energy[ra * Nc + i - 1];
            const float eb = fp->mx_energy[rb * Nc + i - 1];
            const float ec = fp->mx_energy[ra * Nc + i];
            const float ed = fp->mx_energy[rb * Nc + i];
            return (ea + (ec - ea) * t) + ((eb + (ed - eb) * t)
                                          - (ea + (ec - ea) * t)) * rf;
        }
        cos0 = cos1;
    }
    return fp->mx_energy[ra * Nc + Nc - 1] * (1.0f - rf)
           + fp->mx_energy[rb * Nc + Nc - 1] * rf;
}

/// BSDL's `Frame(Z, X)` (tools.h): a right-handed basis with Z along the lobe
/// normal and X as close to the given tangent as orthogonality allows.  It is
/// *not* `pixar_frame`: the conductor is the one family whose data carries an
/// explicit tangent, and the CPU orients the GGX ellipse in that basis.
inline void mx_conductor_frame (const Vec3& N, const Vec3& U, Vec3& X, Vec3& Y)
{
    Vec3 nU = normalize (U);
    float mx = __builtin_fabsf (U.x), my = __builtin_fabsf (U.y),
          mz = __builtin_fabsf (U.z);
    float m = mx > my ? mx : my;
    m       = m > mz ? m : mz;
    if (m < 1e-4f || __builtin_fabsf (dot (N, nU)) > 0.999f) {
        // Frame::ortho_build (Duff et al., tools.h)
        const float s = __builtin_copysignf (1.0f, N.z);
        const float a = -1.0f / (s + N.z);
        const float b = N.x * N.y * a;
        X = mk (1.0f + s * N.x * N.x * a, s * b, -s * N.x);
        Y = mk (b, s + N.y * N.y * a, -N.y);
    } else {
        Y = normalize (cross (N, U));
        X = cross (Y, N);
    }
}

/// `eval_turquin_microms_reflection` (microfacet_tools_impl.h:341-372), the
/// multiple-scattering compensated GGX reflection lobe.
inline void mx_conductor_eval_vk (VK_GLOBAL const FrameParams* fp,
                                  const ClosureEntryVK& p, const Vec3& wo,
                                  const Vec3& wi, Vec3& weight, float& pdf)
{
#pragma clang fp contract(off)
    weight = mk (0.0f, 0.0f, 0.0f);
    pdf    = 0.0f;
    const Vec3 N = visible_normal_vk (wo, p.N);
    Vec3 X, Y;
    mx_conductor_frame (N, p.U, X, Y);
    const Vec3 woL = mk (dot (wo, X), dot (wo, Y), dot (wo, N));
    const Vec3 wiL = mk (dot (wi, X), dot (wi, Y), dot (wi, N));
    if (!(woL.z > 0.0f) || !(wiL.z > 0.0f))
        return;
    const GgxVK d = ggx_make (p.xalpha, p.yalpha, 0.0f);
    const Vec3 mh = normalize (add (woL, wiL));
    const float cosMO = dot (mh, woL);
    const float D     = ggx_D_vk (d, mh);
    const float DrD   = ggx_D_refl_D_vk (d, woL, mh);
    const float D_refl = DrD * D;
    const float G1     = ggx_G1_vk (d, woL);
    pdf                = D_refl / (4.0f * woL.z);
    const float out    = ggx_G2_G1_vk (d, wiL, woL) * G1 / DrD;
    const Vec3 F       = conductor_fresnel_vk (cosMO, p.ior, p.ext);
    // Turquin's "Practical multiple scattering compensation", eq. 16: the
    // single-scatter lobe plus the energy the microfacets re-scatter, scaled by
    // the tabulated multiple-scattering energy.
    const float E_ms = mx_emiss_vk (fp, d.roughness, woL.z);
    const float msf  = E_ms / bmaxf (0.01f, 1.0f - E_ms);
    weight = mul (mulv (F, add (mk (1.0f, 1.0f, 1.0f), mul (F, msf))), out);
}

/// `sample_turquin_microms_reflection`: draw a micro normal from the bounded
/// VNDF, reflect, then evaluate there -- which is why the sampled weight is the
/// full BRDF value rather than the albedo.
inline void mx_conductor_sample_vk (VK_GLOBAL const FrameParams* fp,
                                    const ClosureEntryVK& p, const Vec3& wo,
                                    float rx, float ry, Vec3& wi, Vec3& weight,
                                    float& pdf)
{
#pragma clang fp contract(off)
    weight = mk (0.0f, 0.0f, 0.0f);
    pdf    = 0.0f;
    wi     = mul (wo, -1.0f);
    const Vec3 N = visible_normal_vk (wo, p.N);
    Vec3 X, Y;
    mx_conductor_frame (N, p.U, X, Y);
    const Vec3 woL = mk (dot (wo, X), dot (wo, Y), dot (wo, N));
    if (!(woL.z > 0.0f))
        return;
    const GgxVK d = ggx_make (p.xalpha, p.yalpha, 0.0f);
    const Vec3 m  = ggx_sample_for_refl_vk (d, woL, rx, ry);
    if (!(dot (m, woL) > 0.0f))
        return;
    const Vec3 wiL = sub (mul (m, 2.0f * dot (m, woL)), woL);
    if (!(wiL.z > 0.0f))
        return;
    wi = add (add (mul (X, wiL.x), mul (Y, wiL.y)), mul (N, wiL.z));
    mx_conductor_eval_vk (fp, p, wo, wi, weight, pdf);
}

// ---------------------------------------------------------------------------
// MaterialX dielectric (`dielectric_bsdf`), device side, reflection-only
// ---------------------------------------------------------------------------
//
// testrender turns this closure into BSDL's `mtx::DielectricLobe`
// (shading.cpp:106).  With `Ct == 0` the lobe sets `dorefr = false` and reduces
// to a GGX microfacet reflection whose Fresnel term is the ordinary scalar
// dielectric one,
// compensated for multiple scattering with the *same* Turquin scheme the
// conductor uses (`eval_turquin_microms_reflection`,
// microfacet_tools_impl.h:341-372) driven by the *same* mini-GGX energy table
// (bsdf_dielectric_impl.h:74-78).  Only the Fresnel term and the tint differ.
//
// A non-black `Ct` would add refraction; those components keep the old
// behaviour (a cosine-hemisphere stand-in) rather than silently dropping the
// transmitted half.

/// `DielectricFresnel::DielectricFresnel` (bsdf_dielectric_impl.h:21-28): the
/// backside inverts first, then the IOR is clamped to [1/5, 1/1.001] or
/// [1.001, 5] on the way it faces.
inline float dielec_eta (float eta, int backside)
{
    if (backside)
        eta = 1.0f / eta;
    return eta >= 1.0f ? vk_clampf (eta, 1.001f, 5.0f)
                       : vk_clampf (eta, 1.0f / 5.0f, 1.0f / 1.001f);
}

/// `DielectricFresnel::eval` (bsdf_dielectric_impl.h:30-42): scalar, and `1.0`
/// past total internal reflection where the lobe has no refracted component.
inline float dielec_fresnel (float c, float eta)
{
#pragma clang fp contract(off)
    float g = (eta - 1.0f) * (eta + 1.0f) + c * c;
    if (g > 0.0f) {
        g           = __builtin_sqrtf (g);
        const float A = (g - c) / (g + c);
        const float B = (c * (g + c) - 1.0f) / (c * (g - c) + 1.0f);
        return 0.5f * A * A * (1.0f + B * B);
    }
    return 1.0f;
}


/// `DielectricFresnel::table_index()` (bsdf_dielectric_impl.h:53-64): the IOR
/// mapped onto [0,1] by its squared position in [IOR_MIN, IOR_MAX], with the
/// reciprocal taken on the side the eta faces away from.
inline float dielec_table_index (float eta)
{
#pragma clang fp contract(off)
    const float IOR_MIN = 1.001f, IOR_MAX = 5.0f;
    const float feta    = eta;
    const float seta = vk_clampf (feta < 1.0f ? 1.0f / feta : feta, IOR_MIN,
                                  IOR_MAX);
    const float x = (seta - IOR_MIN) * (1.0f / (IOR_MAX - IOR_MIN));
    return __builtin_sqrtf (vk_clampf (x, 0.0f, 1.0f));
}

/// One (fresnel row, roughness row) of the dielectric energy table, sampled at
/// cosine `c` over the table's own cosine spacing -- the same interpolation
/// `mx_emiss_vk` does, one row at a time.
inline float mx_diel_row_vk (VK_GLOBAL const FrameParams* fp, int frow,
                             int rrow, float c)
{
#pragma clang fp contract(off)
    const int Nc   = 16;
    const int base = (frow * 16 + rrow) * Nc;
    if (c <= 1e-6f)
        return fp->mx_dielec_energy[base];
    float cos0 = 1e-6f;
    for (int i = 1; i < Nc; ++i) {
        const float cos1 = bmaxf ((float)i * (1.0f / 15.0f), 1e-6f);
        if (c < cos1) {
            const float t = (c - cos0) / (cos1 - cos0);
            const float a = fp->mx_dielec_energy[base + i - 1];
            const float b = fp->mx_dielec_energy[base + i];
            return a + (b - a) * t;
        }
        cos0 = cos1;
    }
    return fp->mx_dielec_energy[base + Nc - 1];
}

/// `TabulatedEnergyCurve<DielectricReflFront>::Emiss_eval`
/// (microfacet_tools_impl.h:113-210) with the table's 32 fresnel rows and 16
/// roughness rows: bilinear in (fresnel index, roughness), then in cosine.
inline float mx_emiss_diel_vk (VK_GLOBAL const FrameParams* fp, float roughness,
                               float fresnel_index, float c)
{
#pragma clang fp contract(off)
    float rf = roughness * 15.0f;
    int ra   = (int)rf;
    ra       = ra < 0 ? 0 : (ra > 15 ? 15 : ra);
    int rb   = ra + 1 < 15 ? ra + 1 : 15;
    rf -= (float)ra;
    float ff = vk_clampf (fresnel_index, 0.0f, 1.0f) * 31.0f;
    int fa   = (int)ff;
    fa       = fa < 0 ? 0 : (fa > 31 ? 31 : fa);
    int fb   = fa + 1 < 31 ? fa + 1 : 31;
    ff -= (float)fa;
    const float e00 = mx_diel_row_vk (fp, fa, ra, c);
    const float e01 = mx_diel_row_vk (fp, fa, rb, c);
    const float e10 = mx_diel_row_vk (fp, fb, ra, c);
    const float e11 = mx_diel_row_vk (fp, fb, rb, c);
    const float lo  = e00 + (e01 - e00) * rf;
    const float hi  = e10 + (e11 - e10) * rf;
    return lo + (hi - lo) * ff;
}


/// `DielectricLobe::E_ms` (bsdf_dielectric_impl.h:258-276): the one
/// `albedo_impl()` and `filter_o()` read, which comes from the dielectric's own
/// table and is indexed by the Fresnel term's IOR -- and so depends on which
/// side of the surface the ray is on.  The lobe's *other* E_ms, the one
/// `DielectricBSDF`'s eval/sample multiply in, is the mini-GGX table with
/// fresnel row zero (bsdf_dielectric_impl.h:74-78), which
/// `mx_dielectric_eval_vk` reads from `fp->mx_energy`.  Same roughness and
/// anisotropy parameterisation as that one.
inline float dielec_lobe_E_ms (VK_GLOBAL const FrameParams* fp, const Vec3& N,
                               const Vec3& Ng, const Vec3& wo, float xalpha,
                               float yalpha, float eta)
{
#pragma clang fp contract(off)
    const GgxVK d      = ggx_make (xalpha, yalpha, 0.0f);
    const float cosNO  = dot (N, wo);
    const int backside = dot (Ng, wo) < 0.0f;
    const float fidx   = dielec_table_index (dielec_eta (eta, backside));
    return mx_emiss_diel_vk (fp, d.roughness, fidx,
                             cosNO > 0.0f ? cosNO : 0.0f);
}

/// `eval_turquin_microms_reflection` for a dielectric Fresnel, wrapped in
/// `DielectricLobe::eval_impl`: the tint (Cr) multiplies the lobe value.
inline void mx_dielectric_eval_vk (VK_GLOBAL const FrameParams* fp,
                                   const ClosureEntryVK& p, const Vec3& wo,
                                   const Vec3& wi, Vec3& weight, float& pdf)
{
#pragma clang fp contract(off)
    weight = mk (0.0f, 0.0f, 0.0f);
    pdf    = 0.0f;
    const Vec3 N = visible_normal_vk (wo, p.N);
    Vec3 X, Y;
    mx_conductor_frame (N, p.U, X, Y);
    const Vec3 woL = mk (dot (wo, X), dot (wo, Y), dot (wo, N));
    const Vec3 wiL = mk (dot (wi, X), dot (wi, Y), dot (wi, N));
    if (!(woL.z > 0.0f) || !(wiL.z > 0.0f))
        return;
    const int backside = dot (p.N, wo) < 0.0f;
    const float eta    = dielec_eta (p.eta, backside);
    const GgxVK d      = ggx_make (p.xalpha, p.yalpha, 0.0f);
    const Vec3 mh      = normalize (add (woL, wiL));
    const float cosMO  = dot (mh, woL);
    const float D      = ggx_D_vk (d, mh);
    const float DrD    = ggx_D_refl_D_vk (d, woL, mh);
    const float D_refl = DrD * D;
    const float G1     = ggx_G1_vk (d, woL);
    pdf                = D_refl / (4.0f * woL.z);
    const float out    = ggx_G2_G1_vk (d, wiL, woL) * G1 / DrD;
    const float F      = dielec_fresnel (cosMO, eta);
    const float E_ms   = mx_emiss_vk (fp, d.roughness, woL.z);
    const float msf    = E_ms / bmaxf (0.01f, 1.0f - E_ms);
    // Turquin eq. 16, then the dielectric tint (get_tint(wi.z) = refl_tint for
    // a reflected direction).
    const float lobe = F * (1.0f + F * msf) * out;
    weight           = mul (p.refl_tint, lobe);
}

/// `sample_turquin_microms_reflection` for a dielectric Fresnel.
inline void mx_dielectric_sample_vk (VK_GLOBAL const FrameParams* fp,
                                     const ClosureEntryVK& p, const Vec3& wo,
                                     float rx, float ry, Vec3& wi,
                                     Vec3& weight, float& pdf)
{
#pragma clang fp contract(off)
    weight = mk (0.0f, 0.0f, 0.0f);
    pdf    = 0.0f;
    wi     = mul (wo, -1.0f);
    const Vec3 N = visible_normal_vk (wo, p.N);
    Vec3 X, Y;
    mx_conductor_frame (N, p.U, X, Y);
    const Vec3 woL = mk (dot (wo, X), dot (wo, Y), dot (wo, N));
    if (!(woL.z > 0.0f))
        return;
    const GgxVK d = ggx_make (p.xalpha, p.yalpha, 0.0f);
    const Vec3 m  = ggx_sample_for_refl_vk (d, woL, rx, ry);
    if (!(dot (m, woL) > 0.0f))
        return;
    const Vec3 wiL = sub (mul (m, 2.0f * dot (m, woL)), woL);
    if (!(wiL.z > 0.0f))
        return;
    wi = add (add (mul (X, wiL.x), mul (Y, wiL.y)), mul (N, wiL.z));
    mx_dielectric_eval_vk (fp, p, wo, wi, weight, pdf);
}

/// Scene::sample() from raytracer.h, device side: the Heitz low-distortion
/// triangle/square map, then the solid-angle pdf of the resulting direction.
/// The sample point is expressed in barycentrics (u,v) so the light shader can
/// be given the same surface point the CPU gives it.
struct LightSampleVK {
    Vec3 dir;
    float dist;
    float pdf;
    float u, v;
};

inline LightSampleVK light_sample(const SceneView& s, int primID,
                                  const Vec3& x, float xi, float yi)
{
    if (yi > xi) {
        xi *= 0.5f;
        yi -= xi;
    } else {
        yi *= 0.5f;
        xi -= yi;
    }
    TriangleIndices ti = load_tri(&s.tris[primID]);
    const Vec3 va      = load_vec(&s.verts[ti.a]);
    const Vec3 vb      = load_vec(&s.verts[ti.b]);
    const Vec3 vc      = load_vec(&s.verts[ti.c]);
    Vec3 n             = cross(sub(va, vb), sub(va, vc));

    Vec3 l = sub(add(add(mul(va, 1.0f - xi - yi), mul(vb, xi)), mul(vc, yi)), x);
    float d2 = dot(l, l);
    Vec3 dir = normalize(l);
    // length of n is twice the area
    float pdf = d2 / (0.5f * __builtin_fabsf(dot(dir, n)));

    LightSampleVK out;
    out.dir  = dir;
    out.dist = __builtin_sqrtf(d2);
    out.pdf  = pdf;
    out.u    = xi;
    out.v    = yi;
    return out;
}

/// Scene::shapepdf(): solid-angle pdf of direction `p` seen from `x`, used for
/// the MIS weight on self-emission.
inline float shape_pdf(const SceneView& s, int primID, const Vec3& x,
                       const Vec3& p)
{
    TriangleIndices ti = load_tri(&s.tris[primID]);
    const Vec3 va      = load_vec(&s.verts[ti.a]);
    const Vec3 vb      = load_vec(&s.verts[ti.b]);
    const Vec3 vc      = load_vec(&s.verts[ti.c]);
    Vec3 n             = cross(sub(va, vb), sub(va, vc));

    Vec3 l   = sub(p, x);
    float d2 = dot(l, l);
    Vec3 dir = normalize(l);
    return d2 / (0.5f * __builtin_fabsf(dot(dir, n)));
}


//
// The device runtime allocates closure *components* in the per-invocation arena
// (vulkan_shadeops.cpp) and publishes the root as an offset in `sg.Ci`; every
// component is `{ int id; Vec3 w; }` followed by its parameter block.  For the
// components a testrender scene uses, that is enough to evaluate directly:
//   EMISSION (1)        -> Le = w
//   DIFFUSE  (3)        -> albedo = w, cosine-weighted sampling
//   BACKGROUND (2)      -> Le = w
// ADD/MUL nodes (negative ids) are not representable without pointers in the
// buffer; `osl_mul_closure_color` folds a single component's weight in place,
// so a multiplicative chain still arrives here as one component.

static const int EMISSION_ID_VK   = 1;
static const int BACKGROUND_ID_VK = 2;
static const int DIFFUSE_ID_VK    = 3;
/// `oren_nayar(N, sigma)` is *not* a Lambert lobe with a different colour: its
/// eval depends on both directions, so the accumulated-albedo shortcut shades
/// it as a plain diffuse surface (the roughness only shows up as a slightly
/// different overall level, and the grazing-angle brightening is missing
/// entirely).  testrender does not use the classic four-term formula either --
/// shading.cpp:1498 turns this closure into BSDL's `oren_nayar_diffuse_bsdf`
/// lobe, whose `energy_compensation` flag is off, which leaves the QON model;
/// see `on_eval_vk` below for the exact expression and the CPU/device match.
static const int OREN_NAYAR_ID_VK = 4;
/// `microfacet(dist, N, U, xalpha, yalpha, eta, refract)` is testrender's
/// Masking-Shadowing-Microfacet closure.  `refract` selects the lobe: 0 is the
/// reflection lobe, 1 the transmission lobe, 2 both.  glass is the pair
/// (refract=1, refract=0), which is why two components have to be kept.
static const int MICROFACET_ID_VK = 8;
/// `phong(N, exponent)` is testrender/shading.cpp:320-361: a lobe around the
/// mirror direction with a cosine power falloff.  Its weight is angle
/// dependent (`N.wi * (exponent+2)/(exponent+1)`), so like Ward it cannot be
/// folded into a constant albedo: NEE has to evaluate the real BRDF and the
/// BSDF sample has to come from the lobe itself.
static const int PHONG_ID_VK      = 6;
/// `ward(N, T, ax, ay)` is an anisotropic *glossy* closure: unlike diffuse its
/// weight is angle dependent, so it cannot be folded into a constant albedo --
/// NEE has to evaluate the real BRDF toward the light and the BSDF sample has to
/// come from the Ward lobe itself.
static const int WARD_ID_VK       = 7;
/// testrender/shading.h ClosureIDs.  `reflection(N)` is a perfect mirror and
/// `reflection(N, eta)` adds a Fresnel term; both are a single mirror direction,
/// so no importance sampling is needed to reproduce them.
static const int REFLECTION_ID_VK         = 9;
static const int FRESNEL_REFLECTION_ID_VK = 10;
/// `refraction(N, eta)` bends the ray through the surface and carries the
/// complementary Fresnel term; `transparent()` passes it straight through.  Like
/// the mirrors these are delta closures -- one outgoing direction, infinite pdf.
static const int REFRACTION_ID_VK  = 11;
static const int TRANSPARENT_ID_VK = 12;

// ---- compile-time closure sets ---------------------------------------------
//
// Which closure *families* a group's shader can produce.  This is the one
// dimension that lets a lane shed code: `e.id` is a runtime load out of the
// closure pool, so a plain `if (e.id == WARD_ID_VK)` keeps the Ward lobe, the
// cosine sampler and the Fresnel path in every kernel.  The device runtime is
// also force-inlined before clspv (`vulkan_module.cpp` marks every non-kernel
// function `alwaysinline`, because clspv can only attribute a storage-buffer
// pointer while the chain from the kernel argument is still visible), so each
// of those paths is copied once per material kernel and once per call site --
// which is what pushed the microfacet scene past what the driver will build.
//
// A group's set is read off the `closure_component(sg, id, ...)` calls in its
// IR, so a kernel carries only the families its shader can actually emit.
// `CS_ALL` is the fallback: a set that is not one of the instantiations below
// gets the full body, correct but not slimmed.
static const unsigned CS_BG    = osl_vk_abi::kClosureSetBG;
static const unsigned CS_DIFF  = osl_vk_abi::kClosureSetDiff;
static const unsigned CS_WARD  = osl_vk_abi::kClosureSetWard;
static const unsigned CS_MF    = osl_vk_abi::kClosureSetMF;
static const unsigned CS_DELTA = osl_vk_abi::kClosureSetDelta;
static const unsigned CS_PHONG = osl_vk_abi::kClosureSetPhong;
static const unsigned CS_ON    = osl_vk_abi::kClosureSetON;
static const unsigned CS_MXC   = osl_vk_abi::kClosureSetMXC;
static const unsigned CS_MXD   = osl_vk_abi::kClosureSetMXD;
static const unsigned CS_ALL   = osl_vk_abi::kClosureSetAll;

/// testrender/optics.h:13, verbatim.  TIR returns 1, eta == 0 means "no Fresnel".
inline float fresnel_dielectric_vk(float cosi, float eta)
{
    if (eta == 0.0f)
        return 1.0f;
    if (cosi < 0.0f)
        eta = 1.0f / eta;
    float c = __builtin_fabsf(cosi);
    float g = eta * eta - 1.0f + c * c;
    if (g > 0.0f) {
        g       = __builtin_sqrtf(g);
        float A = (g - c) / (g + c);
        float B = (c * (g + c) - 1.0f) / (c * (g - c) + 1.0f);
        return 0.5f * A * A * (1.0f + B * B);
    }
    return 1.0f;
}

struct ClosureComponentVK {
    int id;
    Vec3 w;
};

/// Root component of a closure.  The device module never produces a null
/// closure pointer (vulkan_shadeops.cpp: a black weight allocates a zero-weight
/// component), which is what lets the null tests be folded away -- a pointer
/// comparison would need `CapabilityVariablePointers`.
inline const ClosureComponentVK* closure_root(const void* ci)
{
    return (const ClosureComponentVK*)ci;
}


// An ADD node as vulkan_shadeops.cpp lays it down: same 24-byte header the CPU
// uses, but the two closure fields hold arena *offsets* (a pointer stored in
// memory is the one thing clspv cannot compile).
struct ClosureAddVK {
    int id;
    long long closureA;
    long long closureB;
};

/// The closure pool's start inside a ShadeBlock, mirroring kOffPool in
/// vulkan_shadeops.cpp.  The kernel section below defines its own copy for the
/// frame body, so this one carries a distinct name.
static const long long kOffPoolClosure = 320;


/// Fill one flattened component from its pool node.
///
/// Split out of `collect_closure_entries` so the two slots can be written by
/// straight-line calls: `(n == 0) ? first : second` is a `select` between two
/// distinct objects, and clspv turns that into an `OpSelect` on a Function
/// pointer -- the same `CapabilityVariablePointers` leak the two-slot stack
/// below avoids, and the Intel driver aborts on it when the module is handed
/// to `vkCreateComputePipelines`.
inline __attribute__((always_inline)) void
fill_closure_entry (ClosureEntryVK& e, int id, long long off, const Vec3& w,
                    VK_GLOBAL const unsigned char* node)
{
    e.id  = id;
    e.off = off;
    e.w   = w;
    e.N   = mk (0.0f, 0.0f, 0.0f);
    e.U   = mk (0.0f, 0.0f, 0.0f);
    e.eta = 0.0f;
    e.xalpha = 1.0f;
    e.yalpha = 1.0f;
    e.refract = 0;
    e.exponent = 0.0f;
    e.sigma    = 0.0f;
    e.ior      = mk (0.0f, 0.0f, 0.0f);
    e.ext      = mk (0.0f, 0.0f, 0.0f);
    e.refl_tint = mk (0.0f, 0.0f, 0.0f);
    e.refr_tint = mk (0.0f, 0.0f, 0.0f);
    if (id == FRESNEL_REFLECTION_ID_VK || id == REFRACTION_ID_VK) {
        // ReflectionParams / RefractionParams { Vec3 N; float eta }
        e.N   = load_vec ((GlobalCVec3)(node + 16));
        e.eta = *(VK_GLOBAL const float*)(node + 28);
    } else if (id == MICROFACET_ID_VK) {
        // MicrofacetParams { ustringhash dist; Vec3 N; Vec3 U; float xalpha;
        // float yalpha; float eta; int refract; }: dist is a 64-bit hash, so N
        // starts at +24.
        e.N       = load_vec ((GlobalCVec3)(node + 24));
        e.U       = load_vec ((GlobalCVec3)(node + 36));
        e.xalpha  = *(VK_GLOBAL const float*)(node + 48);
        e.yalpha  = *(VK_GLOBAL const float*)(node + 52);
        e.eta     = *(VK_GLOBAL const float*)(node + 56);
        e.refract = *(VK_GLOBAL const int*)(node + 60);
    } else if (id == REFLECTION_ID_VK) {
        e.N = load_vec ((GlobalCVec3)(node + 16));
    } else if (id == PHONG_ID_VK) {
        // PhongParams { Vec3 N; float exponent } (testrender/shading.h:74).
        e.N        = load_vec ((GlobalCVec3)(node + 16));
        e.exponent = *(VK_GLOBAL const float*)(node + 28);
    } else if (id == OREN_NAYAR_ID_VK) {
        // OrenNayarParams { Vec3 N; float sigma } (shading.h:70) -- the same
        // byte layout as PhongParams above.
        e.N     = load_vec ((GlobalCVec3)(node + 16));
        e.sigma = *(VK_GLOBAL const float*)(node + 28);
    } else if (id == MX_CONDUCTOR_ID_VK) {
        // mtx::ConductorLobe::Data as register_closure lays it out
        // (bsdf_conductor_decl.h:31-43): Vec3 N (16), Vec3 U (28), float
        // roughness_x (40), float roughness_y (44), C3f IOR (48), C3f
        // extinction (60), Stringhash distribution (72), thinfilm fields.
        // Only the GGX path is modelled here, which is what every test
        // scene's `conductor_bsdf(..., "ggx")` asks for.
        e.N      = load_vec ((GlobalCVec3)(node + 16));
        e.U      = load_vec ((GlobalCVec3)(node + 28));
        e.xalpha = *(VK_GLOBAL const float*)(node + 40);
        e.yalpha = *(VK_GLOBAL const float*)(node + 44);
        e.ior    = load_vec ((GlobalCVec3)(node + 48));
        e.ext    = load_vec ((GlobalCVec3)(node + 60));
    } else if (id == MX_DIELECTRIC_ID_VK) {
        // mtx::DielectricLobe::Data (bsdf_dielectric_decl.h:76-88).  Unlike the
        // conductor it derives from LayeredData, so an 8-byte `const void*
        // closure` precedes N: at +16 the node header ends and Data starts,
        // which puts N at +24, U at +36, refl_tint (Cr) at +48, refr_tint (Ct)
        // at +60, roughness_x at +72, roughness_y at +76, IOR at +80 (its
        // Stringhash follows at +88, so the floats above are the whole prefix
        // this kernel needs).
        e.N         = load_vec ((GlobalCVec3)(node + 24));
        e.U         = load_vec ((GlobalCVec3)(node + 36));
        e.refl_tint = load_vec ((GlobalCVec3)(node + 48));
        e.refr_tint = load_vec ((GlobalCVec3)(node + 60));
        e.xalpha    = *(VK_GLOBAL const float*)(node + 72);
        e.yalpha    = *(VK_GLOBAL const float*)(node + 76);
        e.eta       = *(VK_GLOBAL const float*)(node + 80);
    }
}


/// Opacity of a layer's top closure, i.e. how much of the light reaching the
/// surface the top reflects or absorbs before it can arrive at the base.
///
/// `evaluate_layer_opacity` (testrender/shading.cpp:1195-1284) sums
/// `branch_weight * w * (1 - filter_o)` over every leaf.  Only the dielectric
/// has a `filter_o` this kernel can reproduce -- `E_ms * absorption`
/// (bsdf_dielectric_decl.h:149-153), zero once `Ct` makes the lobe
/// transmissive; every other family is treated as opaque, which is what the
/// CPU's own `default:` case does.
inline Vec3 layer_alpha_vk (VK_GLOBAL ShaderGlobals* sg,
                            VK_GLOBAL const FrameParams* fp, long long root,
                            const Vec3& wo)
{
    Vec3 acc = mk (0.0f, 0.0f, 0.0f);
    if (root < 0)
        return acc;

    long long off0 = 0, off1 = 0;
    Vec3 w0, w1;
    int sp        = 0;
    long long cur = root;
    Vec3 branch   = mk (1.0f, 1.0f, 1.0f);

    for (;;) {
        if (cur >= 0) {
            GlobalCPtr node = (GlobalCPtr)sg + kOffPoolClosure + cur;
            const int id    = *(VK_GLOBAL const int*)node;
            if (id == -2) {  // ADD
                if (sp == 0) {
                    off0 = *(VK_GLOBAL const long long*)(node + 16);
                    w0   = branch;
                } else if (sp == 1) {
                    off1 = *(VK_GLOBAL const long long*)(node + 16);
                    w1   = branch;
                }
                if (sp < 2)
                    ++sp;
                cur = *(VK_GLOBAL const long long*)(node + 8);
                continue;
            }
            if (id == -1) {  // MUL
                branch = mulv (branch, load_vec ((GlobalCVec3)(node + 4)));
                cur    = *(VK_GLOBAL const long long*)(node + 16);
                continue;
            }
            if (id == MX_LAYER_ID_VK) {
                // A nested layer contributes both halves, the base at the
                // layer's own weight -- the same approximation the CPU makes
                // for this case (shading.cpp:1227-1232).
                if (sp == 0) {
                    off0 = *(VK_GLOBAL const long long*)(node + 24);
                    w0   = mulv (branch, load_vec ((GlobalCVec3)(node + 4)));
                } else if (sp == 1) {
                    off1 = *(VK_GLOBAL const long long*)(node + 24);
                    w1   = mulv (branch, load_vec ((GlobalCVec3)(node + 4)));
                }
                if (sp < 2)
                    ++sp;
                cur = *(VK_GLOBAL const long long*)(node + 16);
                continue;
            }
            // A leaf contributes `branch * w * (1 - filter_o)`.  Only the
            // reflection-only dielectric transmits anything here, and its
            // transmittance is the multiple-scattered energy it does not
            // reflect; the transmissive flavour transmits all of it.  Every
            // other closure counts as opaque and contributes its branch
            // weight alone -- note that the CPU does *not* apply `comp->w`
            // for those (shading.cpp:1262-1272), which matters here because
            // the device folds MUL weights into the component.
            Vec3 a = mk (1.0f, 1.0f, 1.0f);
            if (id == MX_DIELECTRIC_ID_VK) {
                const Vec3 ct = load_vec ((GlobalCVec3)(node + 60));
                if (ct.x > 0.0f || ct.y > 0.0f || ct.z > 0.0f) {
                    // A transmissive dielectric has `filter_o = ZERO`
                    // (bsdf_dielectric_decl.h:149-153).
                    a = mk (0.0f, 0.0f, 0.0f);
                } else {
                    const Vec3 N = visible_normal_vk (wo, load_vec ((GlobalCVec3)(node + 24)));
                    const float opa = 1.0f
                                    - dielec_lobe_E_ms (fp, N,
                                                        load_vec ((GlobalCVec3)(node + 24)),
                                                        wo,
                                                        *(VK_GLOBAL const float*)(node + 72),
                                                        *(VK_GLOBAL const float*)(node + 76),
                                                        *(VK_GLOBAL const float*)(node + 80));
                    a = mul (load_vec ((GlobalCVec3)(node + 4)), opa);
                }
            }
            acc  = add (acc, mulv (branch, a));
            cur  = -1;
        }
        if (sp == 0)
            break;
        --sp;
        if (sp == 0) {
            cur    = off0;
            branch = w0;
        } else {
            cur    = off1;
            branch = w1;
        }
    }
    return mk (vk_clampf (acc.x, 0.0f, 1.0f), vk_clampf (acc.y, 0.0f, 1.0f),
               vk_clampf (acc.z, 0.0f, 1.0f));
}


/// Flatten `Ci` into at most two components.
///
/// Two is enough for the closures a render-* material produces: matte and Ward
/// are a single component, and glass is exactly
/// `reflection(N, eta) + refraction(N, eta)`.  A larger tree is truncated to
/// the first two, which dims a path rather than corrupting it.
///
/// The pending work is two scalars, *not* a local array.  An indexed array of
/// `Vec3` leaves the device runtime with a `getelementptr` whose index is
/// dynamic, and clspv lowers that to `OpPtrAccessChain` on a Function pointer:
/// plain float-pointer arithmetic in private memory, which is only legal with
/// `CapabilityVariablePointers`.  That capability in turn makes Intel's
/// compiler abort inside `vkCreateComputePipelines` (0xC0000005) for any
/// module that carries it, however small -- verified with a probe that creates
/// a pipeline for a single 140k-word kernel.  Two slots are enough because the
/// walk stops the moment it has both components, so it can never need a third
/// pending node.
inline __attribute__((always_inline)) int
collect_closure_entries (VK_GLOBAL ShaderGlobals* sg,
                         VK_GLOBAL const FrameParams* fp, long long root,
                         const Vec3& wo, ClosureEntryVK& first,
                         ClosureEntryVK& second)
{
    if (root < 0)
        return 0;

    long long off0 = 0, off1 = 0;
    Vec3 w0, w1;
    int sp        = 0;
    long long cur = root;
    Vec3 branch   = mk (1.0f, 1.0f, 1.0f);
    int n         = 0;

    for (;;) {
        if (cur >= 0) {
            GlobalCPtr node = (GlobalCPtr)sg + kOffPoolClosure + cur;
            const int id    = *(VK_GLOBAL const int*)node;
            if (id == -2) {  // ADD
                if (sp == 0) {
                    off0 = *(VK_GLOBAL const long long*)(node + 16);
                    w0   = branch;
                } else if (sp == 1) {
                    off1 = *(VK_GLOBAL const long long*)(node + 16);
                    w1   = branch;
                }
                if (sp < 2)
                    ++sp;
                cur = *(VK_GLOBAL const long long*)(node + 8);
                continue;
            }
            if (id == -1) {  // MUL
                branch = mulv (branch, load_vec ((GlobalCVec3)(node + 4)));
                cur    = *(VK_GLOBAL const long long*)(node + 16);
                continue;
            }
            if (id == MX_LAYER_ID_VK) {
                // layer(top, base): `top` takes the layer's own weight, `base`
                // what the top leaves of it.  Descending is not optional --
                // leaving the layer as a leaf would hand the shading kernel an
                // id no family implements, and every closure the material
                // actually built would be invisible behind it.
                // `base` is weighted by what the top leaves through:
                // `base_w = w * (1 - clamp(opacity)` (shading.cpp:1314-1326).
                const Vec3 lw    = mulv (branch, load_vec ((GlobalCVec3)(node + 4)));
                const Vec3 alpha = layer_alpha_vk (sg, fp,
                                                  *(VK_GLOBAL const long long*)
                                                      (node + 16),
                                                  wo);
                const Vec3 basew = mulv (lw, mk (1.0f - alpha.x,
                                                  1.0f - alpha.y,
                                                  1.0f - alpha.z));
                if (sp == 0) {
                    off0 = *(VK_GLOBAL const long long*)(node + 24);
                    w0   = basew;
                } else if (sp == 1) {
                    off1 = *(VK_GLOBAL const long long*)(node + 24);
                    w1   = basew;
                }
                if (sp < 2)
                    ++sp;
                cur    = *(VK_GLOBAL const long long*)(node + 16);
                branch = lw;
                continue;
            }
            const Vec3 bw = mulv (branch, load_vec ((GlobalCVec3)(node + 4)));
            if (n == 0) {
                fill_closure_entry (first, id, cur, bw, node);
            } else {
                fill_closure_entry (second, id, cur, bw, node);
            }
            ++n;
            // Both slots are filled: nothing the rest of the tree holds can
            // change the answer, and not walking it keeps the pending depth at
            // two.
            if (n == 2)
                break;
            cur = -1;
        }
        if (sp == 0)
            break;
        --sp;
        if (sp == 0) {
            cur    = off0;
            branch = w0;
        } else {
            cur    = off1;
            branch = w1;
        }
    }
    return n;
}


/// Is this component a delta closure (a single outgoing direction)?  Those are
/// the ones whose pdf is infinite, which is what the CPU tests for to suppress
/// NEE on the next bounce.
inline bool closure_entry_is_delta (const ClosureEntryVK& e)
{
    return e.id == REFLECTION_ID_VK || e.id == FRESNEL_REFLECTION_ID_VK
           || e.id == REFRACTION_ID_VK || e.id == TRANSPARENT_ID_VK;
}


/// `get_albedo(wo)` for one component (shading.cpp:1083-1118 and :1114-1118):
/// mirror carries the Fresnel term, refraction the complement, a matte
/// component its own weight, a microfacet lobe its Fresnel-weighted albedo.
/// `CS` is the kernel's compile-time closure set: a family the material cannot
/// produce is not emitted at all.
template <unsigned CS>
inline Vec3 closure_entry_albedo (VK_GLOBAL const FrameParams* fp,
                                   const ClosureEntryVK& e, const Vec3& wo)
{
    if constexpr (CS & CS_DELTA) {
        if (e.id == REFLECTION_ID_VK || e.id == FRESNEL_REFLECTION_ID_VK) {
            float cosNO = dot (e.N, wo);
            if (cosNO > 0.0f) {
                float F = fresnel_dielectric_vk (cosNO, e.eta);
                return mk (e.w.x * F, e.w.y * F, e.w.z * F);
            }
            return e.w;  // shading.cpp:1088 returns Color3(1) times the weight
        }
        if (e.id == REFRACTION_ID_VK) {
            float cosNO = dot (e.N, wo);
            float F     = 1.0f - fresnel_dielectric_vk (cosNO, e.eta);
            return mk (e.w.x * F, e.w.y * F, e.w.z * F);
        }
    }
    if constexpr (CS & CS_MF) {
        if (e.id == MICROFACET_ID_VK) {
            float a = mf_albedo (e, wo);
            return mk (e.w.x * a, e.w.y * a, e.w.z * a);
        }
    }
    if constexpr (CS & CS_MXD) {
        if (e.id == MX_DIELECTRIC_ID_VK) {
            // `DielectricLobe::albedo_impl()` (bsdf_dielectric_decl.h:104-108):
            // refl_tint * (1 - E_ms), the one supported family whose albedo is
            // *not* its branch weight -- it is what the CPU's get_albedo hands
            // prepare() and the roulette, so it has to match here too.
            if (!(e.refr_tint.x > 0.0f) && !(e.refr_tint.y > 0.0f)
                && !(e.refr_tint.z > 0.0f)) {
                const Vec3 N     = visible_normal_vk (wo, e.N);
                const float E    = dielec_lobe_E_ms (fp, N, e.N, wo, e.xalpha,
                                                    e.yalpha, e.eta);
                const float k    = 1.0f - E;
                return mk (e.w.x * e.refl_tint.x * k,
                           e.w.y * e.refl_tint.y * k,
                           e.w.z * e.refl_tint.z * k);
            }
        }
    }
    return e.w;
}


/// `MIS::update_eval` (sampling.h:173) over device types: folds a component's
/// (weight, pdf) into the running sum of a BSDF mixture with the balance
/// heuristic.  Used to mix a sampled lobe with the others' evaluation.
inline void mis_update_eval_vk (Vec3& w, float& pdf, Vec3 ow, float opdf,
                                float b)
{
    if (b > 1.17549435e-38f) {  // FLT_MIN; 1/b must stay finite
        opdf *= b;
        ow = mul (ow, 1.0f / b);
        float mis;
        if (pdf < opdf)
            mis = 1.0f / (1.0f + pdf / opdf);
        else if (opdf < pdf)
            mis = 1.0f - 1.0f / (1.0f + opdf / pdf);
        else
            mis = 0.5f;  // avoid inf/inf
        w = add (mul (w, 1.0f - mis), mul (ow, mis));
        pdf += opdf;
    }
}


/// One component's `eval(wo, wi)` -- weight already multiplied by the
/// component's branch weight, plus its solid-angle pdf.  `n` is the shading
/// normal the diffuse lobe is defined on (testrender's DiffuseParams.N stores
/// the same vector the shader wrote).  `CS` gates the families at compile time.
template <unsigned CS>
inline Vec3 component_eval (VK_GLOBAL const FrameParams* fp,
                            const ClosureEntryVK& e, const WardVK& wardp,
                            const Vec3& wov, const Vec3& wi, const Vec3& n,
                            float& pdf)
{
    pdf = 0.0f;
    if constexpr (CS & CS_DIFF) {
        if (e.id == DIFFUSE_ID_VK) {
            float NL = dot (n, wi);
            pdf      = (NL > 0.0f ? NL : 0.0f) * 0.31830988618f;  // 1/pi
            return e.w;
        }
    }
    if constexpr (CS & CS_ON) {
        if (e.id == OREN_NAYAR_ID_VK) {
            float lw = 0.0f, lp = 0.0f;
            on_eval_vk (e, wov, wi, lw, lp);
            pdf = lp;
            return mul (e.w, lw);
        }
    }
    if constexpr (CS & CS_WARD) {
        if (e.id == WARD_ID_VK) {
            Vec3 wv;
            float p = 0.0f;
            ward_eval_vk (wardp, wov, wi, wv, p);
            pdf = p;
            return mulv (e.w, wv);
        }
    }
    if constexpr (CS & CS_MF) {
        if (e.id == MICROFACET_ID_VK) {
            float lw = 0.0f;
            mf_eval_vk (e, wov, wi, lw, pdf);
            return mul (e.w, lw);
        }
    }
    if constexpr (CS & CS_PHONG) {
        if (e.id == PHONG_ID_VK) {
            float lw = 0.0f, lp = 0.0f;
            phong_eval_vk (e, wov, wi, lw, lp);
            pdf = lp;
            return mul (e.w, lw);
        }
    }
    if constexpr (CS & CS_MXC) {
        if (e.id == MX_CONDUCTOR_ID_VK) {
            // The conductor's weight is a colour (its Fresnel term is
            // per-channel), so it cannot come back through the scalar path.
            Vec3 lw;
            float lp = 0.0f;
            mx_conductor_eval_vk (fp, e, wov, wi, lw, lp);
            pdf = lp;
            return mulv (e.w, lw);
        }
    }
    if constexpr (CS & CS_MXD) {
        if (e.id == MX_DIELECTRIC_ID_VK
            && !(e.refr_tint.x > 0.0f) && !(e.refr_tint.y > 0.0f)
            && !(e.refr_tint.z > 0.0f)) {
            Vec3 lw;
            float lp = 0.0f;
            mx_dielectric_eval_vk (fp, e, wov, wi, lw, lp);
            pdf = lp;
            return mulv (e.w, lw);
        }
    }
    return mk (0.0f, 0.0f, 0.0f);
}


/// `CompositeBSDF::eval`: the sum over a BSDF's components of their (weight,
/// pdf) toward `wi`, skipping emitters.
///
/// The two slots are two straight-line evaluations rather than a loop over a
/// local array.  Both are needed by clspv: a loop whose body indexes a local
/// array makes clspv diverge on the microfacet kernel (it never returns, at any
/// -O level), and `(k == 0) ? e0 : e1` on the entries is a `select` between two
/// distinct objects, which clspv lowers to an OpSelect on a Function pointer --
/// pointer arithmetic, i.e. CapabilityVariablePointers, which Intel's compiler
/// aborts on at pipeline creation.  Two call sites with constant slot indices
/// have neither shape; the cost is a second copy of `component_eval` in the
/// kernel, which the size measurements have room for.
///
/// Skipping emitters is implicit: `component_eval` only answers for a
/// DIFFUSE/WARD/MICROFACET component and reports pdf 0 otherwise, which is what
/// the old `continue` did.
template <unsigned CS>
inline void components_eval_sum (VK_GLOBAL const FrameParams* fp,
                                 const ClosureEntryVK& e0,
                                 const ClosureEntryVK& e1, int ncomp,
                                 const WardVK& wardp, const Vec3& wov,
                                 const Vec3& wi, const Vec3& n, float b0,
                                 float b1, Vec3& w, float& pdf)
{
    float p0 = 0.0f, p1 = 0.0f;
    Vec3 w0 = mk (0.0f, 0.0f, 0.0f), w1 = mk (0.0f, 0.0f, 0.0f);
    if (ncomp > 0)
        w0 = component_eval<CS> (fp, e0, wardp, wov, wi, n, p0);
    if (ncomp > 1)
        w1 = component_eval<CS> (fp, e1, wardp, wov, wi, n, p1);
    // Not a plain sum: CompositeBSDF::eval (shading.h:364-374) folds each
    // lobe in with MIS::update_eval, using the lobe's own pick probability
    // `b` = pdfs[i] from prepare().  With more than one lobe that is what
    // turns `weight` into the mixture's f/pdf-normalised quantity and what
    // scales the pdf by the pick probability, so the MIS weight the caller
    // computes from it only lines up with the CPU when this matches.
    w   = mk (0.0f, 0.0f, 0.0f);
    pdf = 0.0f;
    if (ncomp > 0)
        mis_update_eval_vk (w, pdf, w0, p0, b0);
    if (ncomp > 1)
        mis_update_eval_vk (w, pdf, w1, p1, b1);
}


/// The NEE evaluation of the whole BSDF toward `wi`: the compile-time set
/// decides which family is consulted, and the runtime flags only pick between
/// the families the set actually contains.  `ar/ag/ab` is the accumulated
/// albedo, which is what the Lambert lobe evaluates to.
template <unsigned CS>
inline void nee_eval_vk (VK_GLOBAL const FrameParams* fp,
                         const ClosureEntryVK& ce0, const ClosureEntryVK& ce1,
                         int ncomp, int has_microfacet, int is_ward,
                         int has_phong, int has_on, int has_mxc, int has_mxd,
                         const WardVK& wardp, const Vec3& wov, const Vec3& wi,
                         const Vec3& n, float ar, float ag, float ab, float b0,
                         float b1, Vec3& w, float& pdf)
{
    if constexpr (CS & CS_MXC) {
        // A conductor lobe is glossy and its eval is angle dependent (GGX
        // distribution times the complex Fresnel term), so the accumulated
        // albedo cannot stand in for it.
        if (has_mxc) {
            components_eval_sum<CS> (fp, ce0, ce1, ncomp, wardp, wov, wi, n, b0,
                                     b1, w, pdf);
            return;
        }
    }
    if constexpr (CS & CS_MXD) {
        // Same for the dielectric: GGX reflectance weighted by its own Fresnel
        // term, nothing a cosine lobe can approximate.
        if (has_mxd) {
            components_eval_sum<CS> (fp, ce0, ce1, ncomp, wardp, wov, wi, n, b0,
                                     b1, w, pdf);
            return;
        }
    }
    if constexpr (CS & CS_ON) {
        // Oren-Nayar's eval depends on the roughness between the two
        // directions, so it cannot ride on the accumulated albedo either.
        if (has_on) {
            components_eval_sum<CS> (fp, ce0, ce1, ncomp, wardp, wov, wi, n, b0,
                                     b1, w, pdf);
            return;
        }
    }
    if constexpr (CS & CS_MF) {
        if (has_microfacet) {
            components_eval_sum<CS> (fp, ce0, ce1, ncomp, wardp, wov, wi, n, b0,
                                     b1, w, pdf);
            return;
        }
    }
    if constexpr (CS & CS_PHONG) {
        // A phong lobe is glossy, not Lambert, so the light sample has to run
        // through the component eval (with the lobe's pick probability folded
        // in) instead of the accumulated-albedo shortcut below.
        if (has_phong) {
            components_eval_sum<CS> (fp, ce0, ce1, ncomp, wardp, wov, wi, n, b0,
                                     b1, w, pdf);
            return;
        }
    }
    if constexpr (CS & CS_WARD) {
        if (is_ward) {
            Vec3 wv;
            float p = 0.0f;
            ward_eval_vk (wardp, wov, wi, wv, p);
            pdf = p;
            w   = mulv (mk (ar, ag, ab), wv);
            return;
        }
    }
    if constexpr (CS & CS_DIFF) {
        // Diffuse: Diffuse<trans>::eval (shading.cpp:305) returns
        // weight = Color3(1) and pdf = max(N.wi, 0)/pi.
        const float NLd = dot (n, wi);
        pdf = (NLd > 0.0f ? NLd : 0.0f) * 0.31830988618f;
        w   = mk (ar, ag, ab);
        return;
    }
    w   = mk (0.0f, 0.0f, 0.0f);
    pdf = 0.0f;
}


/// optics.h:34, verbatim: the refracted direction and its Fresnel term, with
/// the total-internal-reflection case returning zero weight.
inline float
fresnel_refraction_vk (const Vec3& I, const Vec3& N, float eta, Vec3& T)
{
    float cosi = -dot (I, N);
    Vec3 Nn;
    float neta;
    if (cosi > 0.0f) {
        neta = 1.0f / eta;
        Nn   = N;
    } else {
        cosi = -cosi;
        neta = eta;
        Nn   = mul (N, -1.0f);
    }
    float arg = 1.0f - (neta * neta * (1.0f - cosi * cosi));
    if (arg >= 0.0f) {
        float dnp = __builtin_sqrtf (arg);
        float nK  = (neta * cosi) - dnp;
        T         = add (mul (I, neta), mul (Nn, nK));
        return 1.0f - fresnel_dielectric_vk (cosi, eta);
    }
    T = mk (0.0f, 0.0f, 0.0f);
    return 0.0f;
}


/// Sum a background shader's closure tree the way the CPU's
/// process_background_closure() does (src/testrender/shading.cpp:1711): MUL
/// folds a colour into the branch weight, ADD splits the branch, and a
/// BACKGROUND component contributes `weight * colour`.  Anything else is
/// ignored, exactly like the CPU evaluator -- a background shader's Ci is a set
/// of BACKGROUND closures, and an emitter that happens to sit in the tree
/// contributes nothing there.
inline Vec3 background_closure_color(VK_GLOBAL ShaderGlobals* sg, long long root)
{
    Vec3 acc = mk(0.0f, 0.0f, 0.0f);
    if (root < 0)
        return acc;

    long long stack_off[16];
    Vec3 stack_w[16];
    int sp        = 0;
    long long cur = root;
    Vec3 branch   = mk(1.0f, 1.0f, 1.0f);
    for (;;) {
        if (cur >= 0) {
            GlobalCPtr node = (GlobalCPtr)sg + kOffPoolClosure + cur;
            const int id    = *(VK_GLOBAL const int*)node;
            if (id == -2) {  // ADD
                if (sp < 16) {
                    stack_off[sp] = *(VK_GLOBAL const long long*)(node + 16);
                    stack_w[sp]   = branch;
                    ++sp;
                }
                cur = *(VK_GLOBAL const long long*)(node + 8);
                continue;
            }
            if (id == -1) {  // MUL
                branch = mulv(branch, load_vec((GlobalCVec3)(node + 4)));
                cur    = *(VK_GLOBAL const long long*)(node + 16);
                continue;
            }
            if (id == BACKGROUND_ID_VK)
                acc = add(acc, mulv(branch, load_vec((GlobalCVec3)(node + 4))));
            cur = -1;  // a component is a leaf
        }
        if (sp == 0)
            break;
        --sp;
        cur    = stack_off[sp];
        branch = stack_w[sp];
    }
    return acc;
}


/// MIS::power_heuristic<MIS::WEIGHT_WEIGHT> (src/testrender/sampling.h:131),
/// verbatim including the "inf is ok" case: the other pdf is the one that gets
/// multiplied in.  Both the bsdf pdf (inf on a camera ray) and the background
/// pdf (0 wherever the table has no probability mass) reach the extremes.
inline float power_heuristic_weight_weight(float sampled_pdf, float other_pdf)
{
    float r, mis;
    if (sampled_pdf > other_pdf) {
        r   = other_pdf / sampled_pdf;
        mis = 1.0f / (1.0f + r * r);
    } else if (sampled_pdf < other_pdf) {
        r   = sampled_pdf / other_pdf;
        mis = 1.0f - 1.0f / (1.0f + r * r);
    } else {
        mis = 0.5f;
    }
    const float MAXF = 3.4028234663852886e+38f;
    return (other_pdf < MAXF ? other_pdf : MAXF) * mis;
}


/// The CPU's Background::eval (src/testrender/background.h:102), on the table
/// uploaded from Background::prepare: map the direction to the unit square,
/// take the nearest entry, and return the already-pdf-divided value together
/// with that entry's solid-angle pdf.
inline Vec3 background_table_eval(VK_GLOBAL const FrameParams* p,
                                  VK_GLOBAL const unsigned char* base,
                                  const Vec3& dir, float& pdf)
{
    pdf            = 0.0f;
    const int res  = p->bg_res;
    if (res <= 0)
        return mk(0.0f, 0.0f, 0.0f);

    VK_GLOBAL const Vec3* values
        = (VK_GLOBAL const Vec3*)(base + p->bg_values_offset);
    VK_GLOBAL const float* rows
        = (VK_GLOBAL const float*)(base + p->bg_rows_offset);
    VK_GLOBAL const float* cols
        = (VK_GLOBAL const float*)(base + p->bg_cols_offset);

    // 1/(2*pi); through OIIO::fast_atan2 (fmath.h:1775) so the texel index
    // matches the CPU's for bit-identical directions.
    float u = mf_fast_atan2(dir.y, dir.x) * 0.15915494309189535f;
    if (u < 0.0f)
        u += 1.0f;
    float v = (1.0f - dir.z) * 0.5f;
    int x   = (int)(u * (float)res);
    if (x < 0)
        x = 0;
    else if (x >= res)
        x = res - 1;
    int y = (int)(v * (float)res);
    if (y < 0)
        y = 0;
    else if (y >= res)
        y = res - 1;
    const int i    = y * res + x;
    const float row_pdf = rows[y] - (y > 0 ? rows[y - 1] : 0.0f);
    const float col_pdf = cols[i] - (x > 0 ? cols[i - 1] : 0.0f);
    const float pd      = row_pdf * col_pdf * p->bg_invjacobian;
    pdf                 = pd > 0.0f ? pd : 0.0f;
    // Read the three floats through the global lvalue: the element type is
    // address-space-qualified, so copying the whole Vec3 is not well formed.
    Vec3 bgv3 = mk(values[i].x, values[i].y, values[i].z);
    return bgv3;
}


/// Background::sample (src/testrender/background.h:128): draw a row from its
/// cumulative pdf, then a column inside that row, map the sample back to a
/// direction, and return the pre-pdf-divided value with the solid-angle pdf.
/// This is the background *as a light* -- the CPU's explicit background
/// sampling in the NEE block (simpleraytracer.cpp:1138-1155), without which a
/// matte sphere is only lit by the much noisier bsdf-sampling path.
inline Vec3 background_sample(VK_GLOBAL const FrameParams* p,
                              VK_GLOBAL const unsigned char* base, float rx,
                              float ry, Vec3& dir, float& pdf)
{
    dir           = mk(0.0f, 0.0f, 1.0f);
    pdf           = 0.0f;
    const int res = p->bg_res;
    if (res <= 0)
        return mk(0.0f, 0.0f, 0.0f);

    VK_GLOBAL const Vec3* values
        = (VK_GLOBAL const Vec3*)(base + p->bg_values_offset);
    VK_GLOBAL const float* rows
        = (VK_GLOBAL const float*)(base + p->bg_rows_offset);
    VK_GLOBAL const float* cols
        = (VK_GLOBAL const float*)(base + p->bg_cols_offset);
    const float invres = 1.0f / (float)res;

    // Background::sample_cdf, twice: first over the rows, then over the chosen
    // row's columns.  upper_bound() semantics (first entry strictly greater),
    // done inline because the table lives in the storage buffer.
    float row_pdf = 0.0f, col_pdf = 0.0f;
    int y = res - 1;
    {
        int lo = 0, hi = res;
        while (lo < hi) {
            const int mid = (lo + hi) >> 1;
            if (ry < rows[mid])
                hi = mid;
            else
                lo = mid + 1;
        }
        y = lo < res ? lo : res - 1;
        if (y == 0) {
            row_pdf = rows[0];
            ry      = rows[0] > 0.0f ? ry / rows[0] : 0.0f;
        } else {
            const float prev = rows[y - 1];
            row_pdf          = rows[y] - prev;
            ry               = row_pdf > 0.0f ? (ry - prev) / row_pdf : 0.0f;
        }
    }
    int x = res - 1;
    {
        VK_GLOBAL const float* row = cols + (long long)y * (long long)res;
        int lo = 0, hi = res;
        while (lo < hi) {
            const int mid = (lo + hi) >> 1;
            if (rx < row[mid])
                hi = mid;
            else
                lo = mid + 1;
        }
        x = lo < res ? lo : res - 1;
        if (x == 0) {
            col_pdf = row[0];
            rx      = row[0] > 0.0f ? rx / row[0] : 0.0f;
        } else {
            const float prev = row[x - 1];
            col_pdf          = row[x] - prev;
            rx               = col_pdf > 0.0f ? (rx - prev) / col_pdf : 0.0f;
        }
    }
    // Keep the fraction in [0,1), like the CPU's std::min(scaled, 0.99999994).
    if (rx > 0.99999994f)
        rx = 0.99999994f;
    if (ry > 0.99999994f)
        ry = 0.99999994f;

    // Background::map: (x,y) from texel coordinates to a unit direction.
    const float u        = ((float)x + rx) * invres;
    const float v        = ((float)y + ry) * invres;
    const float theta    = u * 6.283185307179586f;
    const float cos_phi  = 1.0f - 2.0f * v;
    const float sin_phi2 = 1.0f - cos_phi * cos_phi;
    const float sin_phi  = sin_phi2 > 0.0f ? __builtin_sqrtf(sin_phi2) : 0.0f;
    float st = 0.0f, ct = 1.0f;
    mf_fast_sincos(theta, st, ct);
    dir = mk(sin_phi * ct, sin_phi * st, cos_phi);

    const float pd = row_pdf * col_pdf * p->bg_invjacobian;
    pdf            = pd > 0.0f ? pd : 0.0f;
    return mk(values[(long long)y * (long long)res + x].x,
              values[(long long)y * (long long)res + x].y,
              values[(long long)y * (long long)res + x].z);
}

// ---------------------------------------------------------------------------
// Ray generation / shading point setup
// ---------------------------------------------------------------------------

inline void fill_globals(VK_GLOBAL ShaderGlobals* sg, const SceneView& s,
                         VK_GLOBAL const FrameParams* p, const Ray& r,
                         const Hit& hit, VK_GLOBAL const unsigned char* frame_base)

{
    // Byte-wise zero: the shader may read fields the renderer does not set, and
    // stale bytes from the previous invocation would leak into its result.
    GlobalPtr raw = (GlobalPtr)sg;
    for (unsigned int i = 0; i < sizeof(ShaderGlobals); ++i)
        raw[i] = 0;

    TriangleIndices ti = load_tri(&s.tris[hit.id]);
    const Vec3 va      = load_vec(&s.verts[ti.a]);
    const Vec3 vb      = load_vec(&s.verts[ti.b]);
    const Vec3 vc      = load_vec(&s.verts[ti.c]);

    const float w0 = 1.0f - hit.u - hit.v;
    // `origin + direction*t`, the CPU's form (raytracer.h:75, `Ray::point`,
    // used by globals_from_hit in simpleraytracer.cpp).  Not the barycentric
    // recurrence: the CPU hands this point to the shader and starts every NEE
    // shadow ray from it, and a shadow ray that grazes the sphere it left is
    // decided by exactly these last bits, so the two tracers have to build the
    // point the same way or the glass silhouettes disagree at ulp level.
    Vec3 P         = add(r.origin, mul(r.direction, hit.t));
    // Same cross product as the CPU, in the same operand order
    // (raytracer.h:270: (va-vb) x (va-vc)).  The equivalent form,
    // (vb-va) x (vc-va), is not bit-identical: its z component comes out as
    // +0.0 where the CPU has -0.0, and pixar_frame() reads that sign bit.
    Vec3 Ng        = normalize(cross(sub(va, vb), sub(va, vc)));
    Vec3 N         = Ng;
    if (s.tri_n) {
        TriangleIndices ni = load_tri(&s.tri_n[hit.id]);
        if (ni.a >= 0) {
            Vec3 na = load_vec(&s.normals[ni.a]);
            Vec3 nb = load_vec(&s.normals[ni.b]);
            Vec3 nc = load_vec(&s.normals[ni.c]);
            N = normalize(add(add(mul(na, w0), mul(nb, hit.u)), mul(nc, hit.v)));
        }
    }

    store_vec(&sg->P, P);
    // The CPU flips N and Ng when the hit is backfacing
    // (simpleraytracer.cpp:946-950).  Skipping that flip leaves a whole
    // silhouette band shaded with an inward normal -- visible as stripes.
    const bool backface = dot(N, r.direction) > 0.0f;
    const Vec3 Ns       = backface ? mul(N, -1.0f) : N;
    store_vec(&sg->Ng, backface ? mul(Ng, -1.0f) : Ng);
    store_vec(&sg->N, Ns);
    store_vec(&sg->I, r.direction);
    sg->raytype        = 0;
    sg->backfacing     = backface ? 1 : 0;

    sg->time           = 0.0f;
    sg->dtime          = 0.0f;

    // u/v and their differentials: barycentric interpolation plus the
    // plane-projection construction scene.uv() uses on the CPU, so a textured
    // shader sees the same footprint.
    // u/v come from the per-corner uv attributes, exactly like
    // Scene::uv (raytracer.h:298): interpolate the corners, and derive
    // dPdu/dPdv from the uv parameterisation rather than from the triangle
    // edges -- a textured shader sees the same footprint either way, but a uv
    // shader ("uv" in render-uv) reads u/v directly.
    float uu   = hit.u;
    float vv   = hit.v;
    Vec3 dPdu  = sub(vb, va);
    Vec3 dPdv  = sub(vc, va);
    if (s.tri_uv) {
        TriangleIndices tuv = load_tri(&s.tri_uv[hit.id]);
        if (tuv.a >= 0 && s.uvs) {
            float tax = s.uvs[2 * tuv.a + 0], tay = s.uvs[2 * tuv.a + 1];
            float tbx = s.uvs[2 * tuv.b + 0], tby = s.uvs[2 * tuv.b + 1];
            float tcx = s.uvs[2 * tuv.c + 0], tcy = s.uvs[2 * tuv.c + 1];
            uu = (1.0f - hit.u - hit.v) * tax + hit.u * tbx + hit.v * tcx;
            vv = (1.0f - hit.u - hit.v) * tay + hit.u * tby + hit.v * tcy;
            // dt02 = ta - tc, dt12 = tb - tc, dp02 = va - vc, dp12 = vb - vc
            float u02x = tax - tcx, u02y = tay - tcy;
            float u12x = tbx - tcx, u12y = tby - tcy;
            Vec3 p02   = sub(va, vc);
            Vec3 p12   = sub(vb, vc);
            float det  = u02x * u12y - u02y * u12x;
            if (det != 0.0f) {
                float invdet = 1.0f / det;
                dPdu = mul(sub(mul(p02, u12y), mul(p12, u02y)), invdet);
                dPdv = mul(sub(mul(p12, u02x), mul(p02, u12x)), invdet);
            }
        }
    }

    sg->u    = uu;
    sg->v    = vv;
    store_vec(&sg->dPdu, dPdu);
    store_vec(&sg->dPdv, dPdv);
    // Screen-space footprint, built exactly like Ray::point(Dual2<float>) and
    // Scene::project do on the CPU (raytracer.h:89-98 and :284-296): the
    // derivatives of a ray point are the two directions perpendicular to the
    // ray scaled by (radius + spread*t), then slid along the view vector onto
    // the surface so they stay tangent to it.  calculatenormal() takes their
    // cross product, so the previous approximation (dPdu/width) tilted every
    // bump away from the reference image.
    Vec3 orthox, orthoy;
    ortho(r.direction, orthox, orthoy);
    const float rds = r.radius + r.spread * hit.t;
    Vec3 dPdx       = mul(orthox, rds);
    Vec3 dPdy       = mul(orthoy, rds);
    const Vec3 nI   = normalize(r.direction);
    const float cosI = -dot(nI, N);
    const float fcos = cosI < 0.0f ? -cosI : cosI;
    if (fcos > 1e-3f) {
        dPdx = add(dPdx, mul(nI, dot(dPdx, N) / cosI));
        dPdy = add(dPdy, mul(nI, dot(dPdy, N) / cosI));
    }
    store_vec(&sg->dPdx, dPdx);
    store_vec(&sg->dPdy, dPdy);
    store_vec(&sg->dPdz, mk(0.0f, 0.0f, 0.0f));
    // Same test the CPU runs after it has flipped N: with a negative
    // handedness `calculatenormal` crosses dy with dx instead of dx with dy
    // (llvm_ops.cpp:919-926), which turns the bumped normal inside out.
    sg->flipHandedness = dot(cross(dPdx, dPdy), Ns) < 0.0f ? 1 : 0;
    sg->dudx  = 1.0f / (float)p->width;
    sg->dudy  = 0.0f;
    sg->dvdx  = 0.0f;
    sg->dvdy  = 1.0f / (float)p->height;
    store_vec(&sg->dIdx, mul(orthox, r.spread));
    store_vec(&sg->dIdy, mul(orthoy, r.spread));
    // Surface area of the whole mesh this primitive belongs to.  emitter.osl
    // computes radiance as power / (M_PI * surfacearea()), so a zero here makes
    // every light in the scene emit nothing at all (and osl_safe_div_fff turns
    // the division into 0 rather than inf, which is why it fails silently).
    sg->surfacearea = ((VK_GLOBAL const float*)(frame_base + p->area_offset))[hit.id];
    sg->shade_index = 0;
    sg->thread_index = 0;
}


// ---------------------------------------------------------------------------
// Wavefront kernels
// ---------------------------------------------------------------------------
//
// The frame is traced in passes over structure-of-arrays buffers, one entry per
// pixel, dispatched in this order for every antialiasing sample:
//
//   osl_kernel_gen        jitter + camera ray            -> ray buffer
//   osl_kernel_intersect  BVH traversal                  ray -> hit buffer
//   osl_kernel_shade      shading, NEE, roulette, next ray
//
// and `intersect` / `shade` repeat once per bounce.  Separating the traversal
// from the shading is what makes it coherent: a launch walks the same node set
// for neighbouring rays instead of one invocation diving into its own path, and
// the shading pass no longer carries the traversal live ranges.
//
// Explicit light sampling and background NEE stay inside the shading pass.
// Their contributions are MIS-weighted against the bsdf pdf that same pass has
// just produced, and the CPU order (emission, NEE, roulette, bsdf sample) is
// reproduced exactly -- splitting them into their own wave would reorder the
// float additions and drift away from the reference image.
//
// Ray buffer contents, one array per component (`width*height` entries each):
// origin x/y/z, direction x/y/z, radius, spread, throughput r/g/b, the sampler
// state (seed, index), depth, the primitive the ray left, a liveness flag and
// the bsdf pdf that produced it.  Hit buffer: t, u, v, primitive id, hit flag.

static const long long kOffRenderstate = 216;
static const long long kOffCi          = 288;
static const long long kOffPool        = 320;
// Where the host puts the texture table inside the storage buffer; must match
// kTextureTableOffset in src/testrender/vulkan_shading.cpp.
static const long long kTexTableOffsetVK = 64 * 1024;

/// One ray-buffer entry, gathered from / scattered back to the SoA arrays.
struct WaveRay {
    Vec3 org, dir;
    float radius, spread;
    float thr_r, thr_g, thr_b;
    float bpdf;                 ///< bsdf pdf of the ray (inf for a delta)
    unsigned int seed, sindex;  ///< sampler state, advanced across bounces
    int depth;                  ///< bounce index
    int prev;                   ///< primitive the ray left (skipped by traversal)
    int active;                 ///< 0 once the path died
};

inline VK_GLOBAL float*
wf_fp(GlobalPtr base, long long off)
{
    return (VK_GLOBAL float*)(base + off);
}

inline VK_GLOBAL const float*
wf_fcp(GlobalPtr base, long long off)
{
    return (VK_GLOBAL const float*)(base + off);
}

inline VK_GLOBAL int*
wf_ip(GlobalPtr base, long long off)
{
    return (VK_GLOBAL int*)(base + off);
}

inline VK_GLOBAL const int*
wf_icp(GlobalPtr base, long long off)
{
    return (VK_GLOBAL const int*)(base + off);
}

inline void
wave_ray_load(VK_GLOBAL const FrameParams* p, GlobalPtr base,
              unsigned long long i, WaveRay& r)
{
    r.org.x  = wf_fcp(base, p->wf_org_x)[i];
    r.org.y  = wf_fcp(base, p->wf_org_y)[i];
    r.org.z  = wf_fcp(base, p->wf_org_z)[i];
    r.dir.x  = wf_fcp(base, p->wf_dir_x)[i];
    r.dir.y  = wf_fcp(base, p->wf_dir_y)[i];
    r.dir.z  = wf_fcp(base, p->wf_dir_z)[i];
    r.radius = wf_fcp(base, p->wf_radius)[i];
    r.spread = wf_fcp(base, p->wf_spread)[i];
    r.thr_r  = wf_fcp(base, p->wf_thr_r)[i];
    r.thr_g  = wf_fcp(base, p->wf_thr_g)[i];
    r.thr_b  = wf_fcp(base, p->wf_thr_b)[i];
    r.bpdf   = wf_fcp(base, p->wf_bpdf)[i];
    r.seed   = (unsigned int)wf_icp(base, p->wf_seed)[i];
    r.sindex = (unsigned int)wf_icp(base, p->wf_sindex)[i];
    r.depth  = wf_icp(base, p->wf_depth)[i];
    r.prev   = wf_icp(base, p->wf_prev)[i];
    r.active = wf_icp(base, p->wf_active)[i];
}

inline void
wave_ray_store(VK_GLOBAL const FrameParams* p, GlobalPtr base,
               unsigned long long i, const WaveRay& r)
{
    wf_fp(base, p->wf_org_x)[i]  = r.org.x;
    wf_fp(base, p->wf_org_y)[i]  = r.org.y;
    wf_fp(base, p->wf_org_z)[i]  = r.org.z;
    wf_fp(base, p->wf_dir_x)[i]  = r.dir.x;
    wf_fp(base, p->wf_dir_y)[i]  = r.dir.y;
    wf_fp(base, p->wf_dir_z)[i]  = r.dir.z;
    wf_fp(base, p->wf_radius)[i] = r.radius;
    wf_fp(base, p->wf_spread)[i] = r.spread;
    wf_fp(base, p->wf_thr_r)[i]  = r.thr_r;
    wf_fp(base, p->wf_thr_g)[i]  = r.thr_g;
    wf_fp(base, p->wf_thr_b)[i]  = r.thr_b;
    wf_fp(base, p->wf_bpdf)[i]   = r.bpdf;
    wf_ip(base, p->wf_seed)[i]   = (int)r.seed;
    wf_ip(base, p->wf_sindex)[i] = (int)r.sindex;
    wf_ip(base, p->wf_depth)[i]  = r.depth;
    wf_ip(base, p->wf_prev)[i]   = r.prev;
    wf_ip(base, p->wf_active)[i] = r.active;
}

inline void
wave_hit_store(VK_GLOBAL const FrameParams* p, GlobalPtr base,
               unsigned long long i, float t, float u, float v, int id, int ok)
{
    wf_fp(base, p->wf_hit_t)[i]  = t;
    wf_fp(base, p->wf_hit_u)[i]  = u;
    wf_fp(base, p->wf_hit_v)[i]  = v;
    wf_ip(base, p->wf_hit_id)[i] = id;
    wf_ip(base, p->wf_hit_ok)[i] = ok;
}

inline void
wave_hit_load(VK_GLOBAL const FrameParams* p, GlobalPtr base,
              unsigned long long i, float& t, float& u, float& v, int& id,
              int& ok)
{
    t  = wf_fcp(base, p->wf_hit_t)[i];
    u  = wf_fcp(base, p->wf_hit_u)[i];
    v  = wf_fcp(base, p->wf_hit_v)[i];
    id = wf_icp(base, p->wf_hit_id)[i];
    ok = wf_icp(base, p->wf_hit_ok)[i];
}

/// The scene view every pass that touches geometry needs: eleven pointers into
/// the storage buffer plus the three counts.  Same construction the megakernel
/// used, so the same pointers clspv can attribute to the buffer argument.
/// The scene view every pass that touches geometry needs.  `idx` is the pixel
/// the invocation works on: it is plain `gid` in the generation and shading
/// passes, but the intersection pass dispatches three slots per pixel (the path
/// ray plus the two shadow rays), so there it is `gid % npix`.
#define WAVE_SCENE_SETUP(scene, idx)                                          \
    scene.tris          = (GlobalTri)(base + p->tri_offset);                  \
    scene.verts         = (GlobalVec3)(base + p->vert_offset);                \
    scene.normals       = (GlobalVec3)(base + p->normal_offset);              \
    scene.tri_n         = (GlobalCTri)(base + p->tri_n_offset);               \
    scene.ntris         = p->num_triangles;                                   \
    scene.lightprims    = (GlobalInt)(base + p->lightprim_offset);            \
    scene.nlightprims   = p->num_lightprims;                                  \
    scene.uvs           = (VK_GLOBAL const float*)(base + p->uv_offset);      \
    scene.tri_uv        = (GlobalCTri)(base + p->tri_uv_offset);              \
    scene.nodes         = (VK_GLOBAL const BvhNodeVK*)(base + p->bvh_offset); \
    scene.bvh_idx       = (VK_GLOBAL const unsigned*)(base                        \
                                + p->bvh_index_offset);                       \
    scene.num_bvh_nodes = p->num_bvh_nodes;                                   \
    scene.stack         = (base + p->wf_stack)                                \
                  + (idx) * (unsigned long long)kTraversalStackStride;


// ---- pass 1: ray generation ------------------------------------------------

extern "C" void osl_vk_wave_gen(GlobalPtr base)
{
    VK_GLOBAL const FrameParams* p
        = (VK_GLOBAL const FrameParams*)(base + kFrameParamsOffset);

    const unsigned long long gid = get_global_id(0);
    const unsigned long long npix
        = (unsigned long long)p->width * (unsigned long long)p->height;
    if (gid >= npix)
        return;

    Camera cam;
    cam.setup(p);

    const int x = (int)(gid % (unsigned long long)p->width);
    const int y = (int)(gid / (unsigned long long)p->width);
    const int si = p->sample_index >= 0 ? p->sample_index : 0;

    Sampler sampler;
    sampler.init(x, y, si);
    Vec3 j = sampler.get();
    // Tent filter warp, identical to antialias_pixel() on the CPU.
    j.x *= 2.0f;
    j.x = j.x < 1.0f ? __builtin_sqrtf(j.x) - 1.0f
                     : 1.0f - __builtin_sqrtf(2.0f - j.x);
    j.y *= 2.0f;
    j.y = j.y < 1.0f ? __builtin_sqrtf(j.y) - 1.0f
                     : 1.0f - __builtin_sqrtf(2.0f - j.y);

    Vec3 dirv = normalize(add(add(mul(cam.cx, ((float)x + 0.5f + j.x) * cam.invw - 0.5f),
                                 mul(cam.cy, 0.5f - ((float)y + 0.5f + j.y) * cam.invh)),
                             cam.dir));
    float cos_a = dot(cam.dir, dirv);
    float spread
        = __builtin_sqrtf(cam.invw * cam.invh * length(cam.cx) * length(cam.cy)
                          * cos_a)
          * cos_a;

    WaveRay r;
    r.org    = cam.eye;
    r.dir    = dirv;
    r.radius = 0.0f;
    r.spread = spread;
    r.thr_r  = 1.0f;
    r.thr_g  = 1.0f;
    r.thr_b  = 1.0f;
    r.bpdf   = inf();  // a camera ray has only one possible direction
    // The sampler state is parked *after* the jitter draw, so the shading pass's
    // first draw is the same one the CPU takes for the first bounce.
    r.seed   = sampler.seed;
    r.sindex = sampler.index;
    r.depth  = 0;
    r.prev   = -1;
    r.active = 1;
    wave_ray_store(p, base, gid, r);

    // No NEE shadow ray is outstanding when a sample starts.
    wf_ip(base, p->wf_nee_pending)[gid] = 0;

    // Accumulation target for this sample: the host sums one image per sample,
    // exactly as it did for the megakernel.
    VK_GLOBAL float* out = wf_fp(base, p->out_offset);
    out[gid * 4 + 0]     = 0.0f;
    out[gid * 4 + 1]     = 0.0f;
    out[gid * 4 + 2]     = 0.0f;
    out[gid * 4 + 3]     = 1.0f;
}


// ---- pass 2: intersection --------------------------------------------------
//
// Three slots per pixel, one dispatch: slot 0 is the path ray of this wave,
// slots 1 and 2 are the background and light NEE shadow rays the previous
// shading wave asked for.  Keeping them in one kernel is what makes the split
// pay: the BVH walk is inlined exactly once, instead of once here plus twice
// inside the shading pass.

extern "C" void osl_vk_wave_intersect(GlobalPtr base)
{
    VK_GLOBAL const FrameParams* p
        = (VK_GLOBAL const FrameParams*)(base + kFrameParamsOffset);

    // 32-bit indices on purpose: an i64 division makes clspv expand a software
    // divide into the middle of an already large kernel, which costs minutes of
    // compile time.  Three slots of a 1k x 1k image fit in 32 bits easily.
    const unsigned gid = (unsigned)get_global_id(0);
    const unsigned npix
        = (unsigned)p->width * (unsigned)p->height;
    if (npix == 0)
        return;
    const unsigned slot = gid / npix;
    if (slot > 2)
        return;
    const unsigned pix = gid - slot * npix;

    const bool is_path = (slot == 0);

    // Gather the ray for whichever slot this invocation belongs to.  The two
    // kinds differ only in where the ray comes from and in the distance cutoff;
    // the walk itself is shared below, so it exists once in the module.
    WaveRay wr;
    wave_ray_load(p, base, pix, wr);

    Ray ray;
    ray.origin = wr.org;
    ray.direction = wr.dir;
    ray.radius = wr.radius;
    ray.spread = wr.spread;
    // A dead path has nothing to hit, so a negative cutoff makes the walk
    // return a miss.  The shadow slots must not inherit that: the path they
    // belong to can die after publishing its NEE request, and a sky shadow ray
    // has no distance cutoff at all.
    float tmax = is_path && !wr.active ? -1.0f : inf();
    int skip_a = wr.prev;
    int skip_b = -1;

    VK_GLOBAL int* vis = wf_ip(base, (slot == 1) ? p->wf_vis_bg : p->wf_vis_l);
    if (!is_path) {
        const int bit     = (slot == 1) ? 1 : 2;
        const bool wanted = (wf_icp(base, p->wf_nee_pending)[pix] & bit) != 0;
        vis[pix]          = 0;
        ray.origin = mk(wf_fcp(base, p->wf_p_x)[pix],
                        wf_fcp(base, p->wf_p_y)[pix],
                        wf_fcp(base, p->wf_p_z)[pix]);
        ray.radius = 0.0f;
        ray.spread = 0.0f;
        skip_a     = wf_icp(base, p->wf_s_skip1)[pix];
        if (slot == 1) {
            ray.direction = mk(wf_fcp(base, p->wf_bgdir_x)[pix],
                               wf_fcp(base, p->wf_bgdir_y)[pix],
                               wf_fcp(base, p->wf_bgdir_z)[pix]);
            // `tmax` is already inf(), the sky is at infinity.
        } else {
            ray.direction = mk(wf_fcp(base, p->wf_ldir_x)[pix],
                               wf_fcp(base, p->wf_ldir_y)[pix],
                               wf_fcp(base, p->wf_ldir_z)[pix]);
            tmax          = wanted ? wf_fcp(base, p->wf_l_tmax)[pix] : -1.0f;
            skip_b        = wf_icp(base, p->wf_l_lid)[pix];
        }
        if (!wanted)
            tmax = -1.0f;  // nothing was requested, so do not walk at all
    }

    // One traversal stack per pixel *and per slot*: the three slots of a pixel
    // are separate work items that the device may run concurrently, so sharing
    // the walk stack between them would corrupt both.
    SceneView scene;
    WAVE_SCENE_SETUP(scene, pix * 3 + slot)

    Hit h = intersect_scene(scene, ray, tmax, skip_a, skip_b);
    if (is_path) {
        wave_hit_store(p, base, pix, h.t, h.u, h.v, h.id, h.id >= 0 ? 1 : 0);
    } else {
        vis[pix] = (h.id < 0) ? 1 : 0;
    }
}


// ---- pass 3: shading -------------------------------------------------------

/// `MAT` selects the material's shader code, `CS` the closure families it can
/// produce.  Everything below that is gated on a closure id is either
/// `if constexpr (CS & ...)` or a runtime flag that is only ever set inside
/// such a block, so a family the material cannot emit costs no code.
template <int MAT, unsigned CS>
void wave_shade_body(GlobalPtr base, unsigned pix)
{
    constexpr bool HAS_MF   = (CS & CS_MF) != 0;
    constexpr bool HAS_WARD = (CS & CS_WARD) != 0;
    constexpr bool HAS_DIFF = (CS & CS_DIFF) != 0;
    constexpr bool HAS_DLTA = (CS & CS_DELTA) != 0;
    constexpr bool HAS_PHONG = (CS & CS_PHONG) != 0;
    constexpr bool HAS_ON    = (CS & CS_ON) != 0;
    constexpr bool HAS_MXC   = (CS & CS_MXC) != 0;
    constexpr bool HAS_MXD   = (CS & CS_MXD) != 0;

    VK_GLOBAL const FrameParams* p
        = (VK_GLOBAL const FrameParams*)(base + kFrameParamsOffset);

    WaveRay r;
    wave_ray_load(p, base, pix, r);

    float h_t, h_u, h_v;
    int h_id, h_ok;
    wave_hit_load(p, base, pix, h_t, h_u, h_v, h_id, h_ok);

    SceneView scene;
    WAVE_SCENE_SETUP(scene, pix)

    GlobalInt tri_shader = (GlobalInt)(base + p->tri_shader_offset);
    GlobalPtr arena      = (GlobalPtr)(base + p->arena_offset)
                           + pix * (unsigned long long)p->arena_stride;
    VK_GLOBAL ShaderGlobals* sg = (VK_GLOBAL ShaderGlobals*)arena;

    // Two facts the device runtime cannot derive from `sg` alone: where this
    // invocation's block sits in the storage buffer, and where the texture
    // table lives.  They ride in the first 16 bytes of the closure pool, which
    // the allocator never hands out (it starts at offset 16).
    const long long sg_abs = (long long)(p->arena_offset
                                         + pix * (unsigned long long)p->arena_stride);
    *(GlobalI64)((GlobalPtr)sg + kOffPool)     = sg_abs;
    *(GlobalI64)((GlobalPtr)sg + kOffPool + 8) = kTexTableOffsetVK;

    const int nsamp = p->sample_count > 0 ? p->sample_count : p->aa * p->aa;
    const float inv = 1.0f / (float)(nsamp > 0 ? nsamp : 1);
    VK_GLOBAL float* out = wf_fp(base, p->out_offset);

    // ---- settle the NEE shadow rays of the previous wave -------------------
    // The shading pass does not traverse the BVH: it published these rays into
    // the buffer and the intersection pass traced them.  Adding them here, at
    // the top of the next wave, is the same arithmetic the megakernel did in
    // place -- only the point in the wave at which it lands has moved.
    const int pending = wf_icp(base, p->wf_nee_pending)[pix];
    if (pending) {
        if ((pending & 1) && wf_icp(base, p->wf_vis_bg)[pix]) {
            out[pix * 4 + 0] += wf_fcp(base, p->wf_bgc_r)[pix] * inv;
            out[pix * 4 + 1] += wf_fcp(base, p->wf_bgc_g)[pix] * inv;
            out[pix * 4 + 2] += wf_fcp(base, p->wf_bgc_b)[pix] * inv;
        }
        if ((pending & 2) && wf_icp(base, p->wf_vis_l)[pix]) {
            // Nothing between the surface and the light: shade the point on the
            // light and multiply in its emission, as the megakernel did inline.
            const int lid           = wf_icp(base, p->wf_l_lid)[pix];
            const int lightShaderID = tri_shader[lid];
            if (lightShaderID >= 0) {
                Ray sray;
                sray.origin
                    = mk(wf_fcp(base, p->wf_p_x)[pix],
                         wf_fcp(base, p->wf_p_y)[pix],
                         wf_fcp(base, p->wf_p_z)[pix]);
                sray.direction = mk(wf_fcp(base, p->wf_ldir_x)[pix],
                                    wf_fcp(base, p->wf_ldir_y)[pix],
                                    wf_fcp(base, p->wf_ldir_z)[pix]);
                sray.radius    = 0.0f;
                sray.spread    = 0.0f;
                Hit lhit;
                lhit.id = lid;
                lhit.u  = wf_fcp(base, p->wf_l_u)[pix];
                lhit.v  = wf_fcp(base, p->wf_l_v)[pix];
                lhit.t  = wf_fcp(base, p->wf_l_tmax)[pix];
                fill_globals(sg, scene, p, sray, lhit,
                             (VK_GLOBAL const unsigned char*)base);
                *(GlobalI64)((GlobalPtr)sg + kOffRenderstate) = 16;
                *(GlobalI64)((GlobalPtr)sg + kOffCi)          = -1;
                wave_shade_dispatch<MAT>((GlobalPtr)sg, lightShaderID);
                long long lci = *(GlobalI64)((GlobalPtr)sg + kOffCi);
                if (lci >= 0) {
                    const VK_GLOBAL ClosureComponentVK* lcomp
                        = (VK_GLOBAL const ClosureComponentVK*)(arena + kOffPool
                                                                + lci);
                    if (lcomp->id == EMISSION_ID_VK
                        || lcomp->id == BACKGROUND_ID_VK) {
                        // Scalar loads: a Vec3 lvalue living in AS1 cannot bind
                        // to a reference.
                        Vec3 lw = mk(lcomp->w.x, lcomp->w.y, lcomp->w.z);
                        out[pix * 4 + 0]
                            += wf_fcp(base, p->wf_lc_r)[pix] * lw.x * inv;
                        out[pix * 4 + 1]
                            += wf_fcp(base, p->wf_lc_g)[pix] * lw.y * inv;
                        out[pix * 4 + 2]
                            += wf_fcp(base, p->wf_lc_b)[pix] * lw.z * inv;
                    }
                }
                // The light shading overwrote the block the surface's closure
                // tree lived in, but that tree is finished with: everything
                // after this pass works off the ray buffer, not `sg`.
            }
        }
        // Consumed.  The next wave publishes a fresh pair.
        wf_ip(base, p->wf_nee_pending)[pix] = 0;
    }

    if (!r.active)
        return;

    // Resume the per-sample sampler where the generation pass left it.
    Sampler sampler;
    sampler.seed  = r.seed;
    sampler.index = r.sindex;

    Ray ray;
    ray.origin    = r.org;
    ray.direction = r.dir;
    ray.radius    = r.radius;
    ray.spread    = r.spread;

    Vec3 w         = mk(r.thr_r, r.thr_g, r.thr_b);
    float bsdf_pdf = r.bpdf;
    const int b    = r.depth;

    Vec3 L = mk(0.0f, 0.0f, 0.0f);

    Hit hit;
    hit.t  = h_t;
    hit.u  = h_u;
    hit.v  = h_v;
    hit.id = h_ok ? h_id : -1;

    // The path is dead unless this pass produces a live next ray.
    r.active = 0;

    if (hit.id < 0) {
        if (p->bg_shader_id < 0) {
            // No background shader: the scene's flat colour, which is the
            // value the CPU path falls back to as well.
            L = add(L, mulv(w, mk(p->bg_color[0], p->bg_color[1],
                                  p->bg_color[2])));
        } else if (b > 0 && p->bg_res > 0) {
            // Indirect ray: the CPU does NOT run the shader here.  It looks the
            // direction up in the importance table and multiplies by the MIS
            // weight against the bsdf pdf.
            float bg_pdf = 0.0f;
            Vec3 bg      = background_table_eval(p, base, ray.direction, bg_pdf);
            const float k = power_heuristic_weight_weight(bsdf_pdf, bg_pdf);
            L = add(L, mulv(w, mk(bg.x * k, bg.y * k, bg.z * k)));
        } else {
            // Primary ray: the shader itself, exactly like the CPU's
            // eval_background().  The globals are zero except for I and
            // raytype, and the arena cursor / root are reset first because this
            // shares the invocation's ShadeBlock with any surface shading.
            GlobalPtr raw = (GlobalPtr)sg;
            for (unsigned int i = 0; i < sizeof(ShaderGlobals); ++i)
                raw[i] = 0;
            store_vec(&sg->I, ray.direction);
            // Ray::CAMERA = 1, Ray::DIFFUSE = 16 (raytracer.h).
            sg->raytype                                   = (b > 0) ? 16 : 1;
            *(GlobalI64)((GlobalPtr)sg + kOffRenderstate) = 16;
            *(GlobalI64)((GlobalPtr)sg + kOffCi)          = -1;
            wave_shade_dispatch<MAT>((GlobalPtr)sg, p->bg_shader_id);
            const Vec3 Le
                = background_closure_color(sg, *(GlobalI64)((GlobalPtr)sg + kOffCi));
            L = add(L, mulv(w, Le));
        }
    } else {
        fill_globals(sg, scene, p, ray, hit,
                     (VK_GLOBAL const unsigned char*)base);

        const int shaderID = tri_shader[hit.id];
        if (shaderID >= 0) {
            // Which kind of ray produced this hit, as `raytype("...")` sees it.
            // The CPU sets `sg.raytype = r.raytype` (simpleraytracer.cpp:961),
            // and the rays it makes are exactly two kinds: Ray::CAMERA for the
            // primary ray (raytracer.h:161) and Ray::DIFFUSE for every ray a
            // scatter or NEE produces (simpleraytracer.cpp:1210, "FIXME? Use
            // DIFFUSE for all indirect rays").  Leaving 0 here -- the value
            // fill_globals parks -- makes *every* query answer false, so a
            // shader that branches on raytype() takes the indirect branch even
            // on the primary hit.
            sg->raytype = (b > 0) ? 16 : 1;
            // Reset the per-sample arena state: cursor at 16 (offset 0 is the
            // two int64s parked at the head of the pool), root sentinel -1.
            *(GlobalI64)((GlobalPtr)sg + kOffRenderstate) = 16;
            *(GlobalI64)((GlobalPtr)sg + kOffCi)          = -1;

            wave_shade_dispatch<MAT>((GlobalPtr)sg, shaderID);

            float er = 0.0f, eg = 0.0f, eb = 0.0f;  // emitted
            float ar = 0.0f, ag = 0.0f, ab = 0.0f;  // albedo / closure weight
            int is_mirror = 0;
            int is_ward   = 0;
            // All the closures this material produced are delta closures
            // (mirror / refraction / transparent).  Then the BSDF has exactly
            // one outgoing direction per component, the pdf is infinite, and
            // NEE has nothing to evaluate toward.
            int all_delta = 0;
            // At least one component is a microfacet lobe: then NEE and the
            // BSDF sample have to go through component_eval / mf_sample_vk
            // instead of the Lambert shortcut below.
            int has_microfacet = 0;
            // At least one component is a phong lobe: same story as the
            // microfacet flag -- the Lambert shortcut below would otherwise
            // shade a glossy plate as a diffuse one (uniform instead of a
            // mirror-direction highlight).
            int has_phong = 0;
            // At least one component is an Oren-Nayar lobe: its eval is not
            // constant in the two directions, so it needs component_eval on
            // the NEE side and its own sample weight on the BSDF side.
            int has_on = 0;
            // At least one component is a MaterialX conductor: GGX with a
            // complex Fresnel term, which is neither a delta lobe nor a
            // Lambert one.
            int has_mxc = 0;
            // At least one component is a MaterialX dielectric (reflection only,
            // i.e. the layer has no transmission tint).
            int has_mxd = 0;
            WardVK wardp;
            long long ci = *(GlobalI64)((GlobalPtr)sg + kOffCi);
            // Flatten ADD/MUL: glass is `Ks*reflection(N,eta) +
            // Cs*refraction(N,eta)`, i.e. two components.
            //
            // The two slots are one array picked by an *index*, not two
            // variables picked by a `select` on their addresses.  A select
            // between two locals is an `OpSelect` on a Function pointer once
            // clspv lowers it (=> `CapabilityVariablePointers`, which Intel's
            // driver aborts on), and the merged local array this front end used
            // to build out of that pair instead makes clspv's
            // SimplifyPointerBitcastPass oscillate between `[2 x
            // ClosureEntryVK]`, `[16 x i64]` and `[32 x float]` views of the
            // same storage: it adds one more GEP every round and never
            // terminates (measured).  Indexing an array clang laid out itself
            // keeps every access an `OpAccessChain` on the array's own type.
            ClosureEntryVK ce[2];
            ce[0].id  = -1;
            ce[1].id  = -1;
            ce[0].off = 0;
            ce[1].off = 0;
            int ncomp   = collect_closure_entries(sg, p, ci,
                                                  mul (ray.direction, -1.0f),
                                                  ce[0], ce[1]);
            Vec3 alb0   = mk(0.0f, 0.0f, 0.0f);
            Vec3 alb1   = mk(0.0f, 0.0f, 0.0f);
            if (ncomp > 0) {
                Vec3 wov0 = mul(ray.direction, -1.0f);
                all_delta = ncomp;
                // Two constant-slot visits instead of a loop over the pair:
                // `(k == 0) ? ce0 : ce1` is a `select` between two distinct
                // objects, which clspv answers with an OpSelect on a Function
                // pointer (pointer arithmetic, i.e. the capability Intel's
                // compiler aborts on), and the merged local array it becomes
                // afterwards is what makes clspv diverge on the microfacet
                // kernel: a dynamic index into a private array of structs
                // inside a loop of that size never comes back.
                auto note_component = [&](const ClosureEntryVK& e,
                                          Vec3& alb_out) {
                    if (e.id == EMISSION_ID_VK || e.id == BACKGROUND_ID_VK) {
                        er += e.w.x;
                        eg += e.w.y;
                        eb += e.w.z;
                        all_delta = 0;  // an emitter is not a delta BSDF
                        return;
                    }
                    Vec3 alb = closure_entry_albedo<CS>(p, e, wov0);
                    alb_out  = alb;
                    ar += alb.x;
                    ag += alb.y;
                    ab += alb.z;
                    if constexpr (HAS_WARD) {
                        if (e.id == WARD_ID_VK) {
                            // shading.cpp:73 WardParams { Vec3 N; Vec3 T;
                            // float ax; float ay; } -- 32 bytes right after the
                            // 16-byte header: N at +16, T at +28, ax at +40,
                            // ay at +44.
                            GlobalCPtr wp = (GlobalCPtr)sg + kOffPoolClosure
                                            + e.off + 16;
                            wardp.N  = load_vec((GlobalCVec3)wp);
                            wardp.T  = load_vec((GlobalCVec3)(wp + 12));
                            wardp.ax = *(GlobalFloatCPtr)(wp + 24);
                            wardp.ay = *(GlobalFloatCPtr)(wp + 28);
                            is_ward   = 1;
                            all_delta = 0;
                        }
                    }
                    if constexpr (HAS_DIFF) {
                        if (e.id == DIFFUSE_ID_VK)
                            all_delta = 0;
                    }
                    if constexpr (HAS_MF) {
                        if (e.id == MICROFACET_ID_VK) {
                            // A microfacet lobe is glossy, not delta, and it
                            // needs its own eval/sample path instead of the
                            // cosine one.
                            has_microfacet = 1;
                            all_delta      = 0;
                        }
                    }
                    if constexpr (HAS_PHONG) {
                        if (e.id == PHONG_ID_VK) {
                            // Same for phong: glossy, its own eval/sample.
                            has_phong = 1;
                            all_delta = 0;
                        }
                    }
                    if constexpr (HAS_MXC) {
                        if (e.id == MX_CONDUCTOR_ID_VK) {
                            has_mxc   = 1;
                            all_delta = 0;
                        }
                    }
                    if constexpr (HAS_MXD) {
                        // Reflection-only dielectric components (no transmission
                        // tint) are sampled and evaluated like the conductor,
                        // just with a scalar Fresnel.  A transmissive one
                        // (`Ct != 0`) keeps the old stand-in: `all_delta` stays
                        // cleared because it is not a delta lobe either way.
                        if (e.id == MX_DIELECTRIC_ID_VK) {
                            if (!(e.refr_tint.x > 0.0f) && !(e.refr_tint.y > 0.0f)
                                && !(e.refr_tint.z > 0.0f))
                                has_mxd = 1;
                            all_delta = 0;
                        }
                    }
                    if constexpr (HAS_ON) {
                        if (e.id == OREN_NAYAR_ID_VK) {
                            // Rough diffuse: cosine sampling, but the weight
                            // carries f(sigma) instead of a flat albedo.
                            has_on    = 1;
                            all_delta = 0;
                        }
                    }
                    if constexpr (HAS_DLTA) {
                        // Anything in this kernel that is not one of the
                        // families above: only the delta closures are left.
                        if (e.id != WARD_ID_VK && e.id != DIFFUSE_ID_VK
                            && e.id != MICROFACET_ID_VK
                            && !closure_entry_is_delta(e))
                            all_delta = 0;
                    } else {
                        // No delta closure exists in this kernel, so every
                        // non-emissive component is glossy by construction.
                        all_delta = 0;
                    }
                };
                if (ncomp > 0)
                    note_component (ce[0], alb0);
                if (ncomp > 1)
                    note_component (ce[1], alb1);
                if constexpr (HAS_DLTA) {
                    if (ncomp == 1
                        && (ce[0].id == REFLECTION_ID_VK
                            || ce[0].id == FRESNEL_REFLECTION_ID_VK))
                        is_mirror = 1;
                }
                if (all_delta > 0)
                    all_delta = 1;
            }

            // CompositeBSDF::prepare's lobe pick probabilities are recomputed
            // per bounce, below, from the path weight as it stands at that
            // bounce -- exactly where the CPU calls prepare()
            // (simpleraytracer.cpp:1155).

            // One random triple per bounce, exactly where the CPU takes it
            // (simpleraytracer.cpp:1103): NEE and the BSDF sample both use it.
            Vec3 s3 = sampler.get();
            float xsv = s3.x, ysv = s3.y;

            // Is the surface we hit one of the light primitives?
            int lid_of_hit = -1;
            for (int i = 0; i < scene.nlightprims; ++i) {
                if (scene.lightprims[i] == hit.id) {
                    lid_of_hit = i;
                    break;
                }
            }
            if (lid_of_hit >= 0) {
                // Self-emission, MIS-weighted against the light sampling that
                // could also have found this point.
                float light_pdf = (1.0f / (float)scene.nlightprims)
                                  * shape_pdf(scene, hit.id, ray.origin,
                                              load_vec(&sg->P));
                float k = power_heuristic_weight_eval(bsdf_pdf, light_pdf);
                L = add(L, mulv(w, mk(er * k, eg * k, eb * k)));
            } else {
                L = add(L, mulv(w, mk(er, eg, eb)));
            }

            Vec3 P = load_vec(&sg->P);
            Vec3 n = load_vec(&sg->N);
            if (dot(n, ray.direction) > 0.0f)
                n = mul(n, -1.0f);
            Vec3 wov = mul(ray.direction, -1.0f);

            do {
                if (b == p->max_bounces)
                    break;
                if (!(ar > 0.0f) && !(ag > 0.0f) && !(ab > 0.0f))
                    break;

                // CompositeBSDF::prepare's lobe pick probabilities
                // (shading.h:322-354): pdfs[i] = weights[i] . (path_weight *
                // albedo_i) / sum(path_weight), normalised by their total the way
                // prepare() does.  The NEE evaluation of a multi-lobe mixture
                // needs them: CompositeBSDF::eval folds each lobe in with
                // MIS::update_eval using its own pick probability, so a plain
                // sum of the lobes gives a different (weight, pdf) pair --
                // visible as a systematic energy difference on glossy two-lobe
                // materials.
                float pk0 = 0.0f, pk1 = 0.0f;
                {
                    const float msum = w.x + w.y + w.z;
                    if (msum > 0.0f && ncomp > 0) {
                        pk0 = (alb0.x * w.x + alb0.y * w.y + alb0.z * w.z) / msum;
                        pk1 = (alb1.x * w.x + alb1.y * w.y + alb1.z * w.z) / msum;
                    }
                    const float ptot = pk0 + pk1;
                    if (ptot > 0.0f) {
                        pk0 /= ptot;
                        pk1 /= ptot;
                    }
                }

                // ---- explicit background sampling (NEE) ---------------------
                // Mirrors simpleraytracer.cpp:1137-1155: sample a direction
                // from the background importance table, evaluate the closure
                // toward it, MIS-weight it against the closure's own pdf, and
                // shoot a shadow ray.
                if (p->bg_res > 0 && !all_delta) {
                    Vec3 bg_dir;
                    float bg_pdf = 0.0f;
                    Vec3 bgv
                        = background_sample(p, base, xsv, ysv, bg_dir, bg_pdf);
                    float bpdf = 0.0f;
                    Vec3 bwt3  = mk(0.0f, 0.0f, 0.0f);
                    nee_eval_vk<CS>(p, ce[0], ce[1], ncomp, has_microfacet, is_ward,
                                    has_phong, has_on, has_mxc, has_mxd, wardp,
                                    wov, bg_dir, n, ar, ag, ab, pk0, pk1, bwt3,
                                    bpdf);
                    const float mis = power_heuristic_weight_weight(bg_pdf, bpdf);
                    Vec3 contrib = mulv(mulv(w, bwt3),
                                        mk(bgv.x * mis, bgv.y * mis,
                                           bgv.z * mis));
                    if (contrib.x + contrib.y + contrib.z > 0.0f) {
                        // Publish the shadow ray instead of tracing it: the
                        // next intersection wave walks it, the next shading
                        // wave adds `contrib` if it got through.
                        wf_fp(base, p->wf_p_x)[pix]      = P.x;
                        wf_fp(base, p->wf_p_y)[pix]      = P.y;
                        wf_fp(base, p->wf_p_z)[pix]      = P.z;
                        wf_ip(base, p->wf_s_skip1)[pix] = hit.id;
                        wf_fp(base, p->wf_bgdir_x)[pix]  = bg_dir.x;
                        wf_fp(base, p->wf_bgdir_y)[pix]  = bg_dir.y;
                        wf_fp(base, p->wf_bgdir_z)[pix]  = bg_dir.z;
                        wf_fp(base, p->wf_bgc_r)[pix]    = contrib.x;
                        wf_fp(base, p->wf_bgc_g)[pix]    = contrib.y;
                        wf_fp(base, p->wf_bgc_b)[pix]    = contrib.z;
                        wf_ip(base, p->wf_nee_pending)[pix] |= 1;
                    }
                }

                // ---- explicit light sampling (NEE) -------------------------
                // Mirrors simpleraytracer.cpp:1129-1187: pick a light
                // primitive uniformly, sample a point on it, evaluate the
                // (Lambert) BSDF toward it, MIS-weight, shoot a shadow ray, and
                // shade the light primitive to get its emitted radiance.
                if (scene.nlightprims > 0 && !all_delta) {
                    const float light_pick_pdf = 1.0f / (float)scene.nlightprims;
                    float xl                   = xsv * (float)scene.nlightprims;
                    int ls                     = (int)__builtin_floorf(xl);
                    xl -= (float)ls;
                    int lid = scene.lightprims[ls];
                    if (lid != hit.id) {
                        LightSampleVK lsmp = light_sample(scene, lid, P, xl, ysv);
                        // CPU: b = result.bsdf.eval(-sg.I, sample.dir), so both
                        // the weight and the pdf come from the closure itself.
                        float bpdf = 0.0f;
                        Vec3 bwt3  = mk(0.0f, 0.0f, 0.0f);
                        nee_eval_vk<CS>(p, ce[0], ce[1], ncomp, has_microfacet, is_ward,
                                        has_phong, has_on, has_mxc, has_mxd,
                                        wardp, wov, lsmp.dir, n, ar, ag, ab, pk0,
                                        pk1, bwt3, bpdf);
                        float mis = power_heuristic_eval_weight(
                            light_pick_pdf * lsmp.pdf, bpdf);
                        Vec3 contrib
                            = mulv(mulv(w, bwt3), mk(mis, mis, mis));
                        if (contrib.x + contrib.y + contrib.z > 0.0f) {
                            // Same hand-off as the background ray above; the
                            // light point is shaded when the result comes back,
                            // because that needs `sg` and the shader dispatch.
                            wf_fp(base, p->wf_p_x)[pix]      = P.x;
                            wf_fp(base, p->wf_p_y)[pix]      = P.y;
                            wf_fp(base, p->wf_p_z)[pix]      = P.z;
                            wf_ip(base, p->wf_s_skip1)[pix] = hit.id;
                            wf_fp(base, p->wf_ldir_x)[pix]   = lsmp.dir.x;
                            wf_fp(base, p->wf_ldir_y)[pix]   = lsmp.dir.y;
                            wf_fp(base, p->wf_ldir_z)[pix]   = lsmp.dir.z;
                            wf_fp(base, p->wf_l_tmax)[pix]   = lsmp.dist;
                            wf_fp(base, p->wf_l_u)[pix]      = lsmp.u;
                            wf_fp(base, p->wf_l_v)[pix]      = lsmp.v;
                            wf_ip(base, p->wf_l_lid)[pix]    = lid;
                            wf_fp(base, p->wf_lc_r)[pix]     = contrib.x;
                            wf_fp(base, p->wf_lc_g)[pix]     = contrib.y;
                            wf_fp(base, p->wf_lc_b)[pix]     = contrib.z;
                            wf_ip(base, p->wf_nee_pending)[pix] |= 2;
                        }
                    }
                }

                // ---- Russian roulette (shading.h:356-386, :411-438) ---------
                float rx = s3.x;
                float pdf0 = 1.0f;
                if (b >= p->rr_depth) {
                    float msum = w.x + w.y + w.z;
                    float sel  = (msum > 0.0f)
                                     ? (ar * w.x + ag * w.y + ab * w.z) / msum
                                     : 0.0f;
                    if (sel > 1.0f)
                        sel = 1.0f;  // total > 1 gets renormalised
                    if (!(rx < sel))
                        break;  // absorbed
                    pdf0 = sel;
                    rx /= sel;
                    if (rx > 0.99999994f)
                        rx = 0.99999994f;
                    ar /= sel;
                    ag /= sel;
                    ab /= sel;
                }

                // ---- BSDF sampling -----------------------------------------
                // Initialised to the mirrored incident direction, the same
                // fallback every sample_* helper uses when it has no valid
                // draw: `r.dir` below must never carry uninitialised private
                // memory, even on the branches that sample nothing (the path
                // dies there anyway -- `r.active` is cleared at the top of the
                // body and only the live branches set it again).
                Vec3 wi       = mul(wov, -1.0f);
                float sweight = 1.0f;
                // Same value as a colour: the conductor lobe's weight carries
                // its per-channel Fresnel term, so it cannot be a scalar.
                Vec3 sweight_v = mk(1.0f, 1.0f, 1.0f);
                float wmulr = ar, wmulg = ag, wmulb = ab;
                if (all_delta) {
                    // CompositeBSDF::sample over the flattened components:
                    // pick one with rx, multiply its weight by
                    // weights[i] * (1/pdfs[i]).
                    float msum = w.x + w.y + w.z;
                    float p0 = 0.0f, p1 = 0.0f;
                    if (msum > 0.0f) {
                        p0 = (alb0.x * w.x + alb0.y * w.y + alb0.z * w.z) / msum;
                        p1 = (alb1.x * w.x + alb1.y * w.y + alb1.z * w.z) / msum;
                    }
                    float ptot = p0 + p1;
                    if (!(ptot > 0.0f))
                        break;  // nothing this BSDF can sample
                    int pick    = 0;
                    float share = p0 / ptot;
                    float prob  = share;
                    if (!(rx < share)) {
                        pick = 1;
                        prob = p1 / ptot;
                        rx   = (rx - share) / prob;
                    } else {
                        rx = rx / share;
                    }
                    // `prob > 0` is not enough: a denormal pick probability
                    // turns the `1 / prob` below into inf, and an infinite
                    // throughput poisons every later bounce.  Same rule
                    // mis_update_eval_vk already applies to `1 / b`.
                    if (!(prob >= 1.17549435e-38f))  // FLT_MIN
                        break;
                    const ClosureEntryVK& e = ce[pick];
                    float cw                = 0.0f;
                    if (e.id == REFLECTION_ID_VK
                        || e.id == FRESNEL_REFLECTION_ID_VK) {
                        // Reflection::sample: the reflected direction is the
                        // only one, weight = Fresnel.
                        float cosNO = dot(e.N, wov);
                        if (cosNO > 0.0f) {
                            wi      = sub(mul(e.N, 2.0f * cosNO), wov);
                            cw      = fresnel_dielectric_vk(cosNO, e.eta);
                            bsdf_pdf = inf();
                        }
                    } else if (e.id == REFRACTION_ID_VK) {
                        float Ft = fresnel_refraction_vk(mul(wov, -1.0f), e.N,
                                                         e.eta, wi);
                        cw       = Ft;
                        bsdf_pdf = inf();
                    } else if (e.id == TRANSPARENT_ID_VK) {
                        wi       = mul(wov, -1.0f);
                        cw       = 1.0f;
                        bsdf_pdf = inf();
                    } else {
                        // A delta closure the device does not implement yet
                        // (the MaterialX dielectric and medium ids are still on
                        // the host only).  Contributes nothing on purpose: a
                        // zero weight with a finite direction, so the path
                        // ends with a well-defined ray instead of a direction
                        // the sampler never wrote.
                        wi       = mul(wov, -1.0f);
                        cw       = 0.0f;
                        bsdf_pdf = 0.0f;
                    }
                    wmulr = e.w.x * cw / prob;
                    wmulg = e.w.y * cw / prob;
                    wmulb = e.w.z * cw / prob;
                } else if (is_mirror) {
                    float cosNO = dot(n, wov);
                    wi          = sub(mul(n, 2.0f * cosNO), wov);
                    bsdf_pdf    = inf();
                } else if ((HAS_MF && has_microfacet)
                           || (HAS_MXD && has_mxd)) {
                    // Glass is a two-lobe mixture (transmission + reflection),
                    // and so is the same shape built with a dielectric over a
                    // diffuse base.  Pick a lobe by its solid-angle share, sample
                    // that lobe, then fold the other lobe's eval toward the
                    // chosen direction in with the balance heuristic -- exactly
                    // CompositeBSDF::sample (shading.h:376-403).
                    float msum = w.x + w.y + w.z;
                    float p0 = 0.0f, p1 = 0.0f;
                    if (msum > 0.0f) {
                        p0 = (alb0.x * w.x + alb0.y * w.y + alb0.z * w.z) / msum;
                        p1 = (alb1.x * w.x + alb1.y * w.y + alb1.z * w.z) / msum;
                    }
                    float ptot = p0 + p1;
                    if (!(ptot > 0.0f))
                        break;
                    p0 /= ptot;
                    p1 /= ptot;
                    int pick   = (rx < p0) ? 0 : 1;
                    float prob = pick ? p1 : p0;
                    // Denormal pick probability would make the `1 / prob` in
                    // `outw` below inf, i.e. a non-finite throughput.  A lobe
                    // the mixture picks with (almost) no probability is
                    // unsamplable: leave the weight at zero and let the path
                    // die.
                    if (!(prob >= 1.17549435e-38f))  // FLT_MIN
                        break;
                    float u1 = pick ? (rx - p0) / prob : rx / p0;
                    if (u1 > 0.99999994f)
                        u1 = 0.99999994f;
                    const ClosureEntryVK& e = ce[pick];
                    Vec3 sw                 = mk(0.0f, 0.0f, 0.0f);
                    float spdf              = 0.0f;
                    if (e.id == MICROFACET_ID_VK) {
                        float lw = 0.0f;
                        mf_sample_vk(e, wov, u1, s3.y, s3.z, wi, lw, spdf);
                        sw = mk(lw, lw, lw);
                    } else if constexpr (HAS_MXD) {
                        // The dielectric lobe samples like the conductor: draw a
                        // micro normal from the bounded VNDF, reflect, then take
                        // the full eval there (sample_turquin_microms_reflection).
                        if (e.id == MX_DIELECTRIC_ID_VK) {
                            Vec3 dw;
                            mx_dielectric_sample_vk(p, e, wov, u1, s3.y, wi, dw,
                                                    spdf);
                            sw = dw;
                        } else if constexpr (HAS_DIFF) {
                            if (e.id == DIFFUSE_ID_VK) {
                                float pc = 0.0f;
                                wi       = sample_cosine_hemisphere(e.N, u1, s3.y,
                                                                    pc);
                                spdf     = pc;
                                sw       = mk(1.0f, 1.0f, 1.0f);
                            } else {
                                break;  // this lobe cannot be sampled
                            }
                        } else {
                            break;  // this lobe cannot be sampled
                        }
                    } else if constexpr (HAS_DIFF) {
                        if (e.id == DIFFUSE_ID_VK) {
                            float pc = 0.0f;
                            wi       = sample_cosine_hemisphere(e.N, u1, s3.y, pc);
                            spdf     = pc;
                            sw       = mk(1.0f, 1.0f, 1.0f);
                        } else {
                            break;  // this lobe cannot be sampled
                        }
                    } else {
                        break;  // this lobe cannot be sampled
                    }
                    Vec3 outw = mul(mulv(e.w, sw), 1.0f / prob);
                    // CompositeBSDF::sample drops the whole sample when the lobe
                    // it picked reports a zero pdf (shading.h:387-388).  It has
                    // to: a micro normal the sampler drew at the very edge of the
                    // lobe has `D_refl_D -> 0`, which makes the lobe's own value
                    // `G2_G1 * G1 / D_refl_D` blow up *and* its pdf vanish, so
                    // the sample is pure noise the CPU never adds.  Merging the
                    // other lobe into it instead (below) would divide that
                    // eval by the pick probability alone -- a firefly.
                    if (!(spdf > 0.0f)) {
                        w = mk(0.0f, 0.0f, 0.0f);
                        break;
                    }
                    spdf *= prob;
                    const ClosureEntryVK& oe = ce[pick ? 0 : 1];
                    if (oe.id >= 0 && oe.id != EMISSION_ID_VK
                        && oe.id != BACKGROUND_ID_VK) {
                        float op = 0.0f;
                        Vec3 ow  = component_eval<CS>(p, oe, wardp, wov, wi, n, op);
                        mis_update_eval_vk(outw, spdf, ow, op, pick ? p0 : p1);
                    }
                    w        = mulv(w, outw);
                    bsdf_pdf = spdf;
                } else if (is_ward) {
                    Vec3 wv;
                    float wp = 0.0f;
                    ward_sample_vk(wardp, wov, rx, s3.y, wi, wv, wp);
                    bsdf_pdf = wp * pdf0;
                    sweight  = wv.x;
                    if (!(wp > 0.0f)) {
                        w.x = 0.0f;
                        w.y = 0.0f;
                        w.z = 0.0f;
                    }
                } else if (HAS_PHONG && has_phong) {
                    // Phong::sample (shading.cpp:342-360): the lobe's own
                    // cosine-power draw around the mirror direction, then eval
                    // at that direction for the (weight, pdf) pair.
                    float pw      = 0.0f;
                    float pweight = 0.0f;
                    phong_sample_vk(ce[0], wov, rx, s3.y, wi, pweight, pw);
                    bsdf_pdf = pw * pdf0;
                    sweight  = pweight;
                    if (!(pw > 0.0f)) {
                        w.x = 0.0f;
                        w.y = 0.0f;
                        w.z = 0.0f;
                    }
                } else if (HAS_ON && has_on) {
                    // OrenNayarDiffuseLobe::sample_impl: cosine hemisphere
                    // about the lobe's visible normal, then eval there.  The
                    // direction pdf is Lambert's, so only the weight differs
                    // from the diffuse branch below (`f` instead of 1).
                    float ow = 0.0f;
                    float op = 0.0f;
                    on_sample_vk(ce[0], wov, rx, s3.y, wi, ow, op);
                    bsdf_pdf = op * pdf0;
                    sweight  = ow;
                    if (!(op > 0.0f)) {
                        w.x = 0.0f;
                        w.y = 0.0f;
                        w.z = 0.0f;
                    }
                } else if (HAS_MXC && has_mxc) {
                    // sample_turquin_microms_reflection: the bounded VNDF draw,
                    // the reflection, then the full BRDF eval at that direction.
                    // The weight is a colour (the Fresnel term is per channel),
                    // so it rides in `sweight_v` instead of `sweight`.
                    Vec3 ow;
                    float op = 0.0f;
                    mx_conductor_sample_vk(p, ce[0], wov, rx, s3.y, wi, ow, op);
                    bsdf_pdf  = op * pdf0;
                    sweight_v = ow;
                    if (!(op > 0.0f)) {
                        w.x = 0.0f;
                        w.y = 0.0f;
                        w.z = 0.0f;
                    }
                } else if constexpr (HAS_DIFF) {
                    // Sampling::sample_cosine_hemisphere (sampling.h:92).
                    float pdf_cos = 0.0f;
                    wi            = sample_cosine_hemisphere(n, rx, s3.y, pdf_cos);
                    bsdf_pdf      = pdf_cos;
                } else {
                    break;  // no family in this kernel can sample this BSDF
                }
                if (all_delta) {
                    w = mulv(w, mk(wmulr, wmulg, wmulb));
                } else if (!((HAS_MF && has_microfacet)
                             || (HAS_MXD && has_mxd))) {
                    // Single-lobe families: the throughput factor is the
                    // component weight, which for all of them except the
                    // dielectric *is* the albedo -- that one samples through the
                    // mixture branch above instead, where the pick probability
                    // and the other lobe's eval are already folded in.
                    if constexpr (HAS_MXC) {
                        if (has_mxc)
                            w = mulv(w, mulv(mk(ar, ag, ab), sweight_v));
                        else
                            w = mulv(w, mk(ar * sweight, ag * sweight,
                                           ab * sweight));
                    } else {
                        w = mulv(w, mk(ar * sweight, ag * sweight, ab * sweight));
                    }
                }
                r.org    = P;
                r.dir    = wi;
                r.radius = ray.radius;
                r.spread = ray.spread;
                r.prev   = hit.id;
                r.depth  = b + 1;
                r.bpdf   = bsdf_pdf;
                r.thr_r  = w.x;
                r.thr_g  = w.y;
                r.thr_b  = w.z;
                r.seed   = sampler.seed;
                r.sindex = sampler.index;
                r.active = (w.x > 0.0f || w.y > 0.0f || w.z > 0.0f) ? 1 : 0;
            } while (0);

        }
    }

    // Accumulate this bounce's radiance into the sample's image.  The vendor
    // path divides by the sample count here, and the host sums one image per
    // sample, so the mean comes out identical to the megakernel's.
    out[pix * 4 + 0] += L.x * inv;
    out[pix * 4 + 1] += L.y * inv;
    out[pix * 4 + 2] += L.z * inv;

    // Publish the next wave's ray (or the dead flag) for the intersect pass.
    wave_ray_store(p, base, pix, r);
}

// ---- pass 3a: material compaction, on the device ---------------------------
//
// The host used to read the wave's hit ids back, bin them by material, fill the
// workgroup tails and upload the buckets -- one round trip per bounce, and the
// one thing that kept a wave from being a single command buffer.  Everything
// the binning needs is already in the frame buffer: `wf_hit_id` from the pass
// that just ran, and the primitive -> material table the shading pass reads
// anyway.  So the sort is one more device pass, and the host only zeroes the
// counters once per sample.
//
// `wf_mat_count` is the per-material cursor, bumped with an atomic add: the
// order inside a bucket is not deterministic, but the shading pass sums
// radiance per pixel, so bucket order does not affect the image (the host's
// sort was ordered only because it happened to walk pixels in order).

extern "C" void osl_vk_wave_compact(GlobalPtr base)
{
    VK_GLOBAL const FrameParams* p
        = (VK_GLOBAL const FrameParams*)(base + kFrameParamsOffset);
    const unsigned npix = (unsigned)p->width * (unsigned)p->height;
    if (npix == 0)
        return;
    const unsigned pix = (unsigned)get_global_id(0);
    if (pix >= npix)
        return;
    const int nm = p->num_materials;
    if (nm <= 0)
        return;

    // The rule the host applied while it owned the sort: a hit carries the
    // material of its primitive, and a miss or a dead path goes to the fallback
    // bucket (the light group), because every shading kernel can shade the
    // light and those pixels still have to settle the previous wave's NEE.
    const int tid = wf_icp(base, p->wf_hit_id)[pix];
    int mat       = -1;
    if (tid >= 0 && tid < p->num_triangles)
        mat = wf_icp(base, p->tri_shader_offset)[tid];
    if (mat < 0 || mat >= nm)
        mat = p->fallback_mat;

    const int slot = __atomic_fetch_add(wf_ip(base, p->wf_mat_count) + mat, 1,
                                        __ATOMIC_RELAXED);
    // Same list layout the shading entry derives at compile time: one
    // `npix + kWorkgroupSize` block per material.
    const int stride = (int)(npix + (unsigned)kWorkgroupSize);
    wf_ip(base, p->wf_bucket_index)[mat * stride + slot] = (int)pix;
}

/// Close the buckets and publish the dispatch sizes: one invocation per
/// material writes the `-1` sentinel into the slots past the last real index,
/// the group count vkCmdDispatchIndirect will read, and resets the cursor for
/// the next wave.  This is the pass that lets the host stop asking how big each
/// bucket is.
extern "C" void osl_vk_wave_bucket_tail(GlobalPtr base)
{
    VK_GLOBAL const FrameParams* p
        = (VK_GLOBAL const FrameParams*)(base + kFrameParamsOffset);
    const unsigned m  = (unsigned)get_global_id(0);
    const int nm      = p->num_materials;
    if (nm <= 0 || (int)m >= nm)
        return;
    const unsigned npix = (unsigned)p->width * (unsigned)p->height;
    const int stride    = (int)(npix + (unsigned)kWorkgroupSize);

    VK_GLOBAL int* count  = wf_ip(base, p->wf_mat_count) + (int)m;
    VK_GLOBAL int* bucket = wf_ip(base, p->wf_bucket_index) + (int)m * stride;
    const int n           = count[0];
    const int g = (n + (int)kWorkgroupSize - 1) / (int)kWorkgroupSize;
    for (int k = n; k < g * (int)kWorkgroupSize; ++k)
        bucket[k] = -1;

    VK_GLOBAL int* args = wf_ip(base, p->wf_dispatch_args) + (int)m * 3;
    args[0]             = g;
    args[1]             = 1;
    args[2]             = 1;

    count[0] = 0;  // armed for the next wave
}

// ---- pass 3, one kernel per material ---------------------------------------
//
// The device sorts the wave by material (pass 3a) and dispatches one bucket
// per material, so every invocation of a given kernel takes the same branch of
// the shader dispatch and the module only has to carry that shader.  The pixel
// index comes from `wf_bucket_index`; past the end of a bucket
// `wave_bucket_tail` leaves a negative sentinel, because a dispatch always
// rounds up to whole workgroups.

template <int MAT, unsigned CS>
inline void wave_shade_entry(GlobalPtr base)
{
    VK_GLOBAL const FrameParams* p
        = (VK_GLOBAL const FrameParams*)(base + kFrameParamsOffset);
    const unsigned npix = (unsigned)p->width * (unsigned)p->height;
    if (npix == 0)
        return;
    const unsigned slot = (unsigned)get_global_id(0);
    // Bucket layout, fixed with the host (simpleraytracer.cpp): one list per
    // material, `npix + kWorkgroupSize` ints each -- the stride rounds the
    // workgroup tail up.  A dispatch only ever reads the slots of the list
    // belonging to the material this entry point was instantiated for, so the
    // base is a compile-time product and the host does not have to rewrite
    // FrameParams between the per-material dispatches.
    const int mat_base = MAT * (int)(npix + (unsigned)kWorkgroupSize);
    const int pix = wf_icp(base, p->wf_bucket_index)[mat_base + (int)slot];
    if (pix < 0 || (unsigned)pix >= npix)
        return;
    wave_shade_body<MAT, CS>(base, (unsigned)pix);
}

//
// Four flavours per material, one per closure set the testrender scenes
// actually produce: `_bg` (background only: the envmap's own shader and the
// light group's emission), `_diff` (matte), `_mf` (microfacet / glass) and
// `_all` (everything, the fallback for any other mix).
//
// This replaces the earlier `MF=true/false` pair, which only covered the
// microfacet family.  The reason it is worth the instantiations: the device
// runtime is force-inlined before clspv, so every `if (e.id == ...)` keeps its
// layer's code in every kernel, and the microfacet scene carries three glass
// groups -- each kernel paid for the cosine sampler, Ward, the Fresnel path
// and the Beckmann lobes whether its shader could reach them or not.
//
// A set that has no instantiation here gets `_all`; the module builder falls
// back to it by name, so an unlisted scene stays correct.
//
// Internal linkage on purpose: the module builder keeps only the bodies a
// kernel actually calls and erases the rest before clspv runs (vulkan_module.
// cpp).  With external linkage every combination would survive the link and
// be carried through the inliner, which turned a 30 second module build into
// 20 minutes once there were four sets instead of two.  The `extern "C"
// { static ... }` block form is what gives an internal symbol with the plain
// C name on both clang and MSVC (`extern "C" static` as a specifier list is
// rejected).
//
// `used` is load bearing twice over: nothing in this translation unit calls
// these, so clang would drop every `static` one as dead before the module is
// ever linked -- and the builder then fails to find them by name.
#if defined(__clang__)
#    define VK_ENTRY_USED __attribute__((used))
#else
#    define VK_ENTRY_USED
#endif
#define OSL_VK_SHADE_ENTRY(n)                                                  \
    extern "C" {                                                               \
    static VK_ENTRY_USED void osl_vk_wave_shade##n##_all(GlobalPtr base)                     \
    {                                                                          \
        wave_shade_entry<n, CS_ALL>(base);                                     \
    }                                                                          \
    static VK_ENTRY_USED void osl_vk_wave_shade##n##_bg(GlobalPtr base)                      \
    {                                                                          \
        wave_shade_entry<n, CS_BG>(base);                                      \
    }                                                                          \
    static VK_ENTRY_USED void osl_vk_wave_shade##n##_diff(GlobalPtr base)                    \
    {                                                                          \
        wave_shade_entry<n, CS_DIFF>(base);                                    \
    }                                                                          \
    static VK_ENTRY_USED void osl_vk_wave_shade##n##_mf(GlobalPtr base)                      \
    {                                                                          \
        wave_shade_entry<n, CS_MF>(base);                                      \
    }                                                                          \
    static VK_ENTRY_USED void osl_vk_wave_shade##n##_delta(GlobalPtr base)                   \
    {                                                                          \
        wave_shade_entry<n, CS_DELTA>(base);                                   \
    }                                                                          \
    }

OSL_VK_SHADE_ENTRY(0)
OSL_VK_SHADE_ENTRY(1)
OSL_VK_SHADE_ENTRY(2)
OSL_VK_SHADE_ENTRY(3)
OSL_VK_SHADE_ENTRY(4)
OSL_VK_SHADE_ENTRY(5)
OSL_VK_SHADE_ENTRY(6)
OSL_VK_SHADE_ENTRY(7)

// Slots 8..15 carry the fallback body only.  The five-set instantiations exist
// to slim down the scenes the shading work is measured on, and every one of
// those has at most eight groups; a scene with nine or more (render-mx-furnace-*
// has twelve, one group per sphere) just needs *a* kernel per material, and the
// `_all` body is what the module builder would fall back to anyway.  Adding
// only the fallback keeps the extra template instantiation cost at eight bodies
// instead of forty.
#define OSL_VK_SHADE_ENTRY_ALL(n)                                               \
    extern "C" {                                                               \
    static VK_ENTRY_USED void osl_vk_wave_shade##n##_all(GlobalPtr base)                     \
    {                                                                          \
        wave_shade_entry<n, CS_ALL>(base);                                     \
    }                                                                          \
    }

OSL_VK_SHADE_ENTRY_ALL(8)
OSL_VK_SHADE_ENTRY_ALL(9)
OSL_VK_SHADE_ENTRY_ALL(10)
OSL_VK_SHADE_ENTRY_ALL(11)
OSL_VK_SHADE_ENTRY_ALL(12)
OSL_VK_SHADE_ENTRY_ALL(13)
OSL_VK_SHADE_ENTRY_ALL(14)
OSL_VK_SHADE_ENTRY_ALL(15)



