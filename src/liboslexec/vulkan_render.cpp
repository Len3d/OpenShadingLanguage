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
    float l = length(a);
    return l > 0.0f ? mul(a, 1.0f / l) : a;
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
        static const unsigned int zmatrix[24] = {
            0x000001u, 0x000003u, 0x000006u, 0x000009u, 0x000017u, 0x00003au,
            0x000071u, 0x0000a3u, 0x000116u, 0x000339u, 0x000677u, 0x0009aau,
            0x001601u, 0x003903u, 0x007706u, 0x00aa09u, 0x010117u, 0x03033au,
            0x060671u, 0x0909a3u, 0x171616u, 0x3a3939u, 0x717777u, 0xa3aaaau
        };
        unsigned int si = vk_owen_scramble(index, vk_hash(seed - 4)) & 0xFFFFFFu;
        unsigned int rx = si, ry = 0, rz = 0, ymatrix = 1;
        for (int c = 0; c < 24; ++c) {
            unsigned int bit = (si >> c) & 1u;
            ry ^= bit * ymatrix;
            rz ^= bit * zmatrix[c];
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

/// Root component of a closure, or null when the shader produced none.
inline const ClosureComponentVK* closure_root(const void* ci)
{
    if (!ci)
        return (const ClosureComponentVK*)0;
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


/// A flattened closure component: the branch weight already folded in (MUL
/// multiplies it, ADD splits it, a component is a leaf -- the same walk
/// testrender's process_bsdf_closure does at shading.cpp:1480-1690).
struct ClosureEntryVK {
    int id;
    Vec3 w;      ///< component weight, MUL folds included
    Vec3 N;      ///< reflection / refraction plane normal
    float eta;   ///< reflection / refraction index of refraction
    long long off;  ///< arena offset of the node, for parameters read on demand
};


/// Flatten `Ci` into at most two components.
///
/// Two is enough for the closures a render-* material produces: matte and Ward
/// are a single component, and glass is exactly
/// `reflection(N, eta) + refraction(N, eta)`.  A larger tree is truncated to
/// the first two, which dims a path rather than corrupting it.  Deliberately
/// not an array: a local array is an `alloca`, and the legalizer rewrites every
/// alloca in the device runtime into `osl_vk_arena_alloc(<enclosing function's
/// first argument>, ...)` -- inside the frame kernel that pointer is the
/// ShadeBlock itself, so the allocation would land in the closure pool.
inline __attribute__((always_inline)) int
collect_closure_entries (VK_GLOBAL ShaderGlobals* sg, long long root,
                         ClosureEntryVK& first, ClosureEntryVK& second)
{
    if (root < 0)
        return 0;

    long long stack_off[8];
    Vec3 stack_w[8];
    int sp        = 0;
    long long cur = root;
    Vec3 branch   = mk (1.0f, 1.0f, 1.0f);
    int n         = 0;

    for (;;) {
        if (cur >= 0) {
            GlobalCPtr node = (GlobalCPtr)sg + kOffPoolClosure + cur;
            const int id    = *(VK_GLOBAL const int*)node;
            if (id == -2) {  // ADD
                if (sp < 8) {
                    stack_off[sp] = *(VK_GLOBAL const long long*)(node + 16);
                    stack_w[sp]   = branch;
                    ++sp;
                }
                cur = *(VK_GLOBAL const long long*)(node + 8);
                continue;
            }
            if (id == -1) {  // MUL
                branch = mulv (branch, load_vec ((GlobalCVec3)(node + 4)));
                cur    = *(VK_GLOBAL const long long*)(node + 16);
                continue;
            }
            if (n < 2) {
                ClosureEntryVK& e = (n == 0) ? first : second;
                e.id              = id;
                e.off             = cur;
                e.w = mulv (branch, load_vec ((GlobalCVec3)(node + 4)));
                e.N   = mk (0.0f, 0.0f, 0.0f);
                e.eta = 0.0f;
                if (id == FRESNEL_REFLECTION_ID_VK
                    || id == REFRACTION_ID_VK) {
                    // ReflectionParams / RefractionParams { Vec3 N; float eta }
                    e.N   = load_vec ((GlobalCVec3)(node + 16));
                    e.eta = *(VK_GLOBAL const float*)(node + 28);
                } else if (id == REFLECTION_ID_VK) {
                    e.N = load_vec ((GlobalCVec3)(node + 16));
                }
                ++n;
            }
            cur = -1;
        }
        if (sp == 0)
            break;
        --sp;
        cur    = stack_off[sp];
        branch = stack_w[sp];
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
/// component its own weight.
inline Vec3 closure_entry_albedo (const ClosureEntryVK& e, const Vec3& wo)
{
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
    return e.w;
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

    // 1/(2*pi); the CPU goes through OIIO::fast_atan2, the device through the
    // libm entry point -- they differ in the last bits, which matters only for
    // a direction that lands exactly on a texel boundary.
    float u = __builtin_atan2f(dir.y, dir.x) * 0.15915494309189535f;
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
    return mk(values[i].x, values[i].y, values[i].z);
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
    dir = mk(sin_phi * __builtin_cosf(theta), sin_phi * __builtin_sinf(theta),
             cos_phi);

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
    Vec3 P         = add(add(mul(va, w0), mul(vb, hit.u)), mul(vc, hit.v));
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

extern "C" void osl_vk_wave_shade(GlobalPtr base)
{
    VK_GLOBAL const FrameParams* p
        = (VK_GLOBAL const FrameParams*)(base + kFrameParamsOffset);

    const unsigned long long gid = get_global_id(0);
    const unsigned long long npix
        = (unsigned long long)p->width * (unsigned long long)p->height;
    if (gid >= npix)
        return;

    WaveRay r;
    wave_ray_load(p, base, gid, r);

    float h_t, h_u, h_v;
    int h_id, h_ok;
    wave_hit_load(p, base, gid, h_t, h_u, h_v, h_id, h_ok);

    SceneView scene;
    WAVE_SCENE_SETUP(scene, gid)

    GlobalInt tri_shader = (GlobalInt)(base + p->tri_shader_offset);
    GlobalPtr arena      = (GlobalPtr)(base + p->arena_offset)
                           + gid * (unsigned long long)p->arena_stride;
    VK_GLOBAL ShaderGlobals* sg = (VK_GLOBAL ShaderGlobals*)arena;

    // Two facts the device runtime cannot derive from `sg` alone: where this
    // invocation's block sits in the storage buffer, and where the texture
    // table lives.  They ride in the first 16 bytes of the closure pool, which
    // the allocator never hands out (it starts at offset 16).
    const long long sg_abs = (long long)(p->arena_offset
                                         + gid * (unsigned long long)p->arena_stride);
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
    const int pending = wf_icp(base, p->wf_nee_pending)[gid];
    if (pending) {
        if ((pending & 1) && wf_icp(base, p->wf_vis_bg)[gid]) {
            out[gid * 4 + 0] += wf_fcp(base, p->wf_bgc_r)[gid] * inv;
            out[gid * 4 + 1] += wf_fcp(base, p->wf_bgc_g)[gid] * inv;
            out[gid * 4 + 2] += wf_fcp(base, p->wf_bgc_b)[gid] * inv;
        }
        if ((pending & 2) && wf_icp(base, p->wf_vis_l)[gid]) {
            // Nothing between the surface and the light: shade the point on the
            // light and multiply in its emission, as the megakernel did inline.
            const int lid           = wf_icp(base, p->wf_l_lid)[gid];
            const int lightShaderID = tri_shader[lid];
            if (lightShaderID >= 0) {
                Ray sray;
                sray.origin
                    = mk(wf_fcp(base, p->wf_p_x)[gid],
                         wf_fcp(base, p->wf_p_y)[gid],
                         wf_fcp(base, p->wf_p_z)[gid]);
                sray.direction = mk(wf_fcp(base, p->wf_ldir_x)[gid],
                                    wf_fcp(base, p->wf_ldir_y)[gid],
                                    wf_fcp(base, p->wf_ldir_z)[gid]);
                sray.radius    = 0.0f;
                sray.spread    = 0.0f;
                Hit lhit;
                lhit.id = lid;
                lhit.u  = wf_fcp(base, p->wf_l_u)[gid];
                lhit.v  = wf_fcp(base, p->wf_l_v)[gid];
                lhit.t  = wf_fcp(base, p->wf_l_tmax)[gid];
                fill_globals(sg, scene, p, sray, lhit,
                             (VK_GLOBAL const unsigned char*)base);
                *(GlobalI64)((GlobalPtr)sg + kOffRenderstate) = 16;
                *(GlobalI64)((GlobalPtr)sg + kOffCi)          = -1;
                osl_vk_shade_dispatch((GlobalPtr)sg, lightShaderID);
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
                        out[gid * 4 + 0]
                            += wf_fcp(base, p->wf_lc_r)[gid] * lw.x * inv;
                        out[gid * 4 + 1]
                            += wf_fcp(base, p->wf_lc_g)[gid] * lw.y * inv;
                        out[gid * 4 + 2]
                            += wf_fcp(base, p->wf_lc_b)[gid] * lw.z * inv;
                    }
                }
                // The light shading overwrote the block the surface's closure
                // tree lived in, but that tree is finished with: everything
                // after this pass works off the ray buffer, not `sg`.
            }
        }
        // Consumed.  The next wave publishes a fresh pair.
        wf_ip(base, p->wf_nee_pending)[gid] = 0;
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

    Vec3 w          = mk(r.thr_r, r.thr_g, r.thr_b);
    float bsdf_pdf  = r.bpdf;
    const int b     = r.depth;

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
            osl_vk_shade_dispatch((GlobalPtr)sg, p->bg_shader_id);
            const Vec3 Le
                = background_closure_color(sg, *(GlobalI64)((GlobalPtr)sg + kOffCi));
            L = add(L, mulv(w, Le));
        }
    } else {
        fill_globals(sg, scene, p, ray, hit,
                     (VK_GLOBAL const unsigned char*)base);

        const int shaderID = tri_shader[hit.id];
        if (shaderID >= 0) {
            // Reset the per-sample arena state: cursor at 16 (offset 0 is the
            // two int64s parked at the head of the pool), root sentinel -1.
            *(GlobalI64)((GlobalPtr)sg + kOffRenderstate) = 16;
            *(GlobalI64)((GlobalPtr)sg + kOffCi)          = -1;

            osl_vk_shade_dispatch((GlobalPtr)sg, shaderID);

            float er = 0.0f, eg = 0.0f, eb = 0.0f;  // emitted
            float ar = 0.0f, ag = 0.0f, ab = 0.0f;  // albedo / closure weight
            int is_mirror = 0;
            int is_ward   = 0;
            // All the closures this material produced are delta closures
            // (mirror / refraction / transparent).  Then the BSDF has exactly
            // one outgoing direction per component, the pdf is infinite, and
            // NEE has nothing to evaluate toward.
            int all_delta = 0;
            WardVK wardp;
            long long ci = *(GlobalI64)((GlobalPtr)sg + kOffCi);
            // Flatten ADD/MUL: glass is `Ks*reflection(N,eta) +
            // Cs*refraction(N,eta)`, i.e. two components.
            ClosureEntryVK ce0, ce1;
            ce0.id  = -1;
            ce1.id  = -1;
            ce0.off = 0;
            ce1.off = 0;
            int ncomp   = collect_closure_entries(sg, ci, ce0, ce1);
            Vec3 alb0   = mk(0.0f, 0.0f, 0.0f);
            Vec3 alb1   = mk(0.0f, 0.0f, 0.0f);
            if (ncomp > 0) {
                Vec3 wov0 = mul(ray.direction, -1.0f);
                all_delta = ncomp;
                for (int k = 0; k < ncomp; ++k) {
                    const ClosureEntryVK& e = (k == 0) ? ce0 : ce1;
                    if (e.id == EMISSION_ID_VK || e.id == BACKGROUND_ID_VK) {
                        er += e.w.x;
                        eg += e.w.y;
                        eb += e.w.z;
                        all_delta = 0;  // an emitter is not a delta BSDF
                        continue;
                    }
                    Vec3 alb = closure_entry_albedo(e, wov0);
                    if (k == 0)
                        alb0 = alb;
                    else
                        alb1 = alb;
                    ar += alb.x;
                    ag += alb.y;
                    ab += alb.z;
                    if (e.id == WARD_ID_VK) {
                        // shading.cpp:73 WardParams { Vec3 N; Vec3 T; float ax;
                        // float ay; } -- 32 bytes right after the 16-byte
                        // header: N at +16, T at +28, ax at +40, ay at +44.
                        GlobalCPtr wp = (GlobalCPtr)sg + kOffPoolClosure
                                        + e.off + 16;
                        wardp.N  = load_vec((GlobalCVec3)wp);
                        wardp.T  = load_vec((GlobalCVec3)(wp + 12));
                        wardp.ax = *(GlobalFloatCPtr)(wp + 24);
                        wardp.ay = *(GlobalFloatCPtr)(wp + 28);
                        is_ward   = 1;
                        all_delta = 0;
                    } else if (e.id == DIFFUSE_ID_VK) {
                        all_delta = 0;
                    } else if (!closure_entry_is_delta(e)) {
                        all_delta = 0;
                    }
                }
                if (ncomp == 1
                    && (ce0.id == REFLECTION_ID_VK
                        || ce0.id == FRESNEL_REFLECTION_ID_VK))
                    is_mirror = 1;
                if (all_delta > 0)
                    all_delta = 1;
            }

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
                    float bwt  = 0.0f;
                    if (is_ward) {
                        Vec3 wv;
                        ward_eval_vk(wardp, wov, bg_dir, wv, bpdf);
                        bwt = wv.x;
                    } else {
                        // Diffuse: Diffuse<trans>::eval (shading.cpp:305)
                        // returns weight = Color3(1) and pdf = max(N.wi,0)/pi.
                        const float NLd = dot(n, bg_dir);
                        bpdf = (NLd > 0.0f ? NLd : 0.0f) * 0.31830988618f;
                        bwt  = 1.0f;
                    }
                    const float mis = power_heuristic_weight_weight(bg_pdf, bpdf);
                    Vec3 contrib = mulv(mulv(w, mk(ar * bwt, ag * bwt, ab * bwt)),
                                        mk(bgv.x * mis, bgv.y * mis, bgv.z * mis));
                    if (contrib.x + contrib.y + contrib.z > 0.0f) {
                        // Publish the shadow ray instead of tracing it: the
                        // next intersection wave walks it, the next shading
                        // wave adds `contrib` if it got through.
                        wf_fp(base, p->wf_p_x)[gid]      = P.x;
                        wf_fp(base, p->wf_p_y)[gid]      = P.y;
                        wf_fp(base, p->wf_p_z)[gid]      = P.z;
                        wf_ip(base, p->wf_s_skip1)[gid] = hit.id;
                        wf_fp(base, p->wf_bgdir_x)[gid]  = bg_dir.x;
                        wf_fp(base, p->wf_bgdir_y)[gid]  = bg_dir.y;
                        wf_fp(base, p->wf_bgdir_z)[gid]  = bg_dir.z;
                        wf_fp(base, p->wf_bgc_r)[gid]    = contrib.x;
                        wf_fp(base, p->wf_bgc_g)[gid]    = contrib.y;
                        wf_fp(base, p->wf_bgc_b)[gid]    = contrib.z;
                        wf_ip(base, p->wf_nee_pending)[gid] |= 1;
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
                        float bwt  = 0.0f;
                        if (is_ward) {
                            Vec3 wv;
                            ward_eval_vk(wardp, wov, lsmp.dir, wv, bpdf);
                            bwt = wv.x;
                        } else {
                            float NLd = dot(n, lsmp.dir);
                            bpdf = (NLd > 0.0f ? NLd : 0.0f) * 0.31830988618f;
                            bwt  = 1.0f;
                        }
                        float mis = power_heuristic_eval_weight(
                            light_pick_pdf * lsmp.pdf, bpdf);
                        Vec3 contrib
                            = mulv(mulv(w, mk(ar * bwt, ag * bwt, ab * bwt)),
                                   mk(mis, mis, mis));
                        if (contrib.x + contrib.y + contrib.z > 0.0f) {
                            // Same hand-off as the background ray above; the
                            // light point is shaded when the result comes back,
                            // because that needs `sg` and the shader dispatch.
                            wf_fp(base, p->wf_p_x)[gid]      = P.x;
                            wf_fp(base, p->wf_p_y)[gid]      = P.y;
                            wf_fp(base, p->wf_p_z)[gid]      = P.z;
                            wf_ip(base, p->wf_s_skip1)[gid] = hit.id;
                            wf_fp(base, p->wf_ldir_x)[gid]   = lsmp.dir.x;
                            wf_fp(base, p->wf_ldir_y)[gid]   = lsmp.dir.y;
                            wf_fp(base, p->wf_ldir_z)[gid]   = lsmp.dir.z;
                            wf_fp(base, p->wf_l_tmax)[gid]   = lsmp.dist;
                            wf_fp(base, p->wf_l_u)[gid]      = lsmp.u;
                            wf_fp(base, p->wf_l_v)[gid]      = lsmp.v;
                            wf_ip(base, p->wf_l_lid)[gid]    = lid;
                            wf_fp(base, p->wf_lc_r)[gid]     = contrib.x;
                            wf_fp(base, p->wf_lc_g)[gid]     = contrib.y;
                            wf_fp(base, p->wf_lc_b)[gid]     = contrib.z;
                            wf_ip(base, p->wf_nee_pending)[gid] |= 2;
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
                Vec3 wi;
                float sweight = 1.0f;
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
                    if (!(prob > 0.0f))
                        break;
                    const ClosureEntryVK& e = (pick == 0) ? ce0 : ce1;
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
                    }
                    wmulr = e.w.x * cw / prob;
                    wmulg = e.w.y * cw / prob;
                    wmulb = e.w.z * cw / prob;
                } else if (is_mirror) {
                    float cosNO = dot(n, wov);
                    wi          = sub(mul(n, 2.0f * cosNO), wov);
                    bsdf_pdf    = inf();
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
                } else {
                    // Sampling::sample_cosine_hemisphere (sampling.h:92).
                    float pdf_cos = 0.0f;
                    wi            = sample_cosine_hemisphere(n, rx, s3.y, pdf_cos);
                    bsdf_pdf      = pdf_cos;
                }
                if (all_delta) {
                    w = mulv(w, mk(wmulr, wmulg, wmulb));
                } else {
                    w = mulv(w, mk(ar * sweight, ag * sweight, ab * sweight));
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
    out[gid * 4 + 0] += L.x * inv;
    out[gid * 4 + 1] += L.y * inv;
    out[gid * 4 + 2] += L.z * inv;

    // Publish the next wave's ray (or the dead flag) for the intersect pass.
    wave_ray_store(p, base, gid, r);
}
