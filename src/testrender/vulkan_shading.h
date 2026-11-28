// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

/// Host side of the Vulkan compute path.
///
/// One instance owns the logical device, the storage buffer and the command
/// machinery.  The whole frame is one pipeline (`add_shader()`), dispatched
/// once (`dispatch()`): every material, the ray generation, the intersection
/// and the accumulation live in that single module, because SPIR-V modules
/// cannot call across modules.
///
/// Vulkan types stay out of this header on purpose, so the rest of testrender
/// never has to include vulkan.h.
class VulkanShading {
public:
    VulkanShading();
    ~VulkanShading();

    /// Create the device for the first compute-capable physical device.
    bool init(std::string& err);

    /// Build a compute pipeline from one SPIR-V module.  Returns a non-negative
    /// id to pass to dispatch(), or -1 on failure.
    int add_shader(const uint32_t* spirv, size_t word_count, const char* entry,
                   std::string& err);

    /// Build one pipeline per entry out of the same SPIR-V module.  The
    /// wavefront path creates ten to sixteen pipelines per scene and the
    /// driver's compiler is what makes setup cost minutes, so the entries are
    /// compiled on `threads` host threads instead of one after another
    /// (serial when the count is not positive).  `ids` comes back in entry
    /// order; the call fails as a whole if any pipeline failed, with `err`
    /// naming the entry.
    bool add_shaders(const uint32_t* spirv, size_t word_count,
                     const std::vector<const char*>& entries,
                     std::vector<int>& ids, int threads, std::string& err);

    /// One decoded texture: RGBA32F, the whole mip chain, largest level first.
    /// `hash` is the ustring hash of the filename, which is exactly what the
    /// compiled shader passes to osl_texture(), so the device can match it
    /// without any name table.
    struct Texture {
        unsigned long long hash = 0;
        int width               = 0;
        int height              = 0;
        std::vector<int> level_width, level_height;
        std::vector<float> rgba;  // all levels, concatenated
    };

    /// Replace the device's texture set.  Uploads the table and the texels
    /// into the same storage buffer the shaders already see.
    bool set_textures(const std::vector<Texture>& textures, std::string& err);

    /// Size of the device storage buffer, in bytes.  The whole-frame path fills
    /// it itself (FrameParams, geometry, output image and the per-pixel arena),
    /// so it has to know how much room there is before laying the scene out.
    size_t capacity() const;

    /// End of the region the textures occupy, as a byte offset from the start
    /// of the buffer.  The whole-frame layout has to start after it.
    size_t texture_end() const;

    /// Copy host bytes into the storage buffer at `offset` (which is measured
    /// from the start of the buffer, exactly like the offsets the device kernel
    /// reads out of FrameParams).
    bool upload(size_t offset, const void* src, size_t bytes, std::string& err);

    /// Copy bytes back out of the storage buffer (the rendered image, mostly).
    bool download(size_t offset, void* dst, size_t bytes, std::string& err);

    /// One dispatch of a whole-frame pipeline: `groups` workgroups of
    /// kWorkgroupSize invocations, one invocation per pixel.  One submit and
    /// one fence wait per call; use the batch below to amortise that.
    bool dispatch(int id, unsigned groups, std::string& err);

    /// Record several dispatches into one command buffer and submit them once.
    /// The wavefront host loop issues one dispatch per material and they all
    /// read the same FrameParams, so between `begin_batch()` and `end_batch()`
    /// they cost one submit instead of one each -- on a 160x120 aa8 render that
    /// is ~5k submissions per frame turning into ~2k.
    bool begin_batch(std::string& err);
    bool record_dispatch(int id, unsigned groups, std::string& err);
    /// Same dispatch, but the group count comes from device memory: `args` is a
    /// byte offset into the storage buffer of a `uint[3]` (groupCountX/Y/Z)
    /// triple, which the device writes to.  This is what lets a bucket whose
    /// size only the device knows still be dispatched by the host.
    bool record_dispatch_indirect(int id, size_t args, std::string& err);
    /// Storage-buffer barrier between two dispatches in the same command
    /// buffer.  Batched passes hand data to each other through the frame
    /// buffer (intersect -> compaction -> shading), and Vulkan only orders them
    /// if the host says so.
    bool record_barrier(std::string& err);
    bool end_batch(std::string& err);

    /// How many queue submissions this instance has made.  Each one costs a
    /// fence round trip (~0.3 ms on the Intel UHD here) on top of its own work,
    /// so this is the part of the frame loop that batching is meant to shrink.
    unsigned long submissions() const;

    bool ready() const;

    /// Device description, for the render banner.
    const std::string& device_name() const;

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};
