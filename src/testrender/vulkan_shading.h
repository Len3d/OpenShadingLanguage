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
/// machinery; each OSL shader group becomes one compute pipeline on it
/// (`add_shader()`), because every group has its own Vulkan SPIR-V module.
///
/// `shade()` runs one shading point: the caller hands over a ShadeBlock (see
/// liboslexec/vulkan_types.h for its layout), already filled with that point's
/// ShaderGlobals; the block is copied into the storage buffer, one workgroup of
/// one invocation is dispatched, and the block is read back so the caller can
/// read the closure root the shader wrote.
///
/// Vulkan types stay out of this header on purpose, so the rest of testrender
/// never has to include vulkan.h.
class VulkanShading {
public:
    VulkanShading();
    ~VulkanShading();

    /// Create the device for the first compute-capable physical device.
    bool init(std::string& err);

    /// Build a compute pipeline from one group's SPIR-V module.  Returns a
    /// non-negative id to pass to shade(), or -1 on failure.
    int add_shader(const uint32_t* spirv, size_t word_count, const char* entry,
                   std::string& err);

    /// Run shader `id` for one shading point.  `block` is input (the
    /// ShaderGlobals the host prepared) and output (whatever the shader wrote,
    /// including its closure arena).
    bool shade(int id, void* block, size_t block_bytes, std::string& err);

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
    /// kWorkgroupSize invocations, one invocation per pixel.
    bool dispatch(int id, unsigned groups, std::string& err);

    bool ready() const;

    /// Device description, for the render banner.
    const std::string& device_name() const;

private:
    struct Impl;
    std::unique_ptr<Impl> m_impl;
};
