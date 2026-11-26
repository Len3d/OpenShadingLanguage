// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#pragma once

#include <atomic>
#include <map>
#include <memory>
#include <unordered_map>

#include <OpenImageIO/imagebuf.h>

#include <OSL/oslexec.h>
#include <OSL/rendererservices.h>
#include "background.h"
#include "raytracer.h"
#include "sampling.h"
// Host side of the Vulkan compute path (declared at global scope, so the
// member below does not create an OSL::VulkanShading that nothing defines).
// Included rather than forward declared because ~SimpleRaytracer is inline
// and therefore needs the complete type to destroy the unique_ptr.
#include "vulkan_shading.h"


OSL_NAMESPACE_BEGIN

struct Material {
    ShaderGroupRef surf;
    ShaderGroupRef disp;
};

using MaterialVec = std::vector<Material>;

class SimpleRaytracer : public RendererServices {
public:
    // Just use 4x4 matrix for transformations
    typedef Matrix44 Transformation;

    SimpleRaytracer();
    virtual ~SimpleRaytracer() {}

    // RendererServices support:
    bool get_matrix(ShaderGlobals* sg, Matrix44& result,
                    TransformationPtr xform, float time) override;
    bool get_matrix(ShaderGlobals* sg, Matrix44& result, ustringhash from,
                    float time) override;
    bool get_matrix(ShaderGlobals* sg, Matrix44& result,
                    TransformationPtr xform) override;
    bool get_matrix(ShaderGlobals* sg, Matrix44& result,
                    ustringhash from) override;
    bool get_inverse_matrix(ShaderGlobals* sg, Matrix44& result, ustringhash to,
                            float time) override;
    bool get_array_attribute(ShaderGlobals* sg, bool derivatives,
                             ustringhash object, TypeDesc type,
                             ustringhash name, int index, void* val) override;
    bool get_attribute(ShaderGlobals* sg, bool derivatives, ustringhash object,
                       TypeDesc type, ustringhash name, void* val) override;
    bool get_userdata(bool derivatives, ustringhash name, TypeDesc type,
                      ShaderGlobals* sg, void* val) override;
    // OSL asks the renderer for a texture handle while compiling a group;
    // recording the name there is how the Vulkan path learns which textures
    // it has to decode and upload.  Returning null keeps the CPU path's
    // behaviour unchanged.
    TextureHandle* get_texture_handle(ustring filename, ShadingContext* context,
                                     const TextureOpt* options = nullptr) override;

    void name_transform(const char* name, const Transformation& xform);

    // Set and get renderer attributes/options
    void attribute(string_view name, TypeDesc type, const void* value);
    void attribute(string_view name, int value)
    {
        attribute(name, TypeDesc::INT, &value);
    }
    void attribute(string_view name, float value)
    {
        attribute(name, TypeDesc::FLOAT, &value);
    }
    void attribute(string_view name, string_view value)
    {
        std::string valstr(value);
        const char* s = valstr.c_str();
        attribute(name, TypeDesc::STRING, &s);
    }
    OIIO::ParamValue* find_attribute(string_view name,
                                     TypeDesc searchtype = OIIO::TypeUnknown,
                                     bool casesensitive  = false);
    const OIIO::ParamValue*
    find_attribute(string_view name, TypeDesc searchtype = OIIO::TypeUnknown,
                   bool casesensitive = false) const;

    // Super simple camera and display parameters.  Many options not
    // available, no motion blur, etc.
    virtual void camera_params(const Matrix44& world_to_camera,
                               ustringhash projection, float hfov, float hither,
                               float yon, int xres, int yres);

    virtual void parse_scene_xml(const std::string& scenefile);
    virtual void prepare_render();
    void prepare_lights();
    void prepare_geometry();
    virtual void warmup() {}
    virtual void render(int xres, int yres);
    virtual void clear();

    // After render, get the pixels into pixelbuf, if they aren't already.
    virtual void finalize_pixel_buffer() {}

    // ShaderGroupRef storage
    MaterialVec& shaders() { return m_shaders; }

    OIIO::ErrorHandler& errhandler() const { return *m_errhandler; }

    /// True when the device path was explicitly requested (OSL_USE_VULKAN) and
    /// failed in a way that would otherwise have been papered over by
    /// re-rendering on the CPU.  The caller should treat the render as failed
    /// rather than keep an image the CPU produced.  Setting
    /// OSL_VULKAN_ALLOW_CPU_FALLBACK=1 restores the old, silent behaviour
    /// (useful when debugging the device path itself).
    bool vulkan_hard_failure() const { return m_vulkan_hard_failure; }

    const std::vector<bool>& shader_is_light() { return m_shader_is_light; }
    const std::vector<unsigned>& lightprims() { return m_lightprims; }

    Camera camera;
    Scene scene;
    Background background;
    ShadingSystem* shadingsys = nullptr;
    OIIO::ParamValueList options;
    OIIO::ImageBuf pixelbuf;

    int getBackgroundShaderID() const { return backgroundShaderID; }
    int getBackgroundResolution() const { return backgroundResolution; }

private:
    // Camera parameters
    Matrix44 m_world_to_camera;
    ustringhash m_projection;
    float m_fov, m_pixelaspect, m_hither, m_yon;
    float m_shutter[2];
    float m_screen_window[4];

    int backgroundShaderID   = -1;
    int backgroundResolution = 1024;
    int aa                   = 1;
    bool no_jitter           = false;
    int max_bounces          = 64;
    int rr_depth             = 5;
    float show_albedo_scale  = 0.0f;
    int show_globals         = 0;
    MaterialVec m_shaders;
    std::vector<bool> m_shader_is_light;
    std::vector<float>
        m_mesh_surfacearea;  // surface area of all triangles in each mesh (one entry per mesh)
    std::vector<unsigned>
        m_lightprims;  // array of all triangles that have a "light" shader on them

    class ErrorHandler;  // subclass ErrorHandler for SimpleRaytracer
    std::unique_ptr<OIIO::ErrorHandler> m_errhandler;
    bool m_had_error = false;

    // ------------------------------------------------------------------
    // Vulkan shading path
    // ------------------------------------------------------------------
    //
    // When the shading system was asked to lower groups to Vulkan SPIR-V
    // (OSL_USE_VULKAN=1), one module is assembled out of every surface group
    // plus the device frame renderer and dispatched once per image: ray
    // generation, intersection, shading, closure sampling and accumulation all
    // run on the device.  There is no per-material pipeline -- a whole-frame
    // kernel has to call any material and SPIR-V cannot call across modules --
    // so the host render loop only runs when the device path was not requested.
    std::unique_ptr<::VulkanShading> m_vk;
    // Which case of the synthesised material dispatch runs a given material.
    // The dispatch numbers the groups it was handed in order, skipping
    // materials without a surface group, so it is *not* necessarily the
    // material index -- and the kernel needs this number, not the index, for
    // the background shader.
    std::vector<int> m_vk_dispatch_id;
    // The wavefront pipelines, all built from the same module: generation,
    // intersection, and one shading pipeline per material (the host sorts the
    // wave by material and dispatches each bucket to its own pipeline).  The
    // host runs gen -> intersect -> {shade buckets} per bounce, per sample.
    int m_vk_gen       = -1;
    int m_vk_intersect = -1;
    std::vector<int> m_vk_shade;
    // The material that shades the light primitives, as a dispatch id.  Every
    // shading kernel has to reach it for the explicit light sample.
    int m_vk_light_group = -1;
    // Textures the shaders referenced while compiling; recorded by the
    // get_texture_handle() override and uploaded to the device.
    std::vector<ustring> m_vk_texture_names;
    /// OSL_USE_VULKAN was set: the caller wants the device path, so a device
    /// failure must be reported instead of quietly becoming a CPU render.
    bool m_vulkan_requested = false;
    /// A failure that has already been reported as fatal; render() then emits
    /// no image and testrender exits non-zero (see vulkan_hard_failure()).
    bool m_vulkan_hard_failure = false;
    /// Only the first failure is reported, so a per-pixel dispatch failure does
    /// not flood stderr from every thread.
    std::atomic<bool> m_vulkan_failure_reported { false };

    void init_vulkan_shading();
    /// OSL_VULKAN_ALLOW_CPU_FALLBACK=1: opt back into the CPU fallback.
    bool vulkan_fallback_allowed() const;
    /// Report a device failure that would otherwise have fallen back to the
    /// CPU, once.  No-op when Vulkan was not requested or the fallback is
    /// allowed.  Also marks the render as a hard failure.
    void vulkan_report_failure(const char* what, const char* detail);

    /// Decode and upload every recorded texture into the device buffer.
    void upload_vulkan_textures();
    /// Render the whole image with the batch pipeline: lay the scene out in the
    /// buffer, dispatch once, read the float image back into `pixelbuf`.
    bool render_vulkan_frame(int xres, int yres);

    // Named transforms
    typedef std::map<ustringhash, std::shared_ptr<Transformation>> TransformMap;
    TransformMap m_named_xforms;

    // Attribute and userdata retrieval -- for fast dispatch, use a hash
    // table to map attribute names to functions that retrieve them. We
    // imagine this to be fairly quick, but for a performance-critical
    // renderer, we would encourage benchmarking various methods and
    // alternate data structures.
    typedef bool (SimpleRaytracer::*AttrGetter)(ShaderGlobals* sg, bool derivs,
                                                ustringhash object,
                                                TypeDesc type, ustringhash name,
                                                void* val);
    typedef std::unordered_map<ustringhash, AttrGetter> AttrGetterMap;
    AttrGetterMap m_attr_getters;

    // Attribute getters
    bool get_osl_version(ShaderGlobals* sg, bool derivs, ustringhash object,
                         TypeDesc type, ustringhash name, void* val);
    bool get_camera_resolution(ShaderGlobals* sg, bool derivs,
                               ustringhash object, TypeDesc type,
                               ustringhash name, void* val);
    bool get_camera_projection(ShaderGlobals* sg, bool derivs,
                               ustringhash object, TypeDesc type,
                               ustringhash name, void* val);
    bool get_camera_fov(ShaderGlobals* sg, bool derivs, ustringhash object,
                        TypeDesc type, ustringhash name, void* val);
    bool get_camera_pixelaspect(ShaderGlobals* sg, bool derivs,
                                ustringhash object, TypeDesc type,
                                ustringhash name, void* val);
    bool get_camera_clip(ShaderGlobals* sg, bool derivs, ustringhash object,
                         TypeDesc type, ustringhash name, void* val);
    bool get_camera_clip_near(ShaderGlobals* sg, bool derivs,
                              ustringhash object, TypeDesc type,
                              ustringhash name, void* val);
    bool get_camera_clip_far(ShaderGlobals* sg, bool derivs, ustringhash object,
                             TypeDesc type, ustringhash name, void* val);
    bool get_camera_shutter(ShaderGlobals* sg, bool derivs, ustringhash object,
                            TypeDesc type, ustringhash name, void* val);
    bool get_camera_shutter_open(ShaderGlobals* sg, bool derivs,
                                 ustringhash object, TypeDesc type,
                                 ustringhash name, void* val);
    bool get_camera_shutter_close(ShaderGlobals* sg, bool derivs,
                                  ustringhash object, TypeDesc type,
                                  ustringhash name, void* val);
    bool get_camera_screen_window(ShaderGlobals* sg, bool derivs,
                                  ustringhash object, TypeDesc type,
                                  ustringhash name, void* val);

    // CPU renderer helpers
    void globals_from_hit(ShaderGlobals& sg, const Ray& r,
                          const Dual2<float>& t, int id, float u, float v);
    Vec3 eval_background(const Dual2<Vec3>& dir, ShadingContext* ctx,
                         int bounce = -1);
    Color3 subpixel_radiance(float x, float y, Sampler& sampler,
                             ShadingContext* ctx);
    Color3 antialias_pixel(int x, int y, ShadingContext* ctx);

    friend class ErrorHandler;
};

OSL_NAMESPACE_END
