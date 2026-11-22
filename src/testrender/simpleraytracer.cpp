// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage


#ifndef __CUDACC__
#    include <OpenImageIO/filesystem.h>
#    include <OpenImageIO/parallel.h>
#    include <OpenImageIO/timer.h>
#    include <pugixml.hpp>

#    ifdef USING_OIIO_PUGI
namespace pugi = OIIO::pugi;
#    endif

#    include <OSL/hashes.h>
#    include "raytracer.h"
#    include "shading.h"
#    include "simpleraytracer.h"
#endif

#include <cmath>

// The host-side Vulkan compute backend lives in liboslexec and pulls in LLVM,
// std::string, <OSL/oslconfig.h> -- none of which can be compiled for the CUDA
// device pass.  __CUDACC__ builds of this file (cuda/optix_raytracer.cu includes
// it) skip the Vulkan headers entirely; all Vulkan code below sits inside
// #ifndef __CUDACC__ anyway.
#ifndef __CUDACC__
#    include "vulkan_shading.h"
#    include "vulkan_types.h"
#    include "vulkan_module.h"
#    include "clspv_util.h"
#endif
#include <algorithm>

// Create ustrings for all strings used by the free function renderer services.
// Required to allow the reverse mapping of hash->string to work when processing messages
namespace RS {
namespace Strings {
#define RS_STRDECL(str, var_name) const OSL::ustring var_name { str };
#include "rs_strdecls.h"
#undef RS_STRDECL
}  // namespace Strings
}  // namespace RS

namespace RS {
namespace {
namespace Hashes {
#define RS_STRDECL(str, var_name) \
    constexpr OSL::ustringhash var_name(OSL::strhash(str));
#include "rs_strdecls.h"
#undef RS_STRDECL
};  //namespace Hashes
}  // unnamed namespace
};  //namespace RS

using namespace OSL;

OSL_NAMESPACE_BEGIN


#ifndef __CUDACC__
using ShaderGlobalsType = OSL::ShaderGlobals;
#else
using ShaderGlobalsType = OSL_CUDA::ShaderGlobals;
#endif


#ifndef __CUDACC__
static TypeDesc TypeFloatArray2(TypeDesc::FLOAT, 2);
static TypeDesc TypeFloatArray4(TypeDesc::FLOAT, 4);
static TypeDesc TypeIntArray2(TypeDesc::INT, 2);



// Subclass ErrorHandler
class SimpleRaytracer::ErrorHandler final : public OIIO::ErrorHandler {
public:
    ErrorHandler(SimpleRaytracer& rend) : m_rend(rend) {}

    virtual void operator()(int errcode, const std::string& msg)
    {
        OIIO::ErrorHandler::operator()(errcode, msg);
        if (errcode & OIIO::ErrorHandler::EH_ERROR
            || errcode & OIIO::ErrorHandler::EH_SEVERE)
            m_rend.m_had_error = true;
        if (errcode & OIIO::ErrorHandler::EH_SEVERE)
            exit(EXIT_FAILURE);
    }

private:
    SimpleRaytracer& m_rend;
};



SimpleRaytracer::SimpleRaytracer()
{
    m_errhandler.reset(new SimpleRaytracer::ErrorHandler(*this));

    Matrix44 M;
    M.makeIdentity();
    camera_params(M, RS::Hashes::perspective, 90.0f, 0.1f, 1000.0f, 256, 256);

    // Set up getters
    m_attr_getters[RS::Hashes::osl_version] = &SimpleRaytracer::get_osl_version;
    m_attr_getters[RS::Hashes::camera_resolution]
        = &SimpleRaytracer::get_camera_resolution;
    m_attr_getters[RS::Hashes::camera_projection]
        = &SimpleRaytracer::get_camera_projection;
    m_attr_getters[RS::Hashes::camera_pixelaspect]
        = &SimpleRaytracer::get_camera_pixelaspect;
    m_attr_getters[RS::Hashes::camera_screen_window]
        = &SimpleRaytracer::get_camera_screen_window;
    m_attr_getters[RS::Hashes::camera_fov]  = &SimpleRaytracer::get_camera_fov;
    m_attr_getters[RS::Hashes::camera_clip] = &SimpleRaytracer::get_camera_clip;
    m_attr_getters[RS::Hashes::camera_clip_near]
        = &SimpleRaytracer::get_camera_clip_near;
    m_attr_getters[RS::Hashes::camera_clip_far]
        = &SimpleRaytracer::get_camera_clip_far;
    m_attr_getters[RS::Hashes::camera_shutter]
        = &SimpleRaytracer::get_camera_shutter;
    m_attr_getters[RS::Hashes::camera_shutter_open]
        = &SimpleRaytracer::get_camera_shutter_open;
    m_attr_getters[RS::Hashes::camera_shutter_close]
        = &SimpleRaytracer::get_camera_shutter_close;
}



OIIO::ParamValue*
SimpleRaytracer::find_attribute(string_view name, TypeDesc searchtype,
                                bool casesensitive)
{
    auto iter = options.find(name, searchtype, casesensitive);
    if (iter != options.end())
        return &(*iter);
    return nullptr;
}



const OIIO::ParamValue*
SimpleRaytracer::find_attribute(string_view name, TypeDesc searchtype,
                                bool casesensitive) const
{
    auto iter = options.find(name, searchtype, casesensitive);
    if (iter != options.end())
        return &(*iter);
    return nullptr;
}



void
SimpleRaytracer::attribute(string_view name, TypeDesc type, const void* value)
{
    if (name.empty())  // Guard against bogus empty names
        return;
    // Don't allow duplicates
    auto f = find_attribute(name);
    if (!f) {
        options.resize(options.size() + 1);
        f = &options.back();
    }
    f->init(name, type, 1, value);
}



void
SimpleRaytracer::camera_params(const Matrix44& world_to_camera,
                               ustringhash projection, float hfov, float hither,
                               float yon, int xres, int yres)
{
    m_world_to_camera  = world_to_camera;
    m_projection       = projection;
    m_fov              = hfov;
    m_pixelaspect      = 1.0f;  // hard-coded
    m_hither           = hither;
    m_yon              = yon;
    m_shutter[0]       = 0.0f;
    m_shutter[1]       = 1.0f;  // hard-coded
    float frame_aspect = float(xres) / float(yres) * m_pixelaspect;
    m_screen_window[0] = -frame_aspect;
    m_screen_window[1] = -1.0f;
    m_screen_window[2] = frame_aspect;
    m_screen_window[3] = 1.0f;
    camera.xres        = xres;
    camera.yres        = yres;
}



inline Vec3
strtovec(string_view str)
{
    Vec3 v(0, 0, 0);
    OIIO::Strutil::parse_float(str, v[0]);
    OIIO::Strutil::parse_char(str, ',');
    OIIO::Strutil::parse_float(str, v[1]);
    OIIO::Strutil::parse_char(str, ',');
    OIIO::Strutil::parse_float(str, v[2]);
    return v;
}

inline bool
strtobool(const char* str)
{
    return strcmp(str, "1") == 0 || strcmp(str, "on") == 0
           || strcmp(str, "yes") == 0;
}

template<int N> struct ParamStorage {
    ParamStorage() : fparamindex(0), iparamindex(0), sparamindex(0) {}

    void* Int(int i)
    {
        OSL_DASSERT(iparamindex < N);
        iparamdata[iparamindex] = i;
        iparamindex++;
        return &iparamdata[iparamindex - 1];
    }

    void* Float(float f)
    {
        OSL_DASSERT(fparamindex < N);
        fparamdata[fparamindex] = f;
        fparamindex++;
        return &fparamdata[fparamindex - 1];
    }

    void* Vec(float x, float y, float z)
    {
        Float(x);
        Float(y);
        Float(z);
        return &fparamdata[fparamindex - 3];
    }

    void* Vec(const float* xyz)
    {
        Float(xyz[0]);
        Float(xyz[1]);
        Float(xyz[2]);
        return &fparamdata[fparamindex - 3];
    }

    void* Str(const char* str)
    {
        OSL_DASSERT(sparamindex < N);
        sparamdata[sparamindex] = ustring(str);
        sparamindex++;
        return &sparamdata[sparamindex - 1];
    }

private:
    // storage for shader parameters
    float fparamdata[N];
    int iparamdata[N];
    ustring sparamdata[N];

    int fparamindex;
    int iparamindex;
    int sparamindex;
};



inline bool
parse_prefix_and_floats(string_view str, string_view prefix, int nvals,
                        float* vals)
{
    bool ok = OIIO::Strutil::parse_prefix(str, prefix);
    for (int i = 0; i < nvals && ok; ++i)
        ok &= OIIO::Strutil::parse_float(str, vals[i]);
    return ok;
}


inline bool
parse_prefix_and_ints(string_view str, string_view prefix, int nvals, int* vals)
{
    bool ok = OIIO::Strutil::parse_prefix(str, prefix);
    for (int i = 0; i < nvals && ok; ++i)
        ok &= OIIO::Strutil::parse_int(str, vals[i]);
    return ok;
}



void
SimpleRaytracer::parse_scene_xml(const std::string& scenefile)
{
    ShaderMap shadermap;
    pugi::xml_document doc;
    pugi::xml_parse_result parse_result;
    if (OIIO::Strutil::ends_with(scenefile, ".xml")
        && OIIO::Filesystem::exists(scenefile)) {
        parse_result = doc.load_file(scenefile.c_str());
    } else {
        parse_result = doc.load_buffer(scenefile.c_str(), scenefile.size());
    }
    if (!parse_result)
        errhandler().severefmt("XML parsed with errors: {} at offset {}",
                               parse_result.description(), parse_result.offset);
    pugi::xml_node root = doc.child("World");
    if (!root)
        errhandler().severefmt(
            "Error reading scene: Root element <World> is missing");

    // loop over all children of world
    for (auto node = root.first_child(); node; node = node.next_sibling()) {
        if (strcmp(node.name(), "Option") == 0) {
            for (auto attr = node.first_attribute(); attr;
                 attr      = attr.next_attribute()) {
                int i = 0;
                if (parse_prefix_and_ints(attr.value(), "int ", 1, &i))
                    attribute(attr.name(), i);
                // TODO: pass any extra options to shading system (or texture system?)
            }
        } else if (strcmp(node.name(), "Camera") == 0) {
            // defaults
            Vec3 eye(0, 0, 0);
            Vec3 dir(0, 0, -1);
            Vec3 up(0, 1, 0);
            float fov = 90.f;

            // load camera (only first attribute counts if duplicates)
            pugi::xml_attribute eye_attr = node.attribute("eye");
            pugi::xml_attribute dir_attr = node.attribute("dir");
            pugi::xml_attribute at_attr  = node.attribute("look_at");
            pugi::xml_attribute up_attr  = node.attribute("up");
            pugi::xml_attribute fov_attr = node.attribute("fov");

            if (eye_attr)
                eye = strtovec(eye_attr.value());
            if (dir_attr)
                dir = strtovec(dir_attr.value());
            else if (at_attr)
                dir = strtovec(at_attr.value()) - eye;
            if (up_attr)
                up = strtovec(up_attr.value());
            if (fov_attr)
                fov = OIIO::Strutil::from_string<float>(fov_attr.value());

            camera.lookat(eye, dir, up, fov);
        } else if (strcmp(node.name(), "Sphere") == 0) {
            // load sphere
            pugi::xml_attribute center_attr = node.attribute("center");
            pugi::xml_attribute radius_attr = node.attribute("radius");

            if (center_attr && radius_attr) {
                Vec3 center  = strtovec(center_attr.value());
                float radius = OIIO::Strutil::from_string<float>(
                    radius_attr.value());
                if (radius > 0) {
                    pugi::xml_attribute resolution_attr = node.attribute(
                        "resolution");
                    int resolution = 64;
                    if (resolution_attr) {
                        OIIO::string_view str = resolution_attr.value();
                        OIIO::Strutil::parse_int(str, resolution);
                    }
                    scene.add_sphere(center, radius, int(shaders().size()) - 1,
                                     resolution);
                }
            }
        } else if (strcmp(node.name(), "Quad") == 0) {
            // load quad
            pugi::xml_attribute corner_attr = node.attribute("corner");
            pugi::xml_attribute edge_x_attr = node.attribute("edge_x");
            pugi::xml_attribute edge_y_attr = node.attribute("edge_y");
            if (corner_attr && edge_x_attr && edge_y_attr) {
                Vec3 co = strtovec(corner_attr.value());
                Vec3 ex = strtovec(edge_x_attr.value());
                Vec3 ey = strtovec(edge_y_attr.value());

                int resolution                      = 1;
                pugi::xml_attribute resolution_attr = node.attribute(
                    "resolution");
                if (resolution_attr) {
                    OIIO::string_view str = resolution_attr.value();
                    OIIO::Strutil::parse_int(str, resolution);
                }

                scene.add_quad(co, ex, ey, int(shaders().size()) - 1,
                               resolution);
            }
        } else if (strcmp(node.name(), "Model") == 0) {
            // load .obj model
            pugi::xml_attribute filename_attr = node.attribute("filename");
            if (filename_attr) {
                std::string filename = filename_attr.value();
                std::vector<std::string> searchpath;
                searchpath.emplace_back(
                    OIIO::Filesystem::parent_path(scenefile));
                std::string actual_filename
                    = OIIO::Filesystem::searchpath_find(filename, searchpath,
                                                        false);
                if (actual_filename.empty()) {
                    errhandler().errorfmt("Unable to find model file {}",
                                          filename);
                } else {
                    // we got a valid filename, try to load the model
                    scene.add_model(actual_filename, shadermap,
                                    int(shaders().size() - 1), errhandler());
                }
            }
        } else if (strcmp(node.name(), "Background") == 0) {
            pugi::xml_attribute res_attr = node.attribute("resolution");
            if (res_attr)
                backgroundResolution = OIIO::Strutil::from_string<int>(
                    res_attr.value());
            backgroundShaderID = int(shaders().size()) - 1;
        } else if (strcmp(node.name(), "ShaderGroup") == 0) {
            ShaderGroupRef group;
            pugi::xml_attribute is_light_attr = node.attribute("is_light");
            pugi::xml_attribute name_attr     = node.attribute("name");
            std::string name = name_attr ? name_attr.value() : "group";
            pugi::xml_attribute type_attr = node.attribute("type");
            std::string shadertype = type_attr ? type_attr.value() : "surface";
            pugi::xml_attribute commands_attr = node.attribute("commands");
            std::string commands = commands_attr ? commands_attr.value()
                                                 : node.text().get();
            if (commands.size())
                group = shadingsys->ShaderGroupBegin(name, shadertype,
                                                     commands);
            else
                group = shadingsys->ShaderGroupBegin(name);
            ParamStorage<1024>
                store;  // scratch space to hold parameters until they are read by Shader()
            for (pugi::xml_node gnode = node.first_child(); gnode;
                 gnode                = gnode.next_sibling()) {
                if (strcmp(gnode.name(), "Parameter") == 0) {
                    // handle parameters
                    for (auto attr = gnode.first_attribute(); attr;
                         attr      = attr.next_attribute()) {
                        int i = 0;
                        float f[3];
                        string_view val(attr.value());
                        if (parse_prefix_and_ints(val, "int ", 1, &i))
                            shadingsys->Parameter(*group, attr.name(), TypeInt,
                                                  store.Int(i));
                        else if (parse_prefix_and_floats(val, "float ", 1, f))
                            shadingsys->Parameter(*group, attr.name(),
                                                  TypeFloat, store.Float(f[0]));
                        else if (parse_prefix_and_floats(val, "vector ", 3, f))
                            shadingsys->Parameter(*group, attr.name(),
                                                  TypeVector, store.Vec(f));
                        else if (parse_prefix_and_floats(val, "point ", 3, f))
                            shadingsys->Parameter(*group, attr.name(),
                                                  TypePoint, store.Vec(f));
                        else if (parse_prefix_and_floats(val, "color ", 3, f))
                            shadingsys->Parameter(*group, attr.name(),
                                                  TypeColor, store.Vec(f));
                        else
                            shadingsys->Parameter(*group, attr.name(),
                                                  TypeString,
                                                  store.Str(attr.value()));
                    }
                } else if (strcmp(gnode.name(), "Shader") == 0) {
                    pugi::xml_attribute type_attr  = gnode.attribute("type");
                    pugi::xml_attribute name_attr  = gnode.attribute("name");
                    pugi::xml_attribute layer_attr = gnode.attribute("layer");
                    const char* type = type_attr ? type_attr.value()
                                                 : "surface";
                    if (name_attr && layer_attr)
                        shadingsys->Shader(*group, type, name_attr.value(),
                                           layer_attr.value());
                } else if (strcmp(gnode.name(), "ConnectShaders") == 0) {
                    // FIXME: find a more elegant way to encode this
                    pugi::xml_attribute sl = gnode.attribute("srclayer");
                    pugi::xml_attribute sp = gnode.attribute("srcparam");
                    pugi::xml_attribute dl = gnode.attribute("dstlayer");
                    pugi::xml_attribute dp = gnode.attribute("dstparam");
                    if (sl && sp && dl && dp)
                        shadingsys->ConnectShaders(*group, sl.value(),
                                                   sp.value(), dl.value(),
                                                   dp.value());
                } else {
                    // unknown element?
                }
            }
            shadingsys->ShaderGroupEnd(*group);
            if (name_attr) {
                if (auto it = shadermap.find(name); it != shadermap.end()) {
                    int shaderID = it->second;
                    if (shaderID >= 0 && shaderID < int(shaders().size())) {
                        fprintf(stderr, "Updating shader %d - %s\n", shaderID,
                                shadertype.c_str());
                        // we already have a material under this name,
                        Material& m = shaders()[shaderID];
                        // TODO: could we query the shadertype directly from the ShaderGroup?
                        if (shadertype == "displacement")
                            m.disp = group;
                        else if (shadertype == "surface")
                            m.surf = group;
                        // skip the rest which would add a new material
                        continue;
                    }
                } else {
                    shadermap.emplace(name, int(shaders().size()));
                }
            }
            shaders().emplace_back(Material { group, nullptr });
            m_shader_is_light.emplace_back(
                is_light_attr ? strtobool(is_light_attr.value()) : false);
        } else {
            // unknown element?
        }
    }
    if (root.next_sibling())
        errhandler().severefmt(
            "Error reading {}: Found multiple top-level elements", scenefile);
    if (shaders().empty())
        errhandler().severefmt("No shaders in scene");
    if (scene.num_prims() == 0)
        errhandler().severefmt("No primitives in scene");
    camera.finalize();
}



bool
SimpleRaytracer::get_matrix(ShaderGlobals* /*sg*/, Matrix44& result,
                            TransformationPtr xform, float /*time*/)
{
    // SimpleRaytracer doesn't understand motion blur and transformations
    // are just simple 4x4 matrices.
    if (xform)
        result = *reinterpret_cast<const Matrix44*>(xform);
    else
        result = Matrix44(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);

    return true;
}



bool
SimpleRaytracer::get_matrix(ShaderGlobals* /*sg*/, Matrix44& result,
                            ustringhash from, float /*time*/)
{
    TransformMap::const_iterator found = m_named_xforms.find(from);
    if (found != m_named_xforms.end()) {
        result = *(found->second);
        return true;
    } else {
        return false;
    }
}



bool
SimpleRaytracer::get_matrix(ShaderGlobals* /*sg*/, Matrix44& result,
                            TransformationPtr xform)
{
    // SimpleRaytracer doesn't understand motion blur and transformations
    // are just simple 4x4 matrices.
    if (xform)
        result = *reinterpret_cast<const Matrix44*>(xform);
    else
        result = Matrix44(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);
    return true;
}



bool
SimpleRaytracer::get_matrix(ShaderGlobals* /*sg*/, Matrix44& result,
                            ustringhash from)
{
    // SimpleRaytracer doesn't understand motion blur, so we never fail
    // on account of time-varying transformations.
    TransformMap::const_iterator found = m_named_xforms.find(from);
    if (found != m_named_xforms.end()) {
        result = *(found->second);
        return true;
    } else {
        return false;
    }
}



bool
SimpleRaytracer::get_inverse_matrix(ShaderGlobals* /*sg*/, Matrix44& result,
                                    ustringhash to, float /*time*/)
{
    if (to == OSL::Hashes::camera || to == OSL::Hashes::screen
        || to == OSL::Hashes::NDC || to == RS::Hashes::raster) {
        // clang-format off
        Matrix44 M = m_world_to_camera;
        if (to == OSL::Hashes::screen || to == OSL::Hashes::NDC || to == RS::Hashes::raster) {
            float depthrange = (double)m_yon-(double)m_hither;
            if (m_projection == RS::Hashes::perspective) {
                float tanhalffov = tanf (0.5f * m_fov * M_PI/180.0);
                Matrix44 camera_to_screen (1/tanhalffov, 0, 0, 0,
                                           0, 1/tanhalffov, 0, 0,
                                           0, 0, m_yon/depthrange, 1,
                                           0, 0, -m_yon*m_hither/depthrange, 0);
                M = M * camera_to_screen;
            } else {
                Matrix44 camera_to_screen (1, 0, 0, 0,
                                           0, 1, 0, 0,
                                           0, 0, 1/depthrange, 0,
                                           0, 0, -m_hither/depthrange, 1);
                M = M * camera_to_screen;
            }
            if (to == OSL::Hashes::NDC || to == RS::Hashes::raster) {
                float screenleft = -1.0, screenwidth = 2.0;
                float screenbottom = -1.0, screenheight = 2.0;
                Matrix44 screen_to_ndc (1/screenwidth, 0, 0, 0,
                                        0, 1/screenheight, 0, 0,
                                        0, 0, 1, 0,
                                        -screenleft/screenwidth, -screenbottom/screenheight, 0, 1);
                M = M * screen_to_ndc;
                if (to == RS::Hashes::raster) {
                    Matrix44 ndc_to_raster (camera.xres, 0, 0, 0,
                                            0, camera.yres, 0, 0,
                                            0, 0, 1, 0,
                                            0, 0, 0, 1);
                    M = M * ndc_to_raster;
                }
            }
        }
        // clang-format on
        result = M;
        return true;
    }

    TransformMap::const_iterator found = m_named_xforms.find(to);
    if (found != m_named_xforms.end()) {
        result = *(found->second);
        result.invert();
        return true;
    } else {
        return false;
    }
}



void
SimpleRaytracer::name_transform(const char* name, const OSL::Matrix44& xform)
{
    std::shared_ptr<Transformation> M(new OSL::Matrix44(xform));
    m_named_xforms[ustringhash(name)] = M;
}



bool
SimpleRaytracer::get_array_attribute(ShaderGlobals* sg, bool derivatives,
                                     ustringhash object, TypeDesc type,
                                     ustringhash name, int index, void* val)
{
    AttrGetterMap::const_iterator g = m_attr_getters.find(name);
    if (g != m_attr_getters.end()) {
        AttrGetter getter = g->second;
        return (this->*(getter))(sg, derivatives, object, type, name, val);
    }

    // If no named attribute was found, allow userdata to bind to the
    // attribute request.
    if (object.empty() && index == -1)
        return get_userdata(derivatives, name, type, sg, val);

    return false;
}



bool
SimpleRaytracer::get_attribute(ShaderGlobals* sg, bool derivatives,
                               ustringhash object, TypeDesc type,
                               ustringhash name, void* val)
{
    return get_array_attribute(sg, derivatives, object, type, name, -1, val);
}



bool
SimpleRaytracer::get_userdata(bool derivatives, ustringhash name, TypeDesc type,
                              ShaderGlobals* sg, void* val)
{
    // Just to illustrate how this works, respect s and t userdata, filled
    // in with the uv coordinates.  In a real renderer, it would probably
    // look up something specific to the primitive, rather than have hard-
    // coded names.

    if (name == RS::Hashes::s && type == TypeFloat) {
        ((float*)val)[0] = sg->u;
        if (derivatives) {
            ((float*)val)[1] = sg->dudx;
            ((float*)val)[2] = sg->dudy;
        }
        return true;
    }
    if (name == RS::Hashes::t && type == TypeFloat) {
        ((float*)val)[0] = sg->v;
        if (derivatives) {
            ((float*)val)[1] = sg->dvdx;
            ((float*)val)[2] = sg->dvdy;
        }
        return true;
    }

    return false;
}


// The base class returns null, which is also what the CPU path wants (OSL then
// looks the texture up by name at shade time).  We only piggyback on the call
// to learn the filename, so the Vulkan path knows what to decode and upload.
RendererServices::TextureHandle*
SimpleRaytracer::get_texture_handle(ustring filename, ShadingContext* /*context*/,
                                    const TextureOpt* /*options*/)
{
    for (const auto& n : m_vk_texture_names)
        if (n == filename)
            return nullptr;
    m_vk_texture_names.push_back(filename);
    return nullptr;
}


bool
SimpleRaytracer::get_osl_version(ShaderGlobals* /*sg*/, bool /*derivs*/,
                                 ustringhash /*object*/, TypeDesc type,
                                 ustringhash /*name*/, void* val)
{
    if (type == TypeInt) {
        ((int*)val)[0] = OSL_VERSION;
        return true;
    }
    return false;
}


bool
SimpleRaytracer::get_camera_resolution(ShaderGlobals* /*sg*/, bool /*derivs*/,
                                       ustringhash /*object*/, TypeDesc type,
                                       ustringhash /*name*/, void* val)
{
    if (type == TypeIntArray2) {
        ((int*)val)[0] = camera.xres;
        ((int*)val)[1] = camera.yres;
        return true;
    }
    return false;
}


bool
SimpleRaytracer::get_camera_projection(ShaderGlobals* /*sg*/, bool /*derivs*/,
                                       ustringhash /*object*/, TypeDesc type,
                                       ustringhash /*name*/, void* val)
{
    if (type == TypeString) {
        ((ustringhash*)val)[0] = m_projection;
        return true;
    }
    return false;
}


bool
SimpleRaytracer::get_camera_fov(ShaderGlobals* /*sg*/, bool derivs,
                                ustringhash /*object*/, TypeDesc type,
                                ustringhash /*name*/, void* val)
{
    // N.B. in a real renderer, this may be time-dependent
    if (type == TypeFloat) {
        ((float*)val)[0] = m_fov;
        if (derivs)
            memset((char*)val + type.size(), 0, 2 * type.size());
        return true;
    }
    return false;
}


bool
SimpleRaytracer::get_camera_pixelaspect(ShaderGlobals* /*sg*/, bool derivs,
                                        ustringhash /*object*/, TypeDesc type,
                                        ustringhash /*name*/, void* val)
{
    if (type == TypeFloat) {
        ((float*)val)[0] = m_pixelaspect;
        if (derivs)
            memset((char*)val + type.size(), 0, 2 * type.size());
        return true;
    }
    return false;
}


bool
SimpleRaytracer::get_camera_clip(ShaderGlobals* /*sg*/, bool derivs,
                                 ustringhash /*object*/, TypeDesc type,
                                 ustringhash /*name*/, void* val)
{
    if (type == TypeFloatArray2) {
        ((float*)val)[0] = m_hither;
        ((float*)val)[1] = m_yon;
        if (derivs)
            memset((char*)val + type.size(), 0, 2 * type.size());
        return true;
    }
    return false;
}


bool
SimpleRaytracer::get_camera_clip_near(ShaderGlobals* /*sg*/, bool derivs,
                                      ustringhash /*object*/, TypeDesc type,
                                      ustringhash /*name*/, void* val)
{
    if (type == TypeFloat) {
        ((float*)val)[0] = m_hither;
        if (derivs)
            memset((char*)val + type.size(), 0, 2 * type.size());
        return true;
    }
    return false;
}


bool
SimpleRaytracer::get_camera_clip_far(ShaderGlobals* /*sg*/, bool derivs,
                                     ustringhash /*object*/, TypeDesc type,
                                     ustringhash /*name*/, void* val)
{
    if (type == TypeFloat) {
        ((float*)val)[0] = m_yon;
        if (derivs)
            memset((char*)val + type.size(), 0, 2 * type.size());
        return true;
    }
    return false;
}



bool
SimpleRaytracer::get_camera_shutter(ShaderGlobals* /*sg*/, bool derivs,
                                    ustringhash /*object*/, TypeDesc type,
                                    ustringhash /*name*/, void* val)
{
    if (type == TypeFloatArray2) {
        ((float*)val)[0] = m_shutter[0];
        ((float*)val)[1] = m_shutter[1];
        if (derivs)
            memset((char*)val + type.size(), 0, 2 * type.size());
        return true;
    }
    return false;
}


bool
SimpleRaytracer::get_camera_shutter_open(ShaderGlobals* /*sg*/, bool derivs,
                                         ustringhash /*object*/, TypeDesc type,
                                         ustringhash /*name*/, void* val)
{
    if (type == TypeFloat) {
        ((float*)val)[0] = m_shutter[0];
        if (derivs)
            memset((char*)val + type.size(), 0, 2 * type.size());
        return true;
    }
    return false;
}


bool
SimpleRaytracer::get_camera_shutter_close(ShaderGlobals* /*sg*/, bool derivs,
                                          ustringhash /*object*/, TypeDesc type,
                                          ustringhash /*name*/, void* val)
{
    if (type == TypeFloat) {
        ((float*)val)[0] = m_shutter[1];
        if (derivs)
            memset((char*)val + type.size(), 0, 2 * type.size());
        return true;
    }
    return false;
}


bool
SimpleRaytracer::get_camera_screen_window(ShaderGlobals* /*sg*/, bool derivs,
                                          ustringhash /*object*/, TypeDesc type,
                                          ustringhash /*name*/, void* val)
{
    // N.B. in a real renderer, this may be time-dependent
    if (type == TypeFloatArray4) {
        ((float*)val)[0] = m_screen_window[0];
        ((float*)val)[1] = m_screen_window[1];
        ((float*)val)[2] = m_screen_window[2];
        ((float*)val)[3] = m_screen_window[3];
        if (derivs)
            memset((char*)val + type.size(), 0, 2 * type.size());
        return true;
    }
    return false;
}
#endif  // #ifndef __CUDACC__



void OSL_HOSTDEVICE
SimpleRaytracer::globals_from_hit(ShaderGlobalsType& sg, const Ray& r,
                                  const Dual2<float>& t, int id, float u,
                                  float v)
{
#ifndef __CUDACC__
    memset((char*)&sg, 0, sizeof(ShaderGlobals));
#endif

#ifndef __CUDACC__
    const int meshid = std::upper_bound(scene.last_index.begin(),
                                        scene.last_index.end(), id)
                       - scene.last_index.begin();
#else
    const int meshid = m_meshids[id];
#endif

    Dual2<Vec3> direction = r.dual_direction();
    sg.I                  = direction.val();
    sg.dIdx               = direction.dx();
    sg.dIdy               = direction.dy();
    Dual2<Vec3> P         = r.point(t);
    sg.P                  = P.val();
    sg.N                  = scene.normal(P, sg.Ng, id, u, v);
    // Projecting onto the surface here
    scene.project(P, sg.N, sg.I);
    sg.dPdx        = P.dx();
    sg.dPdy        = P.dy();
    Dual2<Vec2> uv = scene.uv(P, sg.N, sg.dPdu, sg.dPdv, id, u, v);
    sg.u           = uv.val().x;
    sg.dudx        = uv.dx().x;
    sg.dudy        = uv.dy().x;
    sg.v           = uv.val().y;
    sg.dvdx        = uv.dx().y;
    sg.dvdy        = uv.dy().y;
    sg.surfacearea = m_mesh_surfacearea[meshid];
    sg.backfacing  = sg.Ng.dot(sg.I) > 0;
    if (sg.backfacing) {
        sg.N  = -sg.N;
        sg.Ng = -sg.Ng;
    }
    sg.raytype        = r.raytype;
    sg.flipHandedness = sg.dPdx.cross(sg.dPdy).dot(sg.N) < 0;

#ifndef __CUDACC__
    // In our SimpleRaytracer, the "renderstate" itself just a pointer to
    // the ShaderGlobals.
    sg.renderstate = &sg;
#endif
}



OSL_HOSTDEVICE Vec3
SimpleRaytracer::eval_background(const Dual2<Vec3>& dir, ShadingContext* ctx,
                                 int bounce)
{
    ShaderGlobalsType sg;
    memset((char*)&sg, 0, sizeof(ShaderGlobals));
    sg.I    = dir.val();
    sg.dIdx = dir.dx();
    sg.dIdy = dir.dy();
    if (bounce >= 0)
        sg.raytype = bounce > 0 ? Ray::DIFFUSE : Ray::CAMERA;
#ifndef __CUDACC__
    shadingsys->execute(*ctx, *m_shaders[backgroundShaderID].surf, sg);
#else
    StackClosurePool closure_pool;
    execute_shader(sg, render_params.bg_id, closure_pool);
#endif
    return process_background_closure((const ClosureColor*)sg.Ci);
}

Color3
SimpleRaytracer::subpixel_radiance(float x, float y, Sampler& sampler,
                                   ShadingContext* ctx)
{
#ifdef __CUDACC__
    // Scratch space for the output closures
    StackClosurePool closure_pool;
    StackClosurePool light_closure_pool;
#endif

    constexpr float inf = std::numeric_limits<float>::infinity();
    Ray r               = camera.get(x, y);
    Color3 path_weight(1, 1, 1);
    Color3 path_radiance(0, 0, 0);
    int prev_id    = -1;
    float bsdf_pdf = inf;  // camera ray has only one possible direction
    MediumStack medium_stack;


    for (int b = 0; b <= max_bounces; b++) {
        ShaderGlobalsType sg;

        // trace the ray against the scene
        Intersection hit = scene.intersect(r, inf, prev_id);
        if (hit.t == inf) {
            // we hit nothing? check background shader
            if (backgroundShaderID >= 0) {
                if (b > 0 && backgroundResolution > 0) {
                    float bg_pdf = 0;
                    Vec3 bg      = background.eval(r.direction, bg_pdf);
                    path_radiance
                        += path_weight * bg
                           * MIS::power_heuristic<MIS::WEIGHT_WEIGHT>(bsdf_pdf,
                                                                      bg_pdf);
                } else {
                    // we aren't importance sampling the background - so just run it directly
                    path_radiance += path_weight
                                     * eval_background(r.direction, ctx, b);
                }
            }
            break;
        }

        if (medium_stack.integrate(r, sampler, hit, path_weight, path_radiance,
                                   bsdf_pdf)) {
            continue;
        }

        // construct a shader globals for the hit point
        globals_from_hit(sg, r, hit.t, hit.id, hit.u, hit.v);

        if (show_globals) {
            // visualize the main fields of the shader globals
            Vec3 v = sg.Ng;
            if (show_globals == 2)
                v = sg.N;
            if (show_globals == 3)
                v = sg.dPdu.normalize();
            if (show_globals == 4)
                v = sg.dPdv.normalize();
            if (show_globals == 5)
                v = Vec3(sg.u, sg.v, 0);
            Color3 c(v.x, v.y, v.z);
            if (show_globals != 5)
                c = c * 0.5f + Color3(0.5f);
            path_radiance += path_weight * c;
            break;
        }

        const float radius = r.radius + r.spread * hit.t;

        int shaderID = scene.shaderid(hit.id);

#ifndef __CUDACC__
        if (shaderID < 0 || !m_shaders[shaderID].surf)
            break;  // no shader attached? done

        // execute shader and process the resulting list of closures
        if (!m_vulkan_ready || !shade_vulkan(sg, shaderID, ctx)) {
            // When the device path was explicitly requested and shading it on
            // the CPU is not allowed, leave this path unshaded: computing it
            // here would produce a frame that looks plausible while having
            // come from the CPU.  shade_vulkan() has already reported the
            // failure.
            if (m_vulkan_requested && !vulkan_fallback_allowed())
                break;
            shadingsys->execute(*ctx, *m_shaders[shaderID].surf, sg);
            if (getenv("OSL_VULKAN_SHADE_TRACE") && sg.Ci) {
                const int* id = &sg.Ci->id;
                const float* w = (const float*)(id + 1);
                fprintf(stderr, "[cpu] material %d root id=%d w=(%g,%g,%g)\n",
                        shaderID, *id, w[0], w[1], w[2]);
            }
        }
#else
        if (shaderID < 0)
            break;  // no shader attached? done
        execute_shader(sg, shaderID, closure_pool);
#endif
        ShadingResult result;
        bool last_bounce = b == max_bounces;
        process_closure(sg, r.roughness, result, medium_stack,
                        (const ClosureColor*)sg.Ci, last_bounce);

#ifndef __CUDACC__
        const size_t lightprims_size = m_lightprims.size();
#endif

        // add self-emission
        float k = 1;
        if (m_shader_is_light[shaderID] && lightprims_size > 0) {
            const float light_pick_pdf = 1.0f / lightprims_size;
            // figure out the probability of reaching this point
            float light_pdf = light_pick_pdf
                              * scene.shapepdf(hit.id, r.origin, sg.P);
            k = MIS::power_heuristic<MIS::WEIGHT_EVAL>(bsdf_pdf, light_pdf);
        }
        path_radiance += path_weight * k * result.Le;

        // last bounce? nothing left to do
        if (last_bounce)
            break;

        // build internal pdf for sampling between bsdf closures
        result.bsdf.prepare(-sg.I, path_weight, b >= rr_depth);

        if (show_albedo_scale > 0) {
            // Instead of path tracing, just visualize the albedo
            // of the bsdf. This can be used to validate the accuracy of
            // the get_albedo method for a particular bsdf.
            path_radiance += path_weight * result.bsdf.get_albedo(-sg.I)
                             * show_albedo_scale;
            break;
        }

        // get three random numbers
        Vec3 s   = sampler.get();
        float xi = s.x;
        float yi = s.y;
        float zi = s.z;

        // trace one ray to the background
        if (backgroundResolution > 0) {
            Dual2<Vec3> bg_dir;
            float bg_pdf   = 0;
            Vec3 bg        = background.sample(xi, yi, bg_dir, bg_pdf);
            BSDF::Sample b = result.bsdf.eval(-sg.I, bg_dir.val());
            Color3 contrib = path_weight * b.weight * bg
                             * MIS::power_heuristic<MIS::WEIGHT_WEIGHT>(bg_pdf,
                                                                        b.pdf);
            if ((contrib.x + contrib.y + contrib.z) > 0) {
                ShaderGlobalsType shadow_sg;
                Ray shadow_ray          = Ray(sg.P, bg_dir.val(), radius, 0, 0,
                                              Ray::SHADOW);
                Intersection shadow_hit = scene.intersect(shadow_ray, inf,
                                                          hit.id);
                if (shadow_hit.t == inf)  // ray reached the background?
                    path_radiance += contrib;
            }
        }

        // trace a shadow ray to one of the light emitting primitives
        if (lightprims_size > 0) {
            const float light_pick_pdf = 1.0f / lightprims_size;

            // uniform probability for each light
            float xl = xi * lightprims_size;
            int ls   = floorf(xl);
            xl -= ls;

            uint32_t lid = m_lightprims[ls];
            if (lid != hit.id) {
                int shaderID = scene.shaderid(lid);

                // sample a random direction towards the object
                LightSample sample = scene.sample(lid, sg.P, xl, yi);
                BSDF::Sample b     = result.bsdf.eval(-sg.I, sample.dir);
                Color3 contrib     = path_weight * b.weight
                                     * MIS::power_heuristic<MIS::EVAL_WEIGHT>(
                                         light_pick_pdf * sample.pdf, b.pdf);
                if ((contrib.x + contrib.y + contrib.z) > 0) {
                    ShaderGlobalsType light_sg;
                    Ray shadow_ray = Ray(sg.P, sample.dir, radius, 0, 0,
                                         Ray::SHADOW);
                    // trace a shadow ray and see if we actually hit the target
                    // in this tiny renderer, tracing a ray is probably cheaper than evaluating the light shader
                    Intersection shadow_hit
                        = scene.intersect(shadow_ray, sample.dist, hit.id, lid);

#ifndef __CUDACC__
                    const bool did_hit = shadow_hit.t == sample.dist;
#else
                    // The hit distance on the device is not as precise as on
                    // the CPU, so we need to allow a little wiggle room. An
                    // epsilon of 1e-3f empirically gives results that closely
                    // match the CPU for the test scenes, so that's what we're
                    // using.
                    const bool did_hit = fabsf(shadow_hit.t - sample.dist)
                                         < 1e-3f;
#endif
                    if (did_hit) {
                        // setup a shader global for the point on the light
                        globals_from_hit(light_sg, shadow_ray, sample.dist, lid,
                                         sample.u, sample.v);
#ifndef __CUDACC__
                        // execute the light shader (for emissive closures only)
                        shadingsys->execute(*ctx, *m_shaders[shaderID].surf,
                                            light_sg);
#else
                        execute_shader(light_sg, shaderID, light_closure_pool);
#endif
                        ShadingResult light_result;
                        process_closure(light_sg, r.roughness, light_result,
                                        medium_stack,
                                        (const ClosureColor*)light_sg.Ci, true);
                        // accumulate contribution
                        path_radiance += contrib * light_result.Le;
                    }
                }
            }
        }

        // trace indirect ray and continue
        BSDF::Sample p = result.bsdf.sample(-sg.I, xi, yi, zi);
        path_weight *= p.weight;
        bsdf_pdf  = p.pdf;
        r.raytype = Ray::DIFFUSE;  // FIXME? Use DIFFUSE for all indiirect rays
        r.direction = p.wi;
        r.radius    = radius;
        // Just simply use roughness as spread slope
        r.spread    = std::max(r.spread, p.roughness);
        r.roughness = p.roughness;

        // has the sammpled sampled dir crossed surface?
        bool transmitted = sg.Ng.dot(p.wi) < 0;

        if (transmitted) {
            if (!sg.backfacing) {  // if entering and crossing surface
                medium_stack.add_medium(result.medium_data);
            } else {
                medium_stack.pop_medium();
            }
        }

        if (!(path_weight.x > 0) && !(path_weight.y > 0)
            && !(path_weight.z > 0))
            break;  // filter out all 0's or NaNs
        prev_id  = hit.id;
        r.origin = sg.P;
    }
    return path_radiance;
}



OSL_HOSTDEVICE Color3
SimpleRaytracer::antialias_pixel(int x, int y, ShadingContext* ctx)
{
    Color3 result(0, 0, 0);
    for (int si = 0, n = aa * aa; si < n; si++) {
        Sampler sampler(x, y, si);
        // jitter pixel coordinate [0,1)^2
        Vec3 j = no_jitter ? Vec3(0.5f, 0.5f, 0) : sampler.get();
        // warp distribution to approximate a tent filter [-1,+1)^2
        j.x *= 2;
        j.x = j.x < 1 ? sqrtf(j.x) - 1 : 1 - sqrtf(2 - j.x);
        j.y *= 2;
        j.y = j.y < 1 ? sqrtf(j.y) - 1 : 1 - sqrtf(2 - j.y);
        // trace eye ray (apply jitter from center of the pixel)
        Color3 r = subpixel_radiance(x + 0.5f + j.x, y + 0.5f + j.y, sampler,
                                     ctx);
        // mix in result via lerp for numerical stability
        result = OIIO::lerp(result, r, 1.0f / (si + 1));
    }
    return result;
}


#ifndef __CUDACC__
// ---------------------------------------------------------------------------
// Vulkan shading path
// ---------------------------------------------------------------------------

namespace {

// Offsets inside one ShadeBlock.  liboslexec/vulkan_types.h is the authority
// (vulkan_abi_check.cpp static_asserts every one of these against
// OSL::ShaderGlobals at build time); they are repeated here because that
// header is internal to liboslexec and not installed.
const size_t kShadeBlockStride     = 640;  // one invocation
const size_t kShadeBlockCursorOff  = 216;  // sg.renderstate: arena offset
const size_t kShadeBlockCiOff      = 288;  // sg.Ci: root closure offset
const size_t kShadeBlockGlobalsEnd = 320;  // start of the closure arena
// The group-data block (the `groupdata` argument of every layer entry) lives
// past the 256-byte closure pool, so it can never collide with the bump allocator
// that grows upward from the pool.  A shader whose parameter block is not
// SROA-promoted to SSA values -- any body with a branch, i.e. checkerboard -- keeps
// its layer params there and its first word is the "already initialised" flag.
const size_t kShadeBlockGroupDataOff = 576;

}  // namespace


static bool
decode_texture(const std::string& filename, VulkanShading::Texture& out);

// Structural comparison of two closure trees, walking the same way
// process_closure() does.  Pointers only ever point inside the ShadeBlock that
// produced them, so a mismatch of shape, id or weight is all we need to report.
static bool
closure_equal(const ClosureColor* a, const ClosureColor* b, int depth)
{
    if (a == b)
        return true;
    if (!a || !b || depth > 16)
        return false;
    if (a->id != b->id)
        return false;
    if (a->id == ClosureColor::MUL) {
        const ClosureMul *am = a->as_mul(), *bm = b->as_mul();
        for (int i = 0; i < 3; ++i)
            if (fabsf(am->weight[i] - bm->weight[i]) > 1e-6f)
                return false;
        return closure_equal(am->closure, bm->closure, depth + 1);
    }
    if (a->id == ClosureColor::ADD)
        return closure_equal(a->as_add()->closureA, b->as_add()->closureA,
                             depth + 1)
            && closure_equal(a->as_add()->closureB, b->as_add()->closureB,
                             depth + 1);
    const ClosureComponent* ac = a->as_comp();
    const ClosureComponent* bc = b->as_comp();
    for (int i = 0; i < 3; ++i)
        if (fabsf(ac->w[i] - bc->w[i]) > 1e-5f)
            return false;
    return true;
}

static void
dump_closure(const char* tag, const ClosureColor* c, int depth)
{
    if (!c || depth > 8) {
        fprintf(stderr, "%s (nil)\n", tag);
        return;
    }
    if (c->id == ClosureColor::MUL) {
        fprintf(stderr, "%s mul w=(%.6g,%.6g,%.6g)\n", tag,
                c->as_mul()->weight[0], c->as_mul()->weight[1],
                c->as_mul()->weight[2]);
        dump_closure(tag, c->as_mul()->closure, depth + 1);
    } else if (c->id == ClosureColor::ADD) {
        fprintf(stderr, "%s add\n", tag);
        dump_closure(tag, c->as_add()->closureA, depth + 1);
        dump_closure(tag, c->as_add()->closureB, depth + 1);
    } else {
        const ClosureComponent* cc = c->as_comp();
        fprintf(stderr, "%s id=%d w=(%.6g,%.6g,%.6g)\n", tag, cc->id, cc->w[0],
                cc->w[1], cc->w[2]);
    }
}

bool
SimpleRaytracer::vulkan_fallback_allowed() const
{
    const char* allow = getenv("OSL_VULKAN_ALLOW_CPU_FALLBACK");
    return allow && OIIO::Strutil::stoi(std::string(allow)) != 0;
}


void
SimpleRaytracer::vulkan_report_failure(const char* what, const char* detail)
{
    // Nothing to report when the device path was not asked for, or when the
    // caller explicitly opted back into the CPU fallback.
    if (!m_vulkan_requested || vulkan_fallback_allowed())
        return;
    m_vulkan_hard_failure = true;
    if (m_vulkan_failure_reported.exchange(true))
        return;  // first failure only, no stderr flood from every thread
    const char* tail = ". The CPU fallback is disabled because OSL_USE_VULKAN "
                       "was set; export OSL_VULKAN_ALLOW_CPU_FALLBACK=1 to "
                       "permit it";
    if (detail && *detail)
        errhandler().errorfmt("[vulkan] {}: {}{}", what, detail, tail);
    else
        errhandler().errorfmt("[vulkan] {}{}", what, tail);
}


void
SimpleRaytracer::init_vulkan_shading()
{
    m_vulkan_ready            = false;
    m_vulkan_failed           = false;
    m_vulkan_requested        = false;
    m_vulkan_hard_failure     = false;
    m_vulkan_failure_reported = false;
    m_vk.reset();
    m_vk_pipeline.assign(m_shaders.size(), -1);

    const char* enable = getenv("OSL_USE_VULKAN");
    m_vulkan_requested = enable && OIIO::Strutil::stoi(std::string(enable)) != 0;
    if (!m_vulkan_requested)
        return;

    m_vk.reset(new VulkanShading);
    std::string err;
    if (!m_vk->init(err)) {
        m_vulkan_failed = true;
        m_vk.reset();
        fprintf(stderr, "[vulkan] device init failed: %s\n", err.c_str());
        vulkan_report_failure("device init failed", err.c_str());
        return;
    }
    fprintf(stderr, "[vulkan] device: %s\n", m_vk->device_name().c_str());

    int nok = 0, nfail = 0;

    // Batch path: instead of compiling each material on its own, assemble one
    // module out of every group plus the device frame renderer, so the whole
    // frame (ray generation, sampling, intersection, accumulation) can run in
    // one dispatch.  See src/liboslexec/vulkan_module.cpp.
    if (getenv("OSL_VULKAN_BATCH")) {
        std::vector<std::string> irs;
        std::vector<std::vector<std::string>> ents;
        m_vk_dispatch_id.assign(m_shaders.size(), -1);
        for (size_t i = 0; i < m_shaders.size(); ++i) {
            if (!m_shaders[i].surf)
                continue;
            shadingsys->optimize_group(&*m_shaders[i].surf, nullptr);
            // The dispatch numbers the groups it is handed in order, so the
            // material's case index is the position just before this push.
            m_vk_dispatch_id[i] = (int)irs.size();
            irs.push_back(OSL::pvt::vulkan_group_ir(*m_shaders[i].surf));
            ents.push_back(
                OSL::pvt::vulkan_group_entry_names(*m_shaders[i].surf));
            if (irs.back().empty()) {
                fprintf(stderr,
                        "[vulkan] material %zu kept no Vulkan IR "
                        "(is OSL_VULKAN_BATCH set before compiling?)\n",
                        i);
                vulkan_report_failure("a material kept no Vulkan IR", nullptr);
            }
        }
        std::string mod_ir, log;
        if (!OSL::pvt::vulkan_build_render_module(irs, ents, mod_ir, log)) {
            fprintf(stderr, "[vulkan] batch module assembly failed: %s\n",
                    log.c_str());
            vulkan_report_failure("batch module assembly failed", log.c_str());
            return;
        }
        fprintf(stderr, "[vulkan] batch module: %zu group(s), %zu bytes of IR\n",
                irs.size(), mod_ir.size());
        if (const char* dump = getenv("OSL_VULKAN_DUMP_BATCH_IR")) {
            OIIO::ofstream f;
            OIIO::Filesystem::open(f, dump);
            if (f)
                f << mod_ir;
            // Also write each group's own IR, so a per-group bisect of a clspv
            // failure does not need a debugger.
            for (size_t i = 0; i < irs.size(); ++i) {
                std::string one = std::string(dump) + ".g" + std::to_string(i)
                                  + ".ll";
                OIIO::ofstream gf;
                OIIO::Filesystem::open(gf, one);
                if (gf)
                    gf << irs[i];
            }
        }
        std::vector<uint32_t> spirv;
        if (!OSL::pvt::clspv_compile(mod_ir, "-x ir --arch=spirv64 --spv-version=1.4", spirv,
                                     log)) {
            fprintf(stderr, "[vulkan] batch clspv failed: %s\n", log.c_str());
            vulkan_report_failure("clspv failed", log.c_str());
            return;
        }
        fprintf(stderr, "[vulkan] batch SPIR-V: %zu words\n", spirv.size());
        if (const char* spvpath = getenv("OSL_VULKAN_DUMP_SPV")) {
            FILE* fp = fopen(spvpath, "wb");
            if (fp) {
                fwrite(spirv.data(), sizeof(uint32_t), spirv.size(), fp);
                fclose(fp);
                fprintf(stderr, "[vulkan] wrote %s\n", spvpath);
            }
        }
        // One pipeline for the whole frame: entry `osl_kernel` sits at the top
        // of the module and calls osl_vk_raytrace_frame, which reaches every
        // material through the synthesised dispatch.
        int frame_id = m_vk->add_shader(spirv.data(), spirv.size(), "osl_kernel",
                                        err);
        if (frame_id < 0) {
            fprintf(stderr, "[vulkan] frame pipeline failed: %s\n",
                    err.c_str());
            vulkan_report_failure("frame pipeline failed", err.c_str());
            return;
        }
        m_vk_frame = frame_id;
        upload_vulkan_textures();
        fprintf(stderr,
                "[vulkan] whole-frame pipeline ready (pipeline %d): raytracing, "
                "sampling and accumulation run on the device\n",
                frame_id);
        return;
    }

    for (size_t i = 0; i < m_shaders.size(); ++i) {
        if (!m_shaders[i].surf)
            continue;
        // Groups are compiled lazily (OSL's ShaderGroupEnd only builds the
        // group data; the LLVM/clspv work happens on demand), so the SPIR-V
        // does not exist yet when the renderer sets up.  Force the compile,
        // exactly like the OptiX renderer does before it asks for PTX.
        shadingsys->optimize_group(&*m_shaders[i].surf, nullptr);
        const std::vector<uint32_t>* spv = nullptr;
        if (!shadingsys->getattribute(&*m_shaders[i].surf, "vulkan_spirv",
                                      TypeDesc(TypeDesc::PTR), &spv)
            || !spv || spv->empty()) {
            ++nfail;
            fprintf(stderr,
                    "[vulkan] material %zu has no SPIR-V; keeping it on the "
                    "CPU path\n",
                    i);
            vulkan_report_failure("material has no SPIR-V", nullptr);
            continue;
        }
        ustring entry;
        shadingsys->getattribute(&*m_shaders[i].surf, "vulkan_entry_point",
                                 TypeDesc::STRING, &entry);
        int id = m_vk->add_shader(spv->data(), spv->size(), entry.c_str(),
                                  err);
        if (id < 0) {
            ++nfail;
            fprintf(stderr, "[vulkan] material %zu pipeline failed: %s\n", i,
                    err.c_str());
            vulkan_report_failure("material pipeline failed", err.c_str());
            continue;
        }
        m_vk_pipeline[i] = id;
        ++nok;
    }

    m_vulkan_ready = nok > 0;

    // Textures the shaders asked for, decoded once and uploaded into the same
    // storage buffer the kernels read their ShadeBlock from.
    if (nok > 0)
        upload_vulkan_textures();

    fprintf(stderr, "[vulkan] %d pipeline(s) ready, %d material(s) on CPU\n",
            nok, nfail);
}


/// Decode every texture the shaders referenced and hand it to the device.
/// Shared by the per-material path and the whole-frame (batch) path, because
/// the texture table lives at the same buffer offsets either way.
void
SimpleRaytracer::upload_vulkan_textures()
{
    if (!m_vk || m_vk_texture_names.empty())
        return;
    std::string err;
    std::vector<VulkanShading::Texture> texs;
    for (const auto& name : m_vk_texture_names) {
        VulkanShading::Texture t;
        t.hash = (unsigned long long)name.hash();
        if (!decode_texture(name.string(), t)) {
            fprintf(stderr,
                    "[vulkan] texture '%s' could not be decoded on the host; "
                    "the device will sample it as missing\n",
                    name.c_str());
            continue;
        }
        texs.push_back(std::move(t));
    }
    if (!m_vk->set_textures(texs, err))
        fprintf(stderr, "[vulkan] texture upload failed: %s\n", err.c_str());
    else
        fprintf(stderr, "[vulkan] %zu texture(s) uploaded to the device\n",
                texs.size());
}


// Decode one texture into RGBA32F, mip 0, for the device.  The .tx files the
// test scenes use are plain images as far as ImageInput is concerned, so the
// first subimage at mip 0 is the level the kernel samples.
static bool
decode_texture(const std::string& filename, VulkanShading::Texture& out)
{
    auto in = OIIO::ImageInput::open(filename);
    if (!in)
        return false;

    // The .tx files carry a full mip chain, and the CPU path picks a level per
    // pixel from the texture derivatives.  Uploading every level lets the
    // device do the same, instead of aliasing mip 0 at low sampling rates.
    for (int level = 0;; ++level) {
        OIIO::ImageSpec spec = in->spec(0, level);
        if (spec.width <= 0 || spec.height <= 0)
            break;
        int nchans = spec.nchannels;
        std::vector<float> raw((size_t)spec.width * spec.height * nchans);
        if (!in->read_image(0, level, 0, nchans, OIIO::TypeDesc::FLOAT,
                            raw.data()))
            break;
        if (level == 0) {
            out.width  = spec.width;
            out.height = spec.height;
        }
        if (getenv("OSL_VULKAN_TEXTURE_TRACE"))
            fprintf(stderr, "[vulkan] texture level %d: %dx%d", level, spec.width,
                    spec.height);
        out.level_width.push_back(spec.width);
        out.level_height.push_back(spec.height);
        out.rgba.resize(out.rgba.size() + (size_t)spec.width * spec.height * 4);
        float* dst0 = out.rgba.data() + out.rgba.size()
            - (size_t)spec.width * spec.height * 4;
        for (size_t i = 0, n = (size_t)spec.width * spec.height; i < n; ++i) {
            const float* src = raw.data() + i * nchans;
            float* dst       = dst0 + i * 4;
            dst[0]           = src[0];
            dst[1]           = nchans > 1 ? src[1] : src[0];
            dst[2]           = nchans > 2 ? src[2] : src[0];
            dst[3]           = nchans > 3 ? src[3] : 1.0f;
        }
        // A level of 1x1 is the end of the chain.
        if (spec.width == 1 && spec.height == 1)
            break;
    }
    in->close();
    return !out.level_width.empty();
}


bool
SimpleRaytracer::shade_vulkan(ShaderGlobals& sg, int shaderID, ShadingContext* ctx)
{
    if (!m_vulkan_ready || shaderID < 0
        || shaderID >= (int)m_vk_pipeline.size()
        || m_vk_pipeline[shaderID] < 0)
        return false;

    // One block per thread: the renderer shades pixels from several threads at
    // once, and a shared block would have one thread's dispatch overwrite the
    // closure another thread is about to evaluate.  It has to stay alive
    // until process_closure() returns, so it cannot be a local.
    static thread_local std::vector<unsigned char> tls_block;
    std::vector<unsigned char>& blk = tls_block;

    blk.assign(kShadeBlockStride, 0);
    memcpy(blk.data(), &sg, kShadeBlockGlobalsEnd);
    // The two fields the shader treats as *offsets* must start clean, and the
    // stale host pointers in the rest of the runtime part of ShaderGlobals
    // (tracedata, objdata, context, ...) are meaningless on the device.
    memset(blk.data() + kShadeBlockCursorOff, 0,
           kShadeBlockGlobalsEnd - kShadeBlockCursorOff);
    // Offset 0 is a perfectly valid arena offset -- it is where the *first*
    // component lands, which is exactly what a simple shader like matte
    // produces.  So the root field cannot use 0 to mean "no closure": mark it
    // with -1 instead, and the shader only ever writes a non-negative offset.
    long long no_root = -1;
    memcpy(blk.data() + kShadeBlockCiOff, &no_root, sizeof(no_root));

    std::string err;
    if (getenv("OSL_VULKAN_SHADE_TRACE")) {
        static std::atomic<unsigned long long> s_n(0);
        fprintf(stderr, "[vulkan] #%llu dispatch material %d\n",
                s_n.fetch_add(1), shaderID);
    }
    if (!m_vk->shade(m_vk_pipeline[shaderID], blk.data(), kShadeBlockStride,
                     err)) {
        m_vulkan_failed = true;
        fprintf(stderr, "[vulkan] shade failed: %s\n", err.c_str());
        vulkan_report_failure("shade dispatch failed", err.c_str());
        return false;
    }

    // The shader stored the root closure as an offset into the block's arena;
    // point the host evaluator at it.  blk outlives this call because
    // process_closure() runs right after it, while sg.Ci is still live.
    long long ci = -1;
    memcpy(&ci, blk.data() + kShadeBlockCiOff, sizeof(ci));
    if (getenv("OSL_VULKAN_SHADE_TRACE"))
        fprintf(stderr, "[vulkan]   ci offset = %lld\n", ci);
    if (getenv("OSL_VULKAN_SHADE_TRACE") && ci >= 0) {
        const int* id = (const int*)(blk.data() + kShadeBlockGlobalsEnd + ci);
        const float* w = (const float*)(id + 1);
        fprintf(stderr, "[vulkan]     root id=%d w=(%g,%g,%g)\n", *id, w[0],
                w[1], w[2]);
    }
    sg.Ci = (ci >= 0) ? (ClosureColor*)(blk.data() + kShadeBlockGlobalsEnd + ci)
                      : nullptr;

    // Verification hook: run the same shading point through the CPU path and
    // compare the two closure trees.  A difference here is a *shader*
    // difference, not a filtering/esimation difference, which makes "why do
    // the images differ" answerable without staring at pixels.
    if (getenv("OSL_VULKAN_VERIFY")) {
        static thread_local std::atomic<unsigned long long> s_mismatch(0);
        ShaderGlobals cpu_sg = sg;
        // Leave whatever closure the alias already held: this object belongs
        // to the render loop, and clearing Ci makes matching shaders look like
        // mismatches (the matte path never assigns if the texture is what set
        // the color).
        if (m_vk_pipeline[shaderID] >= 0)
            shadingsys->execute(*ctx, *m_shaders[shaderID].surf, cpu_sg);
        if (!closure_equal(sg.Ci, cpu_sg.Ci, 0)) {
            long long n = s_mismatch.fetch_add(1);
            if (n < 12) {
                fprintf(stderr, "[verify] material %d mismatch:\n", shaderID);
                dump_closure("  gpu", sg.Ci, 0);
                dump_closure("  cpu", cpu_sg.Ci, 0);
                long long cursor = 0;
                memcpy(&cursor, blk.data() + kShadeBlockCursorOff,
                       sizeof(cursor));
                fprintf(stderr, "  arena cursor = %lld, root ci = %lld\n",
                        cursor, ci);
                for (int i = 0; i < 3; ++i) {
                    const int* c = (const int*)(blk.data() + kShadeBlockGlobalsEnd
                                                + 16 * i);
                    fprintf(stderr,
                            "  pool[%d] id=%d w=(%.6g,%.6g,%.6g)\n", i, c[0],
                            ((const float*)c)[1], ((const float*)c)[2],
                            ((const float*)c)[3]);
                }
            }
        }
    }
    return true;
}


void
SimpleRaytracer::prepare_render()
{
    // Retrieve and validate options
    aa                = std::max(1, options.get_int("aa"));
    no_jitter         = options.get_int("no_jitter") != 0;
    max_bounces       = options.get_int("max_bounces");
    rr_depth          = options.get_int("rr_depth");
    show_albedo_scale = options.get_float("show_albedo_scale");
    show_globals      = options.get_int("show_globals");

    // Prepare the background importance table (if requested) *before*
    // init_vulkan_shading(): executing the background shader is what makes the
    // renderer see the texture it samples (get_texture_handle records the name),
    // and the Vulkan path can only upload textures it learned about.
    if (backgroundResolution > 0 && backgroundShaderID >= 0) {
        // get a context so we can make several background shader calls
        OSL::PerThreadInfo* thread_info = shadingsys->create_thread_info();
        ShadingContext* ctx             = shadingsys->get_context(thread_info);

        // build importance table to optimize background sampling
        auto evaler = [this](const Dual2<Vec3>& dir, ShadingContext* ctx) {
            return this->eval_background(dir, ctx);
        };
        background.prepare(backgroundResolution, evaler, ctx);

        // release context
        shadingsys->release_context(ctx);
        shadingsys->destroy_thread_info(thread_info);
    } else {
        // we aren't directly evaluating the background
        backgroundResolution = 0;
    }

    init_vulkan_shading();

    prepare_geometry();

    // build bvh and prepare triangles
    scene.prepare(errhandler());
    prepare_lights();

#    if 0
    // dump scene to disk as obj for debugging purposes
    // TODO: make this a feature?
    FILE* fp = fopen("/tmp/test.obj", "w");
    for (Vec3 v : scene.verts)
        fprintf(fp, "v %.9g %.9g %.9g\n", v.x, v.y, v.z);
    for (TriangleIndices t : scene.triangles)
        fprintf(fp, "f %d %d %d\n", 1 + t.a, 1 + t.b, 1 + t.c);
    fclose(fp);
#    endif
}



void
SimpleRaytracer::prepare_lights()
{
    m_mesh_surfacearea.reserve(scene.last_index.size());

    // measure the total surface area of each mesh
    int first_index = 0;
    for (int last_index : scene.last_index) {
        float area = 0;
        for (int index = first_index; index < last_index; index++) {
            area += scene.primitivearea(index);
        }
        m_mesh_surfacearea.emplace_back(area);
        first_index = last_index;
    }
    // collect all light emitting triangles
    for (unsigned t = 0, n = scene.num_prims(); t < n; t++) {
        int shaderID = scene.shaderid(t);
        if (shaderID < 0 || !m_shaders[shaderID].surf)
            continue;  // no shader attached
        if (m_shader_is_light[shaderID])
            m_lightprims.emplace_back(t);
    }
    if (!m_lightprims.empty())
        errhandler().infofmt("Found {} triangles to be treated as lights",
                             m_lightprims.size());
}


void
SimpleRaytracer::prepare_geometry()
{
    bool have_displacement = false;
    for (const Material& m : shaders()) {
        if (m.disp) {
            have_displacement = true;
            break;
        }
    }
    if (have_displacement) {
        errhandler().infofmt("Evaluating displacement shaders");
        // Loop through all triangles and run displacement shader if there is one
        // or copy the input point if there is none
        std::vector<Vec3> disp_verts(scene.verts.size(), Vec3(0, 0, 0));
        std::vector<int> valance(
            scene.verts.size(),
            0);  // number of times each vertex has been displaced

        OSL::PerThreadInfo* thread_info = shadingsys->create_thread_info();
        ShadingContext* ctx             = shadingsys->get_context(thread_info);

        bool has_smooth_normals = false;
        for (int primID = 0, nprims = scene.triangles.size(); primID < nprims;
             primID++) {
            Vec3 p[3], n[3];
            Vec2 uv[3];

            p[0] = scene.verts[scene.triangles[primID].a];
            p[1] = scene.verts[scene.triangles[primID].b];
            p[2] = scene.verts[scene.triangles[primID].c];

            valance[scene.triangles[primID].a]++;
            valance[scene.triangles[primID].b]++;
            valance[scene.triangles[primID].c]++;

            int shaderID = scene.shaderid(primID);
            if (shaderID < 0 || !m_shaders[shaderID].disp) {
                disp_verts[scene.triangles[primID].a] += p[0];
                disp_verts[scene.triangles[primID].b] += p[1];
                disp_verts[scene.triangles[primID].c] += p[2];
                continue;
            }


            Vec3 Ng    = (p[0] - p[1]).cross(p[0] - p[2]);
            float area = 0.5f * Ng.length();
            Ng         = Ng.normalize();
            if (scene.n_triangles[primID].a >= 0) {
                n[0]               = scene.normals[scene.n_triangles[primID].a];
                n[1]               = scene.normals[scene.n_triangles[primID].b];
                n[2]               = scene.normals[scene.n_triangles[primID].c];
                has_smooth_normals = true;
            } else {
                n[0] = n[1] = n[2] = Ng;
            }

            if (scene.uv_triangles[primID].a >= 0) {
                uv[0] = scene.uvs[scene.uv_triangles[primID].a];
                uv[1] = scene.uvs[scene.uv_triangles[primID].b];
                uv[2] = scene.uvs[scene.uv_triangles[primID].c];
            } else {
                uv[0] = uv[1] = uv[2] = Vec2(0, 0);
            }

            // displace each vertex
            for (int i = 0; i < 3; i++) {
                ShaderGlobals sg = {};
                sg.P             = p[i];
                sg.Ng            = Ng;
                sg.N             = n[i];
                sg.u             = uv[i].x;
                sg.v             = uv[i].y;
                sg.I             = (p[i] - camera.eye).normalize();
                sg.surfacearea   = area;
                sg.renderstate   = &sg;

                shadingsys->execute(*ctx, *m_shaders[shaderID].disp, sg);

                p[i] = sg.P;
            }
            disp_verts[scene.triangles[primID].a] += p[0];
            disp_verts[scene.triangles[primID].b] += p[1];
            disp_verts[scene.triangles[primID].c] += p[2];
        }

        // release context
        shadingsys->release_context(ctx);
        shadingsys->destroy_thread_info(thread_info);

        // average each vertex by the number of times it was displaced
        for (int i = 0, n = scene.verts.size(); i < n; i++) {
            if (valance[i] > 0)
                disp_verts[i] /= float(valance[i]);
            else
                disp_verts[i] = scene.verts[i];
        }
        // replace old data with the new
        scene.verts = std::move(disp_verts);

        if (has_smooth_normals) {
            // Recompute the vertex normals (if we had some)
            std::vector<Vec3> disp_normals(scene.normals.size(), Vec3(0, 0, 0));
            for (int primID = 0, nprims = scene.triangles.size();
                 primID < nprims; primID++) {
                if (scene.n_triangles[primID].a >= 0) {
                    Vec3 p[3];
                    p[0] = scene.verts[scene.triangles[primID].a];
                    p[1] = scene.verts[scene.triangles[primID].b];
                    p[2] = scene.verts[scene.triangles[primID].c];
                    // don't normalize to weight by area
                    Vec3 Ng = (p[0] - p[1]).cross(p[0] - p[2]);
                    disp_normals[scene.n_triangles[primID].a] += Ng;
                    disp_normals[scene.n_triangles[primID].b] += Ng;
                    disp_normals[scene.n_triangles[primID].c] += Ng;
                }
            }
            for (Vec3& n : disp_normals)
                n = n.normalize();
            scene.normals = std::move(disp_normals);
        }
    }
}


/// Lay the whole scene out in the device storage buffer, dispatch the batch
/// kernel once and read the floating-point image back.
///
/// Buffer layout (all offsets from the start of the buffer, and all of them
/// mirrored into FrameParams so the kernel can find them):
///
///   [0, sizeof(FrameParams))                 FrameParams
///   [after the texture region, 16-aligned)   triangles, vertices, uvs,
///                                            uv_triangles, normals,
///                                            n_triangles, tri shader ids,
///                                            the float4 output image, and one
///                                            ShadeBlock per pixel (arena)
///
/// Nothing in this layout overlaps the texture table and texels, which is why
/// the arena starts at texture_end(): the shaders sample textures through the
/// same buffer.
bool
SimpleRaytracer::render_vulkan_frame(int xres, int yres)
{
    using namespace osl_vk_abi;
    std::string err;
    if (!m_vk || !m_vk->ready() || m_vk_frame < 0)
        return false;

    size_t cursor = (m_vk->texture_end() + 15) & ~(size_t)15;
    if (cursor < kFrameParamsOffset + sizeof(FrameParams))
        cursor = kFrameParamsOffset + sizeof(FrameParams);
    auto take = [&cursor](size_t bytes) {
        cursor = (cursor + 15) & ~(size_t)15;
        size_t at = cursor;
        cursor += bytes;
        return at;
    };

    size_t ntris   = scene.triangles.size();
    size_t nverts  = scene.verts.size();
    size_t nuv     = scene.uvs.size();
    size_t nnorm   = scene.normals.size();
    size_t npix    = (size_t)xres * (size_t)yres;

    size_t tri_off  = take(ntris * sizeof(TriangleIndices));
    size_t vert_off = take(nverts * sizeof(float) * 3);
    size_t uv_off   = take(nuv * sizeof(float) * 2);
    size_t tuv_off  = take(scene.uv_triangles.size() * sizeof(TriangleIndices));
    size_t nrm_off  = take(nnorm * sizeof(float) * 3);
    size_t tn_off   = take(scene.n_triangles.size() * sizeof(TriangleIndices));
    size_t sh_off   = take(scene.shaderids.size() * sizeof(int));
    size_t lp_count = m_lightprims.size();
    size_t lp_off   = take(lp_count * sizeof(int));
    /// sg->surfacearea, one float per primitive: the total area of the mesh the
    /// primitive belongs to, exactly what testrender hands the shader
    /// (simpleraytracer.cpp:945).  emitter.osl divides its power by it, so a
    /// zero here silences every light in the scene.
    std::vector<float> tri_area(ntris, 0.0f);
    {
        int first = 0;
        for (size_t m = 0; m < scene.last_index.size(); ++m) {
            int last   = scene.last_index[m];
            float area = (m < m_mesh_surfacearea.size())
                             ? (float)m_mesh_surfacearea[m]
                             : 0.0f;
            for (int t = first; t < last && (size_t)t < ntris; ++t)
                tri_area[t] = area;
            first = last;
        }
    }
    size_t area_off = take(tri_area.size() * sizeof(float));
    /// The CPU BVH, uploaded as the two flat arrays it already is: nodes are
    /// byte-identical to the device's BvhNodeVK, and the leaf triangle ids are
    /// the same `unsigned` order the CPU traversal walks.  This is the whole
    /// point of the Vulkan path sharing the CPU's acceleration structure --
    /// the device visits the same primitives in the same order.
    size_t nbnodes  = scene.bvh ? (size_t)scene.bvh->num_nodes : 0;
    size_t nbvidx   = scene.bvh ? (size_t)scene.bvh->num_indices : 0;
    // OSL_VULKAN_LINEAR=1 keeps the old linear sweep, so the BVH can be A/B'd
    // against it with the exact same sampler and shading code.
    if (getenv("OSL_VULKAN_LINEAR")) {
        nbnodes = 0;
        nbvidx  = 0;
    }
    size_t bvh_off  = take(nbnodes * sizeof(BvhNodeVK));
    size_t bidx_off = take(nbvidx * sizeof(unsigned));
    if (getenv("OSL_VULKAN_DEBUG_PIXEL")) {
        fprintf(stderr,
                "[vulkan] surfacearea: %zu meshes, tri_area[0]=%g tri_area[%zu]=%g\n",
                m_mesh_surfacearea.size(), tri_area.empty() ? -1.0f : tri_area[0],
                ntris ? ntris - 1 : (size_t)0,
                tri_area.empty() ? -1.0f : tri_area[tri_area.size() - 1]);
    }
    size_t out_off  = take(npix * sizeof(float) * 4);
    size_t arena_off = take(npix * (size_t)kShadeBlockStride);
    /// The background importance table (Background::prepare), uploaded so the
    /// device's *indirect* rays can evaluate the background the way the CPU
    /// does: table lookup plus the MIS weight against the bsdf pdf.  Evaluating
    /// the shader directly on those bounces is the right mean but the wrong
    /// weight, and it lights every path up by 1/mis.
    size_t bg_res = 0, bg_vals_off = 0, bg_rows_off = 0, bg_cols_off = 0;
    if (backgroundShaderID >= 0 && backgroundResolution > 0) {
        bg_res      = (size_t)background.table_res();
        bg_vals_off = take(bg_res * bg_res * sizeof(Vec3));
        bg_rows_off = take(bg_res * sizeof(float));
        bg_cols_off = take(bg_res * bg_res * sizeof(float));
    }

    std::function<void()> verify_geometry;  // filled in by the debug block below
    if (cursor > m_vk->capacity()) {
        errhandler().errorfmt(
            "[vulkan] frame does not fit: {} bytes needed, {} available",
            cursor, m_vk->capacity());
        return false;
    }

    if (getenv("OSL_VULKAN_NORMAL_CHECK")) {
        // Are the normal indices and the vertex indices the same index space?
        // For the matte sphere (triangles 16384..32767) each corner normal
        // should point along the radial direction of that corner's vertex; a
        // large residual means the two tables are indexed differently.
        double sum   = 0.0;
        double worst = 0.0;
        int    nbad  = 0, ntested = 0;
        int    max_na = -1, max_ta = -1;
        for (size_t t = 16384; t < scene.n_triangles.size() && t < 32768; ++t) {
            TriangleIndices nt = scene.n_triangles[t];
            TriangleIndices vt = scene.triangles[t];
            // All three corners matter: the device blends all three, so a
            // single valid corner is not enough.
            for (int corner = 0; corner < 3; ++corner) {
                int ni_i = (corner == 0) ? nt.a : (corner == 1) ? nt.b : nt.c;
                int vi_i = (corner == 0) ? vt.a : (corner == 1) ? vt.b : vt.c;
                if (ni_i < 0 || ni_i >= (int)scene.normals.size())
                    continue;
                if (vi_i < 0 || vi_i >= (int)scene.verts.size())
                    continue;
                Vec3 na  = scene.normals[ni_i];
                Vec3 va  = scene.verts[vi_i];
                Vec3 rad(va.x - 0.0f, va.y - 30.0f, va.z - 60.0f);
                rad.normalize();
                double d = (na.normalize() - rad).length();
                sum += d;
                ntested++;
                if (d > 0.05)
                    nbad++;
                if (d > worst)
                    worst = d;
                max_na = std::max(max_na, ni_i);
                max_ta = std::max(max_ta, vi_i);
            }
        }
        fprintf(stderr,
                "[vulkan] normal check (matte sphere): n=%d mean=%.4f max=%.4f "
                "bad(>0.05)=%d  max_ni=%d max_vi=%d  nnormals=%zu nverts=%zu\n",
                ntested, ntested ? sum / ntested : 0.0, worst, nbad, max_na,
                max_ta, scene.normals.size(), scene.verts.size());
    }

    if (getenv("OSL_VULKAN_DEBUG_PIXEL")) {
        // Is the triangle -> material mapping sane?  A sphere should own one
        // contiguous run of triangles and every triangle should carry the
        // material index of its own ShaderGroup.
        int hist[16] = { 0 };
        for (int sid : scene.shaderids)
            if (sid >= 0 && sid < 16)
                hist[sid]++;
        fprintf(stderr,
                "[vulkan] tri_shader histogram over %zu tris:",
                scene.shaderids.size());
        // Is the per-corner normal table real, or are these face-normal meshes?
        if (!scene.n_triangles.empty()) {
            for (size_t t : { (size_t)0, (size_t)16384, (size_t)32768 }) {
                if (t >= scene.n_triangles.size())
                    continue;
                fprintf(stderr,
                        "\n[vulkan] n_triangles[%zu]=(%d,%d,%d)", t,
                        scene.n_triangles[t].a, scene.n_triangles[t].b,
                        scene.n_triangles[t].c);
            }
            fprintf(stderr, "  normals=%zu verts=%zu", scene.normals.size(),
                    scene.verts.size());
        }
        for (int i = 0; i < 16; ++i)
            if (hist[i])
                fprintf(stderr, " mat%d=%d", i, hist[i]);
        fprintf(stderr, "  (lightprims=%zu)\n", m_lightprims.size());
        // triangle indices back and compare with the host copy -- a broken
        // upload shows up as "no hits at all" for whole spheres, not as an error.
        verify_geometry = [&]() {
        auto check_vert = [&](size_t idx) {
            if (idx >= scene.verts.size())
                return;
            float back[3] = { 0, 0, 0 };
            if (!m_vk->download(vert_off + idx * 12, back, 12, err))
                return;
            const float* host = reinterpret_cast<const float*>(&scene.verts[idx]);
            fprintf(stderr,
                    "[vulkan] vert %zu: host (%g,%g,%g) device (%g,%g,%g)%s\n",
                    idx, host[0], host[1], host[2], back[0], back[1], back[2],
                    (host[0] == back[0] && host[1] == back[1] && host[2] == back[2])
                        ? ""
                        : "   <-- MISMATCH");
        };
        auto check_tri = [&](size_t idx) {
            if (idx >= scene.triangles.size())
                return;
            int back[3] = { 0, 0, 0 };
            if (!m_vk->download(tri_off + idx * 12, back, 12, err))
                return;
            const int* host = reinterpret_cast<const int*>(&scene.triangles[idx]);
            fprintf(stderr, "[vulkan] tri %zu: host (%d,%d,%d) device (%d,%d,%d)%s\n",
                    idx, host[0], host[1], host[2], back[0], back[1], back[2],
                    (host[0] == back[0] && host[1] == back[1] && host[2] == back[2])
                        ? ""
                        : "   <-- MISMATCH");
        };
        // The material id per triangle is the one thing a wrong offset shows up
        // as silently: every hit still reports material 0.
        auto check_shader = [&](size_t idx) {
            if (idx >= scene.shaderids.size())
                return;
            int back = -12345;
            if (!m_vk->download(sh_off + idx * 4, &back, 4, err))
                return;
            fprintf(stderr, "[vulkan] shaderid %zu: host %d device %d%s\n", idx,
                    scene.shaderids[idx], back,
                    (scene.shaderids[idx] == back) ? "" : "   <-- MISMATCH");
        };
        check_vert(0);
        check_vert(scene.verts.size() / 3);
        check_vert(scene.verts.size() - 1);
        check_tri(0);
        check_tri(16384);
        check_tri(32768);
        check_tri(scene.triangles.size() - 1);
        check_shader(0);
        check_shader(16384);
        check_shader(32768);
        check_shader(scene.shaderids.size() - 1);
        };
    }

    // FrameParams goes first, at offset 0, because that is where the kernel
    // reads it from its raw buffer argument.
    FrameParams fp {};
    fp.width       = xres;
    fp.height      = yres;
    fp.aa           = aa;
    fp.max_bounces  = std::min(max_bounces, 256);
    fp.sample_index = 0;
    fp.sample_count = 0;
    fp.sample_batch = 0;
    fp.eye[0]      = camera.eye.x;
    fp.eye[1]      = camera.eye.y;
    fp.eye[2]      = camera.eye.z;
    fp.dir[0]      = camera.dir.x;
    fp.dir[1]      = camera.dir.y;
    fp.dir[2]      = camera.dir.z;
    fp.up[0]       = camera.up.x;
    fp.up[1]       = camera.up.y;
    fp.up[2]       = camera.up.z;
    fp.fov         = camera.fov;
    // The finalized camera basis, so the device casts exactly the CPU's rays:
    // Camera::finalize() (raytracer.h:128-137) builds cx/cy from
    // OIIO::fast_tan, not from tan(), and the two differ in the last bits.
    fp.cx[0] = camera.cx.x;
    fp.cx[1] = camera.cx.y;
    fp.cx[2] = camera.cx.z;
    fp.cy[0] = camera.cy.x;
    fp.cy[1] = camera.cy.y;
    fp.cy[2] = camera.cy.z;
    fp.invw  = camera.invw;
    fp.invh  = camera.invh;
    fp.num_triangles = (int)ntris;
    fp.num_vertices  = (int)nverts;
    // A background shader runs on the device through the batch dispatch, and the
    // number the kernel needs is the dispatch *case* the group was given, not
    // the material index.  Outside batch mode there is no dispatch table at all
    // (the kernel does not run either), so -1 keeps the flat bg_color.
    fp.bg_shader_id = -1;
    if (backgroundShaderID >= 0
        && backgroundShaderID < (int)m_vk_dispatch_id.size()
        && m_vk_frame >= 0)
        fp.bg_shader_id = m_vk_dispatch_id[backgroundShaderID];
    fp.bg_color[0] = fp.bg_color[1] = fp.bg_color[2] = 0.0f;
    fp.tri_offset        = (long long)tri_off;
    fp.vert_offset       = (long long)vert_off;
    fp.uv_offset         = (long long)uv_off;
    fp.tri_uv_offset     = (long long)tuv_off;
    fp.normal_offset     = (long long)nrm_off;
    fp.tri_n_offset      = (long long)tn_off;
    fp.tri_shader_offset = (long long)sh_off;
    fp.out_offset        = (long long)out_off;
    fp.arena_offset      = (long long)arena_off;
    fp.arena_stride      = (long long)kShadeBlockStride;
    fp.lightprim_offset  = (long long)lp_off;
    fp.num_lightprims    = (int)lp_count;
    fp.area_offset       = (long long)area_off;
    fp.bvh_offset        = (long long)bvh_off;
    fp.bvh_index_offset  = (long long)bidx_off;
    fp.num_bvh_nodes     = (int)nbnodes;
    fp.num_bvh_indices   = (int)nbvidx;
    // Texture filter variant: 0 trilinear over the mip chain, 1 mip 0 bilinear,
    // 2 nearest mip level, 3 mip 0 without a chain index.
    fp.tex_mode = 0;
    if (const char* tm = getenv("OSL_VULKAN_TEX_MODE"))
        fp.tex_mode = atoi(tm);
    // Calibration of the device mip selector against OIIO (see vulkan_types.h):
    // footprint_scale folds OIIO's filter-width factor, lod_bias its blur terms.
    fp.tex_foot_scale = 2.0f;
    fp.tex_lod_bias   = -1.0f;
    if (const char* fs = getenv("OSL_VULKAN_TEX_FOOT_SCALE"))
        fp.tex_foot_scale = (float)atof(fs);
    if (const char* lb = getenv("OSL_VULKAN_TEX_LOD_BIAS"))
        fp.tex_lod_bias = (float)atof(lb);
    // Russian roulette depth, same attribute the CPU tracer uses
    // (simpleraytracer.cpp:1092 passes absorb = b >= rr_depth).
    fp.rr_depth = rr_depth;
    // Background importance table (allocated above, once the layout was known).
    fp.bg_res           = (int)bg_res;
    fp.bg_invjacobian   = bg_res ? background.table_invjacobian() : 0.0f;
    fp.bg_values_offset = (long long)bg_vals_off;
    fp.bg_rows_offset   = (long long)bg_rows_off;
    fp.bg_cols_offset   = (long long)bg_cols_off;
    fp.pad_light         = getenv("OSL_VULKAN_DEBUG_PIXEL") ? 1 : 0;
    if (const char* dm = getenv("OSL_VULKAN_DEBUG_MODE"))
        fp.pad_light = atoi(dm);
    if (!m_vk->upload(kFrameParamsOffset, &fp, sizeof(fp), err)) {
        errhandler().errorfmt("[vulkan] FrameParams upload failed: {}", err);
        return false;
    }

    auto put = [&](size_t off, const void* src, size_t bytes) {
        if (bytes && !m_vk->upload(off, src, bytes, err)) {
            errhandler().errorfmt("[vulkan] scene upload failed: {}", err);
            return false;
        }
        return true;
    };
    if (!put(tri_off, scene.triangles.data(), ntris * sizeof(TriangleIndices))
        || !put(vert_off, scene.verts.data(), nverts * sizeof(Vec3))
        || !put(uv_off, scene.uvs.data(), nuv * sizeof(Vec2))
        || !put(tuv_off, scene.uv_triangles.data(),
                scene.uv_triangles.size() * sizeof(TriangleIndices))
        || !put(nrm_off, scene.normals.data(), nnorm * sizeof(Vec3))
        || !put(tn_off, scene.n_triangles.data(),
                scene.n_triangles.size() * sizeof(TriangleIndices))
        || !put(sh_off, scene.shaderids.data(),
                scene.shaderids.size() * sizeof(int))
        || !put(area_off, tri_area.data(), tri_area.size() * sizeof(float)))
        return false;

    if (bg_res) {
        if (!put(bg_vals_off, background.table_values(),
                 bg_res * bg_res * sizeof(Vec3))
            || !put(bg_rows_off, background.table_rows(),
                    bg_res * sizeof(float))
            || !put(bg_cols_off, background.table_cols(),
                    bg_res * bg_res * sizeof(float)))
            return false;
        fprintf(stderr,
                "[vulkan] background importance table uploaded: %zux%zu "
                "(%zu KB values, %zu KB cols)\n",
                bg_res, bg_res,
                bg_res * bg_res * sizeof(Vec3) / 1024,
                bg_res * bg_res * sizeof(float) / 1024);
    }

    // The acceleration structure.  Layout is shared with the CPU (BVHNode ==
    // BvhNodeVK: six bounds floats then child/nprims), so this is a raw copy.
    if (nbnodes) {
        if (!put(bvh_off, scene.bvh->nodes.get(),
                 nbnodes * sizeof(BvhNodeVK))
            || !put(bidx_off, scene.bvh->indices.get(),
                    nbvidx * sizeof(unsigned)))
            return false;
        fprintf(stderr,
                "[vulkan] bvh uploaded: %zu nodes (%zu B), %zu triangle ids "
                "(%zu B)\n",
                nbnodes, nbnodes * sizeof(BvhNodeVK), nbvidx,
                nbvidx * sizeof(unsigned));
    } else {
        fprintf(stderr, "[vulkan] no BVH: falling back to the linear sweep\n");
    }

    // Light primitives, as plain triangle ids (m_lightprims is a vector of
    // unsigned, and the device reads them as int).
    {
        std::vector<int> lp(lp_count);
        for (size_t i = 0; i < lp_count; ++i)
            lp[i] = (int)m_lightprims[i];
        if (!put(lp_off, lp.data(), lp.size() * sizeof(int)))
            return false;
    }

    // The arena is scratch: every (pixel, sample) resets its own ShadeBlock, so
    // the contents do not matter, but the memory has to exist.
    {
        std::vector<char> zeros(npix * (size_t)kShadeBlockStride, 0);
        if (!zeros.empty() && !put(arena_off, zeros.data(), zeros.size()))
            return false;
    }

    fprintf(stderr,
            "[vulkan] frame: %zu tris, %zu verts, %zu lightprims, %zu px, %u groups\n",
            ntris, nverts, lp_count, npix, (unsigned)((npix + kWorkgroupSize - 1) / kWorkgroupSize));
    unsigned groups = (unsigned)((npix + kWorkgroupSize - 1) / kWorkgroupSize);
    OIIO::Timer t;
    std::vector<float> image(npix * 4);
    if (fp.pad_light) {
        // Probe runs keep the single-dispatch shape: their counters are
        // per-sample statistics and only make sense with every sample in one
        // thread.
        if (!m_vk->dispatch(m_vk_frame, groups, err)) {
            errhandler().errorfmt("[vulkan] frame dispatch failed: {}", err);
            return false;
        }
        if (!m_vk->download(out_off, image.data(), image.size() * sizeof(float),
                            err)) {
            errhandler().errorfmt("[vulkan] image read-back failed: {}", err);
            return false;
        }
    } else {
        // One dispatch per sample, accumulated on the host.  Each dispatch runs
        // the same kernel with a thread that shades exactly one sample, which is
        // the structure the CUDA path has (one closure pool per subpixel thread)
        // -- and the device-side radiance is already weighted by 1/nsamples, so
        // the host only has to add them up.
        const int frame_samples = std::max(1, fp.aa * fp.aa);
        fp.sample_count         = frame_samples;
        // One sample per thread is what the CUDA path does; batching a few more
        // per dispatch only trims the host round trip, and it has to stay small
        // because it is exactly the reuse that used to stripe the image.
        int batch = 1;
        if (const char* sb = getenv("OSL_VULKAN_SAMPLES_PER_DISPATCH"))
            batch = std::max(1, atoi(sb));
        batch = std::min(batch, frame_samples);
        fp.sample_batch = batch;
        if (frame_samples > 1)
            fprintf(stderr,
                    "[vulkan] aa split: %d sample(s) per dispatch, %d dispatch(es)\n",
                    batch, frame_samples / batch);
        std::fill(image.begin(), image.end(), 0.0f);
        std::vector<float> one(npix * 4);
        for (int s = 0; s < frame_samples; s += batch) {
            fp.sample_index = s;
            if (!m_vk->upload(kFrameParamsOffset, &fp, sizeof(fp), err)) {
                errhandler().errorfmt("[vulkan] FrameParams upload failed: {}",
                                      err);
                return false;
            }
            if (!m_vk->dispatch(m_vk_frame, groups, err)) {
                errhandler().errorfmt("[vulkan] sample {} dispatch failed: {}", s,
                                      err);
                return false;
            }
            if (!m_vk->download(out_off, one.data(), one.size() * sizeof(float),
                                err)) {
                errhandler().errorfmt("[vulkan] sample {} read-back failed: {}", s,
                                      err);
                return false;
            }
            for (size_t i = 0; i < image.size(); ++i)
                image[i] += one[i];
        }
        fp.sample_count = 0;
        fp.sample_index = 0;
        fp.sample_batch = 0;
    }
    (void)t;

    if (fp.pad_light) {
        // Mode 2: per-material radiance, written with the material in alpha.
        if (fp.pad_light == 3) {
            // What the light-material dispatch returned, over every pixel that
            // took a NEE shadow ray: how many had a usable component, how many
            // of those emitted nothing, and the mean radiance.
            int nnee = 0, ncomp = 0, nzero = 0;
            double le = 0;
            for (int y = 0; y < yres; ++y)
                for (int x = 0; x < xres; ++x) {
                    size_t di = ((size_t)y * (size_t)xres + (size_t)x) * 4;
                    if (image[di + 2] >= 0) {
                        ++nnee;
                        if ((int)image[di + 2] == 1)
                            ++ncomp;
                        if (!(image[di + 3] > 0.0f))
                            ++nzero;
                        le += image[di + 3];
                    }
                }
            fprintf(stderr,
                    "[vulkan] nee probe: %d px, %d EMISSION comps, %d Le==0, "
                    "mean Le %.4f\n",
                    nnee, ncomp, nzero, nnee ? le / nnee : 0.0);
        }
        if (fp.pad_light == 4) {
            // Does the black striping correlate with the arena cursor peaking
            // past kClosurePoolSize, or with samples failing to leave a Ci?
            // Per-material: how many pixels are covered, how many of those came
            // out black, how high the arena cursor ever got, and how many of the
            // aa samples managed to leave a closure.  A black pixel with max
            // cursor well under 256 and 16/16 closures is a *sampling* problem,
            // not an arena one.
            struct PStat {
                int npix = 0, nblack = 0;
                float cmax = 0, csum = 0, gsum = 0;
            } ps[16];
            for (int y = 0; y < yres; ++y) {
                std::string row;
                for (int x = 0; x < xres; ++x) {
                    size_t di = ((size_t)y * (size_t)xres + (size_t)x) * 4;
                    int mat   = (int)image[di + 3];
                    if (mat < 0 || mat > 15) {
                        row += ' ';
                        continue;
                    }
                    PStat& s = ps[mat];
                    s.npix++;
                    s.csum += image[di + 1];
                    s.gsum += image[di + 2];
                    if (image[di + 1] > s.cmax)
                        s.cmax = image[di + 1];
                    if (!(image[di + 0] > 0.0f)) {
                        s.nblack++;
                        row += 'X';
                    } else {
                        row += '.';
                    }
                }
                if (fp.pad_light == 4)
                    fprintf(stderr, "[vulkan] px %2d %s\n", y, row.c_str());
            }
            for (int i = 0; i < 16; ++i) {
                if (!ps[i].npix)
                    continue;
                fprintf(stderr,
                        "[vulkan] aa mat %d: %d px, %d black, cursor max %.0f "
                        "mean %.1f, closures/pixel %.1f (aa2=%d)\n",
                        i, ps[i].npix, ps[i].nblack, ps[i].cmax,
                        ps[i].csum / ps[i].npix, ps[i].gsum / ps[i].npix,
                        fp.aa * fp.aa);
            }
        }

        if (fp.pad_light == 5) {
            // Is the blackness a NaN poisoning the aa average?
            struct NStat {
                int npix = 0, nblack = 0, nnan = 0;
                float nmax = 0;
            } ns[16];
            for (int y = 0; y < yres; ++y)
                for (int x = 0; x < xres; ++x) {
                    size_t di = ((size_t)y * (size_t)xres + (size_t)x) * 4;
                    int mat   = (int)image[di + 3];
                    if (mat < 0 || mat > 15)
                        continue;
                    NStat& s = ns[mat];
                    s.npix++;
                    if (!(image[di + 0] > 0.0f))
                        s.nblack++;
                    int nn = (int)image[di + 1];
                    s.nnan += nn;
                    if ((float)nn > s.nmax)
                        s.nmax = (float)nn;
                }
            for (int i = 0; i < 16; ++i) {
                if (!ns[i].npix)
                    continue;
                fprintf(stderr,
                        "[vulkan] nan mat %d: %d px, %d black, %d NaN samples "
                        "total, max per pixel %.0f (aa2=%d)\n",
                        i, ns[i].npix, ns[i].nblack, ns[i].nnan, ns[i].nmax,
                        fp.aa * fp.aa);
            }
        }
        if (fp.pad_light == 6) {
            // Per material: samples that missed the scene, samples that left a
            // closure, and pixels that came out black.
            struct MStat {
                int npix = 0, nblack = 0, nmiss = 0, ngood = 0, nsamp = 0;
                int nmissing = 0;
            } ms[16];
            for (int y = 0; y < yres; ++y)
                for (int x = 0; x < xres; ++x) {
                    size_t di = ((size_t)y * (size_t)xres + (size_t)x) * 4;
                    int mat   = (int)image[di + 3];
                    if (mat < -1 || mat > 15)
                        continue;
                    if (mat < 0) {
                        // pixel where even sample 0 missed: count it separately
                        ms[15].nmissing++;
                        continue;
                    }
                    MStat& s = ms[mat];
                    s.npix++;
                    s.nsamp += fp.aa * fp.aa;
                    s.nmiss += (int)image[di + 0];
                    s.ngood += (int)image[di + 1];
                    if (!(image[di + 2] > 0.0f))
                        s.nblack++;
                }
            for (int i = 0; i < 15; ++i) {
                if (!ms[i].npix)
                    continue;
                fprintf(stderr,
                        "[vulkan] m6 mat %d: %d px, %d black, miss %.1f%%, "
                        "closure %.1f%% (aa2=%d)\n",
                        i, ms[i].npix, ms[i].nblack,
                        100.0 * ms[i].nmiss / ms[i].nsamp,
                        100.0 * ms[i].ngood / ms[i].nsamp, fp.aa * fp.aa);
            }
            fprintf(stderr, "[vulkan] m6: %d px where sample 0 missed\n",
                    ms[15].nmissing);
        }
        if (fp.pad_light == 7) {
            // Sample 0's material vs the last sample's material, per pixel.
            // They have to agree on where the jittered samples land; if sample 0
            // suddenly reports material 0 on pixels it called matte at aa=1,
            // then the sample loop itself is misbehaving.
            int n00 = 0, nmis = 0, npix = 0;
            for (int y = 0; y < yres; ++y) {
                std::string row0, rowL;
                for (int x = 0; x < xres; ++x) {
                    size_t di = ((size_t)y * (size_t)xres + (size_t)x) * 4;
                    int m0   = (int)image[di + 0];
                    int mL   = (int)image[di + 1];
                    int miss = (int)image[di + 3];
                    ++npix;
                    n00 += (m0 == 0);
                    nmis += miss;
                    row0 += (m0 < 0 ? '.' : (char)('0' + (m0 % 10)));
                    rowL += (mL < 0 ? '.' : (char)('0' + (mL % 10)));
                }
                fprintf(stderr, "[vulkan] m7 x0 %2d %s\n", y, row0.c_str());
                if (y % 2 == 0)
                    fprintf(stderr, "[vulkan] m7 xL %2d %s\n", y, rowL.c_str());
            }
            fprintf(stderr,
                    "[vulkan] m7: %d px, %d with sample0 mat 0, %d misses "
                    "(aa2=%d)\n",
                    npix, n00, nmis, fp.aa * fp.aa);
        }
        if (fp.pad_light == 10) {
            // Is the closure cursor reset to 0 at every sample boundary?  Both
            // the peak seen at a sample *start* and the peak seen after shading
            // are reported, per material.
            struct CStat {
                int npix = 0, nblack = 0;
                float cstart = 0, cend = 0;
                int nsamp = 0, nmiss = 0;
            } cs[16];
            for (int y = 0; y < yres; ++y)
                for (int x = 0; x < xres; ++x) {
                    size_t di = ((size_t)y * (size_t)xres + (size_t)x) * 4;
                    int mat   = (int)image[di + 3];
                    if (mat < 0 || mat > 15)
                        continue;
                    CStat& s = cs[mat];
                    s.npix++;
                    s.nsamp += fp.aa * fp.aa;
                    s.nmiss += (int)image[di + 3];
                    if (image[di + 0] > s.cstart)
                        s.cstart = image[di + 0];
                    if (image[di + 1] > s.cend)
                        s.cend = image[di + 1];
                    if (!(image[di + 2] > 0.0f))
                        s.nblack++;
                }
            for (int i = 0; i < 16; ++i) {
                if (!cs[i].npix)
                    continue;
                fprintf(stderr,
                        "[vulkan] m10 mat %d: %d px, %d black, cursor at sample "
                        "start max %.0f B, after shading max %.0f B (pool "
                        "headroom 256)\n",
                        i, cs[i].npix, cs[i].nblack, cs[i].cstart, cs[i].cend);
            }
        }
        if (fp.pad_light == 11) {
            // Ci == 0 (which in the offset model points at the pool header, not
            // at a closure) vs unrecognised component ids, per material.
            struct ZStat {
                int npix = 0, nblack = 0, nz = 0, nbad = 0, nsamp = 0, nhit = 0;
                int nmissing = 0;
            } zs[16];
            for (int y = 0; y < yres; ++y)
                for (int x = 0; x < xres; ++x) {
                    size_t di = ((size_t)y * (size_t)xres + (size_t)x) * 4;
                    int mat   = (int)image[di + 3];
                    if (mat < 0) {
                        zs[15].nmissing++;
                        continue;
                    }
                    if (mat > 14)
                        continue;
                    ZStat& s = zs[mat];
                    s.npix++;
                    s.nsamp += fp.aa * fp.aa;
                    s.nz += (int)image[di + 0];
                    s.nbad += (int)image[di + 1];
                    s.nhit += (int)image[di + 3];
                    if (!(image[di + 2] > 0.0f))
                        s.nblack++;
                }
            for (int i = 0; i < 15; ++i) {
                if (!zs[i].npix)
                    continue;
                fprintf(stderr,
                        "[vulkan] m11 mat %d: %d px, %d black, Ci==0 on %d "
                        "samples, bad id on %d samples (of %d)\n",
                        i, zs[i].npix, zs[i].nblack, zs[i].nz, zs[i].nbad,
                        zs[i].nsamp);
            }
            fprintf(stderr, "[vulkan] m11: %d px with no sample-0 hit\n",
                    zs[15].nmissing);
        }
        if (fp.pad_light == 12) {
            // Jitter sanity: sample 0's x, the spread of x over the samples, and
            // the hit count.  A pixel whose spread sits near 0 on the device is
            // still sampling one fixed point.
            float j0min = 1e9f, j0max = -1e9f, spread_min = 1e9f, spread_max = -1e9f;
            int nhitpx = 0, nz = 0;
            for (int y = 0; y < yres; ++y)
                for (int x = 0; x < xres; ++x) {
                    size_t di = ((size_t)y * (size_t)xres + (size_t)x) * 4;
                    j0min = std::min(j0min, image[di + 0]);
                    j0max = std::max(j0max, image[di + 0]);
                    spread_min = std::min(spread_min, image[di + 1]);
                    spread_max = std::max(spread_max, image[di + 1]);
                    if ((int)image[di + 3] > 0)
                        ++nhitpx;
                    else
                        ++nz;
                }
            fprintf(stderr,
                    "[vulkan] m12: j0.x in [%.3f, %.3f], j.x spread in [%.3f, "
                    "%.3f], %d px with >=1 hit, %d with none (aa2=%d)\n",
                    j0min, j0max, spread_min, spread_max, nhitpx, nz,
                    fp.aa * fp.aa);
            // Also dump the pixel the earlier probe used.
            size_t di = ((size_t)36 * (size_t)xres + (size_t)11) * 4;
            fprintf(stderr, "[vulkan] m12 pixel (11,36): j0.x %.4f spread %.4f "
                    "j0.y %.4f hits %d\n",
                    image[di + 0], image[di + 1], image[di + 2], (int)image[di + 3]);
        }
        if (fp.pad_light == 13) {
            // Does sample 0 hit while the others miss?  Count pixels by that
            // pattern, and report how many hits the "sample 0 missed" pixels got.
            int a = 0, b = 0, c = 0, d = 0;
            for (int y = 0; y < yres; ++y)
                for (int x = 0; x < xres; ++x) {
                    size_t di = ((size_t)y * (size_t)xres + (size_t)x) * 4;
                    int hit0 = (int)image[di + 0];
                    int rest = (int)image[di + 1];
                    if (hit0 && rest > 0)
                        ++a;  // sample 0 hit, later samples hit too
                    else if (hit0)
                        ++b;  // sample 0 hit, later samples all missed
                    else if (rest > 0)
                        ++c;  // sample 0 missed, a later sample hit
                    else
                        ++d;  // nothing hit
                }
            fprintf(stderr,
                    "[vulkan] m13: hit0&&rest %d, hit0 only %d, rest only %d, "
                    "none %d (aa2=%d)\n",
                    a, b, c, d, fp.aa * fp.aa);
        }
        if (fp.pad_light == 14) {
            // How many full-scene scans a sample performs, and what that
            // implies for the linear triangle scan.
            double sum = 0;
            int smax = 0, npix2 = 0, nz = 0;
            for (int y = 0; y < yres; ++y)
                for (int x = 0; x < xres; ++x) {
                    size_t di = ((size_t)y * (size_t)xres + (size_t)x) * 4;
                    int sc = (int)image[di + 0];
                    sum += sc;
                    smax = std::max(smax, sc);
                    ++npix2;
                    if (sc == 0)
                        ++nz;
                }
            double mean = npix2 ? sum / npix2 : 0;
            fprintf(stderr,
                    "[vulkan] m14: scans/pixel mean %.2f max %d, %d px with none "
                    "(aa2=%d)\n",
                    mean, smax, nz, fp.aa * fp.aa);
            fprintf(stderr,
                    "[vulkan] m14: this run does %.3f G triangle tests "
                    "(ntris %d)\n",
                    mean * npix2 * (double)ntris / 1e9, (int)ntris);
        }
        if (fp.pad_light == 2) {
            struct RStat {
                int npix = 0;
                double r = 0, g = 0, b = 0;
                double rmax = 0;
            } rs[16];
            for (int y = 0; y < yres; ++y)
                for (int x = 0; x < xres; ++x) {
                    size_t di = ((size_t)y * (size_t)xres + (size_t)x) * 4;
                    int mat   = (int)image[di + 3];
                    if (mat < 0 || mat > 15)
                        continue;
                    RStat& s = rs[mat];
                    s.npix++;
                    s.r += image[di + 0];
                    s.g += image[di + 1];
                    s.b += image[di + 2];
                    s.rmax = std::max(s.rmax, (double)image[di + 0]);
                }
            for (int i = 0; i < 16; ++i) {
                if (!rs[i].npix)
                    continue;
                fprintf(stderr,
                        "[vulkan] radiance mat %d: %d px, mean rgb %.5f %.5f %.5f, "
                        "max r %.5f\n",
                        i, rs[i].npix, rs[i].r / rs[i].npix,
                        rs[i].g / rs[i].npix, rs[i].b / rs[i].npix, rs[i].rmax);
            }
        }

        // The device must still hold what the host uploaded -- check it *after*
        // the dispatch, since that is the state the kernel actually ran against.
        if (verify_geometry)
            verify_geometry();

        // Per-material census of the diagnostics the kernel wrote into the
        // image buffer: for every material that got hit, how many pixels it
        // covers, how many of those ended up with a real closure, and which
        // component ids it produced.
        struct Stat {
            int npix = 0, nclos = 0, nci_minus1 = 0;
            int comp[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };
            int hb[8] = { 0, 0, 0, 0, 0, 0, 0, 0 };  // hit-id buckets of 16384
            float ci_sum = 0.0f, cur_sum = 0.0f;
        };
        Stat st[16];
        for (int y = 0; y < yres; ++y) {
            for (int x = 0; x < xres; ++x) {
                size_t di  = ((size_t)y * (size_t)xres + (size_t)x) * 4;
                int hid    = (int)image[di];
                int mat    = (int)image[di + 2];
                if (mat < 0 || mat > 15)
                    continue;
                Stat& s = st[mat];
                s.npix++;
                s.ci_sum += image[di + 1];  // hit distance
                s.cur_sum += image[di + 3];  // ntris
                if (image[di + 1] < 0.0f)
                    s.nci_minus1++;
                else
                    s.nclos++;
                int c = mat;
                if (c >= 0 && c < 8)
                    s.comp[c]++;
                int hbk = hid < 0 ? 0 : (int)(hid / 16384.0f);
                if (hbk >= 0 && hbk < 8)
                    s.hb[hbk]++;
            }
        }
        // What the kernel reports for one specific pixel: triangle id, hit
        // distance, material, and the ntris the kernel believes in.
        {
            size_t di = ((size_t)36 * (size_t)xres + (size_t)11) * 4;
            fprintf(stderr,
                    "[vulkan] probe (11,36): tri %.0f t %.1f mat %.0f "
                    "ntris %.0f\n",
                    image[di], image[di + 1], image[di + 2], image[di + 3]);
        }

        // Two ASCII maps: '.' = no primary hit, digit = material id; and for
        // the second one, 'x' marks a pixel whose arena cursor came back 0
        // (i.e. its shading never saw the 16 the kernel stored).
        for (int y = 0; y < yres; ++y) {
            std::string ids, cur;
            for (int x = 0; x < xres; ++x) {
                size_t di = ((size_t)y * (size_t)xres + (size_t)x) * 4;
                float sid = image[di];
                ids += (sid < 0.0f) ? '.' : (char)('0' + (int)sid);
                cur += (sid < 0.0f) ? '.' : (image[di + 3] == 0.0f ? 'x' : '#');
            }
            fprintf(stderr, "[vulkan] ids %2d %s\n", y, ids.c_str());
            fprintf(stderr, "[vulkan] cur %2d %s\n", y, cur.c_str());
        }

        for (int i = 0; i < 16; ++i) {
            if (!st[i].npix)
                continue;
            fprintf(stderr,
                    "[vulkan] mat %d: %d px, closure %d, no-closure %d, "
                    "Ci avg %.1f, cursor avg %.1f, comps[0..3]=%d/%d/%d/%d "
                    "hitbucket=%d/%d/%d/%d\n",
                    i, st[i].npix, st[i].nclos, st[i].nci_minus1,
                    st[i].ci_sum / (float)st[i].npix,
                    st[i].cur_sum / (float)st[i].npix, st[i].comp[0],
                    st[i].comp[1], st[i].comp[2], st[i].comp[3],
                    st[i].hb[0], st[i].hb[1], st[i].hb[2], st[i].hb[3]);
        }
    }
    OIIO::ImageBuf::Iterator<float> p(pixelbuf, OIIO::ROI(0, xres, 0, yres));
    for (; !p.done(); ++p) {
        size_t idx = ((size_t)p.y() * (size_t)xres + (size_t)p.x()) * 4;
        p[0] = image[idx + 0];
        p[1] = image[idx + 1];
        p[2] = image[idx + 2];
    }
    errhandler().infofmt("Rendered {}x{} image with {} samples in {} on the "
                         "device ({} workgroups)",
                         xres, yres, aa * aa,
                         OIIO::Strutil::timeintervalformat(t(), 2), groups);
    return true;
}


void
SimpleRaytracer::render(int xres, int yres)
{
    OIIO::Timer timer;
    if (m_vk_frame >= 0) {
        if (render_vulkan_frame(xres, yres))
            return;
        // The whole-frame pipeline was requested and produced nothing.  Re-
        // rendering on the CPU here would report a pass for a device path that
        // never ran, which is exactly how a broken kernel used to look healthy.
        if (m_vulkan_requested && !vulkan_fallback_allowed()) {
            vulkan_report_failure("device frame rendering produced no image",
                                  nullptr);
            return;
        }
        if (m_vulkan_requested)
            errhandler().warningfmt(
                "[vulkan] falling back to the CPU because "
                "OSL_VULKAN_ALLOW_CPU_FALLBACK is set");
    }
    ShadingSystem* shadingsys = this->shadingsys;
    OIIO::parallel_for_chunked(
        0, yres, 0, [&, this](int64_t ybegin, int64_t yend) {
            // Request an OSL::PerThreadInfo for this thread.
            OSL::PerThreadInfo* thread_info = shadingsys->create_thread_info();

            // Request a shading context so that we can execute the shader.
            // We could get_context/release_context for each shading point,
            // but to save overhead, it's more efficient to reuse a context
            // within a thread.
            ShadingContext* ctx = shadingsys->get_context(thread_info);

            OIIO::ImageBuf::Iterator<float> p(pixelbuf,
                                              OIIO::ROI(0, xres, ybegin, yend));
            for (; !p.done(); ++p) {
                Color3 c = antialias_pixel(p.x(), p.y(), ctx);
                p[0]     = c.x;
                p[1]     = c.y;
                p[2]     = c.z;
            }

            // We're done shading with this context.
            shadingsys->release_context(ctx);
            shadingsys->destroy_thread_info(thread_info);
        });
    double rendertime = timer();
    errhandler().infofmt("Rendered {}x{} image with {} samples in {}", xres,
                         yres, aa * aa,
                         OIIO::Strutil::timeintervalformat(rendertime, 2));
}



void
SimpleRaytracer::clear()
{
    shaders().clear();
}

#endif  // #ifndef __CUDACC__

OSL_NAMESPACE_END
