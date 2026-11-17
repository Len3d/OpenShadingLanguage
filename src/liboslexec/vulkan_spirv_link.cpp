// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#include <spirv-tools/libspirv.hpp>
#include <spirv-tools/linker.hpp>

#include "vulkan_spirv_link.h"

OSL_NAMESPACE_ENTER

namespace pvt {

namespace {

// The target environment has to match what the modules were built for: the
// linker refuses to merge modules that disagree about the addressing or memory
// model, and validation is only meaningful against the environment the driver
// will use.
const spv_target_env kEnv = SPV_ENV_VULKAN_1_3;

const char* level_name(spv_message_level_t level)
{
    switch (level) {
    case SPV_MSG_FATAL: return "fatal";
    case SPV_MSG_INTERNAL_ERROR: return "internal";
    case SPV_MSG_ERROR: return "error";
    case SPV_MSG_WARNING: return "warning";
    case SPV_MSG_INFO: return "info";
    case SPV_MSG_DEBUG: return "debug";
    }
    return "?";
}

void append_message(std::string& log, spv_message_level_t level, const char*,
                    const spv_position_t& pos, const char* message)
{
    log += "spirv-link ";
    log += level_name(level);
    log += " (line " + std::to_string(pos.line)
           + ", column " + std::to_string(pos.column) + "): ";
    log += message ? message : "";
    log += "\n";
}

}  // namespace


bool vulkan_link_spirv_modules(const std::vector<std::vector<uint32_t>>& modules,
                               std::vector<uint32_t>& out_spirv,
                               std::string& out_log, bool create_library)
{
    out_log.clear();
    out_spirv.clear();
    if (modules.empty()) {
        out_log = "spirv-link: nothing to link\n";
        return false;
    }

    spvtools::LinkerOptions options;
    options.SetCreateLibrary(create_library);
    // Catch the easy mistakes early: id collisions between modules are exactly
    // what the id pass would have caught before the modules were ever separate.
    options.SetVerifyIds(true);

    spvtools::Context context(kEnv);
    context.SetMessageConsumer([&out_log](spv_message_level_t level,
                                          const char* source,
                                          const spv_position_t& pos,
                                          const char* message) {
        append_message(out_log, level, source, pos, message);
    });

    const spv_result_t result
        = spvtools::Link(context, modules, &out_spirv, options);
    if (result != SPV_SUCCESS) {
        if (out_spirv.empty()) out_spirv.clear();
        if (out_log.empty())
            out_log = "spirv-link failed with result "
                      + std::to_string(int(result)) + "\n";
        return false;
    }

    // A module that links but does not validate would fail at
    // vkCreateShaderModule instead, where the diagnostic is much worse.
    spvtools::SpirvTools tools(kEnv);
    tools.SetMessageConsumer([&out_log](spv_message_level_t level,
                                        const char* source,
                                        const spv_position_t& pos,
                                        const char* message) {
        append_message(out_log, level, source, pos, message);
    });
    if (!tools.Validate(out_spirv)) {
        out_log += "spirv-link: the linked module does not validate\n";
        return false;
    }
    return true;
}

}  // namespace pvt

OSL_NAMESPACE_EXIT
