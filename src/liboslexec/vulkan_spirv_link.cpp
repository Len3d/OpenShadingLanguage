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

// Drop the Linkage capability and every LinkageAttributes decoration from a
// linked (non-library) module.  Both come from the inputs, which are compiled as
// libraries so that their symbols can be resolved across modules; after the
// merge they describe a relationship that no longer exists, and Vulkan rejects
// the capability.  Every affected id keeps its OpVariable/OpFunction, so the
// symbols simply become ordinary module scope objects.
void strip_linkage(std::vector<uint32_t>& binary)
{
    // Word 0 is the magic number, 1..4 the header; instructions start at 5.
    if (binary.size() <= 5)
        return;
    const size_t header = 5;
    std::vector<uint32_t> out;
    out.reserve(binary.size());
    out.insert(out.end(), binary.begin(), binary.begin() + header);
    size_t i = header;
    while (i < binary.size()) {
        const uint16_t opcode = uint16_t(binary[i] & 0xffff);
        const uint16_t wcount = uint16_t(binary[i] >> 16);
        if (wcount == 0 || i + wcount > binary.size()) {
            // Malformed tail: keep it verbatim and stop, so a broken input
            // produces a validation error rather than a silent truncation.
            out.insert(out.end(), binary.begin() + i, binary.end());
            break;
        }
        bool drop = false;
        if (opcode == 17 /* OpCapability */ && wcount >= 2
            && binary[i + 1] == 5 /* Linkage */)
            drop = true;
        // `Addresses` comes from the same library inputs.  The modules already
        // declare VariablePointersStorageBuffer, which is what the remaining
        // storage-buffer pointer chains need on Vulkan, so the wider capability
        // is redundant -- and Vulkan 1.3 rejects it outright.
        if (opcode == 17 /* OpCapability */ && wcount >= 2
            && binary[i + 1] == 4 /* Addresses */)
            drop = true;
        if (opcode == 71 /* OpDecorate */ && wcount >= 3
            && binary[i + 2] == 41 /* LinkageAttributes */)
            drop = true;
        if (!drop)
            out.insert(out.end(), binary.begin() + i, binary.begin() + i + wcount);
        i += wcount;
    }
    binary.swap(out);
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
    // Every group entry point takes StorageBuffer pointers.  The linker treats
    // pointer parameters across a module boundary as a mismatch unless it is
    // told otherwise ("Allow pointer function parameters to mismatch the target
    // link"), which is what `spirv-link --allow-pointer-mismatch` switches off:
    // the two sides describe the same function, the pointers are just opaque in
    // both and the linker refuses to prove them equal on its own.
    options.SetAllowPtrTypeMismatch(true);
    // The arena view is declared by the device runtime and only *used* by the
    // material modules, so several modules import it and none defines it.  A
    // StorageBuffer variable is a resource declaration in SPIR-V, not a symbol
    // with a body, so leaving that import unresolved is the normal outcome here
    // -- otherwise the link stops at "Unresolved external reference to
    // osl_arena_i32" before anything is produced at all.
    options.SetAllowPartialLinkage(true);

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

    // The inputs are library modules, so they carry the Linkage capability and
    // LinkageAttributes decorations, and the linker keeps them in the result.
    // Nothing outside the link needs them once all modules have been merged
    // into one, and Vulkan 1.3 rejects the capability outright ("Capability
    // Linkage is not allowed by Vulkan 1.3").  Removing the capability and the
    // decorations is enough: every symbol the merge kept is now a plain module
    // scope definition, including the arena view the material modules only
    // imported, which becomes the resource the host binds.
    strip_linkage(out_spirv);

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
