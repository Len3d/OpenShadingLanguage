// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

// Diagnostic: run a device IR file through LLVM's Vulkan-flavor SPIR-V backend
// exactly as the runtime does (null-cast folding + arena typing included), and
// write the module out.  This is the only way to look at what the backend made
// of a group module without a running render.
//
//   vulkan_emit_test <in.ll> <out.spv>

#include <cstdio>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include <llvm/Support/TargetSelect.h>

#include "vulkan_spirv_llvm.h"

int main(int argc, char** argv)
{
    // The target has to be registered before the emitter looks it up; in the
    // library this is done once by LLVM_Util, here it has to be explicit.
    llvm::InitializeAllTargets();
    llvm::InitializeAllTargetMCs();
    llvm::InitializeAllDisassemblers();
    llvm::InitializeAllAsmPrinters();
    llvm::InitializeAllAsmParsers();


    if (argc != 3) {
        std::fprintf(stderr, "usage: vulkan_emit_test <in.ll> <out.spv>\n");
        return 2;
    }

    std::ifstream in(argv[1]);
    if (!in) {
        std::fprintf(stderr, "cannot open %s\n", argv[1]);
        return 2;
    }
    std::string ir((std::istreambuf_iterator<char>(in)),
                   std::istreambuf_iterator<char>());

    std::vector<uint32_t> spirv;
    std::string log;
    if (!OSL::pvt::vulkan_compile_ir_via_llvm_vulkan(ir, spirv, log)) {
        std::fprintf(stdout, "FAILED\n%s", log.c_str());
        return 1;
    }
    if (!log.empty()) std::fprintf(stdout, "%s", log.c_str());

    std::ofstream out(argv[2], std::ios::binary);
    if (!out) {
        std::fprintf(stderr, "cannot write %s\n", argv[2]);
        return 1;
    }
    out.write(reinterpret_cast<const char*>(spirv.data()),
              std::streamsize(spirv.size() * 4));
    std::fprintf(stdout, "OK %zu words -> %s\n", spirv.size(), argv[2]);
    return 0;
}
