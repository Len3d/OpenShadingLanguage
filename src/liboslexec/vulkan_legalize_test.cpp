// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

// Standalone driver for the Vulkan legalisation pass, so the transformation can
// be exercised on a dumped module without going through oslc/testrender.
//
//   vulkan_legalize_test <in.ll|in.bc> <out.ll>

#include "vulkan_legalize.h"

#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/IR/Verifier.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/raw_ostream.h>

#include <cstdio>
#include <string>

int main(int argc, char** argv)
{
    if (argc < 3) {
        std::fprintf(stderr,
                     "usage: %s <in.ll|in.bc> <out.ll>\n", argv[0]);
        return 2;
    }

    llvm::LLVMContext ctx;
    llvm::SMDiagnostic err;
    std::unique_ptr<llvm::Module> M = llvm::parseIRFile(argv[1], err, ctx);
    if (!M) {
        err.print("vulkan_legalize_test", llvm::errs());
        return 1;
    }

    std::string log;
    if (!OSL::pvt::vulkan_legalize_module(*M, log)) {
        std::fprintf(stderr, "legalize FAILED:\n%s\n", log.c_str());
        return 1;
    }

    std::printf("legalize returned\n");
    std::fflush(stdout);
    std::printf("%s\n", OSL::pvt::vulkan_legalize_summary().c_str());
    std::fflush(stdout);

    // Write the module out before verifying: a corrupted module can make the
    // verifier itself crash, and then the dump is all we have to look at.
    {
        std::error_code ec;
        llvm::raw_fd_ostream os(argv[2], ec);
        if (ec) {
            std::fprintf(stderr, "cannot write %s: %s\n", argv[2],
                         ec.message().c_str());
            return 1;
        }
        M->print(os, nullptr);
        std::printf("wrote %s\n", argv[2]);
        std::fflush(stdout);
    }

    {
        // Print each function on its own, so a corrupt function can be named.
        std::error_code ec2;
        llvm::raw_fd_ostream os2(argv[2] + std::string(".probe.ll"), ec2);
        for (auto& F : *M) {
            std::fprintf(stderr, "printing %s ...\n",
                         F.getName().str().c_str());
            std::fflush(stderr);
            F.print(os2);
        }
    }

    std::string verr;
    llvm::raw_string_ostream vos(verr);
    if (llvm::verifyModule(*M, &vos)) {
        std::fprintf(stderr, "verification FAILED:\n%s\n", verr.c_str());
        return 1;
    }
    std::printf("module verified\n");
    std::fflush(stdout);
    return 0;
}
