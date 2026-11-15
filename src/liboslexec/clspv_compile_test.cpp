// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

// Unit test for the clspv-based SPIR-V lowering used by the SPIR-V backend.
//
// Usage: clspv_compile_test [out.spv]
// Exits 0 when the kernel lowers successfully and the emitted module starts
// with the SPIR-V magic number, non-zero otherwise.

#include "clspv_util.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

int
main(int argc, char* argv[])
{
    printf("clspv available: %d\n", (int)OSL::pvt::clspv_available());

    // `clspv_compile_test -ir <in.ll> <out.spv>` lowers a textual LLVM IR
    // module through clspv's IR entry point, which is what the backend feeds
    // it in-process.  Keeping this mode separate from the OpenCL C kernel
    // below means a failure here cannot be blamed on the source path.
    if (argc > 1 && std::string(argv[1]) == "-ir") {
        if (argc < 4) {
            printf("usage: %s -ir <in.ll> <out.spv>\n", argv[0]);
            return 2;
        }
        FILE* in = fopen(argv[2], "rb");
        if (!in) {
            printf("FAILED: cannot read %s\n", argv[2]);
            return 1;
        }
        std::string ir;
        char buf[4096];
        size_t n;
        while ((n = fread(buf, 1, sizeof(buf), in)) > 0)
            ir.append(buf, n);
        fclose(in);

        std::vector<uint32_t> ir_spirv;
        std::string ir_log;
        bool ir_ok = OSL::pvt::clspv_compile(ir, "-x ir --arch=spirv64", ir_spirv,
                                             ir_log);
        if (!ir_log.empty())
            printf("clspv log:\n%s\n", ir_log.c_str());
        if (!ir_ok || ir_spirv.empty()) {
            printf("FAILED: no SPIR-V produced from IR\n");
            return 1;
        }
        printf("OK (IR): %zu SPIR-V words, magic 0x%08x\n", ir_spirv.size(),
               ir_spirv[0]);
        FILE* out = fopen(argv[3], "wb");
        if (!out) {
            printf("FAILED: cannot write %s\n", argv[3]);
            return 1;
        }
        fwrite(ir_spirv.data(), sizeof(uint32_t), ir_spirv.size(), out);
        fclose(out);
        printf("wrote %s\n", argv[3]);
        return 0;
    }

    const char* kernel = R"(
__kernel void osl_kernel(__global float* out, __global const float* in)
{
    int i = get_global_id(0);
    out[i] = in[i] * 2.0f + 1.0f;
}
)";

    std::vector<uint32_t> spirv;
    std::string log;
    bool ok = OSL::pvt::clspv_compile(kernel, "-O2", spirv, log);
    if (!log.empty())
        printf("clspv log:\n%s\n", log.c_str());
    if (!ok || spirv.empty()) {
        printf("FAILED: no SPIR-V produced\n");
        return 1;
    }
    if (spirv[0] != 0x07230203u) {
        printf("FAILED: bad magic 0x%08x\n", spirv[0]);
        return 1;
    }

    printf("OK: %zu SPIR-V words, version %u, bound %u\n", spirv.size(),
           spirv[1], spirv[3]);

    if (argc > 1) {
        FILE* f = fopen(argv[1], "wb");
        if (!f) {
            printf("FAILED: cannot write %s\n", argv[1]);
            return 1;
        }
        fwrite(spirv.data(), sizeof(uint32_t), spirv.size(), f);
        fclose(f);
        printf("wrote %s\n", argv[1]);
    }
    return 0;
}
