// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

// Diagnostic: link N SPIR-V binaries into one, the way the runtime will once
// the materials come from LLVM's backend and the render kernel from clspv.
//
//   spirv_link_test <out.spv> <in0.spv> [<in1.spv> ...]

#include <cstdio>
#include <cstring>
#include <fstream>
#include <iterator>
#include <string>
#include <vector>

#include "vulkan_spirv_link.h"

static bool read_spv(const char* path, std::vector<uint32_t>& words,
                     std::string& err)
{
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        err = std::string("cannot open ") + path;
        return false;
    }
    std::vector<char> raw;
    raw.assign(std::istreambuf_iterator<char>(in),
               std::istreambuf_iterator<char>());
    const std::size_t nbytes = raw.size();
    if (nbytes < 20 || nbytes % 4 != 0) {
        err = std::string("not a SPIR-V binary: ") + path;
        return false;
    }
    words.assign(nbytes / 4, 0u);
    std::memcpy(words.data(), raw.data(), nbytes);
    if (words[0] != 0x07230203u) {
        err = std::string("bad SPIR-V magic in ") + path;
        return false;
    }
    return true;
}

static bool write_spv(const char* path, const std::vector<uint32_t>& words)
{
    std::ofstream out(path, std::ios::binary);
    if (!out) return false;
    out.write(reinterpret_cast<const char*>(words.data()),
              std::streamsize(words.size() * 4));
    return bool(out);
}

int main(int argc, char** argv)
{
    if (argc < 3) {
        std::fprintf(stderr,
                     "usage: spirv_link_test <out.spv> <in0.spv> [<in1.spv> ...]\n");
        return 2;
    }

    std::vector<std::vector<uint32_t>> modules;
    for (int i = 2; i < argc; ++i) {
        std::vector<uint32_t> words;
        std::string err;
        if (!read_spv(argv[i], words, err)) {
            std::fprintf(stderr, "%s\n", err.c_str());
            return 2;
        }
        std::fprintf(stdout, "input  %-40s %zu words\n", argv[i], words.size());
        modules.push_back(std::move(words));
    }

    std::vector<uint32_t> linked;
    std::string log;
    if (!OSL::pvt::vulkan_link_spirv_modules(modules, linked, log, false)) {
        std::fprintf(stdout, "FAILED\n%s", log.c_str());
        return 1;
    }
    if (!log.empty()) std::fprintf(stdout, "%s", log.c_str());
    if (!write_spv(argv[1], linked)) {
        std::fprintf(stderr, "cannot write %s\n", argv[1]);
        return 1;
    }
    std::fprintf(stdout, "OK     %-40s %zu words\n", argv[1], linked.size());
    return 0;
}
