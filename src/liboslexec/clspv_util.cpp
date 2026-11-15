// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#include "clspv_util.h"

#include <OSL/oslconfig.h>

#include <mutex>

#if OSL_USE_CLSPV
#    include <clspv/Compiler.h>
#endif

OSL_NAMESPACE_BEGIN

namespace pvt {

bool
clspv_available()
{
#if OSL_USE_CLSPV
    return true;
#else
    return false;
#endif
}

bool
clspv_compile(const std::string& opencl_source, const std::string& options,
              std::vector<uint32_t>& out_spirv, std::string& out_log)
{
#if OSL_USE_CLSPV
    out_spirv.clear();
    out_log.clear();
    if (opencl_source.empty()) {
        out_log = "clspv: empty source";
        return false;
    }

    char* binary    = nullptr;
    size_t bin_size = 0;
    char* log       = nullptr;
    const char* programs[1] = { opencl_source.c_str() };
    // clspv keeps its whole command line in process-global state
    // (clspv::Option::*): two concurrent compiles would overwrite each other's
    // language, arch and optimization flags, which shows up as heap
    // corruption inside the DLL.  OSL compiles groups on several threads, so
    // the call has to be serialized.
    static std::mutex clspv_mutex;
    ClspvError err;
    {
        std::lock_guard<std::mutex> lock(clspv_mutex);
        err = clspvCompileFromSourcesString(1, nullptr, programs, options.c_str(),
                                            &binary, &bin_size, &log);
    }
    if (log)
        out_log = log;

    bool ok = (err == CLSPV_SUCCESS && binary != nullptr && bin_size != 0
               && (bin_size % 4) == 0);
    if (ok) {
        const uint32_t* words = reinterpret_cast<const uint32_t*>(binary);
        out_spirv.assign(words, words + bin_size / 4);
    }

    clspvFreeOutputBuildObjs(binary, log);
    return ok;
#else
    OSL_UNUSED(opencl_source);
    OSL_UNUSED(options);
    OSL_UNUSED(out_spirv);
    out_log = "OSL was built without clspv (set CLSPV_ROOT)";
    return false;
#endif
}

}  // namespace pvt
OSL_NAMESPACE_END
