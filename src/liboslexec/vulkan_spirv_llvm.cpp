// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#include "vulkan_spirv_llvm.h"

#include <llvm/IR/Constants.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/IR/Metadata.h>
#include <llvm/IR/Module.h>
#include <llvm/IRReader/IRReader.h>
#include <llvm/MC/TargetRegistry.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Target/TargetMachine.h>
#include <llvm/Target/TargetOptions.h>
#include <llvm/TargetParser/Triple.h>

#include <OSL/oslconfig.h>

OSL_NAMESPACE_BEGIN

namespace pvt {

namespace {

// The Vulkan flavor is selected purely by the OS component of the triple
// (SPIRVSubtarget: `getOS() == Triple::Vulkan` -> Shader environment, Logical
// addressing, GLSL450 memory model).  The trailing "compute" and the 1.4 pin
// are what LLVM's own Vulkan tests use: the environment names the default
// execution model, and Vulkan 1.4/SPIR-V 1.4 is the first version where the
// legalised module's null pointer comparisons are legal.
static const char* kTriple = "spirv64-unknown-vulkan1.4-compute";

// "8,1,1" style, which is the spelling the backend parses for the local size.
std::string workgroup_size_string(const llvm::Function& F)
{
    const llvm::MDNode* md = F.getMetadata("reqd_work_group_size");
    if (!md || md->getNumOperands() < 3)
        return std::string();
    std::string s;
    for (unsigned i = 0; i < 3; ++i) {
        const llvm::ConstantInt* ci = llvm::dyn_cast_or_null<llvm::ConstantInt>(
            llvm::dyn_cast<llvm::ConstantAsMetadata>(md->getOperand(i))
                ? llvm::cast<llvm::ConstantAsMetadata>(md->getOperand(i))
                      ->getValue()
                : nullptr);
        if (!ci)
            return std::string();
        if (i)
            s += ",";
        s += std::to_string(ci->getZExtValue());
    }
    return s;
}

}  // namespace

bool
vulkan_compile_ir_via_llvm_vulkan(const std::string& ir_text,
                                  std::vector<uint32_t>& out_spirv,
                                  std::string& out_log)
{
    out_spirv.clear();
    out_log.clear();

    llvm::SMDiagnostic diag;
    auto buf = llvm::MemoryBuffer::getMemBuffer(ir_text, "vulkan-device.ll",
                                                /*RequiresNullTerminator=*/false);
    llvm::LLVMContext ctx;
    std::unique_ptr<llvm::Module> M = llvm::parseIR(*buf, diag, ctx);
    if (!M) {
        out_log = "IR parse failed: " + diag.getMessage().str();
        return false;
    }

    // Retarget the module at the Vulkan flavor.
#if OSL_LLVM_VERSION < 210
    M->setTargetTriple(kTriple);
#else
    M->setTargetTriple(llvm::Triple(kTriple));
#endif

    // In the shader environment the backend refuses to lower an entry point
    // that does not carry "hlsl.shader" ("This entry point lacks mandatory
    // hlsl.shader attribute", SPIRVCallLowering::getExecutionModel), and it
    // takes the local size from "hlsl.numthreads" rather than from the clspv
    // metadata we currently emit.
    for (auto& F : *M) {
        if (F.getCallingConv() != llvm::CallingConv::SPIR_KERNEL)
            continue;
        F.addFnAttr("hlsl.shader", "compute");
        std::string local = workgroup_size_string(F);
        if (!local.empty())
            F.addFnAttr("hlsl.numthreads", local);
    }

    std::string err;
    const llvm::Triple TT(kTriple);
    const llvm::Target* target = llvm::TargetRegistry::lookupTarget(TT, err);
    if (!target) {
        out_log = "no spirv64 target registered: " + err;
        return false;
    }
    llvm::TargetOptions opts;
    std::unique_ptr<llvm::TargetMachine> TM(target->createTargetMachine(
        TT, "generic", /*features=*/"", opts, std::nullopt));
    if (!TM) {
        out_log = "createTargetMachine failed for " + std::string(kTriple);
        return false;
    }

    llvm::SmallString<8192> bytes;
    llvm::raw_svector_ostream os(bytes);
    llvm::legacy::PassManager pm;
    if (TM->addPassesToEmitFile(pm, os, nullptr,
                                llvm::CodeGenFileType::ObjectFile)) {
        out_log = "the SPIR-V backend cannot emit an object file here";
        return false;
    }
    pm.run(*M);

    if (bytes.size() < 20 || (bytes.size() % 4) != 0) {
        out_log = "codegen produced " + std::to_string(bytes.size())
                  + " bytes, which is not a SPIR-V module";
        return false;
    }
    std::vector<uint32_t> words(bytes.size() / 4);
    memcpy(words.data(), bytes.data(), bytes.size());
    if (words[0] != 0x07230203u) {
        out_log = "codegen output does not start with the SPIR-V magic";
        return false;
    }
    out_spirv = std::move(words);
    return true;
}

}  // namespace pvt

OSL_NAMESPACE_END
