// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

// Batch module assembly for the Vulkan flavor.
//
// SPIR-V modules cannot call into each other, so a renderer that dispatches
// whole frames on the device (one invocation per pixel, shading whatever
// material a ray happens to hit) needs *all* the materials in the same module
// as the frame kernel.  The CUDA flavor gets away with a direct-call table;
// Vulkan has no function pointers, so the equivalent here is a synthesised
// switch:
//
//   vulkan_render.bc  (frame kernel, camera/sampling/intersection/accumulation)
//   + osl_layer_<g>_layer0/1/...  for every group of the scene
//   -> one module
//   -> osl_vk_shade_dispatch(sg, id)   switch over the material ids
//   -> osl_kernel(ptr addrspace(1))    calls osl_vk_raytrace_frame
//
// That module is then handed to clspv (`-x ir`) exactly like a single group
// would have been.

#include "vulkan_module.h"

#include "oslexec_pvt.h"
#include "vulkan_legalize.h"
#include "vulkan_types.h"

#include <string>
#include <vector>

#include <llvm/ADT/StringRef.h>
#include <llvm/Analysis/ConstantFolding.h>
#include <llvm/AsmParser/Parser.h>
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Linker/Linker.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/Transforms/Utils/Cloning.h>

#include <OSL/oslconfig.h>

// The serialised device modules live in the global namespace (MAKE_EMBEDDED_CPP
// emits them outside any namespace), hence the explicit `::` at the use site.
extern unsigned char vulkan_render_llvm_compiled_ops_block[];
extern int vulkan_render_llvm_compiled_ops_size;

OSL_NAMESPACE_BEGIN

namespace pvt {

namespace {

// clspv is built against a newer LLVM whose verifier demands the three-operand
// alias.scope form (`!{!scope, i1 false, !"name"}`).  This build's clang/LLVM
// prints the old two-operand form instead, and the verifier then rejects the
// whole module ("second domain operand must be an i1 constant").  Alias scopes
// are pure optimisation hints for a module that is about to be consumed by
// clspv, so drop both the attachments and the `noalias.scope.decl` calls that
// keep those nodes alive in the metadata table.
void strip_alias_metadata (llvm::Module& M)
{
    for (auto& F : M) {
        for (auto& BB : F) {
            for (auto it = BB.begin(), e = BB.end(); it != e;) {
                auto cur = it++;
                llvm::CallBase* cb = llvm::dyn_cast<llvm::CallBase>(&*cur);
                llvm::Function* callee
                    = cb ? cb->getCalledFunction() : nullptr;
                if (callee
                    && callee->getName() == "llvm.experimental.noalias.scope.decl") {
                    cur->eraseFromParent();
                    continue;
                }
                cur->setMetadata(llvm::LLVMContext::MD_alias_scope, nullptr);
                cur->setMetadata(llvm::LLVMContext::MD_noalias, nullptr);
            }
        }
    }
}


// Pick the functions a caller has to invoke for one group.
//
// A compiled group can carry two shapes: a non-trivial
// `osl_init_group_<group>` that runs the layers itself, or a set of
// `osl_layer_...` functions when the init function was folded to `ret void`.  In
// the second case the layers have to be called in order, which is what the
// switch below does.
bool collect_entry_names (llvm::Module& M, std::vector<std::string>& out,
                          std::string& log)
{
    llvm::Function* group_entry = nullptr;
    std::vector<llvm::Function*> layers;
    for (auto& F : M) {
        if (F.isDeclaration())
            continue;
        llvm::StringRef name = F.getName();
        if (name.starts_with("osl_init_group_"))
            group_entry = &F;
        else if (name.starts_with("osl_layer_"))
            layers.push_back(&F);
    }

    auto is_trivial = [](llvm::Function* F) {
        return F->size() == 1 && F->front().size() == 1
               && F->front().getTerminator()->getNumOperands() == 0;
    };

    // Does the init function already invoke the layers itself?  OSL emits
    // `osl_init_group_<group>` as the group's run function, so normally it does,
    // and calling it is the whole job.  But legalization rewrites part of the
    // group into a layer-called-one-by-one shape and leaves an init stub whose
    // only body is `groupdata[0] = 0` (the "layer not yet initialised" flag).  An
    // init that neither calls a layer nor is trivial must therefore still be
    // followed by the layers: calling the stub alone is a no-op, which is exactly
    // how a whole material ends up with no closure while linking cleanly.
    bool init_calls_layers = false;
    if (group_entry && !is_trivial(group_entry)) {
        for (auto& bb : *group_entry) {
            for (auto& inst : bb) {
                auto* cb = llvm::dyn_cast<llvm::CallBase>(&inst);
                llvm::Function* callee
                    = cb ? cb->getCalledFunction() : nullptr;
                if (callee
                    && callee->getName().starts_with("osl_layer_")) {
                    init_calls_layers = true;
                    break;
                }
            }
        }
    }

    out.clear();
    if (group_entry && !is_trivial(group_entry))
        out.push_back(group_entry->getName().str());
    if (!init_calls_layers) {
        for (auto* F : layers)
            out.push_back(F->getName().str());
    }    if (out.empty()) {
        log = "module has neither a non-empty osl_init_group_* nor an "
              "osl_layer_* function";
        return false;
    }
    for (auto& name : out) {
        llvm::Function* F = M.getFunction(name);
        if (!F || F->arg_size() != 6) {
            log = "group entry '" + name + "' does not take 6 arguments";
            return false;
        }
    }
    return true;
}


/// The module-scope constant a pointer resolves to, looking through the
/// getelementptr / bitcast / addrspacecast chain a shader constant picks up on
/// its way to a shadeop argument.
const llvm::GlobalVariable*
pointed_constant (const llvm::Value* v)
{
    using namespace llvm;
    while (v) {
        if (auto* GV = dyn_cast<GlobalVariable>(v))
            return (GV->isConstant() && GV->hasInitializer()) ? GV : nullptr;
        if (auto* ce = dyn_cast<ConstantExpr>(v)) {
            switch (ce->getOpcode()) {
            case Instruction::GetElementPtr:
            case Instruction::BitCast:
            case Instruction::AddrSpaceCast: v = ce->getOperand(0); continue;
            default: return nullptr;
            }
        }
        if (auto* gep = dyn_cast<GetElementPtrInst>(v)) {
            v = gep->getPointerOperand();
            continue;
        }
        return nullptr;
    }
    return nullptr;
}


/// The same pointer, as a Constant, so a load through it can be folded.  A
/// getelementptr written as an instruction is rebuilt as its ConstantExpr
/// equivalent; anything with a non-constant index stays unrepresentable and
/// returns null.
llvm::Constant*
constant_pointer (llvm::Value* v)
{
    using namespace llvm;
    if (isa<GlobalVariable>(v))
        return cast<Constant>(v);
    if (auto* ce = dyn_cast<ConstantExpr>(v)) {
        switch (ce->getOpcode()) {
        case Instruction::GetElementPtr:
        case Instruction::BitCast:
        case Instruction::AddrSpaceCast: return ce;
        default: return nullptr;
        }
    }
    if (auto* gep = dyn_cast<GetElementPtrInst>(v)) {
        auto* base = dyn_cast<Constant>(gep->getPointerOperand());
        if (!base)
            return nullptr;
        SmallVector<Constant*, 8> idx;
        for (auto& u : gep->indices()) {
            auto* ci = dyn_cast<Constant>(u.get());
            if (!ci)
                return nullptr;
            idx.push_back(ci);
        }
        return ConstantExpr::getGetElementPtr(gep->getSourceElementType(),
                                              base, idx, gep->isInBounds());
    }
    return nullptr;
}


}  // namespace


std::string vulkan_group_ir (const ShaderGroup& group)
{
    return group.vulkan_ir();
}


std::vector<std::string> vulkan_group_entry_names (const ShaderGroup& group)
{
    return group.vulkan_entry_names();
}


bool vulkan_collect_entry_names (llvm::Module& M, std::vector<std::string>& out,
                                 std::string& out_log)
{
    return collect_entry_names(M, out, out_log);
}


bool vulkan_build_render_module (const std::vector<std::string>& group_ir,
                                 const std::vector<std::vector<std::string>>& entries,
                                 std::string& out_ir, std::string& out_log)
{
    if (group_ir.size() != entries.size()) {
        out_log = "group IR and entry list sizes differ";
        return false;
    }

    llvm::LLVMContext ctx;

    // The frame kernel: compiled at build time by the same clang that produces
    // the device runtime, so it already carries the spirv64 triple and layout.
    {
        llvm::StringRef content(
            reinterpret_cast<const char*>(::vulkan_render_llvm_compiled_ops_block),
            (size_t)::vulkan_render_llvm_compiled_ops_size);
        llvm::MemoryBufferRef buf(content, "vulkan_render.bc");
        llvm::Expected<std::unique_ptr<llvm::Module>> mod_or_err
            = llvm::parseBitcodeFile(buf, ctx);
        if (!mod_or_err) {
            out_log = llvm::toString(mod_or_err.takeError());
            return false;
        }
        std::unique_ptr<llvm::Module> render = std::move(*mod_or_err);

        // The render module is compiled by this build's clang, so it carries the
        // same old-style alias scopes as the group modules below.
        strip_alias_metadata (*render);

        // One module per group, parsed from the IR text the group compile kept.
        //
        // Every group carries the same names for its own entry points
        // (`osl_init_group_<name>`, `osl_layer_...`) and the same device runtime
        // symbols, so linking two of them collides ("symbol multiply defined").
        // Prefixing every definition of group i with `g<i>_` makes them unique;
        // call sites follow the renamed Function objects, so nothing else has to
        // be rewritten.
        std::vector<std::string> prefixes(group_ir.size());
        for (size_t i = 0; i < group_ir.size(); ++i) {
            llvm::SMDiagnostic err;
            std::unique_ptr<llvm::Module> g
                = llvm::parseAssemblyString(group_ir[i], err, ctx);
            if (!g) {
                out_log = "cannot parse group IR #" + std::to_string(i) + ": "
                          + err.getMessage().str();
                return false;
            }
            g->setModuleIdentifier("osl_group_" + std::to_string(i));

            const std::string prefix = "g" + std::to_string(i) + "_";
            for (auto& F : *g) {
                if (!F.isDeclaration())
                    F.setName(prefix + F.getName().str());
            }
            for (auto& GV : g->globals()) {
                if (!GV.isDeclaration())
                    GV.setName(prefix + GV.getName().str());
            }
            prefixes[i] = prefix;

            strip_alias_metadata (*g);

            if (llvm::Linker::linkModules(*render, std::move(g))) {
                out_log = "linking group IR #" + std::to_string(i)
                          + " failed (duplicate symbol)";
                return false;
            }
        }

        llvm::Type* ptr1 = llvm::PointerType::get(ctx, 1);
        llvm::Type* i32  = llvm::Type::getInt32Ty(ctx);
        llvm::Type* vd   = llvm::Type::getVoidTy(ctx);
        llvm::Value* null1
            = llvm::ConstantPointerNull::get(llvm::cast<llvm::PointerType>(ptr1));

        // ---- osl_vk_shade_dispatch(ptr AS1 sg, i32 material) ----
        llvm::FunctionType* dtype
            = llvm::FunctionType::get(vd, { ptr1, i32 }, false);
        // The render module *calls* the dispatcher, so it already carries a
        // declaration of it.  Fill that declaration in rather than adding a
        // second function: a fresh one would be renamed to
        // `osl_vk_shade_dispatch.1` and the frame kernel would keep calling the
        // bodiless declaration, i.e. no shading at all.
        llvm::Function* dispatch = render->getFunction("osl_vk_shade_dispatch");
        if (!dispatch)
            dispatch = llvm::Function::Create(dtype,
                                              llvm::GlobalValue::ExternalLinkage,
                                              "osl_vk_shade_dispatch", *render);
        if (!dispatch->empty()) {
            out_log = "osl_vk_shade_dispatch is already defined";
            return false;
        }
        if (dispatch->arg_size() != 2
            || dispatch->getFunctionType() != dtype) {
            out_log = "osl_vk_shade_dispatch has an unexpected signature";
            return false;
        }
        {
            llvm::BasicBlock* bb = llvm::BasicBlock::Create(ctx, "entry", dispatch);
            llvm::IRBuilder<> b(bb);
            llvm::SwitchInst* sw
                = b.CreateSwitch(dispatch->getArg(1), llvm::BasicBlock::Create(
                                                         ctx, "none", dispatch),
                                 (unsigned)entries.size());
            for (size_t i = 0; i < entries.size(); ++i) {
                llvm::BasicBlock* cse
                    = llvm::BasicBlock::Create(ctx, "mat" + std::to_string(i),
                                               dispatch);
                sw->addCase(llvm::cast<llvm::ConstantInt>(
                                llvm::ConstantInt::get(i32, (uint64_t)i)),
                            cse);
                llvm::IRBuilder<> cb(cse);
                // The group entry points take (sg, groupdata, userdata, output,
                // shadeindex, interactive).  groupdata must be *real* memory:
                // a layer whose parameter block survives as memory (any body
                // with more than straight-line code -- checkerboard's
                // `if (int(cs) ^ int(ct))` is one) reads and writes it, and its
                // first word is the "layer already initialised" flag that
                // osl_init_group_* clears.  With a null pointer that flag reads
                // the frame header and the parameter stores land there too, so
                // the material silently renders with a garbage closure weight.
                // userdata / output / interactive stay null: they are proved
                // dead by the optimizer in this backend.
                llvm::Value* gd = cb.CreateConstGEP1_32(
                    llvm::Type::getInt8Ty(ctx), dispatch->getArg(0),
                    (unsigned)osl_vk_abi::kGroupDataOffset);
                for (auto& name : entries[i]) {
                    llvm::Function* F = render->getFunction(prefixes[i] + name);
                    if (!F)
                        continue;
                    cb.CreateCall(F, { dispatch->getArg(0), gd, null1, null1,
                                       cb.getInt32(0), null1 });
                }
                cb.CreateRetVoid();
            }
            llvm::BasicBlock* none = sw->getDefaultDest();
            llvm::IRBuilder<> nb(none);
            nb.CreateRetVoid();
        }

        // ---- the compute kernel ----
        llvm::FunctionType* ktype
            = llvm::FunctionType::get(vd, { ptr1 }, false);
        llvm::Function* kernel
            = llvm::Function::Create(ktype, llvm::GlobalValue::ExternalLinkage,
                                     "osl_kernel", *render);
        kernel->setCallingConv(llvm::CallingConv::SPIR_KERNEL);
        {
            llvm::BasicBlock* bb = llvm::BasicBlock::Create(ctx, "entry", kernel);
            llvm::IRBuilder<> b(bb);
            llvm::Function* frame = render->getFunction("osl_vk_raytrace_frame");
            if (!frame) {
                out_log = "the device render module has no "
                          "osl_vk_raytrace_frame";
                return false;
            }
            b.CreateCall(frame, { kernel->getArg(0) });
            b.CreateRetVoid();
        }
        // clspv reads the workgroup size from *metadata* named
        // "reqd_work_group_size" (that is what clang emits for the OpenCL
        // attribute of the same name), not from a function attribute.  With the
        // attribute form it silently falls back to 1x1x1, which is a silent
        // wrong-answer bug: the dispatch then covers only `groups` pixels.
        kernel->addFnAttr("reqd_work_group_size",
                          fmtformat("{} 1 1", osl_vk_abi::kWorkgroupSize));
        {
            llvm::Type* i32ty = llvm::Type::getInt32Ty(ctx);
            auto* md          = llvm::MDNode::get(
                ctx, { llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                           i32ty, osl_vk_abi::kWorkgroupSize)),
                       llvm::ConstantAsMetadata::get(
                           llvm::ConstantInt::get(i32ty, 1)),
                       llvm::ConstantAsMetadata::get(
                           llvm::ConstantInt::get(i32ty, 1)) });
            kernel->setMetadata("reqd_work_group_size", md);
        }

        // Everything else has to be inlined: clspv can only attribute a pointer
        // to the storage buffer when the chain from the kernel argument is still
        // visible in the IR, and a pointer returned by a call breaks that chain
        // (it makes clspv drop the store, see vulkan_legalize.h).
        for (auto& F : *render) {
            if (F.isDeclaration() || &F == kernel)
                continue;
            if (F.getCallingConv() == llvm::CallingConv::SPIR_KERNEL)
                continue;
            F.addFnAttr(llvm::Attribute::AlwaysInline);
        }

        // A shader constant handed to a shadeop -- `noise(vector(0.3,0.5,0.2))`
        // is the case that exposed this -- arrives as a pointer to a
        // module-scope constant.  Legalization moves every constant global to
        // address space 1 so it can be passed where the shadeop signature now
        // wants a storage-buffer pointer, but clspv has no way to place a
        // constant in the storage-buffer space: the load survives as a read of
        // the bound buffer at an offset nobody wrote, so the shadeop sees
        // whatever the frame data happens to hold there and the result changes
        // from pixel to pixel (measured on render-bumptest: noise went from a
        // constant 0.5 to values between -2.5 and 11).
        //
        // The value belongs in the instruction stream, so inline the call that
        // takes such a pointer and fold the resulting load back to a constant.
        // Only calls with a constant argument are inlined, so the module does
        // not grow any further than clspv's own AlwaysInliner would take it.
        {
            const llvm::DataLayout& DL = render->getDataLayout();
            bool changed               = false;
            int ninline                = 0;
            int nfold                  = 0;

            for (auto& F : *render) {
                for (auto& BB : F) {
                    for (auto it = BB.begin(); it != BB.end();) {
                        auto* cb = llvm::dyn_cast<llvm::CallBase>(&*it);
                        ++it;  // InlineFunction can destroy the instruction
                        if (!cb)
                            continue;
                        bool takes_constant = false;
                        for (auto& a : cb->args()) {
                            if (pointed_constant(a.get())) {
                                takes_constant = true;
                                break;
                            }
                        }
                        if (!takes_constant)
                            continue;
                        llvm::Function* callee = cb->getCalledFunction();
                        if (!callee || callee->isDeclaration())
                            continue;
                        if (callee->getCallingConv()
                            == llvm::CallingConv::SPIR_KERNEL)
                            continue;
                        llvm::InlineFunctionInfo ifi;
                        auto result = llvm::InlineFunction(*cb, ifi);
                        if (result.isSuccess()) {
                            ++ninline;
                            changed = true;
                        } else if (getenv("OSL_VULKAN_LEGALIZE_TRACE")) {
                            fprintf(stderr,
                                    "[const-arg] inline of %s failed: %s\n",
                                    callee->getName().str().c_str(),
                                    result.getFailureReason());
                        }
                    }
                }
            }

            for (auto& F : *render) {
                for (auto& BB : F) {
                    for (auto& I : BB) {
                        auto* li = llvm::dyn_cast<llvm::LoadInst>(&I);
                        if (!li || li->isVolatile() || li->isAtomic())
                            continue;
                        llvm::Constant* ptr = constant_pointer(li->getPointerOperand());
                        if (!ptr)
                            continue;
                        llvm::Constant* val
                            = llvm::ConstantFoldLoadFromConstPtr(ptr, li->getType(),
                                                                 DL);
                        if (val) {
                            li->replaceAllUsesWith(val);
                            ++nfold;
                            changed = true;
                        }
                    }
                }
            }
            if (getenv("OSL_VULKAN_LEGALIZE_TRACE"))
                fprintf(stderr, "[const-arg] inlined=%d folded=%d\n", ninline,
                        nfold);
            (void)changed;
        }

        // The last step the per-group path runs before clspv, and the batch
        // path was missing: rewrite every address-space-0 pointer to generic
        // (AS4) so clspv's InferAddressSpacesPass can push it back down to
        // Global.  Without it the device runtime's `void*` parameters stay
        // Private, which is not an error anywhere -- but then every closure
        // the runtime allocates is written into a private copy of the
        // ShaderGlobals, so the shader's arena looks untouched: sg.Ci comes
        // back 0 and the component at pool+offset is all zeros.
        {
            // clspv's legalizer does not know `freeze` ("unsupported
            // instruction: freeze"), and LLVM 20 emits it for operands that may
            // be poison.  Dropping it -- every use sees the original operand --
            // is the conservative simplification: freeze only pins a value that
            // is already well-defined everywhere we produce one.
            for (auto& F : *render) {
                for (auto& BB : F) {
                    std::vector<llvm::Instruction*> frozen;
                    for (auto& I : BB)
                        if (auto* fr = llvm::dyn_cast<llvm::FreezeInst>(&I)) {
                            fr->replaceAllUsesWith(fr->getOperand(0));
                            frozen.push_back(&I);
                        }
                    for (auto* I : frozen)
                        I->eraseFromParent();
                }
            }

            // NOTE: legalizing the merged batch module (vulkan_legalize_module)
            // looks like the right fix for the runtime's Private pointers, but
            // clspv then reports `undefined reference to 'osl_vk_arena_alloc'`:
            // the pass rewrites function signatures to AS4 while the calls that
            // pass AS1 pointers end up referring to a different function type.
            // The proper fix is to give vulkan_shadeops.cpp explicit AS1
            // parameters (like vulkan_render.cpp already has), instead of
            // running this pass over the merged module.
            std::string legalize_log;
            if (!vulkan_legalize_summary().empty())
                legalize_log = vulkan_legalize_summary();
            (void)legalize_log;
        }

        // clspv has no lowering for the fused `llvm.sincos` intrinsic, which
        // LLVM creates when a shader calls sin(x) and cos(x) on the same value
        // -- envmap.osl's spherical mapping does exactly that, and the failure
        // is fatal: clspv rejects the whole kernel with `undefined reference to
        // 'llvm.sincos.f32'` so the frame never reaches the device.
        //
        // Splitting the call back into `llvm.sin`/`llvm.cos` is not enough.  In
        // a module large enough to pull in clspv's builtin library, LLVM's
        // libcall simplifier sees those two intrinsics as the libc pair and
        // folds them straight back into sincos.  Emitting the OpenCL builtin
        // entry points (`_Z3sinf` / `_Z3cosf`) instead is what works: clspv
        // replaces those with GLSL `Sin`/`Cos` before any folding can look at
        // them.
        {
            std::vector<llvm::CallInst*> calls;
            for (auto& F : *render)
                for (auto& BB : F)
                    for (auto& I : BB)
                        if (auto* ci = llvm::dyn_cast<llvm::CallInst>(&I))
                            if (auto* callee = ci->getCalledFunction())
                                if (callee->getIntrinsicID()
                                    == llvm::Intrinsic::sincos)
                                    calls.push_back(ci);

            auto ocl_builtin = [&](const char* name,
                                   llvm::Type* ty) -> llvm::Function* {
                if (!ty->isFloatTy())
                    return nullptr;
                llvm::Function* f = render->getFunction(name);
                if (!f) {
                    f = llvm::Function::Create(
                        llvm::FunctionType::get(ty, { ty }, false),
                        llvm::GlobalValue::ExternalLinkage, name, *render);
                    f->setCallingConv(llvm::CallingConv::SPIR_FUNC);
                }
                return f;
            };

            for (auto* ci : calls) {
                llvm::IRBuilder<> b(ci);
                llvm::Value* x  = ci->getArgOperand(0);
                llvm::Type* ty  = x->getType();
                llvm::Function* sinf = ocl_builtin("_Z3sinf", ty);
                llvm::Function* cosf = ocl_builtin("_Z3cosf", ty);
                if (!sinf || !cosf) {
                    // Non-float overloads keep the intrinsics; the shaders this
                    // module is built from only ever produce float sincos.
                    sinf = llvm::Intrinsic::getOrInsertDeclaration(
                        render.get(), llvm::Intrinsic::sin, { ty });
                    cosf = llvm::Intrinsic::getOrInsertDeclaration(
                        render.get(), llvm::Intrinsic::cos, { ty });
                }
                llvm::Value* s = b.CreateCall(
                    sinf, { x });
                llvm::Value* c = b.CreateCall(cosf, { x });
                // IRBuilder emits calls in the default C convention; the
                // builtins are declared spir_func, and a call whose convention
                // disagrees with the callee crashes clspv outright.
                llvm::cast<llvm::CallInst>(s)->setCallingConv(
                    sinf->getCallingConv());
                llvm::cast<llvm::CallInst>(c)->setCallingConv(
                    cosf->getCallingConv());
                // Callers only ever pull the pair apart with extractvalue.  Feed
                // those the sin/cos results directly: rebuilding the aggregate
                // with insertvalue would let the simplifier recognise the
                // pattern again.
                bool paired = !ci->use_empty();
                for (auto* u : ci->users())
                    if (!llvm::isa<llvm::ExtractValueInst>(u))
                        paired = false;
                if (paired) {
                    std::vector<llvm::ExtractValueInst*> ex;
                    for (auto* u : ci->users())
                        ex.push_back(llvm::cast<llvm::ExtractValueInst>(u));
                    for (auto* e : ex) {
                        e->replaceAllUsesWith(e->getIndices()[0] == 0 ? s : c);
                        e->eraseFromParent();
                    }
                } else {
                    llvm::Value* v = llvm::UndefValue::get(ci->getType());
                    v = b.CreateInsertValue(v, s, 0);
                    v = b.CreateInsertValue(v, c, 1);
                    ci->replaceAllUsesWith(v);
                }
                ci->eraseFromParent();
            }
            // Drop the unreferenced declaration, so clspv never sees the name.
            std::vector<llvm::Function*> dead;
            for (auto& F : *render)
                if (F.isDeclaration() && F.getIntrinsicID()
                                              == llvm::Intrinsic::sincos
                    && F.use_empty())
                    dead.push_back(&F);
            for (auto* F : dead)
                F->eraseFromParent();
        }

        llvm::raw_string_ostream os(out_ir);
        render->print(os, nullptr);
        os.flush();
    }

    return true;
}

}  // namespace pvt

OSL_NAMESPACE_END
