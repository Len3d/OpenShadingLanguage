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
//   -> osl_kernel_gen / osl_kernel_intersect / osl_kernel_shade
//      (ptr addrspace(1)), one wavefront pass each
//
// That module is then handed to clspv (`-x ir`) exactly like a single group
// would have been.

#include "vulkan_module.h"

#include "oslexec_pvt.h"
#include "vulkan_legalize.h"
#include "vulkan_types.h"

#include <set>
#include <string>
#include <vector>

#include <llvm/ADT/StringRef.h>
#include <llvm/ADT/SmallPtrSet.h>
#include <llvm/Analysis/ConstantFolding.h>
#include <llvm/AsmParser/Parser.h>
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/IR/DebugInfo.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Intrinsics.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Linker/Linker.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/raw_ostream.h>
#include <llvm/IR/LegacyPassManager.h>
#include <llvm/Passes/PassBuilder.h>
#include <llvm/Transforms/IPO/GlobalDCE.h>
#include <llvm/Transforms/InstCombine/InstCombine.h>
#include <llvm/Transforms/Scalar.h>
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


// Turn "select between two same-typed allocas" into "index into one array".
//
// OSL's group code stages a layer's output into one of two local buffers and
// picks between them with a `select` on the pointer.  clspv has no select on
// pointer values in temporary storage, so it lowers the IR `select` into an
// SPIR-V `OpSelect` on a Function pointer and then has to declare
// `CapabilityVariablePointers`.  Intel's compiler aborts inside
// vkCreateComputePipelines on any module that carries that capability -- the
// module is rejected as a whole, so even an unrelated kernel (the `gen` pass)
// dies with it, which is what made this look like a size problem.
//
// Merging the two allocas into one `[2 x T]` and replacing the select with a
// select of the *index* keeps every pointer op an ordinary `OpAccessChain`
// over a variable, which needs no capability at all.  The merged array stays a
// local: every use of the original buffers becomes `arr[0]` / `arr[1]`.
int merge_selected_allocas (llvm::Module& M)
{
    int nmerged = 0;
    for (auto& F : M) {
        if (F.isDeclaration())
            continue;
        // Lifetime markers only accept a plain alloca, and the merge below
        // rewrites the allocas' uses into GEPs -- keep the markers and the
        // module no longer verifies ("can only be used on alloca or poison").
        // They are optimisation hints for a module that clspv is about to own.
        for (auto& BB : F) {
            std::vector<llvm::CallInst*> marks;
            for (auto& I : BB)
                if (auto* cb = llvm::dyn_cast<llvm::CallInst>(&I))
                    if (auto* c = cb->getCalledFunction())
                        if (c->getName().contains("llvm.lifetime."))
                            marks.push_back(cb);
            for (auto* c : marks)
                c->eraseFromParent();
        }
        // A pair can be selected many times; merge it once.
        std::vector<llvm::SelectInst*> sels;
        for (auto& BB : F)
            for (auto& I : BB)
                if (auto* si = llvm::dyn_cast<llvm::SelectInst>(&I))
                    if (si->getType()->isPointerTy())
                        sels.push_back(si);
        if (sels.empty())
            continue;

        for (auto* si : sels) {
            auto* a = llvm::dyn_cast<llvm::AllocaInst>(si->getTrueValue());
            auto* b = llvm::dyn_cast<llvm::AllocaInst>(si->getFalseValue());
            if (!a || !b || a == b || a->getFunction() != &F
                || b->getFunction() != &F
                || a->getAllocatedType() != b->getAllocatedType())
                continue;

            llvm::Type* elem = a->getAllocatedType();
            llvm::ArrayType* arrty
                = llvm::ArrayType::get(elem, 2);
            llvm::Instruction* at = &*F.getEntryBlock().getFirstInsertionPt();
            llvm::IRBuilder<> eb(at);
            llvm::AllocaInst* arr = eb.CreateAlloca(arrty, nullptr, "staged");
            arr->setAlignment(a->getAlign());
            // Replace every use of the pair with the corresponding slot.
            llvm::Value* slot0 = eb.CreateConstInBoundsGEP2_32(arrty, arr, 0, 0);
            llvm::Value* slot1 = eb.CreateConstInBoundsGEP2_32(arrty, arr, 0, 1);
            a->replaceAllUsesWith(slot0);
            b->replaceAllUsesWith(slot1);

            // The select itself is the one use the replacements above left in a
            // different shape: re-derive it as slot0/slot1 and pick the index.
            llvm::IRBuilder<> sb(si);
            llvm::Value* idx = sb.CreateSelect(
                si->getCondition(), sb.getInt64(0), sb.getInt64(1));
            llvm::Value* chosen = sb.CreateGEP(
                arrty, arr, { sb.getInt32(0), idx });
            si->replaceAllUsesWith(chosen);
            si->eraseFromParent();
            // The originals only live on through lifetime intrinsics.
            if (a->use_empty())
                a->eraseFromParent();
            if (b->use_empty())
                b->eraseFromParent();
            ++nmerged;
        }

        // Every select that chose between the same pair now chooses between two
        // slots of the merged array (`gep arr, 0, 0` vs `gep arr, 0, 1`).  Turn
        // those into a select of the *index*: the array stays one object, so
        // clspv only ever sees an OpAccessChain.
        for (auto& BB : F) {
            for (auto it = BB.begin(); it != BB.end();) {
                auto* si = llvm::dyn_cast<llvm::SelectInst>(&*it++);
                if (!si || !si->getType()->isPointerTy())
                    continue;
                auto* g0 = llvm::dyn_cast<llvm::GetElementPtrInst>(
                    si->getTrueValue());
                auto* g1 = llvm::dyn_cast<llvm::GetElementPtrInst>(
                    si->getFalseValue());
                if (!g0 || !g1
                    || g0->getPointerOperand() != g1->getPointerOperand()
                    || g0->getSourceElementType() != g1->getSourceElementType()
                    || g0->getNumIndices() != g1->getNumIndices())
                    continue;
                const unsigned nidx = (unsigned)g0->getNumIndices();
                std::vector<llvm::Value*> idxs;
                bool ok = true;
                for (unsigned k = 0; k < nidx && ok; ++k) {
                    llvm::Value* i0 = g0->getOperand(k + 1);
                    llvm::Value* i1 = g1->getOperand(k + 1);
                    if (i0 == i1) {
                        idxs.push_back(i0);
                        continue;
                    }
                    // Only the innermost index may differ, and only as a
                    // constant: that is the array-slot choice this pass created.
                    auto* c0 = llvm::dyn_cast<llvm::ConstantInt>(i0);
                    auto* c1 = llvm::dyn_cast<llvm::ConstantInt>(i1);
                    if (!c0 || !c1 || k + 1 != nidx
                        || c0->getBitWidth() != c1->getBitWidth())
                        ok = false;
                }
                if (!ok)
                    continue;
                llvm::IRBuilder<> b(si);
                llvm::Value* last = b.CreateSelect(si->getCondition(),
                                                   g0->getOperand(nidx),
                                                   g1->getOperand(nidx));
                idxs.push_back(last);
                llvm::Value* chosen = b.CreateGEP(g0->getSourceElementType(),
                                                  g0->getPointerOperand(), idxs);
                si->replaceAllUsesWith(chosen);
                si->eraseFromParent();
                ++nmerged;
            }
        }
    }
    return nmerged;
}


// Which `gep (gep ...)` pairs can be written as one GEP, and with what indices.
//
// Two shapes fold, and they are the two the device code actually produces:
//
//   A. `gep TY, (gep ..., ix...), 0, jx...`  -- the outer starts at element
//      offset 0 of the type the inner already points at.
//
//   B. `gep S, (gep [N x S], arr, 0, c...), %i` -- this is LLVM's lowering of
//      `&arr[i].c`: the outer steps by `sizeof(S)` even though its base points
//      *inside* S (so the type walk does not agree with the C level operation).  It
//      becomes `gep [N x S], arr, 0, %i, c...`, which walks the aggregate
//      properly and gives the same address.
//
// Either way the point is that clspv can emit a single OpAccessChain: an
// OpPtrAccessChain on temporary storage is pointer arithmetic, which needs
// CapabilityVariablePointers, and Intel's compiler aborts on that capability.
bool fold_one (llvm::GetElementPtrInst* inner, llvm::GetElementPtrInst* g,
               std::vector<llvm::Value*>& idxs)
{
    llvm::Type* pointee = inner->getResultElementType();
    auto* outer_first = llvm::dyn_cast<llvm::ConstantInt>(g->getOperand(1));
    if (!pointee && !outer_first)
        return false;

    // Shape A.
    if (pointee && pointee == g->getSourceElementType() && outer_first
        && outer_first->isZero()) {
        for (llvm::Value* v : inner->indices())
            idxs.push_back(v);
        for (unsigned k = 1; k < g->getNumOperands(); ++k)
            idxs.push_back(g->getOperand(k));
        return true;
    }

    // Shape B: the inner indices must start at element 0 of an array whose
    // element type is exactly the type the outer steps by.
    auto* arrty = llvm::dyn_cast<llvm::ArrayType>(inner->getSourceElementType());
    if (!arrty || arrty->getElementType() != g->getSourceElementType())
        return false;
    if (g->getNumOperands() != 2 || inner->getNumOperands() < 2)
        return false;
    auto* inner_first = llvm::dyn_cast<llvm::ConstantInt>(
        inner->getOperand(1));
    if (!inner_first || !inner_first->isZero())
        return false;
    idxs.push_back(inner->getOperand(1));  // 0: stay on the array
    idxs.push_back(g->getOperand(1));      // the element index
    for (unsigned k = 2; k < inner->getNumOperands(); ++k)
        idxs.push_back(inner->getOperand(k));
    return true;
}


// Fold `gep (gep base, ix...), ...` into one GEP.
//
// clspv emits a single `OpAccessChain` only when one LLVM `getelementptr`
// covers the whole chain from the variable.  Two chained GEPs become
// `OpAccessChain` + `OpPtrAccessChain`, and `OpPtrAccessChain` is *pointer
// arithmetic*: on temporary storage that is only legal with
// `CapabilityVariablePointers`, which the Intel driver refuses outright.  LLVM
// only folds the pair when the inner one is inbounds, so the staging accesses
// that OSL's group code generates (an offset into a local buffer, then a
// constant field inside it) stay split.  Folding them here is a pure rewrite of
// address arithmetic: same base, indices concatenated.
int fold_gep_chains (llvm::Module& M)
{
    int nfolded = 0;
    for (auto& F : M) {
        if (F.isDeclaration())
            continue;
        bool changed = true;
        while (changed) {
            changed = false;
            std::vector<llvm::GetElementPtrInst*> work;
            for (auto& BB : F)
                for (auto& I : BB)
                    if (auto* g = llvm::dyn_cast<llvm::GetElementPtrInst>(&I))
                        if (llvm::isa<llvm::GetElementPtrInst>(
                                g->getPointerOperand()))
                            work.push_back(g);
            for (auto* g : work) {
                auto* inner = llvm::cast<llvm::GetElementPtrInst>(
                    g->getPointerOperand());
                std::vector<llvm::Value*> idxs;
                if (!fold_one (inner, g, idxs))
                    continue;
                llvm::IRBuilder<> b(g);
                llvm::Value* folded = b.CreateGEP(
                    inner->getSourceElementType(), inner->getPointerOperand(),
                    idxs, g->getName());
                g->replaceAllUsesWith(folded);
                g->eraseFromParent();
                if (inner->use_empty())
                    inner->eraseFromParent();
                ++nfolded;
                changed = true;
            }
        }
    }
    return nfolded;
}


// clang/InstCombine canonicalizes `arr[i]` into a byte-stride GEP:
//
//     gep [sizeof(T) x i8], ptr arr, i64 i        // arr + i*sizeof(T)
//
// On a private (Function storage class) array that is poison for clspv: it
// cannot tell what the extent of the array is, so it lowers the index to
// OpPtrAccessChain -- pointer arithmetic -- which needs
// CapabilityVariablePointers, and Intel's compiler aborts on that capability.
// Restore the array-typed GEP, so clspv emits one OpAccessChain whose index is
// dynamic.  The address arithmetic is identical because the byte stride of the
// canonical form is exactly the allocation size of the array element.
int restore_array_geps (llvm::Module& M)
{
    const llvm::DataLayout& dl = M.getDataLayout();
    int nfixed                  = 0;
    for (auto& F : M) {
        if (F.isDeclaration())
            continue;
        std::vector<llvm::GetElementPtrInst*> work;
        for (auto& BB : F)
            for (auto& I : BB)
                if (auto* g = llvm::dyn_cast<llvm::GetElementPtrInst>(&I))
                    if (g->getPointerAddressSpace() == 0)
                        work.push_back(g);
        for (auto* g : work) {
            auto* stride = llvm::dyn_cast<llvm::ArrayType>(
                g->getSourceElementType());
            if (!stride || !stride->getElementType()->isIntegerTy(8))
                continue;
            // base has to be a private array whose element size is the stride.
            llvm::Type* pointee = nullptr;
            llvm::Value* base   = g->getPointerOperand();
            if (auto* ai = llvm::dyn_cast<llvm::AllocaInst>(base))
                pointee = ai->getAllocatedType();
            else if (auto* bi = llvm::dyn_cast<llvm::GetElementPtrInst>(base))
                pointee = bi->getResultElementType();
            auto* arr            = llvm::dyn_cast_or_null<llvm::ArrayType>(pointee);
            if (!arr || arr->getNumElements() < 2)
                continue;
            if (dl.getTypeAllocSize(arr->getElementType())
                != stride->getNumElements())
                continue;
            std::vector<llvm::Value*> idxs;
            idxs.push_back(llvm::ConstantInt::get(
                llvm::Type::getInt32Ty(M.getContext()), 0));
            for (unsigned k = 1; k < g->getNumOperands(); ++k)
                idxs.push_back(g->getOperand(k));
            llvm::IRBuilder<> b(g);
            llvm::Value* fixed
                = b.CreateGEP(pointee, base, idxs, g->getName());
            g->replaceAllUsesWith(fixed);
            g->eraseFromParent();
            ++nfixed;
        }
    }
    return nfixed;
}


// Drop `noundef` from every signature and call site.
//
// The device ABI is hand-written: the arena allocator returns a pointer into
// the storage buffer and the shader layer writes through it, so the parameter
// attributes clang inferred upstairs ("this argument is never undef/poison")
// do not hold once the module is assembled this way.  clang's attribute says a
// call that receives poison is immediate UB, and instcombine below is entitled
// to conclude exactly that: it turns the whole function into `store i1 true, ptr
// poison` plus `br i1 poison`, which is an empty kernel once clspv sees it.
// Dropping the attribute costs nothing here -- the IR is not being optimised
// for a language-level guarantee, it is being handed to a compiler that reads
// types, not attribute promises.
int strip_noundef (llvm::Module& M)
{
    int n = 0;
    for (auto& F : M) {
        for (auto& A : F.args())
            A.removeAttr(llvm::Attribute::NoUndef);
        for (auto& BB : F)
            for (auto& I : BB)
                if (auto* cb = llvm::dyn_cast<llvm::CallBase>(&I)) {
                    if (cb->hasRetAttr(llvm::Attribute::NoUndef)) {
                        cb->removeRetAttr(llvm::Attribute::NoUndef);
                        ++n;
                    }
                    for (unsigned i = 0; i < cb->arg_size(); ++i)
                        if (cb->paramHasAttr(i, llvm::Attribute::NoUndef)) {
                            cb->removeParamAttr(i, llvm::Attribute::NoUndef);
                            ++n;
                        }
                }
    }
    return n;
}


// InstCombine's rewrite of a body it has proven unreachable: it leaves the
// function behind as `store i1 true, ptr poison` plus a branch on that poison,
// keeping only the CFG shape.  Nothing else in these modules ever stores
// through a poison pointer, so this is a reliable marker.
static bool body_is_poisoned (const llvm::Function& F)
{
    for (const auto& BB : F)
        for (const auto& I : BB)
            if (const auto* si = llvm::dyn_cast<llvm::StoreInst>(&I))
                if (llvm::isa<llvm::PoisonValue>(si->getPointerOperand()))
                    return true;
    return false;
}


// Run instcombine over every definition, but treat it as an optimization and
// not as an oracle.
//
// The module is hand-assembled: declarations and parameters carry attributes
// that inference copied from a compilation that never saw a storage-buffer
// pointer, and several of them are simply wrong here.  InstCombine believes
// them, concludes that a body can never execute, and rewrites it to poison --
// which clspv then lowers to an empty kernel.  A poisoned body is a false
// positive, so the fix is not to chase each attribute: the pass runs on a
// clone, and only a clone that survives replaces the original.  Bodies it
// collapses keep exactly the code the front end emitted.
int instcombine_guarded (llvm::Module& M)
{
    llvm::PassBuilder pb(M.getContext(), nullptr,
                         llvm::PipelineTuningOptions());
    llvm::LoopAnalysisManager lam;
    llvm::FunctionAnalysisManager fam;
    llvm::CGSCCAnalysisManager cgam;
    llvm::ModuleAnalysisManager mam;
    pb.registerModuleAnalyses(mam);
    pb.registerCGSCCAnalyses(cgam);
    pb.registerFunctionAnalyses(fam);
    pb.registerLoopAnalyses(lam);
    pb.crossRegisterProxies(lam, fam, cgam, mam);
    llvm::FunctionPassManager fpm;
    fpm.addPass(llvm::InstCombinePass());

    std::vector<llvm::Function*> todo;
    for (auto& F : M)
        if (!F.isDeclaration())
            todo.push_back(&F);

    int reinstated = 0;
    for (auto* F : todo) {
        llvm::ValueToValueMapTy vmap;
        llvm::Function* clone = llvm::CloneFunction(F, vmap, nullptr);
        clone->setName(F->getName() + ".ic");
        fpm.run(*clone, fam);
        if (body_is_poisoned(*clone)) {
            clone->eraseFromParent();
            ++reinstated;
            continue;
        }
        F->deleteBody();
        llvm::ValueToValueMapTy back;
        for (auto i = clone->arg_begin(), j = F->arg_begin();
             i != clone->arg_end(); ++i, ++j)
            back[i] = j;
        llvm::SmallVector<llvm::ReturnInst*, 4> returns;
        llvm::CloneFunctionInto(F, clone, back,
                                llvm::CloneFunctionChangeType::LocalChangesOnly,
                                returns);
        clone->eraseFromParent();
    }
    return reinstated;
}


// Erase the dead leftovers of the legalization above.
//
// clspv translates what it is given: it does not run a dead-code pass late
// enough to drop an unused `ptrtoint`, so a folded-away address computation
// still reaches SPIR-V as `OpConvertPtrToU` -- an operation this module cannot
// legally contain (see fold_inttoptr_geps).
int
erase_dead_instructions (llvm::Module& M)
{
    int nerased = 0;
    for (auto& F : M) {
        if (F.isDeclaration())
            continue;
        bool changed = true;
        while (changed) {
            changed = false;
            for (auto& BB : F) {
                for (auto it = BB.begin(); it != BB.end();) {
                    llvm::Instruction* I = &*it++;
                    if (!I->use_empty() || I->mayHaveSideEffects()
                        || I->isTerminator() || llvm::isa<llvm::AllocaInst>(I)
                        || llvm::isa<llvm::PHINode>(I))
                        continue;
                    I->eraseFromParent();
                    ++nerased;
                    changed = true;
                }
            }
        }
    }
    return nerased;
}


// InstCombine also rewrites address arithmetic as an integer round trip:
//
//     %i = ptrtoint ptr %p to i64
//     %j = add i64 %i, %k
//     %q = inttoptr i64 %j to ptr
//
// Both ends of that trip are invalid SPIR-V here: `OpConvertPtrToU` and
// `OpConvertUToPtr` on a storage-buffer pointer require the `Addresses`
// capability (or PhysicalStorageBufferAddresses), which this module does not
// and cannot declare -- the validator rejects it, and Intel's compiler aborts
// on the module.  Undo the trip back into the `gep i8` it came from.
int
fold_inttoptr_geps (llvm::Module& M)
{
    int nfixed = 0;
    for (auto& F : M) {
        if (F.isDeclaration())
            continue;
        std::vector<llvm::IntToPtrInst*> work;
        for (auto& BB : F)
            for (auto& I : BB)
                if (auto* itp = llvm::dyn_cast<llvm::IntToPtrInst>(&I))
                    work.push_back(itp);
        for (auto* itp : work) {
            auto* add = llvm::dyn_cast<llvm::BinaryOperator>(itp->getOperand(0));
            if (!add || add->getOpcode() != llvm::Instruction::Add
                || !add->getType()->isIntegerTy(64))
                continue;
            llvm::Value* base   = nullptr;
            llvm::Value* offset = nullptr;
            for (unsigned k = 0; k < 2; ++k) {
                llvm::Value* o = add->getOperand(k);
                if (auto* pti = llvm::dyn_cast<llvm::PtrToIntInst>(o)) {
                    if (!base)
                        base = pti->getPointerOperand();
                    else
                        offset = pti;
                } else {
                    offset = o;
                }
            }
            if (!base || !offset || offset->getType() != add->getType())
                continue;
            llvm::IRBuilder<> b(itp);
            llvm::Value* g = b.CreateGEP(llvm::Type::getInt8Ty(M.getContext()),
                                         base, offset);
            itp->replaceAllUsesWith(g);
            itp->eraseFromParent();
            if (add->use_empty())
                add->eraseFromParent();
            ++nfixed;
        }
    }
    return nfixed;
}


// Recognize a null pointer in any address space, including the
// `addrspacecast (ptr addrspace(4) null to ptr addrspace(1))` constant that the
// front end emits for generic-space nulls.
static bool is_null_pointer (const llvm::Value* v)
{
    if (llvm::isa<llvm::ConstantPointerNull>(v))
        return true;
    if (const auto* ce = llvm::dyn_cast<llvm::ConstantExpr>(v))
        return ce->isCast() && is_null_pointer(ce->getOperand(0));
    return false;
}


// Fold every `icmp eq/ne ptr X, null` away.
//
// No pointer in this device module is ever null, by construction:
//
//  * every closure component and texture handle is an offset into the frame's
//    storage buffer, computed by `arena_bump` (vulkan_shadeops.cpp), which adds
//    a non-zero pool offset to the ShaderGlobals pointer;
//  * a black closure weight now allocates a zero-weight component instead of
//    returning null, and the add/mul helpers no longer short-circuit on a null
//    operand.
//
// On the strength of that the front end's `if (comp) { ... }` guards (and the
// same shape in the generated texture and background code) are dead, and
// folding them is what removes the last pointer *comparisons* from the module.
// SPIR-V can express a pointer comparison only through `OpPtrEqual` /
// `OpPtrNotEqual`, which require `CapabilityVariablePointers` (4442) -- the one
// capability the Intel driver aborts on inside vkCreateComputePipelines, so the
// whole module has to go without it.
int fold_null_pointer_tests (llvm::Module& M, bool verbose)
{
    std::vector<llvm::ICmpInst*> hits;
    for (auto& F : M) {
        for (auto& BB : F) {
            for (auto& I : BB) {
                auto* ic = llvm::dyn_cast<llvm::ICmpInst>(&I);
                if (!ic || !ic->getOperand(0)->getType()->isPointerTy())
                    continue;
                if (!is_null_pointer(ic->getOperand(0))
                    && !is_null_pointer(ic->getOperand(1)))
                    continue;
                if (verbose)
                    fprintf(stderr, "[vulkan] folding null test in %s\n",
                            F.getName().str().c_str());
                hits.push_back(ic);
            }
        }
    }
    llvm::LLVMContext& ctx = M.getContext();
    for (llvm::ICmpInst* ic : hits) {
        const bool eq = ic->getPredicate() == llvm::CmpInst::ICMP_EQ;
        ic->replaceAllUsesWith(
            llvm::ConstantInt::get(llvm::Type::getInt1Ty(ctx), !eq));
        // Erasing matters: leaving the comparison behind as dead code still
        // gives clspv an OpPtrNotEqual to emit, and with it the capability.
        ic->eraseFromParent();
    }
    return (int)hits.size();
}


// Give every call the calling convention of its callee.
//
// The device functions come out of the clang build with the OpenCL `spir_func`
// convention, while the call instructions this file inserts (the `osl_kernel_*`
// entry wrappers) are plain C calls.  A call whose convention disagrees with
// its callee is undefined behaviour by the LangRef, and the optimizer acts on
// it: an instcombine round turns such a wrapper -- literally `call` + `ret void`
// -- into `store i1 true, ptr poison`, and clspv then emits an empty kernel for
// every one of them, so the whole image comes back black.
int align_call_conventions (llvm::Module& M)
{
    int n = 0;
    for (auto& F : M)
        for (auto& BB : F)
            for (auto& I : BB)
                if (auto* cb = llvm::dyn_cast<llvm::CallBase>(&I))
                    if (auto* callee = cb->getCalledFunction())
                        if (cb->getCallingConv() != callee->getCallingConv()) {
                            cb->setCallingConv(callee->getCallingConv());
                            ++n;
                        }
    return n;
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
                                 std::string& out_ir, std::string& out_log,
                                 int light_group)
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
        // Which closure families this group's shader can produce.  The device
        // kernels are instantiated per set (`osl_vk_wave_shade<n>_<tag>`) and
        // the ones that cannot reach a family do not carry its code at all --
        // they are force-inlined before clspv, so a runtime `if (e.id == ...)`
        // would otherwise put every lobe in every kernel.  See the comment on
        // osl_vk_abi::kClosureSetBG for why the bits live in vulkan_types.h.
        std::vector<unsigned> group_cs(group_ir.size(), 0u);
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
            // Look for the literal closure id in the group's own calls: the
            // front end emits `osl_allocate_closure_component(sg, id, size)`
            // and `osl_allocate_weighted_closure_component(sg, id, size,
            // params)`, so the id in such a call names the closure the shader
            // built.  Matching on function *names* does not work (the device
            // runtime's own helper names contain "microfacet" too).
            {
                unsigned cs = 0u;
                std::string seen;
                for (auto& F : *g) {
                    for (auto& BB : F) {
                        for (auto& I : BB) {
                            llvm::CallInst* CI = llvm::dyn_cast<llvm::CallInst>(&I);
                            if (!CI)
                                continue;
                            llvm::Function* callee = CI->getCalledFunction();
                            if (!callee
                                || !callee->getName().contains(
                                    "closure_component"))
                                continue;
                            llvm::ConstantInt* idc
                                = llvm::dyn_cast<llvm::ConstantInt>(
                                    CI->getArgOperand(1));
                            if (!idc)
                                continue;
                            const unsigned long long id = idc->getZExtValue();
                            seen += " " + std::to_string(id);
                            switch (id) {
                            // testrender/shading.h ClosureIDs.
                            case 1:  // emission
                            case 2:  // background
                                cs |= osl_vk_abi::kClosureSetBG;
                                break;
                            case 3: cs |= osl_vk_abi::kClosureSetDiff; break;
                            case 7: cs |= osl_vk_abi::kClosureSetWard; break;
                            case 8: cs |= osl_vk_abi::kClosureSetMF; break;
                            case 9:   // reflection
                            case 10:  // fresnel reflection
                            case 11:  // refraction
                            case 12:  // transparent
                                cs |= osl_vk_abi::kClosureSetDelta;
                                break;
                            default:
                                // A layer this backend does not model: keep the
                                // full body so nothing is silently dropped.
                                cs |= osl_vk_abi::kClosureSetAll;
                                break;
                            }
                        }
                    }
                }
                // Temporary bisect knob: force every group onto the full body
                // (or onto an explicit mask) so a clspv miscompile can be
                // attributed to the set specialization rather than to the
                // merged code itself.
                if (const char* fc = getenv("OSL_VK_FORCE_CS"))
                    cs = (unsigned)strtoul(fc, nullptr, 0);
                group_cs[i] = cs;
                fprintf(stderr, "[vulkan] group %zu closure ids:%s (set 0x%x)\n",
                        i, seen.c_str(), cs);
            }
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
        // ---- the material dispatchers --------------------------------------
        //
        // The group entry points take (sg, groupdata, userdata, output,
        // shadeindex, interactive).  groupdata must be *real* memory: a layer
        // whose parameter block survives as memory (any body with more than
        // straight-line code -- checkerboard's `if (int(cs) ^ int(ct))` is one)
        // reads and writes it, and its first word is the "layer already
        // initialised" flag that osl_init_group_* clears.  With a null pointer
        // that flag reads the frame header and the parameter stores land there
        // too, so the material silently renders with a garbage closure weight.
        // userdata / output / interactive stay null: they are proved dead by
        // the optimizer in this backend.
        auto build_dispatch = [&](const std::string& name,
                                  const std::vector<size_t>& allowed) -> bool {
            llvm::Function* dispatch = render->getFunction(name);
            if (!dispatch)
                dispatch = llvm::Function::Create(
                    dtype, llvm::GlobalValue::ExternalLinkage, name, *render);
            if (!dispatch->empty()) {
                out_log = name + " is already defined";
                return false;
            }
            if (dispatch->arg_size() != 2
                || dispatch->getFunctionType() != dtype) {
                out_log = name + " has an unexpected signature";
                return false;
            }
            llvm::BasicBlock* bb
                = llvm::BasicBlock::Create(ctx, "entry", dispatch);
            llvm::IRBuilder<> b(bb);
            llvm::SwitchInst* sw = b.CreateSwitch(
                dispatch->getArg(1),
                llvm::BasicBlock::Create(ctx, "none", dispatch),
                (unsigned)allowed.size());
            for (size_t i : allowed) {
                llvm::BasicBlock* cse = llvm::BasicBlock::Create(
                    ctx, "mat" + std::to_string(i), dispatch);
                sw->addCase(
                    llvm::cast<llvm::ConstantInt>(
                        llvm::ConstantInt::get(i32, (uint64_t)i)),
                    cse);
                llvm::IRBuilder<> cb(cse);
                llvm::Value* gd = cb.CreateConstGEP1_32(
                    llvm::Type::getInt8Ty(ctx), dispatch->getArg(0),
                    (unsigned)osl_vk_abi::kGroupDataOffset);
                for (auto& ename : entries[i]) {
                    llvm::Function* F = render->getFunction(prefixes[i] + ename);
                    if (!F)
                        continue;
                    cb.CreateCall(F, { dispatch->getArg(0), gd, null1, null1,
                                       cb.getInt32(0), null1 });
                }
                cb.CreateRetVoid();
            }
            llvm::IRBuilder<> nb(sw->getDefaultDest());
            nb.CreateRetVoid();
            return true;
        };

        // One dispatcher per material, each restricted to that group plus the
        // light group.  A per-material shading kernel calls only its own
        // dispatcher, so clspv sees the other groups as unreachable and drops
        // them from that kernel; without the restriction every kernel would
        // pull in every shader and the split would be pointless.
        {
            std::vector<size_t> all;
            for (size_t i = 0; i < entries.size(); ++i)
                all.push_back(i);
            if (!build_dispatch("osl_vk_shade_dispatch", all))
                return false;
            for (size_t i = 0; i < entries.size(); ++i) {
                std::vector<size_t> allowed { i };
                if (light_group >= 0 && (size_t)light_group < entries.size()
                    && (size_t)light_group != i)
                    allowed.push_back((size_t)light_group);
                if (!build_dispatch(
                        "osl_vk_shade_dispatch_" + std::to_string(i), allowed))
                    return false;
            }
        }

        // ---- the compute kernels ----
        //
        // clspv emits one OpEntryPoint per SPIR_KERNEL function (verified on the
        // standalone clspv binary), and the host creates one compute pipeline
        // per entry name via add_shader().  Sharing the module is what lets
        // every pass use the same descriptor set and the same storage buffer.
        //
        // Shading is one entry per material: the host dispatches a bucket of
        // pixels per material, so each kernel only has to carry its own shader.
        // The suffix names the closure set the kernel was instantiated for; a
        // set with no instantiation falls back to `_all` (correct, just not
        // slimmed down).
        // The kernel bodies are `static` in the device runtime (so that the
        // unused closure-set instantiations can be erased before clspv, see the
        // pruning block below), which makes clang emit them as mangled
        // `_ZL..osl_vk_wave_shadeN_tagPU3AS1h` definitions plus an internal
        // alias carrying the plain C name.  `getFunction` only matches real
        // definitions, so the alias has to be resolved explicitly.
        auto find_body = [&](const std::string& name) -> llvm::Function* {
            if (llvm::Function* f = render->getFunction(name))
                return f;
            if (llvm::GlobalAlias* ga = render->getNamedAlias(name))
                if (auto* f
                    = llvm::dyn_cast<llvm::Function>(ga->getAliaseeObject()))
                    return f;
            return nullptr;
        };
        std::vector<std::pair<std::string, std::string>> kernels;
        kernels.push_back({ "osl_kernel_gen", "osl_vk_wave_gen" });
        kernels.push_back({ "osl_kernel_intersect", "osl_vk_wave_intersect" });
        // Material compaction runs on the device (vulkan_render.cpp pass 3a):
        // `compact` bins the wave by material with an atomic cursor per
        // material, `bucket_tail` closes the buckets, publishes the
        // vkCmdDispatchIndirect group counts and resets the cursors step.
        kernels.push_back({ "osl_kernel_compact", "osl_vk_wave_compact" });
        kernels.push_back({ "osl_kernel_bucket_tail",
                            "osl_vk_wave_bucket_tail" });
        auto closure_set_tag = [&](size_t i) -> std::string {
            const unsigned cs = group_cs[i];
            if (cs == osl_vk_abi::kClosureSetBG)
                return "_bg";
            if (cs == osl_vk_abi::kClosureSetDiff)
                return "_diff";
            if (cs == osl_vk_abi::kClosureSetMF)
                return "_mf";
            // A pure delta shader (mirror / refraction / transparent) needs the
            // delta body: without it the group fell back to `_all`, which drags
            // every lobe -- and the layer-output `select` between staging
            // buffers, which clspv turns into a Function-pointer OpSelect and
            // the driver rejects.
            if (cs == osl_vk_abi::kClosureSetDelta)
                return "_delta";
            return "_all";
        };
        for (size_t i = 0; i < entries.size(); ++i) {
            std::string impl
                = "osl_vk_wave_shade" + std::to_string(i) + closure_set_tag(i);
            if (!find_body(impl))
                impl = "osl_vk_wave_shade" + std::to_string(i) + "_all";
            kernels.push_back({ "osl_kernel_shade" + std::to_string(i), impl });
        }
        llvm::FunctionType* ktype
            = llvm::FunctionType::get(vd, { ptr1 }, false);
        for (auto& ks : kernels) {
            llvm::Function* impl = find_body(ks.second);
            if (!impl) {
                out_log = std::string("the device render module has no ")
                          + ks.second;
                return false;
            }
            llvm::Function* kernel = llvm::Function::Create(
                ktype, llvm::GlobalValue::ExternalLinkage, ks.first, *render);
            kernel->setCallingConv(llvm::CallingConv::SPIR_KERNEL);
            {
                llvm::BasicBlock* bb
                    = llvm::BasicBlock::Create(ctx, "entry", kernel);
                llvm::IRBuilder<> b(bb);
                b.CreateCall(impl, { kernel->getArg(0) });
                b.CreateRetVoid();
            }
            // clspv reads the workgroup size from *metadata* named
            // "reqd_work_group_size" (that is what clang emits for the OpenCL
            // attribute of the same name), not from a function attribute.  With
            // the attribute form it silently falls back to 1x1x1, which is a
            // silent wrong-answer bug: the dispatch then covers only `groups`
            // pixels.
            kernel->addFnAttr("reqd_work_group_size",
                              fmtformat("{} 1 1", osl_vk_abi::kWorkgroupSize));
            {
                llvm::Type* i32ty = llvm::Type::getInt32Ty(ctx);
                auto* md          = llvm::MDNode::get(
                    ctx,
                    { llvm::ConstantAsMetadata::get(llvm::ConstantInt::get(
                          i32ty, osl_vk_abi::kWorkgroupSize)),
                      llvm::ConstantAsMetadata::get(
                          llvm::ConstantInt::get(i32ty, 1)),
                      llvm::ConstantAsMetadata::get(
                          llvm::ConstantInt::get(i32ty, 1)) });
                kernel->setMetadata("reqd_work_group_size", md);
            }
        }

        // Erase the material × closure-set bodies no kernel reaches.
        //
        // vulkan_render.cpp instantiates four closure sets per material slot,
        // all with internal linkage; a scene uses one per material.  Everything
        // unreachable is dropped here, *before* the inliner touches it --
        // carrying every dead shade body through the always-inliner is what
        // turned the module build into an hour-long operation once the set
        // dimension was added.
        {
            // `used` parks the bodies in @llvm.used (the group modules each get
            // a `g<i>_` prefixed one after the link) and gives each body an
            // internal alias carrying its plain C name.  Both pin the
            // instantiations no kernel reached, and neither is needed once the
            // kernels hold direct references.
            {
                std::vector<llvm::GlobalVariable*> pinned;
                for (auto& gv : render->globals())
                    if (gv.getName().contains("llvm.used")
                        || gv.getName().contains("llvm.compiler.used"))
                        pinned.push_back(&gv);
                for (auto* gv : pinned)
                    gv->eraseFromParent();
                std::vector<llvm::GlobalAlias*> aliases;
                for (auto& ga : render->aliases())
                    if (ga.hasLocalLinkage()
                        && ga.getName().starts_with("osl_vk_wave_shade"))
                        aliases.push_back(&ga);
                for (auto* ga : aliases)
                    ga->eraseFromParent();
            }

            // Drop everything the kernels cannot reach.  GlobalDCE is the pass
            // that understands this correctly: with @llvm.used gone, an
            // internal-linkage body nothing calls (directly or through another
            // dead body) is unreachable, and it removes the uses along with the
            // body instead of needing them to be empty first.
            {
                size_t nfuncs_mid = render->size();
                llvm::legacy::PassManager pm;
                pm.add(llvm::createGlobalDCEPass());
                pm.run(*render);
                fprintf(stderr,
                        "[vulkan] prune: functions %zu -> %zu\n", nfuncs_mid,
                        render->size());
            }
        }

        // Everything else has to be inlined: clspv can only attribute a pointer
        // to the storage buffer when the chain from the kernel argument is still
        // visible in the IR, and a pointer returned by a call breaks that chain
        // (it makes clspv drop the store, see vulkan_legalize.h).
        for (auto& F : *render) {
            if (F.isDeclaration()
                || F.getCallingConv() == llvm::CallingConv::SPIR_KERNEL)
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

        // clspv never finishes on the microfacet kernel's raw shape: a loop
        // whose trip count is one of two closure slots, each carrying a full
        // Beckmann evaluation.  Spending an instcombine round here is cheaper
        // than diagnosing the pass inside clspv that diverges (mem2reg, sroa,
        // GVN, DCE, simplifycfg, early-cse and full unrolling all still hang;
        // only this one comes back in seconds).  It has to run *before* the
        // pointer legalisation below, because instcombine re-canonicalises
        // array indexing into the byte-stride GEP form those passes undo.
        {
            // Only the shading side: the ray-generation and intersection entry
            // points are left exactly as the compiler emitted them.  Rewriting
            // their pointers makes clspv collapse the whole kernel into an
            // empty `store i1 true, ptr poison` body, which is why the rule is
            // reachability from those two entries and not "everything".
            // Everything the shading bodies reach *is* rewritten, because that
            // is the code that makes clspv diverge on the microfacet kernel.
            if (!getenv("OSL_VK_NO_IC")) {
                const int ncc = align_call_conventions(*render);
                strip_noundef(*render);
                const int nkeep = instcombine_guarded(*render);
                if (getenv("OSL_VK_CC_TRACE"))
                    fprintf(stderr,
                            "[vulkan] realigned %d call convention(s), kept %d "
                            "body(ies) back from instcombine's UB rewrite\n",
                            ncc, nkeep);
            }
        }

        // Last step before clspv: no pointer *values* and no pointer
        // *comparisons* anywhere in the module.  A `select` between two local
        // buffers is a pointer value, and clspv answers it with
        // `CapabilityVariablePointers`; so does an `icmp` on pointers.
        // Merging the pair into one array turns the choice into an index, and
        // folding the null tests (see fold_null_pointer_tests) removes the
        // comparisons.  The Intel driver refuses the whole module at pipeline
        // creation when that capability is present.
        {
            const int nfixed  = restore_array_geps(*render);
            const int nmerged = merge_selected_allocas(*render);
            const int nfolded = fold_gep_chains(*render);
            const int ngp     = fold_inttoptr_geps(*render);
            const int ndead   = erase_dead_instructions(*render);
            const int nnull   = fold_null_pointer_tests(
                *render, getenv("OSL_VULKAN_LEGALIZE_TRACE") != nullptr);
            if (nmerged || nfolded || nfixed || nnull || ngp || ndead
                || getenv("OSL_VULKAN_LEGALIZE_TRACE"))
                fprintf(stderr,
                        "[vulkan] merged %d alloca pair(s), folded %d GEP chain(s),"
                        " restored %d array GEP(s), folded %d null test(s),"
                        " %d inttoptr GEP(s), erased %d dead instruction(s)\n",
                        nmerged, nfolded, nfixed, nnull, ngp, ndead);
        }

        // The device module is never debugged through clspv -- that would need
        // `-g`, which this pipeline does not pass -- so drop the DWARF here
        // rather than carry it into clspv.  (An earlier note here blamed clspv
        // non-termination on the debug info; that was wrong, the non-termination
        // was in SimplifyPointerBitcastPass and is fixed.)
        llvm::StripDebugInfo(*render);

        llvm::raw_string_ostream os(out_ir);
        render->print(os, nullptr);
        os.flush();
    }

    return true;
}

}  // namespace pvt

OSL_NAMESPACE_END
