// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#include "vulkan_spirv_llvm.h"

#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/Transforms/IPO/AlwaysInliner.h>
#include <llvm/Transforms/Scalar.h>
#include <llvm/Transforms/Scalar/ADCE.h>
#include <llvm/Transforms/Utils/ModuleUtils.h>
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
#include <llvm/IR/GlobalVariable.h>
#include <llvm/TargetParser/Triple.h>
#include <functional>
#include <map>
#include <set>

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

// Vulkan rejects OpCapability Addresses, and the backend adds it as soon as an
// address space conversion survives into codegen.  Every conversion our device
// IR has is of the shape
//     addrspacecast (ptr addrspace(4) null to ptr addrspace(1))
// i.e. a null constant from the generic space cast to the buffer space when a
// pointer is compared against null.  A null pointer is a null pointer in every
// space, so those fold away to a plain null of the destination type; what is
// left is genuine, and there is nothing left at all once they are gone.
llvm::Constant* fold_null_casts(llvm::Constant* C)
{
    using namespace llvm;
    ConstantExpr* CE = dyn_cast<ConstantExpr>(C);
    if (!CE)
        return C;
    if (CE->getOpcode() == Instruction::AddrSpaceCast) {
        Constant* op = fold_null_casts(cast<Constant>(CE->getOperand(0)));
        if (isa<ConstantPointerNull>(op))
            return ConstantPointerNull::get(cast<PointerType>(CE->getType()));
        if (op != CE->getOperand(0))
            return ConstantExpr::getAddrSpaceCast(op, CE->getType());
        return CE;
    }
    SmallVector<Constant*, 4> ops;
    bool changed = false;
    for (Value* O : CE->operands()) {
        Constant* folded = fold_null_casts(cast<Constant>(O));
        changed |= (folded != O);
        ops.push_back(folded);
    }
    if (!changed)
        return CE;
    return CE->getWithOperands(ops, CE->getType());
}

bool fold_null_addrspacecasts(llvm::Module& M)
{
    using namespace llvm;
    bool changed = false;
    for (Function& F : M) {
        for (BasicBlock& BB : F) {
            for (auto II = BB.begin(); II != BB.end();) {
                Instruction& I = *II++;
                if (AddrSpaceCastInst* ASC = dyn_cast<AddrSpaceCastInst>(&I)) {
                    if (isa<ConstantPointerNull>(
                            fold_null_casts(cast<Constant>(ASC->getOperand(0))))) {
                        ASC->replaceAllUsesWith(ConstantPointerNull::get(
                            cast<PointerType>(ASC->getType())));
                        ASC->eraseFromParent();
                        changed = true;
                        continue;
                    }
                }
                for (unsigned i = 0; i < I.getNumOperands(); ++i) {
                    Constant* C = dyn_cast<Constant>(I.getOperand(i));
                    if (!C || !isa<ConstantExpr>(C))
                        continue;
                    Constant* folded = fold_null_casts(C);
                    if (folded != C) {
                        I.setOperand(i, folded);
                        changed = true;
                    }
                }
            }
        }
    }
    return changed;
}

// The device arena is byte addressed: every access is
//     %p = getelementptr i8, ptr %base, i64 <byte offset>
// followed by a load/store of a float, an i32 or an i64.  The SPIR-V backend
// lowers a byte GEP to OpPtrAccessChain, and that operation carries a
// requirement for OpCapability Addresses, which Vulkan forbids outright.
//
// Rewriting the GEP to index 32-bit elements instead (`gep i32, %base, off/4`)
// turns each access into a plain OpAccessChain on the SSBO's word array, which
// is what Vulkan wants.  OSL's arena allocator hands out 4-byte aligned slots,
// so the division is exact; anything that is not 4-aligned is left alone and
// reported, because silently rounding it would corrupt memory rather than
// fail loudly.
bool type_arena_accesses(llvm::Module& M, std::string& log)
{
    using namespace llvm;
    Type* i32 = Type::getInt32Ty(M.getContext());
    Type* i64 = Type::getInt64Ty(M.getContext());

    std::vector<GetElementPtrInst*> worklist;
    for (Function& F : M)
        for (BasicBlock& BB : F)
            for (Instruction& I : BB)
                if (GetElementPtrInst* GEP = dyn_cast<GetElementPtrInst>(&I))
                    worklist.push_back(GEP);

    bool changed   = false;
    unsigned kept  = 0;
    for (GetElementPtrInst* GEP : worklist) {
        if (!GEP->getParent())  // already rewritten as part of another chain
            continue;
        if (GEP->getNumIndices() != 1)
            continue;
        const Type* src = GEP->getSourceElementType();
        if (!src->isIntegerTy(8))
            continue;
        const unsigned as = GEP->getPointerAddressSpace();
        if (as != 1 && as != 11)
            continue;

        Value* off         = GEP->getOperand(1);
        Value* word_offset = nullptr;
        if (ConstantInt* ci = dyn_cast<ConstantInt>(off)) {
            const llvm::APInt four(ci->getBitWidth(), 4);
            if (!ci->getValue().urem(four).isZero()) {
                ++kept;
                continue;
            }
            word_offset = ConstantInt::get(i64, ci->getValue().udiv(four));
        } else {
            // A dynamic offset is a live sum of arena allocations and slot
            // sizes; all of those are multiples of four, so shifting is the
            // exact division.  Keeping this as a shift (not a divide) also
            // keeps the value cheap for the backend.
            IRBuilder<> b(GEP);
            word_offset = b.CreateLShr(off, 2, "arena.word");
        }

        IRBuilder<> b(GEP);
        Value* ng = b.CreateGEP(i32, GEP->getPointerOperand(), word_offset);
        if (Instruction* ngi = dyn_cast<Instruction>(ng))
            ngi->setDebugLoc(GEP->getDebugLoc());
        GEP->replaceAllUsesWith(ng);
        GEP->eraseFromParent();
        changed = true;
    }

    if (kept)
        log += "arena typing: " + std::to_string(kept)
               + " byte access(es) are not 4-aligned and were left as byte GEPs\n";
    return changed;
}

// Vulkan refuses OpCapability Linkage, and the backend emits it as soon as the
// module still exports a symbol (it decorates every externally visible function
// and global with LinkageAttributes).  These modules are self-contained --
// nothing outside them calls in -- so everything except an entry point is made
// internal.  Entry points stay external on purpose: the SPIR-V linker later
// resolves them by name.
bool drop_exported_symbols(llvm::Module& M)
{
    using namespace llvm;
    bool changed = false;
    for (Function& F : M) {
        if (F.isDeclaration()
            || F.getCallingConv() == CallingConv::SPIR_KERNEL)
            continue;
        if (F.hasExternalLinkage()) {
            F.setLinkage(GlobalValue::InternalLinkage);
            changed = true;
        }
    }
    for (GlobalVariable& G : M.globals()) {
        if (G.isDeclaration())
            continue;
        if (G.hasExternalLinkage()) {
            G.setLinkage(GlobalValue::InternalLinkage);
            changed = true;
        }
    }

    // Internal symbols with no caller are dead code, and codegen's GlobalDCE
    // would delete the whole module (the experiment: 15 words out).  These have
    // to survive until the linker picks them up, so they are pinned with
    // llvm.used.
    if (changed) {
        llvm::SmallVector<llvm::GlobalValue*, 16> keep;
        for (llvm::Function& F : M)
            if (!F.isDeclaration() && F.hasInternalLinkage())
                keep.push_back(&F);
        for (llvm::GlobalVariable& G : M.globals())
            if (!G.isDeclaration() && G.hasInternalLinkage())
                keep.push_back(&G);
        if (!keep.empty())
            llvm::appendToUsed(M, keep);
    }
    return changed;
}

// The backend turns llvm.memset/memcpy into calls to spirv.llvm_memset_pNN_*
// and spirv.llvm_memcpy_pNN_pNN_*, which it emits as imported SPIR-V strings
// decorated with LinkageAttributes -- and Linkage is a capability Vulkan does
// not allow.  Byte-granular lowering would only trade the problem for
// StorageBuffer8BitAccess, so these are expanded into 32-bit word stores here:
// the arena is 4-byte aligned by construction, and every length that reaches
// one of these intrinsics is a multiple of four.
bool expand_memory_intrinsics(llvm::Module& M, std::string& log)
{
    using namespace llvm;
    Type* i32 = Type::getInt32Ty(M.getContext());
    Type* i64 = Type::getInt64Ty(M.getContext());

    std::vector<CallInst*> worklist;
    for (Function& F : M)
        for (BasicBlock& BB : F)
            for (Instruction& I : BB)
                if (CallInst* CI = dyn_cast<CallInst>(&I))
                    if (Function* callee = CI->getCalledFunction())
                        if (callee->isIntrinsic()
                            && (callee->getIntrinsicID()
                                    == Intrinsic::memset
                                || callee->getIntrinsicID()
                                       == Intrinsic::memcpy))
                            worklist.push_back(CI);

    bool changed = false;
    for (CallInst* CI : worklist) {
        const bool is_set = CI->getCalledFunction()->getIntrinsicID()
                            == Intrinsic::memset;
        Value* dst = CI->getArgOperand(0);
        Value* src = is_set ? nullptr : CI->getArgOperand(1);
        Value* len = CI->getArgOperand(is_set ? 2 : 2);
        const unsigned as = cast<PointerType>(dst->getType())->getAddressSpace();
        if (as != 1 && as != 11) {
            log += "memory intrinsic on address space "
                   + std::to_string(as) + " left alone\n";
            continue;
        }

        IRBuilder<> b(CI);
        Value* word_count = nullptr;
        std::optional<uint64_t> const_words;
        if (ConstantInt* ci = dyn_cast<ConstantInt>(len)) {
            if (!ci->getValue().urem(APInt(ci->getBitWidth(), 4)).isZero()) {
                log += "memory intrinsic length is not a multiple of four, "
                       "left alone\n";
                continue;
            }
            const_words = ci->getValue().udiv(APInt(ci->getBitWidth(), 4))
                              .getZExtValue();
        } else {
            IRBuilder<> lb(CI);
            word_count = lb.CreateLShr(len, 2, "memset.words");
        }

        // memset value: one byte smeared across the word.
        Value* word_value = nullptr;
        if (is_set) {
            Value* v = CI->getArgOperand(1);
            IRBuilder<> vb(CI);
            Value* v32 = vb.CreateZExt(v, i32);
            word_value = vb.CreateMul(v32, ConstantInt::get(i32, 0x01010101u),
                                      "memset.splat");
        }

        if (const_words) {
            // Small, constant lengths: straight-line stores/loads, which is
            // also what the backend wants (no extra control flow on a hot path).
            for (uint64_t i = 0; i < *const_words; ++i) {
                IRBuilder<> ib(CI);
                Value* p = ib.CreateGEP(i32, dst,
                                        ConstantInt::get(i64, i), "");
                if (is_set) {
                    ib.CreateStore(word_value, p);
                } else {
                    Value* v = ib.CreateLoad(i32, p);
                    Value* sp = ib.CreateGEP(i32, src, ConstantInt::get(i64, i),
                                             "");
                    ib.CreateStore(v, sp);
                }
            }
        } else {
            // Variable length: a counted loop over words.  The block is split
            // at the call so the loop is a proper loop with one header, and
            // everything that followed the intrinsic continues in the tail.
            BasicBlock* bb   = CI->getParent();
            BasicBlock* tail = bb->splitBasicBlock(CI, "mem.cont");
            BasicBlock* loop = BasicBlock::Create(M.getContext(), "mem.words",
                                                  bb->getParent(), tail);
            BasicBlock* exit = BasicBlock::Create(M.getContext(), "mem.done",
                                                  bb->getParent(), tail);
            bb->getTerminator()->eraseFromParent();
            IRBuilder<> hb(bb);
            hb.CreateBr(loop);

            IRBuilder<> lb(loop);
            PHINode* idx = lb.CreatePHI(i64, 2, "mem.i");
            Value* p     = lb.CreateGEP(i32, dst, idx, "");
            if (is_set) {
                lb.CreateStore(word_value, p);
            } else {
                Value* v  = lb.CreateLoad(i32, p);
                Value* sp = lb.CreateGEP(i32, src, idx, "");
                lb.CreateStore(v, sp);
            }
            Value* next  = lb.CreateAdd(idx, ConstantInt::get(i64, 1), "mem.next");
            Value* more  = lb.CreateICmpULT(next, word_count, "mem.more");
            lb.CreateCondBr(more, loop, exit);
            idx->addIncoming(ConstantInt::get(i64, 0), bb);
            idx->addIncoming(next, loop);

            IRBuilder<> xb(exit);
            xb.CreateBr(tail);
        }

        CI->eraseFromParent();
        changed = true;
    }

    return changed;
}

}  // namespace

// ---------------------------------------------------------------------------
// The index-based arena ABI.
//
// OSL has no pointers in its language; the pointers in the device IR are an
// artefact of using host pointers as the arena currency.  Vulkan will not have
// them: a pointer flowing through a function or stored in memory drags in
// OpPtrAccessChain / VariablePointers, and the backend lowers every GEP to
// OpPtrAccessChain even when the base is a storage buffer variable.
//
// So the arena speaks offsets instead: every pointer value becomes an i64 byte
// offset from the start of the arena, and every access becomes an index into
// a typed view of the arena, rooted at that view:
//     getelementptr %Arena, ptr addrspace(11) @osl_arena_i32, i64 0, i32 0, i32 %word
// which the backend lowers to OpAccessChain (Element is always 0) and needs no
// capability beyond Shader.
//
// A null pointer is -1 in this encoding: it cannot collide with a real offset
// (those are non-negative), and the null tests that OSL emits for "no closure"
// keep working as an integer compare.
namespace {

/// The declared extent of each arena view, in elements (256 KB at 4 bytes).
static const unsigned kArenaViewWords = 65536;

static llvm::GlobalVariable*
make_arena_view(llvm::Module& M, llvm::Type* elem, unsigned binding,
                unsigned stride, const char* name)
{
    using namespace llvm;
    LLVMContext& C = M.getContext();
    // A fixed extent: the backend wraps a storage-buffer global's *element*
    // type in a struct of its own and decorates that struct with Block, so the
    // global holds the bare array and we only add the descriptor decorations.
    // The extent states the largest arena the shader may touch; the host has to
    // bind at least that much.
    ArrayType* AT = ArrayType::get(elem, kArenaViewWords);
    SmallVector<Type*, 1> members;
    members.push_back(AT);
    StructType* ST = StructType::get(C, members);
    // No initializer and no Block decoration: Vulkan wants the variable typed
    // as a struct, and the backend decides where Block goes for that struct.
    GlobalVariable* GV = new GlobalVariable(
        M, ST, /*isConstant=*/false, GlobalValue::InternalLinkage,
        /*Initializer=*/nullptr, name,
        /*insertBefore=*/nullptr, GlobalVariable::NotThreadLocal,
        /*AddressSpace=*/11);

    Type* i32 = Type::getInt32Ty(C);
    auto md_i32 = [&](int v) {
        return ConstantAsMetadata::get(ConstantInt::get(i32, v));
    };
    Metadata* deco[] = {
        MDNode::get(C, {md_i32(34), md_i32(binding)}),      // DescriptorSet
        MDNode::get(C, {md_i32(33), md_i32(binding)}),      // Binding
    };
    GV->addMetadata("spirv.Decorations",
                    *MDNode::get(C, ArrayRef<Metadata*>(deco)));
    return GV;
}

}  // namespace

bool
arena_index_accesses(llvm::Module& M, std::string& out_log)
{
    using namespace llvm;
    LLVMContext& C   = M.getContext();
    Type* i32        = Type::getInt32Ty(C);
    Type* i64        = Type::getInt64Ty(C);
    Type* f32        = Type::getFloatTy(C);
    const DataLayout DL = M.getDataLayout();

    GlobalVariable* v32 = make_arena_view(M, i32, 0, 4, "osl_arena_i32");
    GlobalVariable* vf  = make_arena_view(M, f32, 1, 4, "osl_arena_f32");
    GlobalVariable* v64 = make_arena_view(M, i64, 2, 8, "osl_arena_i64");

    // ---- 1. every pointer parameter and return value becomes an offset ----
    unsigned sig_changed = 0;
    for (Function& F : M) {
        bool changed = F.getReturnType()->isPointerTy();
        for (const Argument& A : F.args())
            changed = changed || A.getType()->isPointerTy();
        if (!changed)
            continue;
        SmallVector<Type*, 8> params;
        for (const Argument& A : F.args())
            params.push_back(A.getType()->isPointerTy() ? i64 : A.getType());
        Type* ret = F.getReturnType()->isPointerTy() ? i64 : F.getReturnType();
        FunctionType* NFT = FunctionType::get(ret, params, F.isVarArg());
        // NOTE: a real ABI change cannot be made this way -- Function exposes no
        // mutateFunctionType(), so getFunctionType() keeps the old signature and
        // callers keep a pointer-typed result.  The arena ABI is therefore meant
        // to be emitted as offsets by codegen, not repaired here.
        F.mutateType(NFT);
        for (Argument& A : F.args()) {
            if (A.getType()->isPointerTy())
                A.mutateType(i64);
        }
        // Pointer-only attributes (nonnull, dereferenceable, captures...) would
        // be nonsense on an offset.
        F.setAttributes(AttributeList::get(C, {}, {}, {}));
        ++sig_changed;
    }

    // ---- 2. offsets for every pointer value ----
    std::map<Value*, Value*> offsets;  // pointer value -> i64 byte offset

    auto lookup_off = [&](Value* P) -> Value* {
        auto it = offsets.find(P);
        return it == offsets.end() ? nullptr : it->second;
    };

    auto null_sentinel = [&]() { return ConstantInt::get(i64, -1); };

    // The constant part and the dynamic terms of a GEP's own displacement.
    auto gep_delta = [&](GEPOperator* G, std::vector<Value*>& dyn) -> int64_t {
        int64_t cst = 0;
        Type* ty     = G->getSourceElementType();
        for (unsigned i = 0; i < G->getNumIndices(); ++i) {
            Value* idx = G->getOperand(1 + i);
            if (StructType* st = dyn_cast<StructType>(ty)) {
                ConstantInt* ci = cast<ConstantInt>(idx);
                unsigned fld    = (unsigned)ci->getZExtValue();
                cst += (int64_t)DL.getStructLayout(st)->getElementOffset(fld);
                ty = st->getElementType(fld);
                continue;
            }
            Type* elem = nullptr;
            ArrayType* at = dyn_cast<ArrayType>(ty);
            VectorType* vt = dyn_cast<VectorType>(ty);
            if (at)
                elem = at->getElementType();
            else if (vt)
                elem = vt->getElementType();
            else
                elem = ty;  // scalar: the index strides the scalar itself
            int64_t stride = (int64_t)DL.getTypeAllocSize(elem);
            if (ConstantInt* ci = dyn_cast<ConstantInt>(idx)) {
                cst += ci->getSExtValue() * stride;
            } else {
                dyn.push_back(idx);
                dyn.push_back(ConstantInt::get(i64, stride));
            }
            ty = elem;
        }
        return cst;
    };

    std::function<Value*(Value*)> offset_of = [&](Value* P) -> Value* {
        auto it = offsets.find(P);
        if (it != offsets.end())
            return it->second;
        Value* result = nullptr;
        if (isa<ConstantPointerNull>(P)) {
            result = null_sentinel();
        } else if (isa<GlobalVariable>(P)) {
            result = ConstantInt::get(i64, 0);
        } else if (isa<Argument>(P)) {
            // Signature conversion already replaced this with an i64 argument.
            result = P;
        } else if (GEPOperator* G = dyn_cast<GEPOperator>(P)) {
            std::vector<Value*> dyn;
            int64_t cst   = gep_delta(G, dyn);
            Instruction* GI = cast<Instruction>(G);
            IRBuilder<> B(GI);
            Value* base = offset_of(G->getPointerOperand());
            Value* acc  = B.CreateAdd(base, ConstantInt::get(i64, cst), "arena.off");
            for (size_t i = 0; i < dyn.size(); i += 2) {
                Value* idx = B.CreateSExtOrTrunc(dyn[i], i64);
                Value* term = B.CreateMul(idx, dyn[i + 1], "arena.term");
                acc         = B.CreateAdd(acc, term, "arena.off");
            }
            result = acc;
        } else if (PHINode* PH = dyn_cast<PHINode>(P)) {
            result = lookup_off(PH);  // pre-created in step 3
        } else if (SelectInst* SI = dyn_cast<SelectInst>(P)) {
            IRBuilder<> B(SI);
            result = B.CreateSelect(SI->getCondition(),
                                    offset_of(SI->getTrueValue()),
                                    offset_of(SI->getFalseValue()), "arena.sel");
        } else if (LoadInst* LI = dyn_cast<LoadInst>(P)) {
            result = lookup_off(LI);  // rewritten in step 3
        }
        if (!result) {
            out_log += "\n  cannot express an offset for: "
                       + std::string(P->getName()) + " ("
                       + std::to_string(P->getValueID()) + ")";
            return nullptr;
        }
        offsets[P] = result;
        return result;
    };

    // ---- 3. rewrite phis, loads, stores and the remaining pointer operands ----
    // Pointer phis first: their replacement exists before anything looks at it.
    for (Function& F : M) {
        if (F.isDeclaration())
            continue;
        for (BasicBlock& BB : F) {
            for (Instruction& I : BB) {
                PHINode* PH = dyn_cast<PHINode>(&I);
                if (!PH || !PH->getType()->isPointerTy())
                    continue;
                IRBuilder<> B(PH);
                PHINode* NP = B.CreatePHI(i64, PH->getNumIncomingValues(),
                                          PH->getName() + ".off");
                offsets[PH] = NP;
            }
        }
    }
    // Fill the new phis once every value has an offset.
    for (auto& kv : offsets) {
        PHINode* OP = dyn_cast<PHINode>(kv.first);
        PHINode* NP = dyn_cast<PHINode>(kv.second);
        if (!OP || !NP)
            continue;
        for (unsigned i = 0; i < OP->getNumIncomingValues(); ++i) {
            Value* off = offset_of(OP->getIncomingValue(i));
            if (!off)
                return false;
            NP->addIncoming(off, OP->getIncomingBlock(i));
        }
    }

    // The 32-bit view's pointer for the word that holds byte offset `off`.
    auto word_ptr = [&](IRBuilder<>& B, Value* off) -> Value* {
        Value* word = B.CreateLShr(off, ConstantInt::get(i64, 2), "arena.word");
        Value* idx  = B.CreateTrunc(word, i32, "arena.idx");
        return B.CreateGEP(v32->getValueType(), v32,
                           {ConstantInt::get(i64, 0),
                            ConstantInt::get(i32, 0), idx},
                           "arena.ptr");
    };

    std::vector<Instruction*> dead;
    unsigned loads = 0, stores = 0;
    for (Function& F : M) {
        if (F.isDeclaration())
            continue;
        for (BasicBlock& BB : F) {
            for (Instruction& II : BB) {
                Instruction* I = &II;
                if (!I->getParent())
                    continue;
                if (isa<PHINode>(I) || isa<GetElementPtrInst>(I))
                    continue;

                if (LoadInst* LI = dyn_cast<LoadInst>(I)) {
                    if (LI->getPointerAddressSpace() != 1
                        && LI->getPointerAddressSpace() != 11)
                        continue;
                    Value* off = offset_of(LI->getPointerOperand());
                    if (!off)
                        return false;
                    Type* vt = LI->getType();
                    if (vt->isPointerTy())
                        vt = i64;  // a stored pointer is a stored offset

                    // Sub-word reads have no view of their own; the word that
                    // contains the byte is loaded and shifted down.  The arena
                    // is byte addressable in principle, but a view of `i8`
                    // would need the optional storageBuffer8BitAccess feature.
                    if (vt->isIntegerTy(8) || vt->isIntegerTy(16)) {
                        IRBuilder<> B(LI);
                        Value* w = B.CreateLoad(i32, word_ptr(B, off));
                        Value* lane = B.CreateAnd(off, ConstantInt::get(i64, 3),
                                                 "arena.lane");
                        Value* sh = B.CreateTrunc(
                            B.CreateShl(lane, ConstantInt::get(i64, 3)),
                            i32, "arena.shift");
                        Value* shr = B.CreateLShr(w, sh, "arena.byte");
                        Value* NV  = B.CreateTrunc(shr, vt, LI->getName());
                        LI->replaceAllUsesWith(NV);
                        dead.push_back(LI);
                        ++loads;
                        continue;
                    }
                    GlobalVariable* view = nullptr;
                    uint64_t shift       = 2;
                    if (vt->isIntegerTy(64)) {
                        view  = v64;
                        shift = 3;
                    } else if (vt->isFloatTy()) {
                        view = vf;
                    } else if (vt->isIntegerTy(32)) {
                        view = v32;
                    } else {
                        out_log += "\n  arena access of an unsupported width";
                        return false;
                    }
                    IRBuilder<> B(LI);
                    Value* word = B.CreateLShr(off, ConstantInt::get(i64, shift),
                                               "arena.word");
                    Value* idx  = B.CreateTrunc(word, i32, "arena.idx");
                    Value* p    = B.CreateGEP(view->getValueType(), view,
                                              {ConstantInt::get(i64, 0),
                                               ConstantInt::get(i32, 0), idx},
                                              "arena.ptr");
                    LoadInst* NV = B.CreateLoad(vt, p, LI->isVolatile(),
                                                LI->getName());
                    if (LI->getType()->isPointerTy()) {
                        // A stored pointer is a stored offset: retype first so
                        // the replacement agrees with every user's new type.
                        LI->mutateType(i64);
                        offsets[LI] = NV;
                    }
                    LI->replaceAllUsesWith(NV);
                    dead.push_back(LI);
                    ++loads;
                    continue;
                }

                if (StoreInst* SI = dyn_cast<StoreInst>(I)) {
                    if (SI->getPointerAddressSpace() != 1
                        && SI->getPointerAddressSpace() != 11)
                        continue;
                    Value* off = offset_of(SI->getPointerOperand());
                    if (!off)
                        return false;
                    Value* val = SI->getValueOperand();
                    Type* vt   = val->getType();
                    if (vt->isPointerTy()) {
                        val = offset_of(val);
                        if (!val)
                            return false;
                        vt = i64;
                    }
                    if (vt->isIntegerTy(8) || vt->isIntegerTy(16)) {
                        IRBuilder<> B(SI);
                        Value* w = B.CreateLoad(i32, word_ptr(B, off));
                        Value* lane = B.CreateAnd(off, ConstantInt::get(i64, 3),
                                                  "arena.lane");
                        Value* sh = B.CreateTrunc(
                            B.CreateShl(lane, ConstantInt::get(i64, 3)),
                            i32, "arena.shift");
                        Value* mask = B.CreateShl(ConstantInt::get(i32, 0xFF),
                                                  sh, "arena.mask");
                        Value* shifted
                            = B.CreateShl(B.CreateZExt(val, i32), sh, "arena.bits");
                        Value* cleared = B.CreateAnd(w, B.CreateNot(mask));
                        Value* merged  = B.CreateOr(cleared,
                                                    B.CreateAnd(shifted, mask));
                        B.CreateStore(merged, word_ptr(B, off));
                        dead.push_back(SI);
                        ++stores;
                        continue;
                    }
                    GlobalVariable* view = nullptr;
                    uint64_t shift       = 2;
                    if (vt->isIntegerTy(64)) {
                        view  = v64;
                        shift = 3;
                    } else if (vt->isFloatTy()) {
                        view = vf;
                    } else if (vt->isIntegerTy(32)) {
                        view = v32;
                    } else {
                        out_log += "\n  arena store of an unsupported width";
                        return false;
                    }
                    IRBuilder<> B(SI);
                    Value* word = B.CreateLShr(off, ConstantInt::get(i64, shift),
                                               "arena.word");
                    Value* idx  = B.CreateTrunc(word, i32, "arena.idx");
                    Value* p    = B.CreateGEP(view->getValueType(), view,
                                              {ConstantInt::get(i64, 0),
                                               ConstantInt::get(i32, 0), idx},
                                              "arena.ptr");
                    B.CreateStore(val, p, SI->isVolatile());
                    dead.push_back(SI);
                    ++stores;
                    continue;
                }

                // Anything else that consumes a pointer now consumes an offset.
                // Values that still *produce* a pointer (a phi, a select, a GEP)
                // are left alone here: their offset expression is what replaces
                // them, and they are erased once they have no users.
                CallInst* CI = dyn_cast<CallInst>(I);
                if (I->getType()->isPointerTy() && !CI)
                    continue;
                for (unsigned opi = 0; opi < I->getNumOperands(); ++opi) {
                    Value* V = I->getOperand(opi);
                    if (!V || !V->getType()->isPointerTy())
                        continue;
                    if (isa<Function>(V))
                        continue;
                    Value* off = offset_of(V);
                    if (!off)
                        return false;
                    I->setOperand(opi, off);
                }
                // A call that used to hand back a pointer now hands back the
                // offset itself: its result type follows the callee's.
                if (CI && CI->getType()->isPointerTy()) {
                    CI->mutateType(i64);
                    offsets[CI] = CI;
                }
                // A direct call keeps its own copy of the callee's signature,
                // and the backend reads parameter types from it, so it has to
                // follow the callee we retyped.
                if (CI) {
                    if (Function* callee = CI->getCalledFunction())
                        CI->mutateFunctionType(callee->getFunctionType());
                }
            }
        }
    }

    // A second sweep over calls: a CallBase reports its result type through the
    // function type it stores, so `mutateType` alone does not move it, and an
    // instruction visited before its callee was retyped can be left pointing at
    // the old signature.  Both are settled here, once, after every signature is
    // final.
    unsigned calls_fixed = 0;
    for (Function& F : M) {
        if (F.isDeclaration())
            continue;
        for (BasicBlock& BB : F) {
            for (Instruction& I : BB) {
                CallBase* CB = dyn_cast<CallBase>(&I);
                if (!CB)
                    continue;
                Function* callee = CB->getCalledFunction();
                if (!callee)
                    continue;
                bool touched = false;
                for (unsigned opi = 0; opi < CB->arg_size(); ++opi) {
                    Value* V = CB->getArgOperand(opi);
                    if (!V || !V->getType()->isPointerTy())
                        continue;
                    if (Value* off = offset_of(V)) {
                        CB->setArgOperand(opi, off);
                        touched = true;
                    }
                }
                if (CB->getFunctionType() != callee->getFunctionType()) {
                    CB->mutateFunctionType(callee->getFunctionType());
                    touched = true;
                }
                if (touched)
                    ++calls_fixed;
            }
        }
    }
    if (calls_fixed)
        out_log += ", " + std::to_string(calls_fixed) + " call sites settled";

    for (Instruction* I : dead) {
        if (I->getParent())
            I->eraseFromParent();
    }

    // Retuning the signatures leaves the old `llvm.used` array behind: its
    // entries still spell the previous function types, which the textual form
    // cannot express (a function is not a value).  Rebuilding it is cheap.
    if (llvm::GlobalVariable* U = M.getGlobalVariable("llvm.used")) {
        U->eraseFromParent();
        llvm::SmallVector<llvm::GlobalValue*, 32> pinned;
        for (Function& F : M)
            if (!F.isDeclaration())
                pinned.push_back(&F);
        if (!pinned.empty())
            llvm::appendToUsed(M, pinned);
    }

    // The pointers themselves are now dead weight: every GEP, phi and select
    // that produced one has been replaced by its offset expression.  A simple
    // "no users" sweep cannot retire them, because a pointer phi and the
    // pointer select that feeds its back edge can keep each other alive, so
    // liveness is computed backwards from the instructions that have effects.
    for (Function& F : M) {
        if (F.isDeclaration())
            continue;
        std::set<Instruction*> live;
        SmallVector<Instruction*, 16> stack;
        for (BasicBlock& BB : F) {
            for (Instruction& I : BB) {
                if (I.mayHaveSideEffects() || I.isTerminator()) {
                    live.insert(&I);
                    stack.push_back(&I);
                }
            }
        }
        while (!stack.empty()) {
            Instruction* I = stack.pop_back_val();
            for (Value* Op : I->operands()) {
                if (Instruction* OI = dyn_cast<Instruction>(Op)) {
                    if (live.insert(OI).second)
                        stack.push_back(OI);
                }
            }
        }
        SmallVector<Instruction*, 64> drop;
        for (BasicBlock& BB : F)
            for (Instruction& I : BB)
                if (!live.count(&I))
                    drop.push_back(&I);
        for (Instruction* I : drop) {
            if (I->getParent())
                I->eraseFromParent();
        }
    }

    out_log += "\n  arena ABI: " + std::to_string(sig_changed)
               + " signatures, " + std::to_string(loads) + " loads, "
               + std::to_string(stores) + " stores converted to indices";
    if (getenv("OSL_VULKAN_ARENA_DUMP"))
        M.print(llvm::errs(), nullptr);
    return true;
}

namespace {

// The handful of SPIR-V opcodes and capability numbers this file needs.
enum : uint32_t {
    kOpCapability            = 17,
    kOpConstant              = 43,
    kOpConstantNull          = 46,
    kOpAccessChain           = 65,
    kOpInBoundsAccessChain   = 66,
    kOpPtrAccessChain        = 67,
    kOpInBoundsPtrAccessChain = 70,
    kCapabilityAddresses     = 4,
};

/// Rewrite `Op*PtrAccessChain` into `Op*AccessChain` when the Element operand
/// is the constant zero, and drop the now-unused `Addresses` capability.
///
/// Vulkan forbids `Op*PtrAccessChain` (it requires the `Addresses` capability),
/// yet the LLVM backend lowers every GEP to one -- including the plain
/// `getelementptr %Arena, @osl_arena, i64 0, i32 0, i32 %word` shape.  With a
/// zero Element the two instructions mean the same thing: the Element is the
/// offset in elements of the base pointer's pointee type, and zero of them
/// leaves the indices to do all the addressing, so dropping the operand is
/// lossless.  An access that carries a non-zero Element (byte addressing in
/// practice) is left alone and shows up as a surviving `Addresses` capability,
/// which is the signal that the IR feeding us still uses byte offsets.
bool
rewrite_ptr_access_chains(std::vector<uint32_t>& words, std::string& log)
{
    // Walk once to learn which ids are the constant zero, so the second walk
    // can tell a rewritable access from a byte-addressed one.
    std::set<uint32_t> zero_ids;
    for (size_t i = 5; i < words.size();) {
        uint32_t wc  = words[i] >> 16;
        uint32_t op  = words[i] & 0xffffu;
        if (wc == 0)
            break;
        if (op == kOpConstant && wc >= 3 && i + wc <= words.size()) {
            bool all_zero = true;
            for (size_t k = 3; k < wc; ++k)
                all_zero = all_zero && words[i + k] == 0;
            if (all_zero)
                zero_ids.insert(words[i + 2]);
        } else if (op == kOpConstantNull && wc >= 3) {
            zero_ids.insert(words[i + 2]);
        }
        i += wc;
    }

    std::vector<uint32_t> out;
    out.reserve(words.size() + 8);
    out.insert(out.end(), words.begin(), words.begin() + 5);

    size_t rewritten = 0;
    size_t untouched = 0;
    size_t total     = 0;
    for (size_t i = 5; i < words.size();) {
        uint32_t wc  = words[i] >> 16;
        uint32_t op  = words[i] & 0xffffu;
        if (wc == 0)
            break;
        size_t end   = i + wc;
        bool is_ptr_chain
            = (op == kOpPtrAccessChain || op == kOpInBoundsPtrAccessChain);
        bool rewrite = false;
        if (is_ptr_chain) {
            ++total;
            // [wc|op, result type, result id, base, element, indices...]
            if (wc >= 5 && zero_ids.count(words[i + 4])) {
                out.push_back((wc - 1) << 16
                              | (op == kOpPtrAccessChain ? kOpAccessChain
                                                         : kOpInBoundsAccessChain));
                out.push_back(words[i + 1]);  // result type
                out.push_back(words[i + 2]);  // result id
                out.push_back(words[i + 3]);  // base
                for (size_t k = 5; k < wc; ++k)
                    out.push_back(words[i + k]);  // indices, element dropped
                rewrite = true;
                ++rewritten;
            } else {
                ++untouched;
            }
        }
        if (!rewrite) {
            for (size_t k = i; k < end; ++k)
                out.push_back(words[k]);
        }
        i = end;
    }

    // Only now, with a fresh count of surviving pointer accesses, can the
    // Addresses capability go.
    if (untouched == 0) {
        std::vector<uint32_t> out2;
        out2.reserve(out.size());
        out2.insert(out2.end(), out.begin(), out.begin() + 5);  // header
        for (size_t i = 5; i < out.size();) {
            uint32_t wc = out[i] >> 16;
            uint32_t op = out[i] & 0xffffu;
            if (wc == 0)
                break;
            if (op == kOpCapability && wc == 2 && out[i + 1] == kCapabilityAddresses) {
                i += 2;
                continue;
            }
            for (size_t k = i; k < i + wc; ++k)
                out2.push_back(out[k]);
            i += wc;
        }
        out.swap(out2);
    }

    log += "ptr access chains: " + std::to_string(rewritten) + "/"
           + std::to_string(total) + " rewritten to access chains";
    if (untouched)
        log += ", " + std::to_string(untouched) + " byte-addressed left alone";

    words.swap(out);
    return true;
}

/// Put the `Block` decoration where Vulkan wants it.
///
/// A storage buffer variable must be typed as a struct, and that *struct type*
/// -- not the variable -- carries `Block`.  LLVM attaches decorations only to
/// the global, so the struct the backend synthesises ends up undecorated and
/// the module fails validation with "StorageBuffer id N is missing Block
/// decoration".  This walks the type section, finds every struct used by a
/// StorageBuffer variable, and inserts the decoration the backend left out.
bool
add_block_decorations(std::vector<uint32_t>& words, std::string& log)
{
    enum : uint32_t {
        kOpTypePointer = 32,
        kOpVariable    = 59,
        kOpDecorate    = 71,
        kStorageBuffer = 12,
        kDecorationBlock = 2,
    };

    std::set<uint32_t> ssbo_structs;
    std::map<uint32_t, uint32_t> pointee_of;
    std::set<uint32_t> decorated;
    size_t types_start = 0;
    for (size_t i = 5; i < words.size();) {
        uint32_t wc = words[i] >> 16;
        uint32_t op = words[i] & 0xffffu;
        if (wc == 0)
            break;
        if (op == kOpDecorate && wc >= 3 && words[i + 2] == kDecorationBlock)
            decorated.insert(words[i + 1]);
        if (op == kOpTypePointer && wc == 4)
            pointee_of[words[i + 1]] = words[i + 3];
        if (op == kOpVariable && wc == 4 && words[i + 3] == kStorageBuffer) {
            auto it = pointee_of.find(words[i + 1]);
            if (it != pointee_of.end())
                ssbo_structs.insert(it->second);
        }
        // Annotations must precede the type declarations, so insert right before
        // the first instruction that belongs to the type section.
        if (types_start == 0 && op >= 19 && op <= 39)
            types_start = i;
        i += wc;
    }
    if (types_start == 0)
        return true;

    unsigned added = 0;
    for (uint32_t st : ssbo_structs) {
        if (decorated.count(st))
            continue;
        // OpDecorate <struct> Block
        words.insert(words.begin() + types_start, kDecorationBlock);
        words.insert(words.begin() + types_start, st);
        words.insert(words.begin() + types_start, (3u << 16) | kOpDecorate);
        added += 3;
    }
    if (added)
        log += ", " + std::to_string(added / 3) + " Block decorations added";
    return true;
}

}  // namespace

bool
vulkan_compile_ir_via_llvm_vulkan(
    const std::string& ir_text,
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
    // Both must happen before codegen: the capability set is decided while
    // the module is being lowered, not by anything downstream.  The null casts
    // go first so the arena rewrite has one shape to deal with.
    fold_null_addrspacecasts(*M);
    type_arena_accesses(*M, out_log);
    expand_memory_intrinsics(*M, out_log);
    drop_exported_symbols(*M);

    // Opt-in: re-express the arena in offsets instead of pointers.  The device
    // runtime still hands out pointers, so this is only exercised by the probe
    // tool until the runtime and the render kernel speak offsets as well.
    if (getenv("OSL_VULKAN_ARENA_INDEX")) {
        llvm::legacy::PassManager inline_pm;
        inline_pm.add(llvm::createAlwaysInlinerLegacyPass());
        inline_pm.run(*M);
        if (!arena_index_accesses(*M, out_log)) {
            out_log += "\n  arena index ABI failed";
            return false;
        }
        // Retyped functions do not carry their new parameter types into the
        // backend when the type is mutated in place, so the module is
        // laundered through its textual form before codegen.  The arena views
        // come back without an initializer, which is what a storage buffer
        // variable needs anyway.
        std::string txt;
        {
            llvm::raw_string_ostream ros(txt);
            M->print(ros, nullptr);
            ros.flush();
        }
        size_t pos = 0;
        while ((pos = txt.find("global { [65536 x ", pos)) != std::string::npos) {
            size_t close = txt.find('}', pos);
            txt.insert(close + 1, " zeroinitializer");
            pos = close + 1;
        }
        // A function whose signature was retyped prints inside an old cast
        // expression as "ptr bitcast (void (...) @f to ptr)", which the parser
        // rejects with "functions are not values".  With opaque pointers the
        // cast is redundant, so it is spelled away.
        {
            std::string fixed;
            size_t cur = 0, bpos = 0;
            while ((bpos = txt.find("ptr bitcast (", cur)) != std::string::npos) {
                size_t at  = txt.find(" @", bpos);
                size_t end = txt.find(" to ptr)", at == std::string::npos ? bpos : at);
                if (at == std::string::npos || end == std::string::npos)
                    break;
                fixed += txt.substr(cur, bpos - cur);
                fixed += "ptr" + txt.substr(at, end - at);
                cur = end + 8;
            }
            fixed += txt.substr(cur);
            txt.swap(fixed);
        }
        if (const char* dump_path = getenv("OSL_VULKAN_ARENA_DUMP")) {
            std::error_code ec;
            llvm::raw_fd_ostream out(dump_path, ec);
            out << txt;
        }
        llvm::SMDiagnostic diag2;
        auto buf2 = llvm::MemoryBuffer::getMemBuffer(txt, "after-arena.ll",
                                                    /*RequiresNullTerminator=*/false);
        std::unique_ptr<llvm::Module> M2 = llvm::parseIR(*buf2, diag2, ctx);
        if (!M2) {
            out_log += "\n  arena ABI: re-parse failed: "
                       + diag2.getMessage().str();
            return false;
        }
        for (llvm::GlobalVariable& G : M2->globals()) {
            if (G.getName().starts_with("osl_arena_"))
                G.setInitializer(nullptr);
        }
        M = std::move(M2);
    }

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
    // The arena accessors are `alwaysinline` on purpose: only once they are
    // gone does every arena read turn into a GEP rooted at the single arena
    // global, which is the shape this backend can turn into an access chain.
    pm.add(llvm::createAlwaysInlinerLegacyPass());
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
    rewrite_ptr_access_chains(words, out_log);
    add_block_decorations(words, out_log);
    out_spirv = std::move(words);
    return true;
}

}  // namespace pvt

OSL_NAMESPACE_END



