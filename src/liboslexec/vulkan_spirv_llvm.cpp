// Copyright Contributors to the Open Shading Language project.
// SPDX-License-Identifier: BSD-3-Clause
// https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

#include "vulkan_spirv_llvm.h"

#ifdef _WIN32
#    include <windows.h>
#endif

#include <llvm/ADT/SmallVector.h>
#include <llvm/IR/Constants.h>
#include <llvm/IR/DataLayout.h>
#include <llvm/IR/DerivedTypes.h>
#include <llvm/IR/NoFolder.h>
#include <llvm/IR/Function.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/Instructions.h>
#include <llvm/Transforms/IPO/AlwaysInliner.h>
#include <llvm/Transforms/InstCombine/InstCombine.h>
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
#include <llvm/IR/Verifier.h>
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

        // An access that already goes through one of the arena views is left
        // alone: re-typing it to i32 leaves a byte-scaled chain on a view whose
        // element is 64-bit, and the backend then emits OpPtrAccessChain over
        // uchar -- which needs the Addresses capability.  arena_index_accesses
        // turns those into word-indexed OpAccessChains instead.
        Value* base = GEP->getPointerOperand()->stripPointerCasts();
        if (GlobalVariable* GV = dyn_cast<GlobalVariable>(base))
            if (GV->getName().starts_with("osl_arena_"))
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
    // If the device runtime already declared this view, keep its *type*.  The
    // runtime names its structure ("struct.ArenaViewI64") while a literal
    // struct built here would be a different type, and LLVM treats named and
    // literal structures as unequal -- the two sides would no longer be talking
    // about the same global.  Reusing the declaration's type is what keeps the
    // two in sync.
    Type* view_ty = nullptr;
    if (GlobalVariable* Existing = M.getGlobalVariable(name)) {
        view_ty = Existing->getValueType();
    } else {
        ArrayType* AT = ArrayType::get(elem, kArenaViewWords);
        SmallVector<Type*, 1> members;
        members.push_back(AT);
        view_ty = StructType::get(C, members);
    }
    // No Block decoration: Vulkan wants the variable typed as a struct, and
    // the backend decides where Block goes for that struct.
    GlobalVariable* GV = M.getGlobalVariable(name);
    if (GV) {
        // The runtime's declaration is the one the accessors already point at;
        // keep the object (and its type) and only decorate it here.  It must
        // become a *definition*: a leftover external declaration makes the back
        // end add `OpCapability Linkage`, which Vulkan rejects outright.  The
        // zero initializer is stripped again right after the round trip.
        GV->setInitializer(Constant::getNullValue(GV->getValueType()));
        GV->setLinkage(GlobalValue::InternalLinkage);
    } else {
        GV = new GlobalVariable(
            M, view_ty, /*isConstant=*/false, GlobalValue::InternalLinkage,
            /*Initializer=*/nullptr, name,
            /*insertBefore=*/nullptr, GlobalVariable::NotThreadLocal,
            /*AddressSpace=*/11);
    }

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

/// Turn constant-offset GEPs on the arena views into instructions.
///
/// A `getelementptr` with all-constant indices folds into a ConstantExpr, which
/// is not an instruction: the rewrite below walks instructions only, so such an
/// access survives untouched and the backend lowers it straight to
/// `OpPtrAccessChain` with a byte-scaled element -- which needs `Addresses` and
/// drags `Int8` in with it.  Materialising it first puts constant and dynamic
/// offsets on the same path.
static bool
materialize_const_geps(llvm::Module& M)
{
    using namespace llvm;
    std::vector<std::pair<Instruction*, unsigned>> uses;
    for (Function& F : M) {
        if (F.isDeclaration())
            continue;
        for (BasicBlock& BB : F) {
            for (Instruction& I : BB) {
                for (unsigned i = 0; i < I.getNumOperands(); ++i) {
                    ConstantExpr* CE = dyn_cast<ConstantExpr>(I.getOperand(i));
                    if (!CE || CE->getOpcode() != Instruction::GetElementPtr)
                        continue;
                    Value* Base = CE->getOperand(0)->stripPointerCasts();
                    GlobalVariable* GV = dyn_cast<GlobalVariable>(Base);
                    if (!GV || !GV->getName().starts_with("osl_arena_"))
                        continue;
                    uses.push_back({ &I, i });
                }
            }
        }
    }
    for (auto& u : uses) {
        ConstantExpr* CE = cast<ConstantExpr>(u.first->getOperand(u.second));
        IRBuilder<> B(u.first);
        // The source element type comes from the GEP, not from its result: the
        // result is a pointer type, and using that as the element type produces
        // a chain the backend can only express as a byte-scaled
        // OpPtrAccessChain.
        GEPOperator* GO  = cast<GEPOperator>(CE);
        std::vector<Value*> idx;
        for (unsigned i = 1; i < GO->getNumOperands(); ++i)
            idx.push_back(GO->getOperand(i));
        Value* GEP = B.CreateGEP(GO->getSourceElementType(), GO->getPointerOperand(),
                                 idx, "arena.const.gep");
        u.first->setOperand(u.second, GEP);
    }
    return !uses.empty();
}

}  // namespace

namespace {
// Crash probe: the rewrite walks IR by hand, and an access violation gives no
// stack without a debugger.  A step number published here is the last thing the
// filter can print, which localises the fault to a single statement.
volatile long g_arena_step = -1;
LONG CALLBACK
arena_crash_filter(EXCEPTION_POINTERS*)
{
    fprintf(stderr, "[arena] access violation at step %ld\n", g_arena_step);
    return EXCEPTION_CONTINUE_SEARCH;
}
}  // namespace

bool
arena_index_accesses(llvm::Module& M, std::string& out_log)
{
    using namespace llvm;
    SetUnhandledExceptionFilter(arena_crash_filter);
    // The diagnostics below only matter when the process dies: a buffered stderr
    // loses the last few writes exactly when they matter most, so it is made
    // unbuffered for the duration of this rewrite.
    setvbuf(stderr, nullptr, _IONBF, 0);

    // An access violation says nothing about where the IR went wrong.  The
    // verifier does, so every phase ends with a check that names the offending
    // instruction instead of leaving a bare address to chase.
    auto check_module = [&](const char* stage) {
        std::string rep;
        llvm::raw_string_ostream os(rep);
        bool bad = llvm::verifyModule(M, &os);
        fprintf(stderr, "  verify[%s]: %s\n", stage,
                bad ? os.str().c_str() : "ok");
    };
    LLVMContext& C   = M.getContext();
    Type* i32        = Type::getInt32Ty(C);
    Type* i64        = Type::getInt64Ty(C);
    Type* f32        = Type::getFloatTy(C);
    const DataLayout DL = M.getDataLayout();

    GlobalVariable* v32 = make_arena_view(M, i32, 0, 4, "osl_arena_i32");
    GlobalVariable* vf  = make_arena_view(M, f32, 1, 4, "osl_arena_f32");
    GlobalVariable* v64 = make_arena_view(M, i64, 2, 8, "osl_arena_i64");

    if (materialize_const_geps(M))
        out_log += ", constant view GEPs materialised";

    // ---- 1. every pointer parameter and return value becomes an offset ----
    fprintf(stderr, "arena phase 1: ---- 1. every pointer parameter and return value becomes an offset ----\n");
    unsigned sig_changed = 0;
    // Rewriting a signature in place is not something a Function supports:
    // mutateType changes the pointer type but leaves getFunctionType() alone, so
    // every argument stops matching its parameter and the module is invalid from
    // that point on.  The offset ABI therefore has to be emitted by codegen, and
    // this pass only reports what it found instead of mutating it.
    if (!getenv("OSL_ARENA_NO_SIGREWRITE"))
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
    fprintf(stderr, "arena phase 2: ---- 2. offsets for every pointer value ----\n");
    std::map<Value*, Value*> offsets;  // pointer value -> i64 byte offset

    auto lookup_off = [&](Value* P) -> Value* {
        auto it = offsets.find(P);
        return it == offsets.end() ? nullptr : it->second;
    };

    auto null_sentinel = [&]() { return ConstantInt::get(i64, -1); };

    // The constant part and the dynamic terms of a GEP's own displacement.
    auto gep_delta = [&](GEPOperator* G, std::vector<Value*>& dyn,
                         bool& ok) -> int64_t {
        ok          = true;
        int64_t cst = 0;
        Type* ty     = G->getSourceElementType();
        for (unsigned i = 0; i < G->getNumIndices(); ++i) {
            Value* idx = G->getOperand(1 + i);
            if (StructType* st = dyn_cast<StructType>(ty)) {
                // An opaque struct has no layout at all: DataLayout dereferences
                // a null StructLayout for it, which is an access violation rather
                // than a diagnostic.  A field index that is not a constant is not
                // a struct GEP either, so both cases are handled as "cannot
                // express", not as a crash.
                ConstantInt* ci = dyn_cast<ConstantInt>(idx);
                if (!ci || st->isOpaque()) {
                    ok = false;
                    return 0;
                }
                unsigned fld = (unsigned)ci->getZExtValue();
                // A field index past the end of the struct is not a field: the
                // layout lookup indexes its element table unchecked and reads
                // past it, which shows up as an access violation with no IR.
                if (fld >= st->getNumElements()) {
                    ok = false;
                    return 0;
                }
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
            // An unsized element (an opaque struct reached through an array or a
            // vector of them) makes the size query llvm_unreachable, which is
            // undefined behaviour in a release build rather than an error.
            if (!elem->isSized()) {
                ok = false;
                return 0;
            }
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
            // Only a parameter that phase 1 actually retyped is an offset.  One
            // that was left alone (a declaration that never got rewritten, or a
            // parameter the mutation did not reach) is still a pointer, and
            // returning it here would hand the builder a `ptr` to add an i64 to.
            result = P->getType()->isIntegerTy(64) ? P : nullptr;
        } else if (GEPOperator* G = dyn_cast<GEPOperator>(P)) {
            std::vector<Value*> dyn;
            bool ok      = true;
            fprintf(stderr, "  gep: delta start\n");
            int64_t cst  = gep_delta(G, dyn, ok);
            if (!ok) {
                out_log += "\n  cannot compute the displacement of a GEP";
                return nullptr;
            }
            // A constant-offset GEP folds into a ConstantExpr, which is not an
            // instruction: inserting the offset arithmetic "at" it is undefined.
            // Such a value only survives when it has a single user, so the
            // instruction reading or writing through it is the anchor.
            Instruction* anchor = dyn_cast<Instruction>(G);
            if (!anchor) {
                // Exactly one use is what makes the reader or writer usable as
                // the insertion point.  Zero uses leaves this a dead constant,
                // and dereferencing user_begin() then walks off the end of the
                // list instead of finding an instruction.
                if (G->getNumUses() != 1) {
                    out_log += G->getNumUses() ? "\n  constant GEP with several users"
                                               : "\n  constant GEP with no users";
                    return nullptr;
                }
                anchor = dyn_cast<Instruction>(*G->user_begin());
                if (!anchor) {
                    out_log += "\n  constant GEP without an instruction user";
                    return nullptr;
                }
            }
            IRBuilder<> B(anchor);
            Value* base = offset_of(G->getPointerOperand());
            fprintf(stderr, "  gep: recursing base\n");
            if (!base) {
                // The recursion can decline to express the base; handing that
                // nullptr to the builder dereferences it, so this has to be a
                // refusal here rather than a crash further down.
                out_log += "\n  cannot express the base of a GEP";
                return nullptr;
            }
            fprintf(stderr, "  off_of gep: recursing to base id=%d cst=%lld\n", (int)G->getPointerOperand()->getValueID(), (long long)cst);
            Value* acc  = B.CreateAdd(base, ConstantInt::get(i64, cst), "arena.off");
            fprintf(stderr, "  gep: added\n");
            for (size_t i = 0; i < dyn.size(); i += 2) {
                Value* idx = B.CreateSExtOrTrunc(dyn[i], i64);
                Value* term = B.CreateMul(idx, dyn[i + 1], "arena.term");
                acc         = B.CreateAdd(acc, term, "arena.off");
            }
            result = acc;
        } else if (PHINode* PH = dyn_cast<PHINode>(P)) {
            result = lookup_off(PH);  // pre-created in step 3
        } else if (SelectInst* SI = dyn_cast<SelectInst>(P)) {
            // Both arms have to exist before a select can be built: passing a
            // null through CreateSelect produces a malformed instruction, not a
            // diagnostic.
            Value* tv = offset_of(SI->getTrueValue());
            Value* fv = tv ? offset_of(SI->getFalseValue()) : nullptr;
            if (!tv || !fv) {
                out_log += "\n  a select arm has no offset";
                return nullptr;
            }
            IRBuilder<> B(SI);
            result = B.CreateSelect(SI->getCondition(), tv, fv, "arena.sel");
        } else if (LoadInst* LI = dyn_cast<LoadInst>(P)) {
            result = lookup_off(LI);  // rewritten in step 3
        } else if (isa<BitCastInst>(P)
                   || (isa<ConstantExpr>(P) && cast<ConstantExpr>(P)->isCast()
                       && cast<ConstantExpr>(P)->getOpcode()
                              != Instruction::AddrSpaceCast)
                   || (isa<AddrSpaceCastInst>(P)
                       && (cast<AddrSpaceCastInst>(P)->getSrcAddressSpace() == 1
                           || cast<AddrSpaceCastInst>(P)->getSrcAddressSpace() == 11))) {
            // A bitcast never moves a pointer inside the arena.  An address space
            // cast is only offset-preserving when it starts in the arena itself:
            // a cast out of private memory (an alloca) has no arena offset at
            // all, and following it would invent one for a stack slot.
            result = offset_of(cast<Operator>(P)->getOperand(0));
        } else if (isa<Constant>(P) && P->getType()->isPointerTy()) {
            // Any other pointer constant (undef, poison, an aggregate zero) is
            // not a place in the arena; it carries the null sentinel so that
            // whatever loads it can never look like a valid offset.
            result = null_sentinel();
        }
        // Every offset handed back has to be an i64.  A pointer here would make
        // the arithmetic below a `ptr + i64`, which does not assert in a release
        // build: it builds invalid IR and the fault surfaces much later, with no
        // trace of where it came from.  Refusing is the honest outcome.
        if (result && !result->getType()->isIntegerTy(64)) {
            out_log += "\n  offset is not an i64: "
                       + std::string(P->getName()) + " ("
                       + std::to_string(P->getValueID()) + ")";
            return nullptr;
        }
        if (!result) {
            std::string what = "\n  cannot express an offset for: ";
            if (auto* I = dyn_cast<Instruction>(P))
                what += std::string("op=") + I->getOpcodeName() + " ";
            else if (auto* CE = dyn_cast<ConstantExpr>(P))
                what += std::string("constexpr=") + CE->getOpcodeName() + " ";
            else
                what += std::string("class=") + std::to_string(P->getValueID())
                        + " ";
            what += P->getName().str() + " ("
                    + std::to_string(P->getValueID()) + ")";
            out_log += what;
            return nullptr;
        }
        offsets[P] = result;
        return result;
    };

    // ---- 2a. stack slots are private memory, not arena memory ----
    //
    // The module's data layout says a bare `ptr` is address space 1, so every
    // alloca and every chain off it came out typed as an arena pointer.  A stack
    // slot has no arena offset at all: retyping the slot and the GEP/bitcast
    // chain that addresses it into address space 0 leaves it as the private
    // memory it really is, which the backend emits without any capability of
    // its own.
    unsigned privatized = 0;
    for (Function& F : M) {
        if (F.isDeclaration())
            continue;
        for (BasicBlock& BB : F) {
            for (Instruction& I : BB) {
                AllocaInst* A = dyn_cast<AllocaInst>(&I);
                Type* pt       = A ? A->getType() : nullptr;
                if (!A || !pt->isPointerTy() || pt->getPointerAddressSpace() == 0)
                    continue;
                A->mutateType(PointerType::get(C, 0));
                std::vector<Instruction*> work;
                for (User* U : A->users())
                    if (Instruction* UI = dyn_cast<Instruction>(U))
                        work.push_back(UI);
                std::set<Instruction*> seen;
                while (!work.empty()) {
                    Instruction* W = work.back();
                    work.pop_back();
                    if (!seen.insert(W).second)
                        continue;
                    if (GetElementPtrInst* G = dyn_cast<GetElementPtrInst>(W)) {
                        G->mutateType(PointerType::get(C, 0));
                    } else if (BitCastInst* B = dyn_cast<BitCastInst>(W)) {
                        B->mutateType(PointerType::get(C, 0));
                    } else {
                        continue;  // loads and stores need no retype of their own
                    }
                    for (User* U : W->users())
                        if (Instruction* UI = dyn_cast<Instruction>(U))
                            work.push_back(UI);
                }
                ++privatized;
            }
        }
    }
    if (privatized)
        out_log += ", stack slots privatised: " + std::to_string(privatized);

    check_module("2a privatised");
    // ---- 2b. private spill slots that hold arena pointers ----
    //
    // The inlined group IR spills an arena pointer into an `alloca ptr` and
    // reloads it later.  The slot itself is private memory, which needs no arena
    // and no capability, but its payload is a pointer that no longer exists in
    // the offset ABI.  Such a slot is retyped to i64 and carries the offset; a
    // slot that sees any other use is left alone, and the load that reads it
    // will then refuse to produce an offset rather than be rewritten wrongly.
    std::set<Value*> spill_slots;
    for (Function& F : M) {
        if (F.isDeclaration())
            continue;
        for (BasicBlock& BB : F) {
            for (Instruction& I : BB) {
                AllocaInst* A = dyn_cast<AllocaInst>(&I);
                if (!A || !A->getAllocatedType()->isPointerTy())
                    continue;
                bool usable = true, used = false;
                for (User* U : A->users()) {
                    if (StoreInst* SU = dyn_cast<StoreInst>(U)) {
                        if (SU->getPointerOperand()->stripPointerCasts() != A) {
                            usable = false;
                            break;
                        }
                        Type* vt = SU->getValueOperand()->getType();
                        unsigned as
                            = vt->isPointerTy() ? vt->getPointerAddressSpace() : 0;
                        if (!vt->isPointerTy() || (as != 1 && as != 11)) {
                            usable = false;
                            break;
                        }
                        used = true;
                    } else if (LoadInst* LU = dyn_cast<LoadInst>(U)) {
                        if (LU->getPointerOperand()->stripPointerCasts() != A) {
                            usable = false;
                            break;
                        }
                        used = true;
                    } else {
                        usable = false;
                        break;
                    }
                }
                if (!usable || !used)
                    continue;
                A->setAllocatedType(i64);
                A->mutateType(PointerType::get(C, A->getAddressSpace()));
                spill_slots.insert(A);
            }
        }
    }
    if (!spill_slots.empty())
        out_log += ", spill slots retyped: "
                   + std::to_string(spill_slots.size());

    // The module's data layout defaults a bare `ptr` to address space 1, so a
    // stack slot -- `%9 = alloca ptr` -- is *typed* as an arena pointer even
    // though it lives in private memory.  Such a chain is private and is left
    // alone: it has no arena offset, and inventing one for it would be wrong.
    auto rooted_in_alloca = [](Value* P) -> bool {
        if (getenv("OSL_ARENA_NO_ROOTCHECK"))
            return false;
        Value* V = P;
        for (unsigned guard = 0; guard < 64; ++guard) {
            V = V->stripPointerCasts();
            if (isa<AllocaInst>(V))
                return true;
            if (GEPOperator* G = dyn_cast<GEPOperator>(V)) {
                V = G->getPointerOperand();
                continue;
            }
            return false;
        }
        return false;
    };

    check_module("2b spill slots");
    // ---- 3. rewrite phis, loads, stores and the remaining pointer operands ----
    fprintf(stderr, "arena phase 3: ---- 3. rewrite phis, loads, stores and the remaining pointer operands ----\n");
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
    //
    // The pairs are collected first: offset_of() adds entries to `offsets` for
    // values it has not seen yet (constants, directive results), and inserting
    // into a std::map while iterating it invalidates the iterator this loop
    // would otherwise be sitting on -- which is exactly the kind of crash that
    // shows up as a bare access violation with no IR to look at.
    std::vector<std::pair<PHINode*, PHINode*>> pending_phis;
    for (auto& kv : offsets) {
        PHINode* OP = dyn_cast<PHINode>(kv.first);
        PHINode* NP = dyn_cast<PHINode>(kv.second);
        if (OP && NP)
            pending_phis.push_back({ OP, NP });
    }
    for (auto& pr : pending_phis) {
        PHINode* OP = pr.first;
        PHINode* NP = pr.second;
        for (unsigned i = 0; i < OP->getNumIncomingValues(); ++i) {
            Value* off = offset_of(OP->getIncomingValue(i));
            if (!off)
                return false;
            NP->addIncoming(off, OP->getIncomingBlock(i));
        }
    }

    /// Build a word-indexed GEP on a view *as an instruction*.
    ///
    /// IRBuilder constant-folds a GEP whose indices are all constant into a
    /// ConstantExpr, and LLVM canonicalises that into byte form --
    /// `getelementptr (i8, ptr @view, i64 N)` -- which the backend can only
    /// express as OpPtrAccessChain over uchar, needing the Addresses capability
    /// (and Int8).  Creating the instruction directly keeps the struct-indexed
    /// shape the backend turns into OpAccessChain.
    auto view_gep = [&](IRBuilder<>& B, GlobalVariable* view, Value* idx,
                        const char* nm) -> Value* {
        // NoFolder keeps the builder from constant-folding the GEP into a
        // ConstantExpr, which is what turns into a byte-addressed chain.
        llvm::NoFolder NF;
        IRBuilder<llvm::NoFolder> NB(&*B.GetInsertPoint());
        SmallVector<Value*, 3> idxs;
        idxs.push_back(ConstantInt::get(i64, 0));
        idxs.push_back(ConstantInt::get(i32, 0));
        idxs.push_back(idx);
        return NB.CreateGEP(view->getValueType(), view, idxs, nm);
    };

    // The 32-bit view's pointer for the word that holds byte offset `off`.
    auto word_ptr = [&](IRBuilder<>& B, Value* off) -> Value* {
        Value* word = B.CreateLShr(off, ConstantInt::get(i64, 2), "arena.word");
        Value* idx  = B.CreateTrunc(word, i32, "arena.idx");
        return view_gep(B, v32, idx, "arena.ptr");
    };

    std::vector<Instruction*> dead;
    unsigned loads = 0, stores = 0;

    /// Replace a constant-offset GEP operand with a real instruction in front of
    /// `At`.  A constant GEP folds into a ConstantExpr, and the backend lowers a
    /// ConstantExpr GEP to OpPtrAccessChain with a byte-scaled element (which
    /// needs Addresses and pulls Int8 in).  The pointer rewrite below only looks
    /// at instructions, so the operand is materialised before it is asked for an
    /// offset.
    auto materialize_operand_gep = [](Instruction* At, unsigned OpNo) {
        ConstantExpr* CE = dyn_cast<ConstantExpr>(At->getOperand(OpNo));
        if (!CE || CE->getOpcode() != Instruction::GetElementPtr)
            return;
        IRBuilder<> B(At);
        std::vector<Value*> idx;
        for (unsigned i = 1; i < CE->getNumOperands(); ++i)
            idx.push_back(CE->getOperand(i));
        At->setOperand(OpNo,
                       B.CreateGEP(cast<GEPOperator>(CE)->getSourceElementType(),
                                   CE->getOperand(0), idx, "arena.const.gep"));
    };
    fprintf(stderr, "  p3 sub-loop at line 771\n");
    for (Function& F : M) {
        if (F.isDeclaration())
            continue;
        for (BasicBlock& BB : F) {
            for (Instruction& II : BB) {
                Instruction* I = &II;
                if (!I->getParent())
                    continue;
                fprintf(stderr, "  instr %s @ %s\n", I->getOpcodeName(), F.getName().str().c_str());
                if (isa<PHINode>(I) || isa<GetElementPtrInst>(I))
                    continue;

                if (LoadInst* LI = dyn_cast<LoadInst>(I)) {
                    if (LI->getPointerAddressSpace() == 0
                        && LI->getType()->isPointerTy()
                        && (LI->getType()->getPointerAddressSpace() == 1
                            || LI->getType()->getPointerAddressSpace() == 11)) {
                        // A private slot that holds an arena pointer now holds
                        // its offset, so reading the slot *produces* an offset.
                        LI->mutateType(i64);
                        offsets[LI] = LI;
                        continue;
                    }
                    if (spill_slots.count(
                            LI->getPointerOperand()->stripPointerCasts())) {
                        // The slot now holds an i64 offset; reading it *is* the
                        // offset, so the load stays where it is and only its
                        // result type changes.
                        LI->mutateType(i64);
                        offsets[LI] = LI;
                        continue;
                    }
                    if (LI->getPointerAddressSpace() != 1
                        && LI->getPointerAddressSpace() != 11)
                        continue;
                    if (rooted_in_alloca(LI->getPointerOperand())) {
                        fprintf(stderr, "  load: private, left alone\n");
                        continue;
                    }
                    materialize_operand_gep(LI, 0);
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
                    fprintf(stderr, "  store: vt id=%d bits=%u\n", (int)vt->getTypeID(), (unsigned)(vt->isIntegerTy() ? vt->getIntegerBitWidth() : 0));
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
                    Value* p    = view_gep(B, view, idx, "arena.ptr");
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
                    out_log += "\n  store seen: as="
                               + std::to_string(SI->getPointerAddressSpace())
                               + " val=" + std::to_string(SI->getValueOperand()
                                                              ->getType()
                                                              ->getTypeID());
                    if (SI->getPointerAddressSpace() != 1
                        && SI->getPointerAddressSpace() != 11) {
                        Type* vtp = SI->getValueOperand()->getType();
                        if (vtp->isPointerTy()
                            && (vtp->getPointerAddressSpace() == 1
                                || vtp->getPointerAddressSpace() == 11)) {
                            // Storing an arena pointer into a private slot stores
                            // its offset; private pointers get the null sentinel.
                            Value* off = offset_of(SI->getValueOperand());
                            if (!off)
                                off = null_sentinel();
                            SI->setOperand(0, off);
                            ++stores;
                            continue;
                        }
                        Value* slot
                            = SI->getPointerOperand()->stripPointerCasts();
                        if (spill_slots.count(slot)) {
                            // Storing the arena pointer becomes storing its
                            // offset into the retyped slot.
                            Value* off = offset_of(SI->getValueOperand());
                            if (!off)
                                return false;
                            SI->setOperand(0, off);
                            ++stores;
                        }
                        continue;
                    }
                    g_arena_step = 200;
                    materialize_operand_gep(SI, 0);
                    g_arena_step = 202;
                    fprintf(stderr, "  store: as=%d\n", SI->getPointerAddressSpace());
                    g_arena_step = 204;
                    if (rooted_in_alloca(SI->getPointerOperand())) {
                    fprintf(stderr, "  store: root check\n");
                        fprintf(stderr, "  store: private, left alone\n");
                        continue;
                    }
                    g_arena_step = 206;
                    {
                        Value* PO = SI->getPointerOperand();
                        fprintf(stderr,
                                "  store: operand op=%s tyid=%d as=%d\n",
                                isa<Instruction>(PO)
                                    ? cast<Instruction>(PO)->getOpcodeName()
                                    : "not-instruction",
                                (int)PO->getType()->getTypeID(),
                                PO->getType()->isPointerTy()
                                    ? (int)PO->getType()->getPointerAddressSpace()
                                    : -1);
                    }
                    Value* off = offset_of(SI->getPointerOperand());
                    fprintf(stderr, "  store: off computed\n");
                    if (!off)
                        return false;
                    g_arena_step = 210;
                    Value* val = SI->getValueOperand();
                    Type* vt   = val->getType();
                    if (vt->isPointerTy()) {
                    g_arena_step = 212;
                        val = offset_of(val);
                        if (!val) {
                            // A pointer into private memory (an alloca) has no
                            // arena offset: the device cannot address it.  It is
                            // stored as the arena's null sentinel, which is the
                            // same value a null pointer gets, so anything that
                            // later loads and dereferences it is invalid by
                            // construction rather than silently reading the
                            // wrong arena word.
                            val = null_sentinel();
                            fprintf(stderr,
                                    "  store: private pointer -> null\n");
                            out_log
                                += "\n  private pointer stored into the arena as"
                                   " the null sentinel";
                        }
                        vt = i64;
                    }
                    if (vt->isIntegerTy(8) || vt->isIntegerTy(16)) {
                    fprintf(stderr, "  store: vt id=%d bits=%u\n", (int)vt->getTypeID(), (unsigned)(vt->isIntegerTy() ? vt->getIntegerBitWidth() : 0));
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
                    Value* p    = view_gep(B, view, idx, "arena.ptr");
                    B.CreateStore(val, p, SI->isVolatile());
                    fprintf(stderr, "  store: viewed store\n");
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
    fprintf(stderr, "  p3 sub-loop at line 951\n");
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
    fprintf(stderr, "  p3: instruction loops done\n");
    check_module("3 instruction loops");
        out_log += ", " + std::to_string(calls_fixed) + " call sites settled";

    for (Instruction* I : dead) {
        if (I->getParent())
            I->eraseFromParent();
    }

    // Retuning the signatures leaves the old `llvm.used` array behind: its
    // entries still spell the previous function types, which the textual form
    // cannot express (a function is not a value).  Rebuilding it is cheap.
    if (llvm::GlobalVariable* U = M.getGlobalVariable("llvm.used")) {
    fprintf(stderr, "  p3: dead erased\n");
        U->eraseFromParent();
        llvm::SmallVector<llvm::GlobalValue*, 32> pinned;
    fprintf(stderr, "  p3 sub-loop at line 995\n");
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
    fprintf(stderr, "  p3 sub-loop at line 1007\n");
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

    fprintf(stderr, "  p3: liveness done\n");
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
                                  std::string& out_log, bool as_library)
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

    // Observation hook: the module as the passes first see it.  Constant GEPs
    // are folded by the parser, so what is in the text and what the passes can
    // see differ; this dump is how that difference gets measured.
    if (const char* dump_path = getenv("OSL_VULKAN_ARENA_DUMP")) {
        std::string txt;
        llvm::raw_string_ostream ros(txt);
        M->print(ros, nullptr);
        ros.flush();
        std::string p = std::string(dump_path) + ".pre";
        std::error_code ec;
        llvm::raw_fd_ostream out(p.c_str(), ec);
        if (ec)
            out_log += "\n  pre dump failed: " + ec.message();
        else
            out << txt;
    }

    out_log += "\n  emit path v2";
    std::string err;
    // Both must happen before codegen: the capability set is decided while
    // the module is being lowered, not by anything downstream.  The null casts
    // go first so the arena rewrite has one shape to deal with.
    fold_null_addrspacecasts(*M);
    type_arena_accesses(*M, out_log);
    expand_memory_intrinsics(*M, out_log);
    // Only an executable module drops its export decorations; a module meant to
    // be linked against must keep them so a later spirv-link can resolve the
    // other modules' imports.
    if (!as_library)
        drop_exported_symbols(*M);

    // Opt-in: re-express the arena in offsets instead of pointers.  The device
    // runtime still hands out pointers, so this is only exercised by the probe
    // tool until the runtime and the render kernel speak offsets as well.
    if (getenv("OSL_VULKAN_ARENA_INDEX")) {
        // The device runtime's entry points take storage-buffer pointers, and a
        // pointer parameter is what forces VariablePointersStorageBuffer into
        // the capability set.  Force them inline first: once their bodies are
        // folded into the caller there is no pointer crossing a function
        // boundary, and the capability has no reason to exist.
        for (const char* name :
             { "osl_vk_arena_alloc", "osl_allocate_closure_component",
               "osl_allocate_weighted_closure_component", "osl_mul_closure_color",
               "osl_mul_closure_float", "osl_normalize_vv",
               "osl_init_texture_options", "osl_texture" }) {
            if (llvm::Function* F = M->getFunction(name)) {
                if (!F->isDeclaration()) {
                    F->addFnAttr(llvm::Attribute::AlwaysInline);
                    out_log += "\n  forced inline: " + std::string(name);
                }
            }
        }
        llvm::legacy::PassManager inline_pm;
        inline_pm.add(llvm::createAlwaysInlinerLegacyPass());
        inline_pm.run(*M);

        // Diagnostic: which stage of the rewrite a crash belongs to.  Without
        // it the failure is a bare access violation with no IR to look at.
        auto dump_stage = [&](const char* suffix) {
            if (const char* p = getenv("OSL_VULKAN_DUMP_RETYPED")) {
                std::string txt;
                llvm::raw_string_ostream ros(txt);
                M->print(ros, nullptr);
                ros.flush();
                std::error_code ec;
                llvm::raw_fd_ostream out(
                    (std::string(p) + suffix).c_str(), ec);
                if (!ec)
                    out << txt;
            }
        };
        dump_stage(".inlined");
        // The arena views only exist in the material modules; the frame kernel
        // module reaches its data through its own frame buffer, so it has no
        // arena to rewrite and must not be failed for it.
        const bool has_arena = M->getNamedGlobal("osl_arena_i64") != nullptr
                               || M->getNamedGlobal("osl_arena_i32")
                                      != nullptr
                               || M->getNamedGlobal("osl_arena_f32")
                                      != nullptr;
        // The offset runtime only declares the arena view, so every material
        // module imports it and no module exports it -- spirv-link then stops
        // with "Unresolved external reference to osl_arena_i32".  One module has
        // to carry the definition, and it has to be a module that really uses
        // the view: a global with no users is dropped before codegen and the
        // export never reaches SPIR-V.  So the first module that references it
        // turns the declaration into a definition, and the rest keep importing.
        {
            static bool arena_view_defined = false;
            if (!arena_view_defined) {
                if (llvm::GlobalVariable* v
                    = M->getNamedGlobal("osl_arena_i32")) {
                    if (!v->use_empty()) {
                        v->setInitializer(
                            llvm::UndefValue::get(v->getValueType()));
                        v->setLinkage(llvm::GlobalValue::InternalLinkage);
                        arena_view_defined = true;
                        fprintf(stderr,
                                "[vulkan] arena view exported from this module\n");
                    }
                }
            }
        }
        // With the ABI emitted by codegen there is nothing left for this pass to
        // repair; the switch below runs the module without it so the two can be
        // compared directly.
        if (!getenv("OSL_ARENA_NO_REWRITE") && has_arena
            && !arena_index_accesses(*M, out_log)) {
            out_log += "\n  arena index ABI failed";
            return false;
        }
        dump_stage(".indexed");
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
        // The parser re-derives constants: a fully constant GEP comes back as a
        // ConstantExpr in byte form, which the backend can only express as
        // OpPtrAccessChain over uchar.  Materialise every use again, *after* the
        // round trip, so codegen sees instructions.
        if (materialize_const_geps(*M2))
            out_log += ", view GEPs re-materialised";
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
    // Inlining the arena accessors leaves chains like
    //   %a = gep i8, ptr %root, i64 320
    //   %b = gep i8, %a, i64 %field
    // and the backend has to express the second one as OpPtrAccessChain, which
    // needs the Addresses capability that Vulkan forbids.  InstCombine folds a
    // GEP of a GEP into a single GEP off the same root
    // (gep i8, ptr %root, i64 (320 + %field)), which is an ordinary access
    // chain.  Two rounds so that chains that only become adjacent after the
    // first round are folded too.
    for (int round = 0; round < 2; ++round) {
        llvm::legacy::PassManager ipm;
        ipm.add(llvm::createInstructionCombiningPass());
        ipm.run(*M);
    }
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



