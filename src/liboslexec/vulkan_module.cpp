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
#include <llvm/AsmParser/Parser.h>
#include <llvm/Bitcode/BitcodeReader.h>
#include <llvm/IR/IRBuilder.h>
#include <llvm/IR/LLVMContext.h>
#include <llvm/IR/Module.h>
#include <llvm/Transforms/Utils/ModuleUtils.h>
#include <llvm/Linker/Linker.h>
#include <llvm/Support/MemoryBuffer.h>
#include <llvm/Support/SourceMgr.h>
#include <llvm/Support/raw_ostream.h>

#include <OSL/oslconfig.h>

// The serialised device modules live in the global namespace (MAKE_EMBEDDED_CPP
// emits them outside any namespace), hence the explicit `::` at the use site.
extern unsigned char vulkan_render_llvm_compiled_ops_block[];
extern int vulkan_render_llvm_compiled_ops_size;
// Same kernel compiled with VK_GLOBAL = address_space(11) (StorageBuffer): the
// variant the LLVM SPIR-V backend can turn into a Vulkan-legal module, since it
// keeps no CrossWorkgroup pointers.
extern unsigned char vulkan_render_sb_llvm_compiled_ops_block[];
extern int vulkan_render_sb_llvm_compiled_ops_size;

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
// A compiled group can carry two shapes (see vulkan_add_kernel_entry): a
// non-trivial `osl_init_group_<group>` that runs the layers itself, or a set of
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
                                 std::vector<std::string>* out_group_ir,
                                 std::string* out_entry_ir)
{
    if (group_ir.size() != entries.size()) {
        out_log = "group IR and entry list sizes differ";
        return false;
    }

    // Split mode: the groups are compiled on their own (Vulkan flavor SPIR-V
    // backend, where the arena views are legal) and merged with the kernel
    // afterwards by spirv-link.  Linking them here would put the views into the
    // module handed to clspv, which rejects the extra descriptor binding.
    const bool split = (out_group_ir != nullptr);
    if (split)
        out_group_ir->clear();

    llvm::LLVMContext ctx;

    // The frame kernel: compiled at build time by the same clang that produces
    // the device runtime, so it already carries the spirv64 triple and layout.
    {
        // Split mode compiles this half with the LLVM Vulkan flavor backend, so
        // it needs the StorageBuffer variant of the kernel.
        const unsigned char* block
            = (split && ::vulkan_render_sb_llvm_compiled_ops_size > 0)
                  ? ::vulkan_render_sb_llvm_compiled_ops_block
                  : ::vulkan_render_llvm_compiled_ops_block;
        const int block_size
            = (split && ::vulkan_render_sb_llvm_compiled_ops_size > 0)
                  ? ::vulkan_render_sb_llvm_compiled_ops_size
                  : ::vulkan_render_llvm_compiled_ops_size;
        llvm::StringRef content(reinterpret_cast<const char*>(block),
                                (size_t)block_size);
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
                          + err.getMessage().str() + " (line "
                          + std::to_string(err.getLineNo()) + " col "
                          + std::to_string(err.getColumnNo()) + ")";
                // Handy when the parser complains: keep the exact text so the
                // reported line can be looked at instead of guessed at.
                if (const char* dump = getenv("OSL_VULKAN_DUMP_BAD_IR")) {
                    std::error_code ec;
                    llvm::raw_fd_ostream os(dump, ec);
                    if (!ec)
                        os << group_ir[i];
                }
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

            if (split) {
                // Hand this group's (prefixed) IR text to the caller untouched.
                llvm::raw_string_ostream gos(out_group_ir->emplace_back());
                g->print(gos, nullptr);
                gos.flush();
            } else if (llvm::Linker::linkModules(*render, std::move(g))) {
                out_log = "linking group IR #" + std::to_string(i)
                          + " failed (duplicate symbol)";
                return false;
            }
        }

        llvm::Type* ptr1 = llvm::PointerType::get(ctx, 1);
        // The StorageBuffer kernel variant reaches the frame buffer through
        // address_space(11) pointers, so its `osl_vk_shade_dispatch` declaration
        // (and every group entry) takes AS11 rather than CrossWorkgroup.
        llvm::Type* pbuf  = split ? llvm::PointerType::get(ctx, 11) : ptr1;
        llvm::Type* i32  = llvm::Type::getInt32Ty(ctx);
        llvm::Type* vd   = llvm::Type::getVoidTy(ctx);
        llvm::Value* null1
            = llvm::ConstantPointerNull::get(llvm::cast<llvm::PointerType>(pbuf));

        // ---- osl_vk_shade_dispatch(ptr AS1 sg, i32 material) ----
        llvm::FunctionType* dtype
            = llvm::FunctionType::get(vd, { pbuf, i32 }, false);
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
                // Offset ABI: the last slot carries the block's absolute
                // position in the storage buffer, which the frame kernel stored
                // at sg + kOffPool (320, see vulkan_shadeops.cpp).  Every
                // parameter the group touches is then reached as
                // sg_off + field offset, so no storage-buffer pointer crosses
                // a function boundary.
                llvm::Value* arg5 = null1;
                if (getenv("OSL_VULKAN_ARENA_INDEX"))
                    arg5 = cb.CreateLoad(
                        llvm::Type::getInt64Ty(ctx),
                        cb.CreateConstGEP1_32(llvm::Type::getInt8Ty(ctx),
                                              dispatch->getArg(0), 320u));
                for (auto& name : entries[i]) {
                    llvm::Function* F = render->getFunction(prefixes[i] + name);
                    if (!F) {
                        if (!split)
                            continue;
                        // Split mode: the definition lives in the group's own
                        // SPIR-V module; spirv-link resolves the call afterwards.
                        // The signature must match what the group entry really
                        // is, otherwise the link fails on a type mismatch.
                        llvm::FunctionType* fty = llvm::FunctionType::get(
                            vd,
                            { pbuf, pbuf, pbuf, pbuf, i32,
                              getenv("OSL_VULKAN_ARENA_INDEX")
                                  ? llvm::Type::getInt64Ty(ctx)
                                  : pbuf },
                            false);
                        F = llvm::Function::Create(
                            fty, llvm::GlobalValue::ExternalLinkage,
                            prefixes[i] + name, *render);
                    }
                    cb.CreateCall(F, { dispatch->getArg(0), gd, null1, null1,
                                       cb.getInt32(0), arg5 });
                }
                cb.CreateRetVoid();
            }
            llvm::BasicBlock* none = sw->getDefaultDest();
            llvm::IRBuilder<> nb(none);
            nb.CreateRetVoid();
        }

        // ---- the compute kernel ----
        //
        // A SPIR-V compute entry point cannot take a pointer parameter: with the
        // `Shader` capability the entry function must be void() with no
        // arguments, and the frame buffer has to reach it as a descriptor-backed
        // StorageBuffer variable instead (clspv did that translation itself,
        // which is why the CrossWorkgroup form only ever worked through clspv).
        llvm::Value* frame_base = nullptr;
        if (split && !out_entry_ir) {
            llvm::Type* bytes = llvm::ArrayType::get(llvm::Type::getInt8Ty(ctx),
                                                     (uint64_t)80 << 20);
            llvm::StructType* fbt
                = llvm::StructType::create(ctx, { bytes }, "struct.FrameBuf");
            // Declared, not defined: the host binds this descriptor, and spirv-link
            // resolves the declaration against the module that has the variable.
            llvm::GlobalVariable* gv = new llvm::GlobalVariable(
                *render, fbt, /*isConstant=*/false,
                llvm::GlobalValue::ExternalLinkage, /*init=*/nullptr,
                "osl_frame_buf", /*insertBefore=*/nullptr,
                llvm::GlobalVariable::NotThreadLocal,
                /*AddressSpace=*/11);
            gv->setAlignment(llvm::Align(16));
            // GEP through the block to its first byte.  The struct index has to
            // be an i32 constant (a struct field is not a runtime index); feeding
            // i64 for it builds invalid IR that parses nowhere and crashes the
            // backend without a diagnostic.
            llvm::Constant* idx[3]
                = { llvm::ConstantInt::get(llvm::Type::getInt64Ty(ctx), 0),
                    llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx), 0),
                    llvm::ConstantInt::get(llvm::Type::getInt64Ty(ctx), 0) };
            frame_base = llvm::ConstantExpr::getInBoundsGetElementPtr(
                fbt, gv,
                llvm::ArrayRef<llvm::Constant*>(idx, 3));

            // The arena view itself is defined here, in the kernel module: the
            // material modules only import it, and spirv-link resolves imports
            // against a later module, which is exactly where this one sits.
            llvm::Type* i32m = llvm::Type::getInt32Ty(ctx);
            llvm::StructType* word_m
                = llvm::StructType::create(ctx, { i32m }, "struct.ArenaWord");
            llvm::ArrayType* words_m = llvm::ArrayType::get(word_m, 65536);
            llvm::StructType* view_m
                = llvm::StructType::create(ctx, { words_m },
                                           "struct.ArenaViewI32");
            llvm::GlobalVariable* arena_m = new llvm::GlobalVariable(
                *render, view_m, /*isConstant=*/false,
                llvm::GlobalValue::InternalLinkage,
                llvm::UndefValue::get(view_m), "osl_arena_i32", nullptr,
                llvm::GlobalVariable::NotThreadLocal, /*AddressSpace=*/11);
            arena_m->setAlignment(llvm::Align(16));
        }
        llvm::Function* frame = render->getFunction("osl_vk_raytrace_frame");
        if (!frame) {
            out_log = "the device render module has no osl_vk_raytrace_frame";
            return false;
        }

        llvm::Function* kernel = nullptr;   // only exists in the non-split path
        if (split && out_entry_ir) {
            // The entry goes into a *separate* module.  LLVM's SPIR-V
            // structurizer fails ("No valid candidate in the queue. Is the graph
            // reducible?") when it has to structure the whole kernel module at
            // once, while each half on its own structures fine: the library
            // half emits as a library (nothing to structure) and this entry
            // half is five instructions.  spirv-link joins them afterwards.
            llvm::Module em("osl_entry", ctx);
            em.setTargetTriple(render->getTargetTriple());
            em.setDataLayout(render->getDataLayout());

            llvm::Type* i8t = llvm::Type::getInt8Ty(ctx);
            llvm::Type* ptr11
                = llvm::PointerType::get(ctx, 11);
            llvm::Type* bytes = llvm::ArrayType::get(i8t, (uint64_t)80 << 20);
            llvm::StructType* fbt
                = llvm::StructType::create(ctx, { bytes }, "struct.FrameBuf");
            // Defined here (internal linkage + zero initialiser) so the linker has
            // an export to resolve the kernel module's import against; a second
            // external declaration would leave "osl_frame_buf" unresolved.
            // Defined here (internal linkage) so the linker has an export to
            // resolve any import of it against.  `undef` rather than a zero
            // initialiser: a global with internal linkage needs a value to be a
            // definition, and an aggregate zero reaches SPIR-V as an
            // OpVariable initialiser that the validator rejects ("Initializer
            // type must match the data type").  undef means "no initialiser".
            llvm::GlobalVariable* gv = new llvm::GlobalVariable(
                em, fbt, /*isConstant=*/false,
                llvm::GlobalValue::InternalLinkage,
                llvm::UndefValue::get(fbt), "osl_frame_buf", nullptr,
                llvm::GlobalVariable::NotThreadLocal, /*AddressSpace=*/11);
            gv->setAlignment(llvm::Align(16));

            // The arena view that the material modules and the offset runtime
            // import.  It is defined here, next to the frame buffer, so that
            // spirv-link has one export to resolve those imports against; with
            // no defining module the link stops at "Unresolved external
            // reference to osl_arena_i32".  The shape has to match what
            // vulkan_shadeops_offset.cpp declares -- one array of 32-bit words,
            // in which the union of an int and a float is a single 32-bit
            // member -- because a getelementptr whose source element type
            // disagrees with the global's real type is invalid IR.
            llvm::Type* i32v = llvm::Type::getInt32Ty(ctx);
            llvm::StructType* word_t
                = llvm::StructType::create(ctx, { i32v }, "struct.ArenaWord");
            llvm::ArrayType* words_t = llvm::ArrayType::get(word_t, 65536);
            llvm::StructType* view_t
                = llvm::StructType::create(ctx, { words_t },
                                           "struct.ArenaViewI32");
            llvm::GlobalVariable* arena_gv = new llvm::GlobalVariable(
                em, view_t, /*isConstant=*/false,
                llvm::GlobalValue::InternalLinkage,
                llvm::UndefValue::get(view_t), "osl_arena_i32", nullptr,
                llvm::GlobalVariable::NotThreadLocal, /*AddressSpace=*/11);
            arena_gv->setAlignment(llvm::Align(16));
            // Nothing in this module reads the view, so an internal-linkage
            // global is dropped before codegen and the linker never sees the
            // definition it is supposed to export: pin it with llvm.used.
            {
                llvm::SmallVector<llvm::GlobalValue*, 2> pinned { arena_gv };
                llvm::appendToUsed(em, pinned);
            }

            llvm::Function* fidx = llvm::Function::Create(
                llvm::FunctionType::get(
                    vd, { ptr11, llvm::Type::getInt64Ty(ctx) }, false),
                llvm::GlobalValue::ExternalLinkage, "osl_vk_raytrace_frame", em);

            // The kernel library reaches the invocation index through the
            // mangled OpenCL builtin name.  clspv turned that into
            // gl_GlobalInvocationID; the LLVM backend treats it as an ordinary
            // external, so the definition has to exist here -- and only here,
            // because a builtin belongs to the module that owns the entry point.
            llvm::Type* i32t = llvm::Type::getInt32Ty(ctx);
            llvm::ArrayType* id3 = llvm::ArrayType::get(i32t, 3);
            llvm::GlobalVariable* gidvar = new llvm::GlobalVariable(
                em, id3, /*isConstant=*/false,
                llvm::GlobalValue::InternalLinkage,
                llvm::ConstantAggregateZero::get(id3), "gl_GlobalInvocationID",
                nullptr, llvm::GlobalVariable::NotThreadLocal,
                /*AddressSpace=*/7 /* SPIR-V Input */);
            {
                // Decoration BuiltIn (11), builtin GlobalInvocationId (28).
                // !spirv.Decorations is a list of decoration nodes, so the
                // {kind, operand} pair has to be wrapped one level deeper --
                // attaching the pair itself is rejected as "Invalid decoration".
                llvm::Metadata* ops[2] = {
                    llvm::ConstantAsMetadata::get(
                        llvm::ConstantInt::get(i32t, 11)),
                    llvm::ConstantAsMetadata::get(
                        llvm::ConstantInt::get(i32t, 28))
                };
                llvm::Metadata* list[1] = { llvm::MDNode::get(ctx, ops) };
                gidvar->setMetadata("spirv.Decorations",
                                    llvm::MDNode::get(ctx, list));
            }
            // NOTE: no `_Z13get_global_idj` definition here.  The kernel library
            // takes the index as an argument precisely so that it imports nothing
            // from this module; otherwise the two modules would import each
            // other's exports and spirv-link could not resolve them in one pass.

            llvm::Function* k = llvm::Function::Create(
                llvm::FunctionType::get(vd, {}, false),
                llvm::GlobalValue::ExternalLinkage, "osl_kernel", em);
            k->setCallingConv(llvm::CallingConv::SPIR_KERNEL);
            k->addFnAttr("hlsl.shader", "compute");
            // The backend reads the workgroup size from this attribute, comma
            // separated (the same spelling the working view probe used).
            k->addFnAttr("hlsl.numthreads",
                         fmtformat("{},{},{}", osl_vk_abi::kWorkgroupSize, 1, 1));
            {
                llvm::BasicBlock* bb = llvm::BasicBlock::Create(ctx, "entry", k);
                llvm::IRBuilder<> b(bb);
                llvm::Value* base = llvm::ConstantExpr::getInBoundsGetElementPtr(
                    fbt, gv,
                    llvm::ArrayRef<llvm::Constant*>(
                        { llvm::ConstantInt::get(llvm::Type::getInt64Ty(ctx), 0),
                          llvm::ConstantInt::get(llvm::Type::getInt32Ty(ctx), 0),
                          llvm::ConstantInt::get(llvm::Type::getInt64Ty(ctx), 0) }));
                llvm::Value* gp
                    = b.CreateInBoundsGEP(id3, gidvar,
                                          { b.getInt32(0), b.getInt32(0) });
                llvm::Value* gx = b.CreateZExt(b.CreateLoad(i32t, gp),
                                               llvm::Type::getInt64Ty(ctx));
                b.CreateCall(fidx, { base, gx });
                b.CreateRetVoid();
            }
            llvm::raw_string_ostream eos(*out_entry_ir);
            em.print(eos, nullptr);
            eos.flush();
        } else {
            llvm::FunctionType* ktype = llvm::FunctionType::get(
                vd, split ? std::vector<llvm::Type*>{}
                          : std::vector<llvm::Type*>{ pbuf },
                false);
            // Assign to the enclosing `kernel`, not a new local: the workgroup
            // size metadata below is attached through that variable, and a
            // shadowing declaration here silently leaves it null -- which makes
            // clspv fall back to 1x1x1 and dispatch only `groups` invocations
            // (the rest of the frame is never written).
            kernel = llvm::Function::Create(
                ktype, llvm::GlobalValue::ExternalLinkage, "osl_kernel",
                *render);
            kernel->setCallingConv(llvm::CallingConv::SPIR_KERNEL);
            llvm::BasicBlock* bb
                = llvm::BasicBlock::Create(ctx, "entry", kernel);
            llvm::IRBuilder<> b(bb);
            // The frame now takes the invocation index as an argument.  In the
            // clspv path the synthesised kernel asks for it through the OpenCL
            // builtin (clspv turns that into gl_GlobalInvocationID); in the split
            // path the entry module supplies it.
            llvm::Value* gval;
            if (split) {
                gval = llvm::ConstantInt::get(llvm::Type::getInt64Ty(ctx), 0);
            } else {
                llvm::Function* gget
                    = render->getFunction("_Z13get_global_idj");
                if (!gget)
                    gget = llvm::Function::Create(
                        llvm::FunctionType::get(llvm::Type::getInt64Ty(ctx),
                                                { i32 }, false),
                        llvm::GlobalValue::ExternalLinkage,
                        "_Z13get_global_idj", *render);
                gval = b.CreateCall(gget, { b.getInt32(0) });
            }
            b.CreateCall(frame, { split ? frame_base : kernel->getArg(0), gval });
            b.CreateRetVoid();
        }
        // clspv reads the workgroup size from *metadata* named
        // "reqd_work_group_size" (that is what clang emits for the OpenCL
        // attribute of the same name), not from a function attribute.  With the
        // attribute form it silently falls back to 1x1x1, which is a silent
        // wrong-answer bug: the dispatch then covers only `groups` pixels.
        if (kernel)
            kernel->addFnAttr("reqd_work_group_size",
                              fmtformat("{} 1 1", osl_vk_abi::kWorkgroupSize));
        if (kernel) {
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

        llvm::raw_string_ostream os(out_ir);
        render->print(os, nullptr);
        os.flush();
    }

    return true;
}

}  // namespace pvt

OSL_NAMESPACE_END
