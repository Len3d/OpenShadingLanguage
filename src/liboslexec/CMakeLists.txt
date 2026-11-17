# Copyright Contributors to the Open Shading Language project.
# SPDX-License-Identifier: BSD-3-Clause
# https://github.com/AcademySoftwareFoundation/OpenShadingLanguage

# NOTE: for any additions/deletions from liboslexec_target_srcs,
# please update wide_target_combine_text_and_rodata.ld
set ( liboslexec_target_srcs
    wide/wide_opalgebraic
    wide/wide_opcolor
    wide/wide_opclosure
    wide/wide_opdictionary
    wide/wide_ophash
    wide/wide_opmatrix
    wide/wide_opmessage
    wide/wide_opnoise
    wide/wide_opnoise_cell
    wide/wide_opnoise_gabor_impl
    wide/wide_opnoise_generic_impl
    wide/wide_opnoise_hash
    wide/wide_opnoise_null
    wide/wide_opnoise_null_deriv
    wide/wide_opnoise_periodic_cell
    wide/wide_opnoise_periodic_gabor_impl
    wide/wide_opnoise_periodic_generic_impl
    wide/wide_opnoise_periodic_hash
    wide/wide_opnoise_periodic_perlin_float
    wide/wide_opnoise_periodic_perlin_deriv_float
    wide/wide_opnoise_periodic_perlin_Vec3
    wide/wide_opnoise_periodic_perlin_deriv_Vec3
    wide/wide_opnoise_periodic_uperlin_float
    wide/wide_opnoise_periodic_uperlin_deriv_float
    wide/wide_opnoise_periodic_uperlin_Vec3
    wide/wide_opnoise_periodic_uperlin_deriv_Vec3
    wide/wide_opnoise_perlin_float
    wide/wide_opnoise_perlin_deriv_float
    wide/wide_opnoise_perlin_Vec3
    wide/wide_opnoise_perlin_deriv_Vec3
    wide/wide_opnoise_simplex_float
    wide/wide_opnoise_simplex_deriv_float
    wide/wide_opnoise_simplex_Vec3
    wide/wide_opnoise_simplex_deriv_Vec3
    wide/wide_opnoise_unull
    wide/wide_opnoise_unull_deriv
    wide/wide_opnoise_uperlin_float
    wide/wide_opnoise_uperlin_deriv_float
    wide/wide_opnoise_uperlin_Vec3
    wide/wide_opnoise_uperlin_deriv_Vec3
    wide/wide_opnoise_usimplex_float
    wide/wide_opnoise_usimplex_deriv_float
    wide/wide_opnoise_usimplex_Vec3
    wide/wide_opnoise_usimplex_deriv_Vec3
    wide/wide_oppointcloud
    wide/wide_opspline
    wide/wide_opstring
    wide/wide_optest_float
    wide/wide_optexture
    wide/wide_optranscendental
    wide/wide_optrigonometric
    wide/wide_shadingsys
   ../liboslnoise/wide/wide_gabor_anisotropic_disabled
   ../liboslnoise/wide/wide_gabor_anisotropic_enabled
   ../liboslnoise/wide/wide_gabor_hybrid_disabled
   ../liboslnoise/wide/wide_gabor_hybrid_enabled
   ../liboslnoise/wide/wide_gabor_isotropic_disabled
   ../liboslnoise/wide/wide_gabor_isotropic_enabled   
   ../liboslnoise/wide/wide_gabor3_anisotropic_disabled
   ../liboslnoise/wide/wide_gabor3_anisotropic_enabled
   ../liboslnoise/wide/wide_gabor3_hybrid_disabled
   ../liboslnoise/wide/wide_gabor3_hybrid_enabled
   ../liboslnoise/wide/wide_gabor3_isotropic_disabled
   ../liboslnoise/wide/wide_gabor3_isotropic_enabled   
    )


set ( liboslexec_override_limits
    wide/wide_opnoise_periodic_perlin_deriv_float
    wide/wide_opnoise_periodic_perlin_deriv_Vec3
    wide/wide_opnoise_periodic_perlin_float
    wide/wide_opnoise_periodic_perlin_Vec3
    wide/wide_opnoise_periodic_uperlin_deriv_float
    wide/wide_opnoise_periodic_uperlin_deriv_Vec3
    wide/wide_opnoise_periodic_uperlin_float
    wide/wide_opnoise_periodic_uperlin_Vec3
    wide/wide_opnoise_perlin_deriv_float
    wide/wide_opnoise_perlin_deriv_Vec3
    wide/wide_opnoise_perlin_float
    wide/wide_opnoise_perlin_Vec3
    wide/wide_opnoise_uperlin_deriv_float
    wide/wide_opnoise_uperlin_deriv_Vec3
    wide/wide_opnoise_uperlin_float
    wide/wide_opnoise_uperlin_Vec3
    wide/wide_opspline
   ../liboslnoise/wide/wide_gabor3_anisotropic_disabled
   ../liboslnoise/wide/wide_gabor3_anisotropic_enabled
   ../liboslnoise/wide/wide_gabor3_hybrid_disabled
   ../liboslnoise/wide/wide_gabor3_hybrid_enabled
   ../liboslnoise/wide/wide_gabor3_isotropic_disabled
   ../liboslnoise/wide/wide_gabor3_isotropic_enabled   
    )
    
set ( liboslexec_require_INF_NaN
    shadingsys.cpp
    wide/wide_shadingsys
    wide/wide_optest_float
    )

set ( liboslexec_exceeds_INTELCLANG_inlining_limits
    wide/wide_opspline
    )

set (local_lib oslexec)
set (lib_src
          shadingsys.cpp closure.cpp
          dictionary.cpp
          context.cpp instance.cpp
          loadshader.cpp master.cpp
          opcolor.cpp opfmt.cpp opmatrix.cpp opmessage.cpp
          opnoise.cpp
          opspline.cpp opstring.cpp optexture.cpp
          oslexec.cpp
          pointcloud.cpp rendservices.cpp
          constfold.cpp runtimeoptimize.cpp typespec.cpp
          lpexp.cpp lpeparse.cpp automata.cpp accum.cpp
          opclosure.cpp
          shadeimage.cpp
          backendcpp.cpp
          backendllvm.cpp
          llvm_gen.cpp llvm_instance.cpp llvm_util.cpp
          rs_fallback.cpp
          journal.cpp
    )
if (OSL_BUILD_BATCHED)
    list(APPEND lib_src 
          batched_analysis.cpp
          batched_backendllvm.cpp
          batched_llvm_gen.cpp
          batched_llvm_instance.cpp
          batched_rendservices.cpp
        )
endif()    

if (BUILD_SHARED_LIBS)
    # oslcomp symbols used in oslexec
    list(APPEND lib_src
        ../liboslcomp/symtab.cpp
        )
endif ()

file (GLOB exec_headers "*.h")
file (GLOB compiler_headers "../liboslcomp/*.h")

FLEX_BISON ( osolex.l osogram.y oso lib_src exec_headers )

set ( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -D__STDC_LIMIT_MACROS -D__STDC_CONSTANT_MACROS" )


macro ( REQUIRE_INF_NAN SRC )
    if (CMAKE_COMPILER_IS_INTELCLANG)
        # if -ffast-math is enabled to to the underlying compiler, 
        # inf's and NaN's may not be handled properly (by design)
        if (MSVC)
            set_property(SOURCE ${SRC} APPEND PROPERTY COMPILE_OPTIONS "/fp:precise")
            #message (STATUS "SOURCE ${SRC} APPEND PROPERTY COMPILE_OPTIONS /fp:precise")
        else ()
            #set_property(SOURCE $SRC} APPEND PROPERTY COMPILE_OPTIONS "-fp-model=precise")
            # Disabling fast-math should be enough to enable proper support of INF and NaN
            #set_property(SOURCE ${SRC} APPEND PROPERTY COMPILE_OPTIONS "-fno-fast-math")
            # We can explicitly honor inf, nans, and signed 0's
            set_property(SOURCE ${SRC} APPEND PROPERTY COMPILE_OPTIONS 
                "-fhonor-infinities" 
                "-fhonor-nans" 
                "-fsigned-zeros")
        endif()
    endif()
endmacro ( )

# Apply complilation options as needed
foreach(src ${lib_src})
    if (${src} IN_LIST liboslexec_require_INF_NaN)
        set(SRC "${CMAKE_CURRENT_SOURCE_DIR}/${src}")
        REQUIRE_INF_NAN ( ${SRC} )  
    endif()
endforeach(src)

# Which clang compiles the Vulkan device runtime to bitcode.  It must be a
# clang that knows the `spirv64` target (it only emits IR; clspv makes the
# actual SPIR-V), i.e. the same LLVM we build oslexec against.
# NOTE: the doc string below must stay on one line -- splitting it over several
# quoted strings turns the extra pieces into additional cache *values*.
set (VULKAN_SHADEOPS_CLANG "" CACHE FILEPATH "clang used to compile the Vulkan device runtime (vulkan_shadeops.cpp) to LLVM bitcode; empty means the Vulkan backend is built without one")
if (NOT VULKAN_SHADEOPS_CLANG)
    # LLVM_DIR points at <prefix>/lib/cmake/llvm; LLVM_TOOLS_BINARY_DIR at
    # <prefix>/bin.  Try both spellings before giving up.
    set (_vk_clang_candidates
         "${LLVM_TOOLS_BINARY_DIR}/clang${CMAKE_EXECUTABLE_SUFFIX}"
         "${LLVM_DIR}/../../../bin/clang${CMAKE_EXECUTABLE_SUFFIX}")
    foreach (_candidate ${_vk_clang_candidates})
        if (EXISTS "${_candidate}")
            set (VULKAN_SHADEOPS_CLANG "${_candidate}")
            break()
        endif ()
    endforeach ()
endif ()
message (STATUS "Vulkan device runtime clang: '${VULKAN_SHADEOPS_CLANG}'")

if (USE_LLVM_BITCODE)
    set (llvm_ops_srcs llvm_ops.cpp)

    set(include_dirs ${CMAKE_CURRENT_SOURCE_DIR})
    list(APPEND include_dirs ${CMAKE_SOURCE_DIR}/src/include)
    list(APPEND include_dirs ${CMAKE_BINARY_DIR}/include)
    list(APPEND include_dirs ${IMATH_INCLUDES})
    list(APPEND include_dirs ${OpenImageIO_INCLUDES})
    list(APPEND include_dirs ${ROBINMAP_INCLUDES})

    EMBED_LLVM_BITCODE_IN_CPP ( "${llvm_ops_srcs}" "_host" "osl_llvm_compiled_ops" lib_src "" "${include_dirs}")

    set (rs_dependent_ops_srcs
         opmatrix.cpp opfmt.cpp optexture.cpp pointcloud.cpp opclosure.cpp
         )
    # Achieve the effect of absorbing osl_llvm_compiled_ops by adding its 
    # sources to rs_dependent_ops_srcs which avoids having to do it at runtime.
    list (APPEND rs_dependent_ops_srcs ${llvm_ops_srcs})

    EMBED_LLVM_BITCODE_IN_CPP ( "${rs_dependent_ops_srcs}" "_host_rs" "osl_llvm_compiled_rs_dependent_ops" lib_src "" "${include_dirs}")

    if (CUDAToolkit_FOUND)
        add_compile_definitions (OSL_LLVM_CUDA_BITCODE)

        # The shadeops are compiled to both LLVM bitcode and PTX, and embedded
        # in liboslexec in binary format. The bitcode is used to seed the LLVM
        # module for each shader, and the PTX can be retrieved by the renderer
        # through the shading system.
        set (liboslexec_shadeops_srcs
            ${CMAKE_SOURCE_DIR}/src/liboslexec/llvm_ops.cpp
            ${CMAKE_SOURCE_DIR}/src/liboslexec/opnoise.cpp
            ${CMAKE_SOURCE_DIR}/src/liboslexec/opspline.cpp
            ${CMAKE_SOURCE_DIR}/src/liboslexec/opcolor.cpp
            ${CMAKE_SOURCE_DIR}/src/liboslexec/opmatrix.cpp
            ${CMAKE_SOURCE_DIR}/src/liboslexec/optexture.cpp
            ${CMAKE_SOURCE_DIR}/src/liboslexec/pointcloud.cpp
            ${CMAKE_SOURCE_DIR}/src/liboslexec/opclosure.cpp
            ${CMAKE_SOURCE_DIR}/src/liboslnoise/gabornoise.cpp
            ${CMAKE_SOURCE_DIR}/src/liboslnoise/simplexnoise.cpp
        )

        CUDA_SHADEOPS_COMPILE ( "shadeops_cuda"
            shadeops_cuda_bc
            shadeops_cuda_ptx
            "${liboslexec_shadeops_srcs}"
            ""
        )

        # Serialize the linked bitcode into a CPP file to be embedded in the current target binary
        set (shadeops_bc_cuda_cpp "${CMAKE_CURRENT_BINARY_DIR}/shadeops_cuda.bc.cpp")
        MAKE_EMBEDDED_CPP( "shadeops_cuda_llvm_compiled_ops" ${shadeops_bc_cuda_cpp} ${shadeops_cuda_bc} )
        list (APPEND lib_src ${shadeops_bc_cuda_cpp})

        # Serialize the linked PTX into a CPP file to be embedded in the current target binary
        set (shadeops_ptx_cuda_cpp "${CMAKE_CURRENT_BINARY_DIR}/shadeops_cuda.ptx.cpp")
        MAKE_EMBEDDED_CPP( "shadeops_cuda_ptx_compiled_ops" ${shadeops_ptx_cuda_cpp} ${shadeops_cuda_ptx} )
        list (APPEND lib_src ${shadeops_ptx_cuda_cpp})

        # Installing the shadeops PTX isn't necessary since the renderer can
        # retrieve it through the shading system. But for testing purposes it is
        # handy to make the PTX available offline.
        install (FILES ${shadeops_cuda_ptx}
            DESTINATION ${OSL_PTX_INSTALL_DIR})

    endif ()
    # Also compile llvm_ops.cpp as a native object so that generated shader
    # DSOs (cpp backend, debug_output_cpp >= 2) can link osl_* symbols from
    # liboslexec at load time.  The bitcode copy (above) is still used by the
    # LLVM JIT for full inlining; the native copy provides the same functions
    # as exported symbols.  OSL_COMPILING_TO_BITCODE is NOT set here, so the
    # OSL_SHADEOP macro expands to extern "C" OSL_DLL_EXPORT.
    list (APPEND lib_src llvm_ops.cpp)
else ()
    # With MSVC/Mingw, we don't compile llvm_ops.cpp to LLVM bitcode, due
    # to clang being unable to compile MSVC C++ header files at this time.
    # Instead it is part of the regular build process.
    add_compile_definitions (OSL_LLVM_NO_BITCODE)
    set (lib_src ${lib_src} llvm_ops.cpp)
endif ()

# The Vulkan device runtime is different from llvm_ops.cpp: it is written to be
# self contained (vulkan_types.h includes nothing), so clang *can* compile it
# for the spirv64 target -- it only has to emit IR, clspv produces the actual
# SPIR-V.  Doing it at build time (instead of shipping a .ll in the tree) means
# the bitcode can never drift from the source.
if (VULKAN_SHADEOPS_CLANG)
    set (vk_shadeops_src ${CMAKE_CURRENT_SOURCE_DIR}/vulkan_shadeops.cpp)
    set (vk_shadeops_bc ${CMAKE_CURRENT_BINARY_DIR}/vulkan_shadeops.bc)

    add_custom_command (OUTPUT ${vk_shadeops_bc}
        COMMAND ${VULKAN_SHADEOPS_CLANG}
            -target spirv64 -emit-llvm -c -O2
            -fno-rtti -fno-exceptions
            -I${CMAKE_CURRENT_SOURCE_DIR}
            -o ${vk_shadeops_bc} ${vk_shadeops_src}
        DEPENDS ${vk_shadeops_src} ${CMAKE_CURRENT_SOURCE_DIR}/vulkan_types.h
        COMMENT "Compiling Vulkan device runtime to bitcode"
        VERBATIM)

    # Serialize the bitcode into a .cpp with a byte array, so liboslexec does
    # not have to find a file on disk at runtime.
    MAKE_EMBEDDED_CPP ( "vulkan_shadeops_llvm_compiled_ops"
                        ${CMAKE_CURRENT_BINARY_DIR}/vulkan_shadeops.bc.cpp
                        ${vk_shadeops_bc} )
    list (APPEND lib_src ${CMAKE_CURRENT_BINARY_DIR}/vulkan_shadeops.bc.cpp)

    # The whole-frame render kernel (camera/ray generation/sampling/intersection/
    # accumulation) is built the same way.  It is a second module because it is
    # only linked in when the scene is rendered frame-at-a-time on the device,
    # not when a single shading point is evaluated from the host.
    set (vk_render_src ${CMAKE_CURRENT_SOURCE_DIR}/vulkan_render.cpp)
    set (vk_render_bc ${CMAKE_CURRENT_BINARY_DIR}/vulkan_render.bc)

    add_custom_command (OUTPUT ${vk_render_bc}
        COMMAND ${VULKAN_SHADEOPS_CLANG}
            -target spirv64 -emit-llvm -c -O2
            -fno-rtti -fno-exceptions
            -I${CMAKE_CURRENT_SOURCE_DIR}
            -o ${vk_render_bc} ${vk_render_src}
        DEPENDS ${vk_render_src} ${CMAKE_CURRENT_SOURCE_DIR}/vulkan_types.h
        COMMENT "Compiling Vulkan device render kernel to bitcode"
        VERBATIM)

    MAKE_EMBEDDED_CPP ( "vulkan_render_llvm_compiled_ops"
                        ${CMAKE_CURRENT_BINARY_DIR}/vulkan_render.bc.cpp
                        ${vk_render_bc} )
    list (APPEND lib_src ${CMAKE_CURRENT_BINARY_DIR}/vulkan_render.bc.cpp)
else ()
    message (WARNING "VULKAN_SHADEOPS_CLANG not set: Vulkan backend will run "
                     "without a device runtime and its SPIR-V will not have an "
                     "entry point")
endif ()

# Always build target specific libraries for batched execution as shared
# For each target specific shared library being built, we need
# we need to let a couple source files know so they can build
# function mappings for those targets
set_property(SOURCE "batched_llvm_instance.cpp"  "shadingsys.cpp"  
    APPEND PROPERTY COMPILE_DEFINITIONS ${BATCHED_SUPPORT_DEFINES}) 

set (BATCHED_TARGET_LIBS)
foreach(batched_target ${BATCHED_TARGET_LIST})
    set (batched_target_lib "_${batched_target}_oslexec")
    list (APPEND BATCHED_TARGET_LIBS ${batched_target_lib})
    string (REPLACE "_" ";" TARGET_OPTS ${batched_target})
    list (GET TARGET_OPTS 0 TARGET_OPT_SIZE)
    list (GET TARGET_OPTS 1 TARGET_OPT_ISA)
    list (LENGTH TARGET_OPTS NUM_TARGET_OPTS)
    set (TARGET_OPT_FMA "FMA")
    set (TARGET_ISA ${TARGET_OPT_ISA})
    if (NUM_TARGET_OPTS EQUAL 3)
        list (GET TARGET_OPTS 2 TARGET_OPT_FMA)
        set (TARGET_ISA "${TARGET_OPT_ISA}_${TARGET_OPT_FMA}")
    endif() 
    string (SUBSTRING ${TARGET_OPT_SIZE} 1 -1 TARGET_BATCH_SIZE)
    # Strategy is to make a copy of each cpp in liboslexec_target_srcs for 
    # each target batch width and ISA combination, applying compiler flags to define 
    # -D__OSL_WIDTH=[4|8|16]
    # -D__OSL_TARGET_ISA=[AVX512|AVX2|AVX|SSE4_2]
    # and compiler flags to choose correct target ISA for your compiler
    # You may then add/remove the desired  ${TARGET_SOURCES_B[4|8|16]_[AVX512|AVX2|AVX|SSE4_2]}
    # to your add_library call 
    set (TARGET_LIB_SOURCES)
    set (TARGET_CXX_DEFS)
    set (TARGET_CXX_OPTS)
    
    list (APPEND TARGET_CXX_OPTS
        "-I${CMAKE_CURRENT_SOURCE_DIR}/wide"
        "-I${CMAKE_CURRENT_SOURCE_DIR}/../liboslnoise/wide"
        )
        
    list (APPEND TARGET_CXX_DEFS
        "__OSL_WIDTH=${TARGET_BATCH_SIZE}"
        "__OSL_TARGET_ISA=${TARGET_ISA}"        
        "OSL_OPENMP_SIMD"        
        )

    if (CMAKE_COMPILER_IS_INTEL)
        if (MSVC)
            if (${TARGET_OPT_ISA} STREQUAL "AVX512")
                list (APPEND TARGET_CXX_OPTS "/QxCORE-AVX512")
                if (${TARGET_BATCH_SIZE} STREQUAL "16")
                    list (APPEND TARGET_CXX_OPTS "/Qopt-zmm-usage:high")
                else ()
                    list (APPEND TARGET_CXX_OPTS "/Qopt-zmm-usage:low")
                endif ()
            elseif (${TARGET_OPT_ISA} STREQUAL "AVX2")
                list (APPEND TARGET_CXX_OPTS "/QxCORE-AVX2")
            elseif (${TARGET_OPT_ISA} STREQUAL "AVX")
                list (APPEND TARGET_CXX_OPTS "/QxAVX")
            else ()
                message (FATAL_ERROR "Unknown ISA=${TARGET_OPT_ISA} extract from USE_BATCHED entry ${batched_target}")
            endif ()
            
            if (${TARGET_OPT_FMA} STREQUAL "noFMA")
                list (APPEND TARGET_CXX_OPTS "/Qfma-")
            endif ()
            
            list (APPEND TARGET_CXX_OPTS 
                "/Qprec-div" 
                "/Qopenmp-simd"
                )
            if (VEC_REPORT)
                list (APPEND TARGET_CXX_OPTS "/Qopt-report:5")
            endif ()
                
                
        else ()
            if (${TARGET_OPT_ISA} STREQUAL "AVX512")
                list (APPEND TARGET_CXX_OPTS "-xCORE-AVX512")
                if (${TARGET_BATCH_SIZE} STREQUAL "16")
                    list (APPEND TARGET_CXX_OPTS "-qopt-zmm-usage:high")
                else ()
                    list (APPEND TARGET_CXX_OPTS "-qopt-zmm-usage:low")
                endif ()
            elseif (${TARGET_OPT_ISA} STREQUAL "AVX2")
                list (APPEND TARGET_CXX_OPTS "-xCORE-AVX2")
            elseif (${TARGET_OPT_ISA} STREQUAL "AVX")
                list (APPEND TARGET_CXX_OPTS "-xAVX")
            else ()
                message (FATAL_ERROR "Unknown ISA=${TARGET_OPT_ISA} extract from USE_BATCHED entry ${batched_target}")
            endif ()
            
            if (${TARGET_OPT_FMA} STREQUAL "noFMA")
                list (APPEND TARGET_CXX_OPTS "-no-fma")
            endif ()
            
            list (APPEND TARGET_CXX_OPTS 
                "-prec-div" 
                "-qopenmp-simd"
                "-qno-opt-multiple-gather-scatter-by-shuffles"
                )
                
            if (VEC_REPORT)
                list (APPEND TARGET_CXX_OPTS "-qopt-report=5")
            endif ()
                
        endif ()
    endif ()
    
    if (CMAKE_COMPILER_IS_CLANG OR CMAKE_COMPILER_IS_APPLECLANG)
        # REQUIRES CLANG 7+ to use successfully vectorize with OpenMP simd
        
        if (${TARGET_OPT_ISA} STREQUAL "AVX512")
            list (APPEND TARGET_CXX_OPTS "-march=skylake-avx512")
            if (${TARGET_BATCH_SIZE} STREQUAL "16")
                list (APPEND TARGET_CXX_OPTS "-mprefer-vector-width=512")
            else ()
                list (APPEND TARGET_CXX_OPTS "-mprefer-vector-width=256")
            endif ()
        elseif (${TARGET_OPT_ISA} STREQUAL "AVX2")
            list (APPEND TARGET_CXX_OPTS "-march=core-avx2")
        elseif (${TARGET_OPT_ISA} STREQUAL "AVX")
            list (APPEND TARGET_CXX_OPTS "-march=corei7-avx")
        elseif (${TARGET_OPT_ISA} STREQUAL "SSE2")
            list (APPEND TARGET_CXX_OPTS "-march=x86-64")
        else ()
            message (FATAL_ERROR "Unknown ISA=${TARGET_OPT_ISA} extract from USE_BATCHED entry ${batched_target}")
        endif ()
        
        if (${TARGET_OPT_FMA} STREQUAL "noFMA")
            # Do NOT use "-mno-fma" as it may lower ISA to one without any FMA
            # instructions (AVX512->AVX2).  Our intent is to not generate 
            # FMA's but still use the available instructions and register 
            # width. However math library calls may use still end up using 
            # FMA's, but lets try turning of fused math ops because many OSL 
            # math functions are from OIIO/IMath headers generate non FMA
            list (APPEND TARGET_CXX_OPTS "-ffp-contract=off")
        endif ()
        
        if (NOT CMAKE_COMPILER_IS_INTELCLANG)
            # large SIMD function loops will exceed llvm's -inline-threshold default of 225.
            # remark: loop not vectorized: call instruction cannot be vectorized [-Rpass-analysis]     
            # choose to increase that limit via compiler flags vs. 
            # workaround with __attribute__((flatten))     
            list (APPEND TARGET_CXX_OPTS "-mllvm" "-inline-threshold=100000")
            # NOTE: INTELCLANG we use icx/icc specific "#pragma force inline recursive"
            #       to only inline the SIMD loops contents vs. everything.
        endif ()
            
        # For loops with small loop bodies, clang was unrolling the loop before
        # #pragma omp simd
        # was processed.  Thus there was no loop left for the pragma to apply to
        # To work around this, for these wide SIMD libraries, we will disable
        # clang's loop unrolling.  We really don't want to do this and wish to remove
        # it as soon as clang addresses this issue.
        list (APPEND TARGET_CXX_OPTS "-fno-unroll-loops")

        if (CMAKE_COMPILER_IS_INTELCLANG)
            # utilize Intel compiler's optimized openmp simd generation vs. open source "-qopenmp-simd"
            list (APPEND TARGET_CXX_OPTS "-fiopenmp-simd")
        else ()
            list (APPEND TARGET_CXX_OPTS "-fopenmp")
        endif ()
        
        if (VEC_REPORT)
            if (CMAKE_COMPILER_IS_INTELCLANG)
                list (APPEND TARGET_CXX_OPTS 
                    "-qopt-report=3"
                    )
            else ()
                list (APPEND TARGET_CXX_OPTS 
                    "-Rpass=loop-vectorize" 
                    "-Rpass-analysis=loop-vectorize" 
                    # Uncomment next section to investigate failures in vectorization
                    #"-Rpass-missed=loop-vectorize"
                    )
            endif ()
        endif ()
    endif ()
    
    
    if (CMAKE_COMPILER_IS_GNUCC)
        # Experimental, not exhaustively tested with GCC.
        # More compiler options may be needed to successfully vectorize
        # complex control flow 
        list (APPEND TARGET_CXX_OPTS "-finline-limit=1000000")
        
        if (${TARGET_OPT_ISA} STREQUAL "AVX512")
            list (APPEND TARGET_CXX_OPTS "-march=skylake-avx512")
            if (${TARGET_BATCH_SIZE} STREQUAL "16")
                list (APPEND TARGET_CXX_OPTS "-mprefer-vector-width=512")
            else ()
                list (APPEND TARGET_CXX_OPTS "-mprefer-vector-width=256")
            endif ()
        elseif (${TARGET_OPT_ISA} STREQUAL "AVX2")
            list (APPEND TARGET_CXX_OPTS "-march=haswell")
        elseif (${TARGET_OPT_ISA} STREQUAL "AVX")
            list (APPEND TARGET_CXX_OPTS "-march=sandybridge")
        elseif (${TARGET_OPT_ISA} STREQUAL "SSE2")
            list (APPEND TARGET_CXX_OPTS "-march=x86-64")
        else ()
            message (FATAL_ERROR "Unknown ISA=${TARGET_OPT_ISA} extract from USE_BATCHED entry ${batched_target}")
        endif ()
        
        if (${TARGET_OPT_FMA} STREQUAL "noFMA")
            # Do NOT use "-mno-fma" as it may lower ISA to one without any FMA
            # instructions (AVX512->AVX2).  Our intent is to not generate 
            # FMA's but still use the available instructions and register 
            # width. However math library calls may use still end up using 
            # FMA's, but lets try turning of fused math ops because many OSL 
            # math functions are from OIIO/IMath headers generate non FMA
            list (APPEND TARGET_CXX_OPTS "-ffp-contract=off")
        endif ()
        
        list (APPEND TARGET_CXX_OPTS "-fopenmp-simd")
    endif ()    
        
    foreach(target_src ${liboslexec_target_srcs})
        set(TARGET_SRC "${CMAKE_CURRENT_BINARY_DIR}/${target_src}_${batched_target}.cpp")
        set(SRC "${CMAKE_CURRENT_SOURCE_DIR}/${target_src}.cpp")        
        CONFIGURE_FILE("${SRC}" ${TARGET_SRC} COPYONLY)
        list(APPEND TARGET_LIB_SOURCES "${TARGET_SRC}")
        
        set_property(SOURCE ${TARGET_SRC}
            APPEND PROPERTY COMPILE_DEFINITIONS ${TARGET_CXX_DEFS})
            
        set_property(SOURCE ${TARGET_SRC}  
            APPEND PROPERTY COMPILE_OPTIONS ${TARGET_CXX_OPTS})
            
        if (${target_src} IN_LIST liboslexec_override_limits)
            if (CMAKE_COMPILER_IS_INTEL)
                if (MSVC)
                    set_property(SOURCE ${TARGET_SRC} APPEND PROPERTY COMPILE_OPTIONS "/Qoverride-limits")
                else ()
                    set_property(SOURCE ${TARGET_SRC} APPEND PROPERTY COMPILE_OPTIONS "-qoverride-limits")
                endif()
            endif()
        endif()
        
        if (${target_src} IN_LIST liboslexec_require_INF_NaN)
            REQUIRE_INF_NAN ( ${TARGET_SRC} ) 
        endif()

        if (CMAKE_COMPILER_IS_INTELCLANG)
            if (${target_src} IN_LIST liboslexec_exceeds_INTELCLANG_inlining_limits)
                # NOTE: Some files still require increasing
                #       inlining threshold to successfully vectorize.
                #       Retest with future compiler versions
                set_property(SOURCE ${TARGET_SRC} APPEND PROPERTY COMPILE_OPTIONS 
                    "-mllvm" "-inline-threshold=100000")
            endif()
        endif()


    endforeach(target_src)
    
    add_library ( ${batched_target_lib} MODULE ${TARGET_LIB_SOURCES} )
    if (WIN32)
        set_target_properties( ${batched_target_lib} PROPERTIES PREFIX "lib" )
    endif()

    target_include_directories (${batched_target_lib}
        PUBLIC
            ${CMAKE_INSTALL_FULL_INCLUDEDIR}
            ${IMATH_INCLUDES}
        PRIVATE
            ${ROBINMAP_INCLUDES}
        )
    target_link_libraries (${batched_target_lib}
        PUBLIC
            OpenImageIO::OpenImageIO
            ${local_lib}
        )
    if (partio_FOUND)
        target_link_libraries(${batched_target_lib} PRIVATE partio::partio ZLIB::ZLIB)
        target_compile_definitions (${batched_target_lib} PRIVATE USE_PARTIO=1)
    endif ()
        
endforeach(batched_target)

configure_file (buildopts.h.in "${CMAKE_CURRENT_BINARY_DIR}/buildopts.h" @ONLY)

add_library (${local_lib} ${lib_src})
target_include_directories (${local_lib}
    PUBLIC
        ${CMAKE_INSTALL_FULL_INCLUDEDIR}
        ${IMATH_INCLUDES}
    PRIVATE
        "${CMAKE_SOURCE_DIR}/src/liboslcomp"
        "${CMAKE_CURRENT_BINARY_DIR}"
        ${ROBINMAP_INCLUDES}
    )

target_include_directories (${local_lib}  BEFORE PRIVATE ${OpenImageIO_INCLUDES})

# Collect include paths and platform base flags for compile_to_dso() defaults
get_target_property(_oiio_incs OpenImageIO::OpenImageIO INTERFACE_INCLUDE_DIRECTORIES)
get_target_property(_imath_incs Imath::Imath INTERFACE_INCLUDE_DIRECTORIES)
if (MSVC)
    set(_cpp_base_flags "/LD /O2 /std:c++17")
    set(_I "/I ")
elseif (APPLE)
    # -undefined dynamic_lookup: generated DSO resolves osl_* symbols from the
    # already-loaded liboslexec at dlopen time without needing an explicit -l.
    set(_cpp_base_flags "-shared -fPIC -O2 -std=c++17 -undefined dynamic_lookup")
    set(_I "-I")
else ()
    # On Linux, generated DSOs resolve osl_* symbols from the already-loaded
    # liboslexec at dlopen time. This requires those symbols to be exported in
    # liboslexec's dynamic symbol table — see src/build-scripts/hidesymbols.map,
    # which globally exports osl_* for exactly this reason.
    set(_cpp_base_flags "-shared -fPIC -O2 -std=c++17")
    set(_I "-I")
endif ()
set(_cpp_inc_flags
    "${_I}${CMAKE_CURRENT_SOURCE_DIR}"
    "${_I}${CMAKE_SOURCE_DIR}/src/include"
    "${_I}${CMAKE_BINARY_DIR}/include"
)
foreach(_inc ${_oiio_incs})
    list(APPEND _cpp_inc_flags "${_I}${_inc}")
endforeach()
foreach(_inc ${_imath_incs})
    list(APPEND _cpp_inc_flags "${_I}${_inc}")
endforeach()
list(JOIN _cpp_inc_flags " " _cpp_inc_flags_str)

target_compile_definitions (${local_lib}
    PRIVATE
        OSL_EXPORTS
        CUDA_TARGET_ARCH="${CUDA_TARGET_ARCH}"
        OSL_CUDA_VERSION="${CUDAToolkit_VERSION}"
        OSL_OPTIX_VERSION="${OPTIX_VERSION}"
        OSL_CPP_COMPILER_DEFAULT="${CMAKE_CXX_COMPILER}"
        OSL_CPP_COMPILER_FLAGS_DEFAULT="${_cpp_base_flags} ${_cpp_inc_flags_str}"
    )
if (partio_FOUND)
    target_link_libraries(${local_lib} PRIVATE partio::partio ZLIB::ZLIB)
    target_compile_definitions (${local_lib} PRIVATE USE_PARTIO=1)
endif ()

# Optional: clspv is the lowering engine for the Vulkan-flavor SPIR-V backend.
#
# clspv is linked *statically*, and it is built against the same LLVM 24
# install that OSL itself links.  That is the point of the arrangement: the
# process ends up with exactly one copy of LLVM instead of two (previously
# clspv's own LLVM lived inside clspv_core.dll).  CLSPV_ROOT must therefore
# point at a directory laid out as
#   <CLSPV_ROOT>/include/clspv/Compiler.h
#   <CLSPV_ROOT>/lib/clspv_core.lib
#   <CLSPV_ROOT>/lib/SPIRV-Tools.lib
set (CLSPV_ROOT "" CACHE PATH
     "Directory holding include/clspv and the static clspv_core.lib/SPIRV-Tools.lib")
set (CLSPV_LLVM_LIB_DIR "" CACHE PATH "Directory holding the LLVM/clang static libs clspv_core.lib was built against; empty means derive it from LLVM_DIR")
if (CLSPV_ROOT AND EXISTS "${CLSPV_ROOT}/include/clspv/Compiler.h"
        AND EXISTS "${CLSPV_ROOT}/lib/clspv_core.lib")
    message (STATUS "clspv found at ${CLSPV_ROOT}: Vulkan SPIR-V backend enabled (static)")
    target_sources (${local_lib} PRIVATE clspv_util.cpp)
    target_include_directories (${local_lib} PRIVATE "${CLSPV_ROOT}/include")

    if (NOT CLSPV_LLVM_LIB_DIR)
        get_filename_component (CLSPV_LLVM_LIB_DIR "${LLVM_DIR}/../.." ABSOLUTE)
    endif ()
    # clspv_core.lib calls into clang's frontend/parser, so those objects have
    # to be present too.  They are static libs in the same LLVM install.
    file (GLOB _clspv_clang_libs "${CLSPV_LLVM_LIB_DIR}/clang*.lib")
    if (NOT _clspv_clang_libs)
        message (WARNING "No clang*.lib found in ${CLSPV_LLVM_LIB_DIR}; "
                         "clspv_core.lib will not link")
    endif ()
    message (STATUS "clspv: ${CLSPV_LLVM_LIB_DIR} -> "
                    "${CMAKE_MATCH_COUNT} clang libs")

    target_link_libraries (${local_lib} PRIVATE
        "${CLSPV_ROOT}/lib/clspv_core.lib"
        "${CLSPV_ROOT}/lib/SPIRV-Tools.lib"
        ${_clspv_clang_libs}
        ${LLVM_LIBRARIES})
    target_compile_definitions (${local_lib} PRIVATE OSL_USE_CLSPV=1)
    # Unit test for the lowering path (no gtest dependency). clspv_util.cpp is
    # compiled into the test directly because pvt functions are not exported
    # from the oslexec DLL.
    add_executable (clspv_compile_test clspv_compile_test.cpp clspv_util.cpp)
    target_include_directories (clspv_compile_test PRIVATE
                                "${CMAKE_CURRENT_SOURCE_DIR}"
                                "${CMAKE_SOURCE_DIR}/src/include"
                                "${CLSPV_ROOT}/include"
                                ${OpenImageIO_INCLUDES})
    target_link_libraries (clspv_compile_test PRIVATE
                             "${CLSPV_ROOT}/lib/clspv_core.lib"
                             "${CLSPV_ROOT}/lib/SPIRV-Tools.lib"
                             ${_clspv_clang_libs}
                             ${LLVM_LIBRARIES}
                             # LLVM 24's Support library imports these directly.
                             ntdll
                             OpenImageIO::OpenImageIO ${CMAKE_DL_LIBS})
    target_compile_definitions (clspv_compile_test PRIVATE OSL_USE_CLSPV=1)
    set_target_properties (clspv_compile_test PROPERTIES FOLDER "Unit Tests")
endif ()

# Optional: the SPIR-V linker (SPIRV-Tools' `spv::Link`).
#
# Once the materials are emitted by LLVM's SPIR-V backend and the render kernel
# keeps going through clspv, the two halves are separate modules and SPIR-V has
# no cross-module calls, so they have to be merged before the driver sees them.
# SPIRV_TOOLS_LINK_ROOT must point at a directory laid out as
#   <SPIRV_TOOLS_LINK_ROOT>/include/spirv-tools/linker.hpp
#   <SPIRV_TOOLS_LINK_ROOT>/lib/SPIRV-Tools-link.lib
#   <SPIRV_TOOLS_LINK_ROOT>/lib/SPIRV-Tools-opt.lib
# The libraries have to be built with the same MSVC runtime as this tree
# (/MT here), otherwise the link fails with LNK2038.
set (SPIRV_TOOLS_LINK_ROOT "" CACHE PATH
     "Directory holding include/spirv-tools/linker.hpp and the static SPIRV-Tools-link/opt libs")
if (SPIRV_TOOLS_LINK_ROOT
        AND EXISTS "${SPIRV_TOOLS_LINK_ROOT}/include/spirv-tools/linker.hpp"
        AND EXISTS "${SPIRV_TOOLS_LINK_ROOT}/lib/SPIRV-Tools-link.lib")
    message (STATUS "SPIRV-Tools linker found at ${SPIRV_TOOLS_LINK_ROOT}: material/kernel module linking enabled")
    target_sources (${local_lib} PRIVATE vulkan_spirv_link.cpp)
    target_include_directories (${local_lib} PRIVATE "${SPIRV_TOOLS_LINK_ROOT}/include")
    # SPIRV-Tools-link needs the optimizer's IR (it links PUBLIC SPIRV-Tools-opt),
    # and both sit on top of the SPIRV-Tools core the clspv block already links.
    target_link_libraries (${local_lib} PRIVATE
        "${SPIRV_TOOLS_LINK_ROOT}/lib/SPIRV-Tools-link.lib"
        "${SPIRV_TOOLS_LINK_ROOT}/lib/SPIRV-Tools-opt.lib")
    target_compile_definitions (${local_lib} PRIVATE OSL_USE_SPIRV_LINK=1)

    # Unit test for the linking path, same shape as clspv_compile_test: no
    # gtest, the pvt helper is compiled in directly.
    add_executable (spirv_link_test spirv_link_test.cpp vulkan_spirv_link.cpp)
    target_include_directories (spirv_link_test PRIVATE
                                "${CMAKE_CURRENT_SOURCE_DIR}"
                                "${CMAKE_SOURCE_DIR}/src/include"
                                "${SPIRV_TOOLS_LINK_ROOT}/include"
                                ${OpenImageIO_INCLUDES})
    target_link_libraries (spirv_link_test PRIVATE
        "${SPIRV_TOOLS_LINK_ROOT}/lib/SPIRV-Tools-link.lib"
        "${SPIRV_TOOLS_LINK_ROOT}/lib/SPIRV-Tools-opt.lib"
        "${SPIRV_TOOLS_LINK_ROOT}/lib/SPIRV-Tools.lib")
    set_target_properties (spirv_link_test PROPERTIES FOLDER "Unit Tests")
endif ()

# Diagnostic for the LLVM-flavor emitter: same code path the runtime takes (the
# Vulkan retarget, null-cast folding and arena typing all live in
# vulkan_spirv_llvm.cpp), so what it writes is exactly what the driver would
# have been handed.
add_executable (vulkan_emit_test vulkan_emit_test.cpp vulkan_spirv_llvm.cpp)
target_include_directories (vulkan_emit_test PRIVATE
                            "${CMAKE_CURRENT_SOURCE_DIR}"
                            "${CMAKE_SOURCE_DIR}/src/include"
                            ${OpenImageIO_INCLUDES})
target_link_libraries (vulkan_emit_test PRIVATE
                         ${LLVM_LIBRARIES} ${LLVM_LDFLAGS} ${LLVM_SYSTEM_LIBRARIES}
                         # LLVM 24's Support library imports these directly.
                         ntdll
                         OpenImageIO::OpenImageIO ${CMAKE_DL_LIBS})
set_target_properties (vulkan_emit_test PROPERTIES FOLDER "Unit Tests")

# Disable RTTI from just the one module where we have classes that inherit
# from LLVM classes.
if (MSVC)
    set_source_files_properties (llvm_util.cpp
                                 PROPERTIES COMPILE_FLAGS "/GR-")
else ()
    set_source_files_properties (llvm_util.cpp
                                 PROPERTIES COMPILE_FLAGS "-fno-rtti")
endif()

# link with (system) library to prevent missing symbols inside clangDriver.lib
if (MSVC)
    target_link_libraries (${local_lib} PRIVATE "Version.lib")
endif()

target_sources (${local_lib} PRIVATE vulkan_legalize.cpp vulkan_abi_check.cpp vulkan_module.cpp vulkan_spirv_llvm.cpp)

target_link_libraries (${local_lib}
    PUBLIC
        oslquery oslnoise
        OpenImageIO::OpenImageIO
        $<TARGET_NAME_IF_EXISTS:Imath::Imath>
        $<TARGET_NAME_IF_EXISTS:Imath::Half>
    PRIVATE
        pugixml::pugixml
        ${CMAKE_DL_LIBS}
        ${LLVM_LIBRARIES} ${LLVM_LDFLAGS} ${LLVM_SYSTEM_LIBRARIES}
    )

target_compile_features (${local_lib}
                         INTERFACE cxx_std_${DOWNSTREAM_CXX_STANDARD})

set (link_flags "${VISIBILITY_MAP_COMMAND} ${EXTRA_DSO_LINK_ARGS}")
if (UNIX AND NOT APPLE)
    # Hide symbols from any static dependent libraries embedded here.
    set (link_flags "${link_flags} -Wl,--exclude-libs,ALL")
endif ()

set_target_properties (${local_lib}
        PROPERTIES
        VERSION     ${PROJECT_VERSION_MAJOR}.${PROJECT_VERSION_MINOR}.${PROJECT_VERSION_PATCH}
        SOVERSION   ${SOVERSION}
        OUTPUT_NAME ${local_lib}${OSL_LIBNAME_SUFFIX}
        LINK_FLAGS  ${link_flags}
    )

# add_dependencies (${local_lib} "${CMAKE_SOURCE_DIR}/src/build-scripts/hidesymbols.map")

foreach(batched_target_lib ${BATCHED_TARGET_LIBS})
    install_targets ( ${batched_target_lib} )
endforeach(batched_target_lib)

install_targets (${local_lib})

# Unit tests
if (OSL_BUILD_TESTS AND BUILD_TESTING)
    add_executable (accum_test accum_test.cpp)
    target_link_libraries (accum_test PRIVATE oslexec ${CMAKE_DL_LIBS})
    target_include_directories (accum_test  BEFORE PRIVATE ${OpenImageIO_INCLUDES})
    set_target_properties (accum_test PROPERTIES FOLDER "Unit Tests")
    add_test (unit_accum ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/accum_test)

    add_executable (dual_test dual_test.cpp)
    target_link_libraries (dual_test PRIVATE OpenImageIO::OpenImageIO Imath::Imath ${CMAKE_DL_LIBS})
    target_include_directories (dual_test  BEFORE PRIVATE ${OpenImageIO_INCLUDES})
    set_target_properties (dual_test PROPERTIES FOLDER "Unit Tests")
    add_test (unit_dual ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/dual_test)

    add_executable (llvmutil_test llvmutil_test.cpp)
    target_link_libraries (llvmutil_test PRIVATE oslexec ${CMAKE_DL_LIBS})
    target_include_directories (llvmutil_test  BEFORE PRIVATE ${OpenImageIO_INCLUDES})
    set_target_properties (llvmutil_test PROPERTIES FOLDER "Unit Tests")
    add_test (unit_llvmutil ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/llvmutil_test)
endif ()

# Driver for the Vulkan legalisation pass (the rewrite that makes an OSL module
# consumable by `clspv -x ir`).  Needs only LLVM, not clspv, so it builds even
# when the SPIR-V backend is disabled.
add_executable (vulkan_legalize_test vulkan_legalize_test.cpp vulkan_legalize.cpp)
# vulkan_legalize.cpp links the embedded device shadeops module, which lives in
# the generated .bc.cpp inside liboslexec.  Pull it in here as well when it
# was generated (it is only there when a device clang was configured).
if (VULKAN_SHADEOPS_CLANG)
    target_sources (vulkan_legalize_test PRIVATE
                    ${CMAKE_CURRENT_BINARY_DIR}/vulkan_shadeops.bc.cpp)
endif ()
target_include_directories (vulkan_legalize_test PRIVATE
                            "${CMAKE_CURRENT_SOURCE_DIR}"
                            "${CMAKE_SOURCE_DIR}/src/include"
                            ${OpenImageIO_INCLUDES})
target_link_libraries (vulkan_legalize_test PRIVATE
                       ${LLVM_LIBRARIES} ${LLVM_LDFLAGS} ${LLVM_SYSTEM_LIBRARIES}
                       ${CMAKE_DL_LIBS})
set_target_properties (vulkan_legalize_test PROPERTIES FOLDER "Unit Tests")
