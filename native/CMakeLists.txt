cmake_minimum_required(VERSION 3.24)
project(boojum_cuda_native)
enable_language(CUDA)
if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES native)
endif ()
add_library(boojum_cuda_native STATIC
        ops_cub/common.cuh
        ops_cub/device_radix_sort.cu
        ops_cub/device_reduce.cu
        ops_cub/device_run_length_encode.cu
        ops_cub/device_scan.cu
        barycentric.cu
        blake2s.cu
        carry_chain.cuh
        common.cuh
        context.cu
        context.cuh
        ${CMAKE_INSTALL_PREFIX}/gate_kernels.cuh
        gates.cu
        gates.cuh
        gates_poseidon.cuh
        goldilocks.cuh
        goldilocks_extension.cu
        goldilocks_extension.cuh
        memory.cuh
        ntt.cu
        ntt_b2n.cuh
        ntt_n2b.cuh
        ops_complex.cu
        ops_complex.cuh
        ops_simple.cu
        poseidon2_cooperative.cu
        poseidon2_single_thread.cu
        poseidon2_single_thread.cuh
        poseidon_common.cu
        ${CMAKE_INSTALL_PREFIX}/poseidon_constants.cuh
        poseidon_cooperative.cu
        poseidon_single_thread.cu
        poseidon_single_thread.cuh
        poseidon_utils.cuh
        ptx.cuh
)
target_include_directories(boojum_cuda_native PRIVATE ${CMAKE_INSTALL_PREFIX})
set_target_properties(boojum_cuda_native PROPERTIES CUDA_STANDARD 17)
set_target_properties(boojum_cuda_native PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
set_target_properties(boojum_cuda_native PROPERTIES CUDA_RESOLVE_DEVICE_SYMBOLS ON)
target_compile_options(boojum_cuda_native PRIVATE --expt-relaxed-constexpr)
target_compile_options(boojum_cuda_native PRIVATE --ptxas-options=-v)
#target_compile_options(boojum_cuda_native PRIVATE -lineinfo)
#target_compile_options(boojum_cuda_native PRIVATE --keep)
install(TARGETS boojum_cuda_native DESTINATION .)
