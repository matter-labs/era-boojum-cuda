use std::mem::MaybeUninit;

use cudart_sys::{
    cudaOccupancyAvailableDynamicSMemPerBlock, cudaOccupancyMaxActiveBlocksPerMultiprocessor,
    cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags,
};

use crate::execution::Kernel;
use crate::result::{CudaResult, CudaResultWrap};

pub fn available_dynamic_smem_per_block(
    kernel: impl Kernel,
    num_blocks: i32,
    block_size: i32,
) -> CudaResult<usize> {
    let mut result = MaybeUninit::<usize>::uninit();
    unsafe {
        cudaOccupancyAvailableDynamicSMemPerBlock(
            result.as_mut_ptr(),
            kernel.get_kernel_raw(),
            num_blocks,
            block_size,
        )
        .wrap_maybe_uninit(result)
    }
}

pub fn max_active_blocks_per_multiprocessor(
    kernel: impl Kernel,
    block_size: i32,
    dynamic_smem_size: usize,
) -> CudaResult<i32> {
    let mut result = MaybeUninit::<i32>::uninit();
    unsafe {
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            result.as_mut_ptr(),
            kernel.get_kernel_raw(),
            block_size,
            dynamic_smem_size,
        )
        .wrap_maybe_uninit(result)
    }
}

pub fn max_active_blocks_per_multiprocessor_with_flags(
    kernel: impl Kernel,
    block_size: i32,
    dynamic_smem_size: usize,
    flags: u32,
) -> CudaResult<i32> {
    let mut result = MaybeUninit::<i32>::uninit();
    unsafe {
        cudaOccupancyMaxActiveBlocksPerMultiprocessorWithFlags(
            result.as_mut_ptr(),
            kernel.get_kernel_raw(),
            block_size,
            dynamic_smem_size,
            flags,
        )
        .wrap_maybe_uninit(result)
    }
}
