use std::cmp::min;

use cudart_sys::dim3;

pub const WARP_SIZE: u32 = 32;

pub fn get_grid_block_dims_for_threads_count(
    threads_per_block: u32,
    threads_count: u32,
) -> (dim3, dim3) {
    let block_dim = min(threads_count, threads_per_block);
    let grid_dim = (threads_count + block_dim - 1) / block_dim;
    (grid_dim.into(), block_dim.into())
}
