use boojum::field::goldilocks::GoldilocksField;
use boojum::implementations::poseidon_goldilocks_params::*;

use cudart::execution::{
    KernelFiveArgs, KernelFourArgs, KernelLaunch, KernelSevenArgs, KernelThreeArgs,
};
use cudart::result::CudaResult;
use cudart::slice::DeviceSlice;
use cudart::stream::CudaStream;
use cudart_sys::dim3;

use crate::device_structures::{
    BaseFieldDeviceType, DeviceMatrixChunkImpl, DeviceMatrixChunkMutImpl, MutPtrAndStride,
    PtrAndStride,
};
use crate::utils::{get_grid_block_dims_for_threads_count, WARP_SIZE};
use crate::BaseField;

extern "C" {
    fn poseidon_single_thread_leaves_kernel(
        values: *const GoldilocksField,
        results: *mut GoldilocksField,
        rows_count: u32,
        cols_count: u32,
        count: u32,
        load_intermediate: bool,
        store_intermediate: bool,
    );

    fn poseidon_single_thread_nodes_kernel(
        values: *const GoldilocksField,
        results: *mut GoldilocksField,
        count: u32,
    );

    fn poseidon_cooperative_nodes_kernel(
        values: *const GoldilocksField,
        results: *mut GoldilocksField,
        count: u32,
    );

    fn poseidon2_single_thread_leaves_kernel(
        values: *const GoldilocksField,
        results: *mut GoldilocksField,
        rows_count: u32,
        cols_count: u32,
        count: u32,
        load_intermediate: bool,
        store_intermediate: bool,
    );

    fn poseidon2_single_thread_nodes_kernel(
        values: *const GoldilocksField,
        results: *mut GoldilocksField,
        count: u32,
    );

    fn poseidon2_cooperative_leaves_kernel(
        values: *const GoldilocksField,
        results: *mut GoldilocksField,
        rows_count: u32,
        cols_count: u32,
        count: u32,
        load_intermediate: bool,
        store_intermediate: bool,
    );

    fn poseidon2_cooperative_nodes_kernel(
        values: *const GoldilocksField,
        results: *mut GoldilocksField,
        count: u32,
    );

    fn gather_rows_kernel(
        indexes: *const u32,
        indexes_count: u32,
        values: PtrAndStride<BaseFieldDeviceType>,
        results: MutPtrAndStride<BaseFieldDeviceType>,
    );

    fn gather_merkle_paths_kernel(
        indexes: *const u32,
        indexes_count: u32,
        values: *const GoldilocksField,
        log_leaves_count: u32,
        results: *mut GoldilocksField,
    );
}

type LeavesFn =
    unsafe extern "C" fn(*const GoldilocksField, *mut GoldilocksField, u32, u32, u32, bool, bool);

type NodesFn = unsafe extern "C" fn(*const GoldilocksField, *mut GoldilocksField, u32);

pub struct Poseidon {}

pub struct Poseidon2 {}

pub trait PoseidonRunnable {
    fn nodes_prefer_single_thread_threshold() -> u32 {
        14
    }
    fn get_grid_block_leaves_single_thread(count: u32) -> (dim3, dim3) {
        get_grid_block_dims_for_threads_count(WARP_SIZE * 2, count)
    }
    fn get_grid_block_nodes_single_thread(count: u32) -> (dim3, dim3) {
        get_grid_block_dims_for_threads_count(WARP_SIZE * 2, count)
    }
    fn unique_asserts();
    fn get_grid_block_leaves_cooperative(count: u32) -> (dim3, dim3);
    fn get_grid_block_nodes_cooperative(count: u32) -> (dim3, dim3);
    fn get_kernel_leaves_single_thread() -> LeavesFn;
    fn get_kernel_leaves_cooperative() -> LeavesFn;
    fn get_kernel_nodes_single_thread() -> NodesFn;
    fn get_kernel_nodes_cooperative() -> NodesFn;
}

impl PoseidonRunnable for Poseidon {
    fn unique_asserts() {}
    #[allow(unused_variables)]
    fn get_grid_block_leaves_cooperative(count: u32) -> (dim3, dim3) {
        unimplemented!("leaves_cooperative not implemented for Poseidon");
    }
    fn get_grid_block_nodes_cooperative(count: u32) -> (dim3, dim3) {
        let (grid_dim, mut block_dim) = get_grid_block_dims_for_threads_count(WARP_SIZE, count);
        block_dim.y = 4;
        (grid_dim, block_dim)
    }
    fn get_kernel_leaves_single_thread() -> LeavesFn {
        poseidon_single_thread_leaves_kernel
    }
    fn get_kernel_leaves_cooperative() -> LeavesFn {
        unimplemented!("leaves_cooperative not implemented for Poseidon");
    }
    fn get_kernel_nodes_single_thread() -> NodesFn {
        poseidon_single_thread_nodes_kernel
    }
    fn get_kernel_nodes_cooperative() -> NodesFn {
        poseidon_cooperative_nodes_kernel
    }
}

impl PoseidonRunnable for Poseidon2 {
    fn unique_asserts() {
        // These sizes are what we need for now.
        // I can generalize the kernels if that changes.
        assert_eq!(RATE, 8);
        assert_eq!(CAPACITY, 4);
    }
    fn get_grid_block_leaves_cooperative(count: u32) -> (dim3, dim3) {
        let (grid_dim, mut block_dim) = get_grid_block_dims_for_threads_count(WARP_SIZE, count);
        block_dim.y = 3;
        (grid_dim, block_dim)
    }
    fn get_grid_block_nodes_cooperative(count: u32) -> (dim3, dim3) {
        let (grid_dim, mut block_dim) = get_grid_block_dims_for_threads_count(WARP_SIZE, count);
        block_dim.y = 3;
        (grid_dim, block_dim)
    }
    fn get_kernel_leaves_single_thread() -> LeavesFn {
        poseidon2_single_thread_leaves_kernel
    }
    fn get_kernel_leaves_cooperative() -> LeavesFn {
        poseidon2_cooperative_leaves_kernel
    }
    fn get_kernel_nodes_single_thread() -> NodesFn {
        poseidon2_single_thread_nodes_kernel
    }
    fn get_kernel_nodes_cooperative() -> NodesFn {
        poseidon2_cooperative_nodes_kernel
    }
}

#[allow(clippy::too_many_arguments)]
pub fn launch_leaves_kernel<P: PoseidonRunnable>(
    get_kernel_fn: fn() -> LeavesFn,
    get_grid_block_fn: fn(u32) -> (dim3, dim3),
    values: &DeviceSlice<GoldilocksField>,
    results: &mut DeviceSlice<GoldilocksField>,
    log_rows_per_hash: u32,
    load_intermediate: bool,
    store_intermediate: bool,
    stream: &CudaStream,
) -> CudaResult<()> {
    P::unique_asserts();
    let values_len = values.len();
    let results_len = results.len();
    assert_eq!(results_len % CAPACITY, 0);
    let count = results_len / CAPACITY;
    assert_eq!(values_len % (count << log_rows_per_hash), 0);
    let values = values.as_ptr();
    let results = results.as_mut_ptr();
    let rows_count = 1u32 << log_rows_per_hash;
    let cols_count = values_len / (count << log_rows_per_hash);
    assert!(cols_count <= u32::MAX as usize);
    let cols_count = cols_count as u32;
    assert!(count <= u32::MAX as usize);
    // If this launch computes an intermediate result for a partial set of columns,
    // the kernels assume we'll complete a permutation for a full state before writing
    // the result for the current columns. This imposes a restriction on the number
    // of columns we may include in the partial set.
    assert!(!store_intermediate || ((rows_count * cols_count) % RATE as u32 == 0));
    let count = count as u32;
    let kernel = get_kernel_fn();
    let (grid_dim, block_dim) = get_grid_block_fn(count);
    let args = (
        &values,
        &results,
        &rows_count,
        &cols_count,
        &count,
        &load_intermediate,
        &store_intermediate,
    );
    unsafe { KernelSevenArgs::launch(kernel, grid_dim, block_dim, args, 0, stream) }
}

pub fn launch_single_thread_leaves_kernel<P: PoseidonRunnable>(
    values: &DeviceSlice<GoldilocksField>,
    results: &mut DeviceSlice<GoldilocksField>,
    log_rows_per_hash: u32,
    load_intermediate: bool,
    store_intermediate: bool,
    stream: &CudaStream,
) -> CudaResult<()> {
    let get_kernel_fn = P::get_kernel_leaves_single_thread;
    let get_grid_block_fn = P::get_grid_block_leaves_single_thread;
    launch_leaves_kernel::<P>(
        get_kernel_fn,
        get_grid_block_fn,
        values,
        results,
        log_rows_per_hash,
        load_intermediate,
        store_intermediate,
        stream,
    )
}

pub fn launch_cooperative_leaves_kernel<P: PoseidonRunnable>(
    values: &DeviceSlice<GoldilocksField>,
    results: &mut DeviceSlice<GoldilocksField>,
    log_rows_per_hash: u32,
    load_intermediate: bool,
    store_intermediate: bool,
    stream: &CudaStream,
) -> CudaResult<()> {
    let get_kernel_fn = P::get_kernel_leaves_cooperative;
    let get_grid_block_fn = P::get_grid_block_leaves_cooperative;
    launch_leaves_kernel::<P>(
        get_kernel_fn,
        get_grid_block_fn,
        values,
        results,
        log_rows_per_hash,
        load_intermediate,
        store_intermediate,
        stream,
    )
}

const NODE_REDUCTION_FACTOR: usize = 2;

pub fn launch_single_thread_nodes_kernel<P: PoseidonRunnable>(
    values: &DeviceSlice<GoldilocksField>,
    results: &mut DeviceSlice<GoldilocksField>,
    stream: &CudaStream,
) -> CudaResult<()> {
    P::unique_asserts();
    assert_eq!(RATE, CAPACITY * NODE_REDUCTION_FACTOR);
    let values_len = values.len();
    let results_len = results.len();
    assert_eq!(values_len % RATE, 0);
    assert_eq!(results_len % CAPACITY, 0);
    assert_eq!(values_len, results_len * NODE_REDUCTION_FACTOR);
    let values = values.as_ptr();
    let results = results.as_mut_ptr();
    assert!(results_len / CAPACITY <= u32::MAX as usize);
    let count = (results_len / CAPACITY) as u32;
    let (grid_dim, block_dim) = P::get_grid_block_nodes_single_thread(count);
    let args = (&values, &results, &count);
    unsafe {
        KernelThreeArgs::launch(
            P::get_kernel_nodes_single_thread(),
            grid_dim,
            block_dim,
            args,
            0,
            stream,
        )
    }
}

pub fn launch_cooperative_nodes_kernel<P: PoseidonRunnable>(
    values: &DeviceSlice<GoldilocksField>,
    results: &mut DeviceSlice<GoldilocksField>,
    stream: &CudaStream,
) -> CudaResult<()> {
    P::unique_asserts();
    assert_eq!(RATE, CAPACITY * NODE_REDUCTION_FACTOR);
    let values_len = values.len();
    let results_len = results.len();
    assert_eq!(values_len % RATE, 0);
    assert_eq!(results_len % CAPACITY, 0);
    assert_eq!(values_len, results_len * NODE_REDUCTION_FACTOR);
    let values = values.as_ptr();
    let results = results.as_mut_ptr();
    assert!(results_len / CAPACITY <= u32::MAX as usize);
    let count = (results_len / CAPACITY) as u32;
    let (grid_dim, block_dim) = P::get_grid_block_nodes_cooperative(count);
    let args = (&values, &results, &count);
    unsafe {
        KernelThreeArgs::launch(
            P::get_kernel_nodes_cooperative(),
            grid_dim,
            block_dim,
            args,
            0,
            stream,
        )
    }
}

pub fn build_merkle_tree_nodes<P: PoseidonRunnable>(
    values: &DeviceSlice<GoldilocksField>,
    results: &mut DeviceSlice<GoldilocksField>,
    layers_count: u32,
    stream: &CudaStream,
) -> CudaResult<()> {
    if layers_count == 0 {
        Ok(())
    } else {
        assert_eq!(NODE_REDUCTION_FACTOR, 2);
        let values_len = values.len();
        let results_len = results.len();
        assert_eq!(values_len % RATE, 0);
        let layer = (values_len / RATE).trailing_zeros();
        assert_eq!(values_len, RATE << layer);
        assert_eq!(values_len, results_len);
        let (nodes, nodes_remaining) = results.split_at_mut(results_len >> 1);
        if layer > P::nodes_prefer_single_thread_threshold() {
            launch_single_thread_nodes_kernel::<P>(values, nodes, stream)?;
        } else {
            launch_cooperative_nodes_kernel::<P>(values, nodes, stream)?;
        }
        build_merkle_tree_nodes::<P>(nodes, nodes_remaining, layers_count - 1, stream)
    }
}

pub fn build_merkle_tree_leaves<P: PoseidonRunnable>(
    values: &DeviceSlice<GoldilocksField>,
    results: &mut DeviceSlice<GoldilocksField>,
    log_rows_per_hash: u32,
    load_intermediate: bool,
    store_intermediate: bool,
    stream: &CudaStream,
) -> CudaResult<()> {
    let values_len = values.len();
    let results_len = results.len();
    assert_eq!(results_len % CAPACITY, 0);
    let leaves_count = results_len / CAPACITY;
    assert_eq!(values_len % leaves_count, 0);
    launch_single_thread_leaves_kernel::<P>(
        values,
        results,
        log_rows_per_hash,
        load_intermediate,
        store_intermediate,
        stream,
    )
}

pub fn build_merkle_tree<P: PoseidonRunnable>(
    values: &DeviceSlice<GoldilocksField>,
    results: &mut DeviceSlice<GoldilocksField>,
    log_rows_per_hash: u32,
    stream: &CudaStream,
    layers_count: u32,
) -> CudaResult<()> {
    assert_ne!(layers_count, 0);
    let values_len = values.len();
    let results_len = results.len();
    assert_eq!(results_len % (2 * CAPACITY), 0);
    let leaves_count = results_len / (2 * CAPACITY);
    assert!(1 << (layers_count - 1) <= leaves_count);
    assert_eq!(values_len % leaves_count, 0);
    let (nodes, nodes_remaining) = results.split_at_mut(results.len() >> 1);
    build_merkle_tree_leaves::<P>(values, nodes, log_rows_per_hash, false, false, stream)?;
    build_merkle_tree_nodes::<P>(nodes, nodes_remaining, layers_count - 1, stream)
}

pub fn gather_rows(
    indexes: &DeviceSlice<u32>,
    log_rows_per_index: u32,
    values: &(impl DeviceMatrixChunkImpl<BaseField> + ?Sized),
    result: &mut (impl DeviceMatrixChunkMutImpl<BaseField> + ?Sized),
    stream: &CudaStream,
) -> CudaResult<()> {
    let indexes_len = indexes.len();
    let values_cols = values.cols();
    let result_rows = result.rows();
    let result_cols = result.cols();
    let rows_per_index = 1 << log_rows_per_index;
    assert!(log_rows_per_index < WARP_SIZE);
    assert_eq!(result_cols, values_cols);
    assert_eq!(result_rows, indexes_len << log_rows_per_index);
    assert!(indexes_len <= u32::MAX as usize);
    let indexes_count = indexes_len as u32;
    let (mut grid_dim, block_dim) =
        get_grid_block_dims_for_threads_count(WARP_SIZE >> log_rows_per_index, indexes_count);
    let block_dim = (rows_per_index, block_dim.x).into();
    assert!(result_cols <= u32::MAX as usize);
    grid_dim.y = result_cols as u32;
    let indexes = indexes.as_ptr();
    let values = values.as_ptr_and_stride();
    let result = result.as_mut_ptr_and_stride();
    let args = (&indexes, &indexes_count, &values, &result);
    unsafe { KernelFourArgs::launch(gather_rows_kernel, grid_dim, block_dim, args, 0, stream) }
}

pub fn gather_merkle_paths(
    indexes: &DeviceSlice<u32>,
    values: &DeviceSlice<GoldilocksField>,
    results: &mut DeviceSlice<GoldilocksField>,
    layers_count: u32,
    stream: &CudaStream,
) -> CudaResult<()> {
    assert!(indexes.len() <= u32::MAX as usize);
    let indexes_count = indexes.len() as u32;
    assert_eq!(values.len() % CAPACITY, 0);
    let values_count = values.len() / CAPACITY;
    assert!(values_count.is_power_of_two());
    let log_values_count = values_count.trailing_zeros();
    assert_ne!(log_values_count, 0);
    let log_leaves_count = log_values_count - 1;
    assert!(layers_count < log_leaves_count);
    assert_eq!(
        indexes.len() * layers_count as usize * CAPACITY,
        results.len()
    );
    let (grid_dim, block_dim) = get_grid_block_dims_for_threads_count(WARP_SIZE, indexes_count);
    let grid_dim = (grid_dim.x, CAPACITY as u32, layers_count).into();
    let indexes = indexes.as_ptr();
    let values = values.as_ptr();
    let result = results.as_mut_ptr();
    let args = (
        &indexes,
        &indexes_count,
        &values,
        &log_leaves_count,
        &result,
    );
    unsafe {
        KernelFiveArgs::launch(
            gather_merkle_paths_kernel,
            grid_dim,
            block_dim,
            args,
            0,
            stream,
        )
    }
}

#[cfg(test)]
mod tests {
    use std::cmp;

    use boojum::field::{Field, U64Representable};
    use boojum::implementations::poseidon2::state_generic_impl::State;
    use itertools::Itertools;
    use rand::Rng;

    use cudart::memory::{memory_copy_async, DeviceAllocation};
    use cudart::slice::CudaSlice;

    // use boojum::implementations::poseidon_goldilocks::poseidon_permutation_optimized;
    use crate::device_structures::{DeviceMatrix, DeviceMatrixMut};
    use crate::ops_simple::set_to_zero;

    use super::*;

    pub trait PoseidonTestable: PoseidonRunnable {
        fn poseidon_permutation_cpu_shim(state: &mut [GoldilocksField; STATE_WIDTH]);
    }

    // // Maybe we should ask for poseidon2 in boojum to add a functional API similar to
    // // poseidon_permutation_optimized, in which case we won't need these shims
    // impl PoseidonTestable for Poseidon {
    //     fn poseidon_permutation_cpu_shim(state: &mut [GoldilocksField; STATE_WIDTH]) {
    //         poseidon_permutation_optimized(state);
    //     }
    // }

    impl PoseidonTestable for Poseidon2 {
        fn poseidon_permutation_cpu_shim(state: &mut [GoldilocksField; STATE_WIDTH]) {
            let mut state_vec = State::from_field_array(*state);
            state_vec.poseidon2_permutation();
            state.copy_from_slice(&state_vec.0);
        }
    }

    fn verify_hash(
        state: &[GoldilocksField],
        results: &[GoldilocksField],
        offset: usize,
        stride: usize,
    ) {
        for i in 0..CAPACITY {
            let left = state[i];
            let right = results[i * stride + offset];
            assert_eq!(left.as_u64(), right.as_u64());
        }
    }

    fn verify_leaves<P: PoseidonTestable>(
        values: &[GoldilocksField],
        results: &[GoldilocksField],
        log_rows_per_hash: u32,
    ) {
        let results_len = results.len();
        assert_eq!(results_len % CAPACITY, 0);
        let count = results_len / CAPACITY;
        let values_len = values.len();
        assert_eq!(values_len % (count << log_rows_per_hash), 0);
        let cols_count = values_len / (count << log_rows_per_hash);
        let rows_count = 1 << log_rows_per_hash;
        let values_per_hash = cols_count << log_rows_per_hash;
        for i in 0..count {
            let mut state = [GoldilocksField::ZERO; STATE_WIDTH];
            for j in 0..((values_per_hash + RATE - 1) / RATE) {
                for (k, s) in state.iter_mut().enumerate().take(RATE) {
                    let idx = j * RATE + k;
                    *s = if idx < values_per_hash {
                        let row = idx % rows_count;
                        let col = idx / rows_count;
                        values[(i << log_rows_per_hash) + row + col * rows_count * count]
                    } else {
                        GoldilocksField::ZERO
                    };
                }
                // Inefficient for poseidon2, but convenient
                P::poseidon_permutation_cpu_shim(&mut state);
            }
            verify_hash(&state, results, i, count);
        }
    }

    fn verify_nodes<P: PoseidonTestable>(values: &[GoldilocksField], results: &[GoldilocksField]) {
        let values_len = values.len();
        assert_eq!(values_len % (CAPACITY * NODE_REDUCTION_FACTOR), 0);
        let count = values_len / (CAPACITY * NODE_REDUCTION_FACTOR);
        assert_eq!(results.len(), count * CAPACITY);
        for i in 0..count {
            let mut state = [GoldilocksField::ZERO; STATE_WIDTH];
            for j in 0..NODE_REDUCTION_FACTOR {
                for k in 0..CAPACITY {
                    let state_offset = j * CAPACITY + k;
                    let value_offset = (k * count + i) * 2 + j;
                    state[state_offset] = values[value_offset];
                }
            }
            P::poseidon_permutation_cpu_shim(&mut state);
            verify_hash(&state, results, i, count);
        }
    }

    #[allow(clippy::type_complexity)]
    fn test_leaves<P: PoseidonTestable>(
        launch: fn(
            &DeviceSlice<GoldilocksField>,
            &mut DeviceSlice<GoldilocksField>,
            u32,
            bool,
            bool,
            &CudaStream,
        ) -> CudaResult<()>,
        checkpointed: bool,
    ) {
        const LOG_N: usize = 6;
        const N: usize = 1 << LOG_N;
        const VALUES_PER_ROW: usize = 125;
        const LOG_ROWS_PER_HASH: u32 = 1;
        const COL_CHUNK: usize = 8;
        let mut values_host = [GoldilocksField::ZERO; (N * VALUES_PER_ROW) << LOG_ROWS_PER_HASH];
        let mut rng = rand::thread_rng();
        values_host.fill_with(|| GoldilocksField(rng.gen()));
        let mut results_host = [GoldilocksField::ZERO; N * CAPACITY];
        let stream = CudaStream::default();
        let mut values_device =
            DeviceAllocation::<GoldilocksField>::alloc(values_host.len()).unwrap();
        let mut results_device =
            DeviceAllocation::<GoldilocksField>::alloc(results_host.len()).unwrap();
        memory_copy_async(&mut values_device, &values_host, &stream).unwrap();
        if checkpointed {
            for start_col in (0..VALUES_PER_ROW).step_by(COL_CHUNK) {
                let end_col = start_col + cmp::min(COL_CHUNK, VALUES_PER_ROW - start_col);
                let start_mem_idx = (start_col * N) << LOG_ROWS_PER_HASH;
                let end_mem_idx = (end_col * N) << LOG_ROWS_PER_HASH;
                launch(
                    &values_device[start_mem_idx..end_mem_idx],
                    &mut results_device,
                    LOG_ROWS_PER_HASH,
                    start_col != 0,
                    end_col != VALUES_PER_ROW,
                    &stream,
                )
                .unwrap();
            }
        } else {
            launch(
                &values_device,
                &mut results_device,
                LOG_ROWS_PER_HASH,
                false,
                false,
                &stream,
            )
            .unwrap();
        }
        memory_copy_async(&mut results_host, &results_device, &stream).unwrap();
        stream.synchronize().unwrap();
        verify_leaves::<P>(&values_host, &results_host, LOG_ROWS_PER_HASH);
    }

    // #[test]
    // fn poseidon_single_thread_leaves() {
    //     test_leaves::<Poseidon>(launch_single_thread_leaves_kernel::<Poseidon>);
    // }
    //
    // #[test]
    // #[should_panic(expected = "leaves_cooperative not implemented for Poseidon")]
    // fn poseidon_cooperative_leaves() {
    //     test_leaves::<Poseidon>(launch_cooperative_leaves_kernel::<Poseidon>);
    // }

    #[test]
    fn poseidon2_single_thread_leaves() {
        test_leaves::<Poseidon2>(launch_single_thread_leaves_kernel::<Poseidon2>, false);
    }

    #[test]
    fn poseidon2_single_thread_leaves_checkpointed() {
        test_leaves::<Poseidon2>(launch_single_thread_leaves_kernel::<Poseidon2>, true);
    }

    #[test]
    fn poseidon2_cooperative_leaves() {
        test_leaves::<Poseidon2>(launch_cooperative_leaves_kernel::<Poseidon2>, false);
    }

    #[test]
    fn poseidon2_cooperative_leaves_checkpointed() {
        test_leaves::<Poseidon2>(launch_cooperative_leaves_kernel::<Poseidon2>, true);
    }

    fn test_nodes<P: PoseidonTestable>(
        launch: fn(
            &DeviceSlice<GoldilocksField>,
            &mut DeviceSlice<GoldilocksField>,
            &CudaStream,
        ) -> CudaResult<()>,
    ) {
        const LOG_N: usize = 10;
        const N: usize = 1 << LOG_N;
        let mut values_host = [GoldilocksField::ZERO; N * RATE];
        let mut rng = rand::thread_rng();
        values_host.fill_with(|| GoldilocksField(rng.gen()));
        let mut results_host = [GoldilocksField::ZERO; N * CAPACITY];
        let stream = CudaStream::default();
        let mut values_device =
            DeviceAllocation::<GoldilocksField>::alloc(values_host.len()).unwrap();
        let mut results_device =
            DeviceAllocation::<GoldilocksField>::alloc(results_host.len()).unwrap();
        memory_copy_async(&mut values_device, &values_host, &stream).unwrap();
        launch(&values_device, &mut results_device, &stream).unwrap();
        memory_copy_async(&mut results_host, &results_device, &stream).unwrap();
        stream.synchronize().unwrap();
        verify_nodes::<P>(&values_host, &results_host);
    }

    // #[test]
    // fn poseidon_single_thread_nodes() {
    //     test_nodes::<Poseidon>(launch_single_thread_nodes_kernel::<Poseidon>);
    // }
    //
    // #[test]
    // fn poseidon_cooperative_nodes() {
    //     test_nodes::<Poseidon>(launch_cooperative_nodes_kernel::<Poseidon>);
    // }

    #[test]
    fn poseidon2_single_thread_nodes() {
        test_nodes::<Poseidon2>(launch_single_thread_nodes_kernel::<Poseidon2>);
    }

    #[test]
    fn poseidon2_cooperative_nodes() {
        test_nodes::<Poseidon2>(launch_cooperative_nodes_kernel::<Poseidon2>);
    }

    fn verify_tree_nodes<P: PoseidonTestable>(
        values: &[GoldilocksField],
        results: &[GoldilocksField],
        layers_count: u32,
    ) {
        assert_eq!(values.len(), results.len());
        if layers_count == 0 {
            assert!(results.iter().all(|x| x.is_zero()));
        } else {
            let (nodes, nodes_remaining) = results.split_at(results.len() >> 1);
            verify_nodes::<P>(values, nodes);
            verify_tree_nodes::<P>(nodes, nodes_remaining, layers_count - 1);
        }
    }

    #[allow(non_snake_case)]
    fn merkle_tree<P: PoseidonTestable>(LOG_N: usize) {
        const VALUES_PER_ROW: usize = 125;
        let N: usize = 1 << LOG_N;
        let LAYERS_COUNT: u32 = (LOG_N + 1) as u32;
        let mut values_host = vec![GoldilocksField::ZERO; N * VALUES_PER_ROW];
        let mut rng = rand::thread_rng();
        values_host.fill_with(|| GoldilocksField(rng.gen()));
        let mut results_host = vec![GoldilocksField::ZERO; N * CAPACITY * 2];
        let stream = CudaStream::default();
        let mut values_device =
            DeviceAllocation::<GoldilocksField>::alloc(values_host.len()).unwrap();
        let mut results_device =
            DeviceAllocation::<GoldilocksField>::alloc(results_host.len()).unwrap();
        set_to_zero(&mut results_device, &stream).unwrap();
        memory_copy_async(&mut values_device, &values_host, &stream).unwrap();
        build_merkle_tree::<P>(
            &values_device,
            &mut results_device,
            0,
            &stream,
            LAYERS_COUNT,
        )
        .unwrap();
        memory_copy_async(&mut results_host, &results_device, &stream).unwrap();
        stream.synchronize().unwrap();
        let (nodes, nodes_remaining) = results_host.split_at(results_host.len() >> 1);
        verify_leaves::<P>(&values_host, nodes, 0);
        verify_tree_nodes::<P>(nodes, nodes_remaining, LAYERS_COUNT - 1);
    }

    // #[test]
    // fn merkle_tree_poseidon() {
    //     merkle_tree::<Poseidon>(8);
    // }

    #[test]
    fn merkle_tree_poseidon2() {
        merkle_tree::<Poseidon2>(8);
    }

    // #[test]
    // #[ignore]
    // fn merkle_tree_poseidon_large() {
    //     merkle_tree::<Poseidon>((Poseidon::nodes_prefer_single_thread_threshold() + 3) as usize);
    // }

    #[test]
    #[ignore]
    fn merkle_tree_poseidon2_large() {
        merkle_tree::<Poseidon2>((Poseidon2::nodes_prefer_single_thread_threshold() + 3) as usize);
    }

    fn cooperative_matches_single_thread_leaves<P: PoseidonTestable>() {
        const LOG_N: usize = 6;
        const N: usize = 1 << LOG_N;
        const VALUES_PER_ROW: usize = 125;
        const LOG_ROWS_PER_HASH: u32 = 1;
        let mut values_host = [GoldilocksField::ZERO; (N * VALUES_PER_ROW) << LOG_ROWS_PER_HASH];
        let mut rng = rand::thread_rng();
        values_host.fill_with(|| GoldilocksField(rng.gen()));
        let mut results_host_single_thread = vec![GoldilocksField::ZERO; N * CAPACITY];
        let mut results_host_cooperative = vec![GoldilocksField::ZERO; N * CAPACITY];
        let stream = CudaStream::default();
        let mut values_device =
            DeviceAllocation::<GoldilocksField>::alloc(values_host.len()).unwrap();
        let mut results_device_single_thread =
            DeviceAllocation::<GoldilocksField>::alloc(results_host_single_thread.len()).unwrap();
        let mut results_device_cooperative =
            DeviceAllocation::<GoldilocksField>::alloc(results_host_cooperative.len()).unwrap();
        memory_copy_async(&mut values_device, &values_host, &stream).unwrap();
        launch_single_thread_leaves_kernel::<P>(
            &values_device,
            &mut results_device_single_thread,
            LOG_ROWS_PER_HASH,
            false,
            false,
            &stream,
        )
        .unwrap();
        launch_cooperative_leaves_kernel::<P>(
            &values_device,
            &mut results_device_cooperative,
            LOG_ROWS_PER_HASH,
            false,
            false,
            &stream,
        )
        .unwrap();
        memory_copy_async(
            &mut results_host_single_thread,
            &results_device_single_thread,
            &stream,
        )
        .unwrap();
        memory_copy_async(
            &mut results_host_cooperative,
            &results_device_cooperative,
            &stream,
        )
        .unwrap();
        stream.synchronize().unwrap();
        for i in 0..results_host_single_thread.len() {
            assert_eq!(results_host_single_thread[i], results_host_cooperative[i]);
        }
    }

    // #[test]
    // #[should_panic(expected = "leaves_cooperative not implemented for Poseidon")]
    // fn poseidon_cooperative_matches_single_thread_leaves() {
    //     cooperative_matches_single_thread_leaves::<Poseidon>();
    // }

    #[test]
    fn poseidon2_cooperative_matches_single_thread_leaves() {
        cooperative_matches_single_thread_leaves::<Poseidon2>();
    }

    fn cooperative_matches_single_thread_nodes<P: PoseidonTestable>() {
        const LOG_N: usize = 12;
        const N: usize = 1 << LOG_N;
        let mut values_host = vec![GoldilocksField::ZERO; N * RATE];
        let mut rng = rand::thread_rng();
        values_host.fill_with(|| GoldilocksField(rng.gen()));
        let mut results_host_single_thread = vec![GoldilocksField::ZERO; N * CAPACITY];
        let mut results_host_cooperative = vec![GoldilocksField::ZERO; N * CAPACITY];
        let stream = CudaStream::default();
        let mut values_device =
            DeviceAllocation::<GoldilocksField>::alloc(values_host.len()).unwrap();
        let mut results_device_single_thread =
            DeviceAllocation::<GoldilocksField>::alloc(results_host_single_thread.len()).unwrap();
        let mut results_device_cooperative =
            DeviceAllocation::<GoldilocksField>::alloc(results_host_cooperative.len()).unwrap();
        memory_copy_async(&mut values_device, &values_host, &stream).unwrap();
        launch_single_thread_nodes_kernel::<P>(
            &values_device,
            &mut results_device_single_thread,
            &stream,
        )
        .unwrap();
        launch_cooperative_nodes_kernel::<P>(
            &values_device,
            &mut results_device_cooperative,
            &stream,
        )
        .unwrap();
        memory_copy_async(
            &mut results_host_single_thread,
            &results_device_single_thread,
            &stream,
        )
        .unwrap();
        memory_copy_async(
            &mut results_host_cooperative,
            &results_device_cooperative,
            &stream,
        )
        .unwrap();
        stream.synchronize().unwrap();
        for i in 0..results_host_single_thread.len() {
            assert_eq!(results_host_single_thread[i], results_host_cooperative[i]);
        }
    }

    // #[test]
    // fn poseidon_cooperative_matches_single_thread_nodes() {
    //     cooperative_matches_single_thread_nodes::<Poseidon>();
    // }

    #[test]
    fn poseidon2_cooperative_matches_single_thread_nodes() {
        cooperative_matches_single_thread_nodes::<Poseidon2>();
    }

    #[test]
    fn gather_rows() {
        const SRC_LOG_ROWS: usize = 12;
        const SRC_ROWS: usize = 1 << SRC_LOG_ROWS;
        const COLS: usize = 16;
        const INDEXES_COUNT: usize = 42;
        const LOG_ROWS_PER_INDEX: usize = 1;
        const DST_ROWS: usize = INDEXES_COUNT << LOG_ROWS_PER_INDEX;
        let mut rng = rand::thread_rng();
        let mut indexes_host = vec![0; INDEXES_COUNT];
        indexes_host.fill_with(|| rng.gen_range(0..INDEXES_COUNT as u32));
        let mut values_host = vec![GoldilocksField::ZERO; SRC_ROWS * COLS];
        values_host.fill_with(|| GoldilocksField(rng.gen()));
        let mut results_host = vec![GoldilocksField::ZERO; DST_ROWS * COLS];
        let stream = CudaStream::default();
        let mut indexes_device = DeviceAllocation::<u32>::alloc(indexes_host.len()).unwrap();
        let mut values_device =
            DeviceAllocation::<GoldilocksField>::alloc(values_host.len()).unwrap();
        let mut results_device =
            DeviceAllocation::<GoldilocksField>::alloc(results_host.len()).unwrap();
        memory_copy_async(&mut indexes_device, &indexes_host, &stream).unwrap();
        memory_copy_async(&mut values_device, &values_host, &stream).unwrap();
        super::gather_rows(
            &indexes_device,
            LOG_ROWS_PER_INDEX as u32,
            &DeviceMatrix::new(&values_device, SRC_ROWS),
            &mut DeviceMatrixMut::new(&mut results_device, DST_ROWS),
            &stream,
        )
        .unwrap();
        memory_copy_async(&mut results_host, &results_device, &stream).unwrap();
        stream.synchronize().unwrap();
        for (i, index) in indexes_host.into_iter().enumerate() {
            let src_index = (index as usize) << LOG_ROWS_PER_INDEX;
            let dst_index = i << LOG_ROWS_PER_INDEX;
            for j in 0..1 << LOG_ROWS_PER_INDEX {
                let src_index = src_index + j;
                let dst_index = dst_index + j;
                for k in 0..COLS {
                    let expected = values_host[(k << SRC_LOG_ROWS) + src_index];
                    let actual = results_host[(k * DST_ROWS) + dst_index];
                    assert_eq!(expected, actual);
                }
            }
        }
    }

    #[test]
    fn gather_merkle_paths() {
        const LOG_LEAVES_COUNT: usize = 12;
        const INDEXES_COUNT: usize = 42;
        const LAYERS_COUNT: usize = LOG_LEAVES_COUNT - 4;
        let mut rng = rand::thread_rng();
        let mut indexes_host = vec![0; INDEXES_COUNT];
        indexes_host.fill_with(|| rng.gen_range(0..1u32 << LOG_LEAVES_COUNT));
        let mut values_host = vec![GoldilocksField::ZERO; CAPACITY << (LOG_LEAVES_COUNT + 1)];
        values_host.fill_with(|| GoldilocksField(rng.gen()));
        let mut results_host = vec![GoldilocksField::ZERO; CAPACITY * INDEXES_COUNT * LAYERS_COUNT];
        let stream = CudaStream::default();
        let mut indexes_device = DeviceAllocation::<u32>::alloc(indexes_host.len()).unwrap();
        let mut values_device =
            DeviceAllocation::<GoldilocksField>::alloc(values_host.len()).unwrap();
        let mut results_device =
            DeviceAllocation::<GoldilocksField>::alloc(results_host.len()).unwrap();
        memory_copy_async(&mut indexes_device, &indexes_host, &stream).unwrap();
        memory_copy_async(&mut values_device, &values_host, &stream).unwrap();
        super::gather_merkle_paths(
            &indexes_device,
            &values_device,
            &mut results_device,
            LAYERS_COUNT as u32,
            &stream,
        )
        .unwrap();
        memory_copy_async(&mut results_host, &results_device, &stream).unwrap();
        stream.synchronize().unwrap();
        fn verify_merkle_path(
            indexes: &[u32],
            values: &[GoldilocksField],
            results: &[GoldilocksField],
        ) {
            let (values, values_next) = values.split_at(values.len() >> 1);
            let (results, results_next) = results.split_at(INDEXES_COUNT * CAPACITY);
            values
                .chunks(values.len() / CAPACITY)
                .zip(results.chunks(results.len() / CAPACITY))
                .for_each(|(values, results)| {
                    for (row_index, &index) in indexes.iter().enumerate() {
                        let sibling_index = index ^ 1;
                        let expected = values[sibling_index as usize];
                        let actual = results[row_index];
                        assert_eq!(expected, actual);
                    }
                });
            if !results_next.is_empty() {
                let indexes_next = indexes.iter().map(|&x| x >> 1).collect_vec();
                verify_merkle_path(&indexes_next, &values_next, &results_next);
            }
        }
        verify_merkle_path(&indexes_host, &values_host, &results_host);
    }
}
