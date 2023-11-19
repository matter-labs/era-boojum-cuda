use crate::device_structures::{
    DeviceMatrixChunkImpl, DeviceMatrixChunkMutImpl, DeviceMatrixImpl, DeviceMatrixMutImpl,
    DeviceRepr, DeviceVectorImpl, DeviceVectorMutImpl, MutPtrAndStride, PtrAndStride,
};
use crate::extension_field::{ExtensionField, VectorizedExtensionField};
use crate::ops_cub::device_radix_sort::{get_sort_pairs_temp_storage_bytes, sort_pairs};
use crate::ops_cub::device_run_length_encode::{encode, get_encode_temp_storage_bytes};
use crate::ops_cub::device_scan::{get_scan_temp_storage_bytes, scan_in_place, ScanOperation};
use crate::ops_simple::{set_by_val, set_to_zero};
use crate::utils::{get_grid_block_dims_for_threads_count, WARP_SIZE};
use crate::BaseField;
use cudart::execution::{CudaLaunchConfig, Dim3, KernelFunction};
use cudart::memory::memory_copy_async;
use cudart::paste::paste;
use cudart::result::CudaResult;
use cudart::slice::{DeviceSlice, DeviceVariable};
use cudart::stream::CudaStream;
use cudart::{cuda_kernel, cuda_kernel_declaration, cuda_kernel_signature_arguments_and_function};
use itertools::max;

type BF = BaseField;
type EF = VectorizedExtensionField;

fn get_launch_dims(count: u32) -> (Dim3, Dim3) {
    get_grid_block_dims_for_threads_count(WARP_SIZE * 4, count)
}

cuda_kernel!(
    GetPowerOf,
    get_powers_of_kernel,
    log_degree: u32,
    offset: u32,
    inverse: bool,
    bit_reverse: bool,
    result: *mut BF,
    count: u32,
);

fn get_powers_of(
    kernel_function: GetPowerOfSignature,
    log_degree: u32,
    offset: u32,
    inverse: bool,
    bit_reverse: bool,
    result: &mut DeviceSlice<BF>,
    stream: &CudaStream,
) -> CudaResult<()> {
    assert!(result.len() <= u32::MAX as usize);
    let count = result.len() as u32;
    let (grid_dim, block_dim) = get_launch_dims(count);
    let result = result.as_mut_ptr();
    let config = CudaLaunchConfig::basic(grid_dim, block_dim, stream);
    let args = GetPowerOfArguments::new(log_degree, offset, inverse, bit_reverse, result, count);
    GetPowerOfFunction(kernel_function).launch(&config, &args)
}

macro_rules! get_powers_of_impl {
    ($x:ident) => {
        paste! {
            get_powers_of_kernel!([<get_powers_of_ $x _kernel>]);
            pub fn [<get_powers_of_ $x>](
                log_degree: u32,
                offset: u32,
                inverse: bool,
                bit_reverse: bool,
                result: &mut DeviceSlice<BF>,
                stream: &CudaStream,
            ) -> CudaResult<()> {
                get_powers_of(
                    [<get_powers_of_ $x _kernel>],
                    log_degree,
                    offset,
                    inverse,
                    bit_reverse,
                    result,
                    stream,
                )
            }

        }
    };
}

get_powers_of_impl!(w);
get_powers_of_impl!(g);

cuda_kernel_signature_arguments_and_function!(
    GetPowersByVal<T>,
    base: T,
    offset: u32,
    bit_reverse: bool,
    result: *mut T,
    count: u32,
);

macro_rules! get_powers_by_val_kernel {
    ($type:ty) => {
        paste! {
            cuda_kernel_declaration!(
                [<get_powers_by_val_ $type:lower _kernel>](
                    base: $type,
                    offset: u32,
                    bit_reverse: bool,
                    result: *mut $type,
                    count: u32,
                )
            );
        }
    };
}

pub trait GetPowersByVal: Sized {
    const KERNEL_FUNCTION: GetPowersByValSignature<Self>;
}

pub fn get_powers_by_val<T: GetPowersByVal>(
    base: T,
    offset: u32,
    bit_reverse: bool,
    result: &mut DeviceSlice<T>,
    stream: &CudaStream,
) -> CudaResult<()> {
    assert!(result.len() <= u32::MAX as usize);
    let count = result.len() as u32;
    let (grid_dim, block_dim) = get_launch_dims(count);
    let result = result.as_mut_ptr();
    let config = CudaLaunchConfig::basic(grid_dim, block_dim, stream);
    let args = GetPowersByValArguments::new(base, offset, bit_reverse, result, count);
    GetPowersByValFunction(T::KERNEL_FUNCTION).launch(&config, &args)
}

macro_rules! get_powers_by_val_impl {
    ($type:ty) => {
        paste! {
            get_powers_by_val_kernel!($type);
            impl GetPowersByVal for $type {
                const KERNEL_FUNCTION: GetPowersByValSignature<Self> = [<get_powers_by_val_ $type:lower _kernel>];
            }
        }
    };
}

get_powers_by_val_impl!(BF);
get_powers_by_val_impl!(EF);

cuda_kernel_signature_arguments_and_function!(
    GetPowersByRef<T>,
    base: *const T,
    offset: u32,
    bit_reverse: bool,
    result: *mut T,
    count: u32,
);

macro_rules! get_powers_by_ref_kernel {
    ($type:ty) => {
        paste! {
            cuda_kernel_declaration!(
                [<get_powers_by_ref_ $type:lower _kernel>](
                    base: *const $type,
                    offset: u32,
                    bit_reverse: bool,
                    result: *mut $type,
                    count: u32,
                )
            );
        }
    };
}

pub trait GetPowersByRef: Sized {
    const KERNEL_FUNCTION: GetPowersByRefSignature<Self>;
}

pub fn get_powers_by_ref<T: GetPowersByRef>(
    base: &DeviceVariable<T>,
    offset: u32,
    bit_reverse: bool,
    result: &mut DeviceSlice<T>,
    stream: &CudaStream,
) -> CudaResult<()> {
    assert!(result.len() <= u32::MAX as usize);
    let count = result.len() as u32;
    let (grid_dim, block_dim) = get_launch_dims(count);
    let base = base.as_ptr();
    let result = result.as_mut_ptr();
    let config = CudaLaunchConfig::basic(grid_dim, block_dim, stream);
    let args = GetPowersByRefArguments::new(base, offset, bit_reverse, result, count);
    GetPowersByRefFunction(T::KERNEL_FUNCTION).launch(&config, &args)
}

macro_rules! get_powers_by_ref_impl {
    ($type:ty) => {
        paste! {
            get_powers_by_ref_kernel!($type);
            impl GetPowersByRef for $type {
                const KERNEL_FUNCTION: GetPowersByRefSignature<Self> = [<get_powers_by_ref_ $type:lower _kernel>];
            }
        }
    };
}

get_powers_by_ref_impl!(BF);
get_powers_by_ref_impl!(EF);

cuda_kernel!(
    OmegaShift,
    omega_shift_kernel(
        src: *const BF,
        log_degree: u32,
        offset: u32,
        inverse: bool,
        shift: u32,
        dst: *mut BF,
        count: u32,
    )
);

fn launch_omega_shift(args: &OmegaShiftArguments, stream: &CudaStream) -> CudaResult<()> {
    let (grid_dim, block_dim) = get_launch_dims(args.count);
    let config = CudaLaunchConfig::basic(grid_dim, block_dim, stream);
    OmegaShiftFunction::default().launch(&config, args)
}

pub fn omega_shift(
    src: &DeviceSlice<BF>,
    log_degree: u32,
    offset: u32,
    inverse: bool,
    shift: u32,
    dst: &mut DeviceSlice<BF>,
    stream: &CudaStream,
) -> CudaResult<()> {
    assert_eq!(src.len(), dst.len());
    assert!(dst.len() <= u32::MAX as usize);
    let count = dst.len() as u32;
    let src = src.as_ptr();
    let dst = dst.as_mut_ptr();
    let args = OmegaShiftArguments::new(src, log_degree, offset, inverse, shift, dst, count);
    launch_omega_shift(&args, stream)
}

pub fn omega_shift_in_place(
    values: &mut DeviceSlice<BF>,
    log_degree: u32,
    offset: u32,
    inverse: bool,
    shift: u32,
    stream: &CudaStream,
) -> CudaResult<()> {
    assert!(values.len() <= u32::MAX as usize);
    let count = values.len() as u32;
    let src = values.as_ptr();
    let dst = values.as_mut_ptr();
    let args = OmegaShiftArguments::new(src, log_degree, offset, inverse, shift, dst, count);
    launch_omega_shift(&args, stream)
}

cuda_kernel!(
    BitReverse,
    bit_reverse_kernel,
    src: PtrAndStride<<BF as DeviceRepr>::Type>,
    dst: MutPtrAndStride<<BF as DeviceRepr>::Type>,
    log_count: u32,
);

bit_reverse_kernel!(bit_reverse_naive_kernel);
bit_reverse_kernel!(bit_reverse_kernel);

fn launch_bit_reverse(
    rows: usize,
    cols: usize,
    src: PtrAndStride<BF>,
    dst: MutPtrAndStride<BF>,
    stream: &CudaStream,
) -> CudaResult<()> {
    assert!(rows.is_power_of_two());
    assert!(rows <= u32::MAX as usize);
    assert!(cols <= u32::MAX as usize);
    let log_count = rows.trailing_zeros();
    let half_log_count = log_count >> 1;
    const LOG_TILE_DIM: u32 = 5;
    let args = BitReverseArguments::new(src, dst, log_count);
    let (function, config) = if half_log_count < LOG_TILE_DIM {
        let (mut grid_dim, block_dim) = get_launch_dims(1 << log_count);
        grid_dim.y = cols as u32;
        let config = CudaLaunchConfig::basic(grid_dim, block_dim, stream);
        (BitReverseFunction(bit_reverse_naive_kernel), config)
    } else {
        const TILE_DIM: u32 = 1 << LOG_TILE_DIM;
        const BLOCK_ROWS: u32 = 2;
        assert!(half_log_count >= LOG_TILE_DIM);
        let tiles_per_dim = 1 << (half_log_count - LOG_TILE_DIM);
        let grid_dim_x = tiles_per_dim * (tiles_per_dim + 1) / 2;
        let grid_dim_y = log_count - (half_log_count << 1) + 1;
        let grid_dim_z = cols as u32;
        let grid_dim = (grid_dim_x, grid_dim_y, grid_dim_z);
        let block_dim = (TILE_DIM, BLOCK_ROWS, 2);
        let config = CudaLaunchConfig::basic(grid_dim, block_dim, stream);
        (BitReverseFunction(bit_reverse_kernel), config)
    };
    function.launch(&config, &args)
}

pub fn bit_reverse(
    src: &(impl DeviceMatrixChunkImpl<BF> + ?Sized),
    dst: &mut (impl DeviceMatrixChunkMutImpl<BF> + ?Sized),
    stream: &CudaStream,
) -> CudaResult<()> {
    let rows = dst.rows();
    let cols = dst.cols();
    assert_eq!(src.rows(), rows);
    assert_eq!(src.cols(), cols);
    let src = src.as_ptr_and_stride();
    let dst = dst.as_mut_ptr_and_stride();
    launch_bit_reverse(rows, cols, src, dst, stream)
}

pub fn bit_reverse_in_place(
    values: &mut (impl DeviceMatrixChunkMutImpl<BF> + ?Sized),
    stream: &CudaStream,
) -> CudaResult<()> {
    let rows = values.rows();
    let cols = values.cols();
    let src = values.as_ptr_and_stride();
    let dst = values.as_mut_ptr_and_stride();
    launch_bit_reverse(rows, cols, src, dst, stream)
}

cuda_kernel_signature_arguments_and_function!(
    BatchInv<T: DeviceRepr>,
    src: PtrAndStride<<T as DeviceRepr>::Type>,
    dst: MutPtrAndStride<<T as DeviceRepr>::Type>,
    count: u32,
);

macro_rules! batch_inv_kernel {
    ($type:ty) => {
        paste! {
            cuda_kernel_declaration!(
                [<batch_inv_ $type:lower _kernel>](
                    src: PtrAndStride<<$type as DeviceRepr>::Type>,
                    dst: MutPtrAndStride<<$type as DeviceRepr>::Type>,
                    count: u32,
                )
            );
        }
    };
}

pub trait BatchInv: DeviceRepr {
    const BATCH_SIZE: u32;
    const KERNEL_FUNCTION: BatchInvSignature<Self>;
}

pub fn launch_batch_inv<T: BatchInv>(
    src: PtrAndStride<<T as DeviceRepr>::Type>,
    dst: MutPtrAndStride<<T as DeviceRepr>::Type>,
    count: u32,
    stream: &CudaStream,
) -> CudaResult<()> {
    let block_dim = WARP_SIZE * 4;
    let grid_dim = (count + T::BATCH_SIZE * block_dim - 1) / (T::BATCH_SIZE * block_dim);
    let config = CudaLaunchConfig::basic(grid_dim, block_dim, stream);
    let args = BatchInvArguments::<T>::new(src, dst, count);
    BatchInvFunction::<T>(T::KERNEL_FUNCTION).launch(&config, &args)
}

pub fn batch_inv<T: BatchInv>(
    src: &DeviceSlice<T>,
    dst: &mut DeviceSlice<T>,
    stream: &CudaStream,
) -> CudaResult<()> {
    assert_eq!(src.len(), dst.len());
    assert!(dst.len() <= u32::MAX as usize);
    launch_batch_inv::<T>(
        DeviceVectorImpl::as_ptr_and_stride(src),
        DeviceVectorMutImpl::as_mut_ptr_and_stride(dst),
        dst.len() as u32,
        stream,
    )
}

pub fn batch_inv_in_place<T: BatchInv>(
    values: &mut DeviceSlice<T>,
    stream: &CudaStream,
) -> CudaResult<()> {
    assert!(values.len() <= u32::MAX as usize);
    launch_batch_inv::<T>(
        DeviceVectorImpl::as_ptr_and_stride(values),
        DeviceVectorMutImpl::as_mut_ptr_and_stride(values),
        values.len() as u32,
        stream,
    )
}

macro_rules! batch_inv_impl {
    ($type:ty, $batch_size:expr) => {
        paste! {
            batch_inv_kernel!($type);
            impl BatchInv for $type {
                const BATCH_SIZE: u32 = $batch_size;
                const KERNEL_FUNCTION: BatchInvSignature<Self> = [<batch_inv_ $type:lower _kernel>];
            }
        }
    };
}

batch_inv_impl!(BF, 10);
batch_inv_impl!(EF, 6);

cuda_kernel!(
    PackVariableIndexes,
    pack_variable_indexes_kernel(src: *const u64, dst: *mut u32, count: u32)
);

pub fn pack_variable_indexes(
    src: &DeviceSlice<u64>,
    dst: &mut DeviceSlice<u32>,
    stream: &CudaStream,
) -> CudaResult<()> {
    assert_eq!(src.len(), dst.len());
    assert!(src.len() <= u32::MAX as usize);
    let count = src.len() as u32;
    let (grid_dim, block_dim) = get_launch_dims(count);
    let src = src.as_ptr();
    let dst = dst.as_mut_ptr();
    let config = CudaLaunchConfig::basic(grid_dim, block_dim, stream);
    let args = PackVariableIndexesArguments::new(src, dst, count);
    PackVariableIndexesFunction::default().launch(&config, &args)
}

cuda_kernel!(
    Select,
    select_kernel(
        indexes: *const u32,
        src: *const BF,
        dst: *mut BF,
        count: u32,
    )
);

pub fn select(
    indexes: &DeviceSlice<u32>,
    src: &DeviceSlice<BF>,
    dst: &mut DeviceSlice<BF>,
    stream: &CudaStream,
) -> CudaResult<()> {
    assert!(indexes.len() <= u32::MAX as usize);
    assert!(src.len() <= u32::MAX as usize);
    assert!(dst.len() <= u32::MAX as usize);
    let count = indexes.len() as u32;
    let (grid_dim, block_dim) = get_launch_dims(count);
    let indexes = indexes.as_ptr();
    let src = src.as_ptr();
    let dst = dst.as_mut_ptr();
    let config = CudaLaunchConfig::basic(grid_dim, block_dim, stream);
    let args = SelectArguments::new(indexes, src, dst, count);
    SelectFunction::default().launch(&config, &args)
}

cuda_kernel!(
    MarkEndsOfRuns,
    mark_ends_of_runs_kernel(
        run_lengths: *const u32,
        run_offsets: *const u32,
        result: *mut u32,
        count: u32,
    )
);

pub fn mark_ends_of_runs(
    run_lengths: &DeviceSlice<u32>,
    run_offsets: &DeviceSlice<u32>,
    result: &mut DeviceSlice<u32>,
    stream: &CudaStream,
) -> CudaResult<()> {
    assert_eq!(run_lengths.len(), run_offsets.len());
    assert!(run_lengths.len() <= u32::MAX as usize);
    assert!(result.len() <= u32::MAX as usize);
    let count = run_lengths.len() as u32;
    let (grid_dim, block_dim) = get_launch_dims(count);
    let run_lengths = run_lengths.as_ptr();
    let run_offsets = run_offsets.as_ptr();
    let result = result.as_mut_ptr();
    let config = CudaLaunchConfig::basic(grid_dim, block_dim, stream);
    let args = MarkEndsOfRunsArguments::new(run_lengths, run_offsets, result, count);
    MarkEndsOfRunsFunction::default().launch(&config, &args)
}

cuda_kernel!(
    GeneratePermutationMatrix,
    generate_permutation_matrix_kernel(
        unique_variable_indexes: *const u32,
        run_indexes: *const u32,
        run_lengths: *const u32,
        run_offsets: *const u32,
        cell_indexes: *const u32,
        scalars: *const BF,
        result: *mut BF,
        columns_count: u32,
        log_rows_count: u32,
        )
);

#[allow(clippy::too_many_arguments)]
fn generate_permutation_matrix_raw(
    unique_variable_indexes: &DeviceSlice<u32>,
    run_indexes: &DeviceSlice<u32>,
    run_lengths: &DeviceSlice<u32>,
    run_offsets: &DeviceSlice<u32>,
    cell_indexes: &DeviceSlice<u32>,
    scalars: &DeviceSlice<BF>,
    result: &mut DeviceSlice<BF>,
    stream: &CudaStream,
) -> CudaResult<()> {
    assert!(run_indexes.len() <= u32::MAX as usize);
    assert_eq!(run_lengths.len(), run_offsets.len());
    assert_eq!(run_lengths.len(), unique_variable_indexes.len());
    assert!(run_lengths.len() <= u32::MAX as usize);
    assert!(cell_indexes.len() <= u32::MAX as usize);
    assert!(scalars.len() <= u32::MAX as usize);
    let columns_count = scalars.len() as u32;
    assert!(result.len() <= u32::MAX as usize);
    let count = result.len() as u32;
    let (grid_dim, block_dim) = get_launch_dims(count);
    assert_eq!(count % columns_count, 0);
    let rows_count = count / columns_count;
    assert!(rows_count.is_power_of_two());
    let log_rows_count = rows_count.ilog2();
    let unique_variable_indexes = unique_variable_indexes.as_ptr();
    let run_indexes = run_indexes.as_ptr();
    let run_lengths = run_lengths.as_ptr();
    let run_offsets = run_offsets.as_ptr();
    let cell_indexes = cell_indexes.as_ptr();
    let scalars = scalars.as_ptr();
    let result = result.as_mut_ptr();
    let config = CudaLaunchConfig::basic(grid_dim, block_dim, stream);
    let args = GeneratePermutationMatrixArguments::new(
        unique_variable_indexes,
        run_indexes,
        run_lengths,
        run_offsets,
        cell_indexes,
        scalars,
        result,
        columns_count,
        log_rows_count,
    );
    GeneratePermutationMatrixFunction::default().launch(&config, &args)
}

pub fn get_generate_permutation_matrix_temp_storage_bytes(num_cells: usize) -> CudaResult<usize> {
    let end_bit: i32 = (usize::BITS - num_cells.leading_zeros()) as i32;
    let sort_pairs_tsb =
        get_sort_pairs_temp_storage_bytes::<u32, u32>(false, num_cells as u32, 0, end_bit)?;
    let encode_tsb = get_encode_temp_storage_bytes::<u32>(num_cells as i32)?;
    let scan_tsb =
        get_scan_temp_storage_bytes::<u32>(ScanOperation::Sum, false, false, num_cells as i32)?;
    let cub_tsb = max([sort_pairs_tsb, encode_tsb, scan_tsb]).unwrap();
    Ok((7 * num_cells + 1) * 4 + cub_tsb)
}

pub fn generate_permutation_matrix(
    temp_storage: &mut DeviceSlice<u8>,
    variable_indexes: &DeviceSlice<u32>,
    scalars: &DeviceSlice<BF>,
    result: &mut DeviceSlice<BF>,
    stream: &CudaStream,
) -> CudaResult<()> {
    let num_cells = variable_indexes.len();
    let (sorted_variable_indexes, temp_storage) = temp_storage.split_at_mut(num_cells * 4);
    let (unsorted_cell_indexes, temp_storage) = temp_storage.split_at_mut(num_cells * 4);
    let (sorted_cell_indexes, temp_storage) = temp_storage.split_at_mut(num_cells * 4);
    let (unique_variable_indexes, temp_storage) = temp_storage.split_at_mut(num_cells * 4);
    let (run_lengths, temp_storage) = temp_storage.split_at_mut(num_cells * 4);
    let (run_offsets, temp_storage) = temp_storage.split_at_mut(num_cells * 4);
    let (num_runs_out, temp_storage) = temp_storage.split_at_mut(4);
    let (run_indexes, temp_storage) = temp_storage.split_at_mut(num_cells * 4);
    let sorted_variable_indexes = unsafe { sorted_variable_indexes.transmute_mut() };
    let unsorted_cell_indexes = unsafe { unsorted_cell_indexes.transmute_mut() };
    let sorted_cell_indexes = unsafe { sorted_cell_indexes.transmute_mut() };
    let unique_variable_indexes = unsafe { unique_variable_indexes.transmute_mut() };
    let run_lengths = unsafe { run_lengths.transmute_mut() };
    let run_offsets = unsafe { run_offsets.transmute_mut() };
    let num_runs_out = unsafe { &mut num_runs_out.transmute_mut()[0] };
    let run_indexes = unsafe { run_indexes.transmute_mut() };
    set_by_val(1u32, unsorted_cell_indexes, stream)?;
    scan_in_place(
        ScanOperation::Sum,
        false,
        false,
        temp_storage,
        unsorted_cell_indexes,
        stream,
    )?;
    sort_pairs(
        false,
        temp_storage,
        variable_indexes,
        sorted_variable_indexes,
        unsorted_cell_indexes,
        sorted_cell_indexes,
        0,
        32,
        stream,
    )?;
    set_to_zero(run_lengths, stream)?;
    encode(
        temp_storage,
        sorted_variable_indexes,
        unique_variable_indexes,
        run_lengths,
        num_runs_out,
        stream,
    )?;
    memory_copy_async(run_offsets, run_lengths, stream)?;
    scan_in_place(
        ScanOperation::Sum,
        false,
        false,
        temp_storage,
        run_offsets,
        stream,
    )?;
    set_to_zero(run_indexes, stream)?;
    mark_ends_of_runs(run_lengths, run_offsets, run_indexes, stream)?;
    scan_in_place(
        ScanOperation::Sum,
        false,
        false,
        temp_storage,
        run_indexes,
        stream,
    )?;
    generate_permutation_matrix_raw(
        unique_variable_indexes,
        run_indexes,
        run_lengths,
        run_offsets,
        sorted_cell_indexes,
        scalars,
        result,
        stream,
    )
}

cuda_kernel!(
    SetValuesFromPacketBits,
    set_values_from_packed_bits_kernel(packed_bits: *const u32, result: *mut BF, count: u32,)
);

pub fn set_values_from_packed_bits(
    packed_bits: &DeviceSlice<u32>,
    result: &mut DeviceSlice<BF>,
    stream: &CudaStream,
) -> CudaResult<()> {
    assert!(packed_bits.len() <= u32::MAX as usize);
    assert!(result.len() <= u32::MAX as usize);
    let words_count = packed_bits.len() as u32;
    let count = result.len() as u32;
    let (grid_dim, block_dim) = get_launch_dims(count);
    assert!(words_count * 32 >= count);
    assert!((words_count - 1) * 32 < count);
    let packed_bits = packed_bits.as_ptr();
    let result = result.as_mut_ptr();
    let config = CudaLaunchConfig::basic(grid_dim, block_dim, stream);
    let args = SetValuesFromPacketBitsArguments::new(packed_bits, result, count);
    SetValuesFromPacketBitsFunction::default().launch(&config, &args)
}

cuda_kernel!(
    Fold,
    fold_kernel(
        coset_inverse: BF,
        challenge: *const <ExtensionField as DeviceRepr>::Type,
        src: PtrAndStride<<EF as DeviceRepr>::Type>,
        dst: MutPtrAndStride<<EF as DeviceRepr>::Type>,
        count: u32,
    )
);

pub fn fold<S, D>(
    coset_inverse: BF,
    challenge: &DeviceVariable<ExtensionField>,
    src: &S,
    dst: &mut D,
    stream: &CudaStream,
) -> CudaResult<()>
where
    S: DeviceVectorImpl<EF> + ?Sized,
    D: DeviceVectorMutImpl<EF> + ?Sized,
{
    assert!(src.slice().len().is_power_of_two());
    assert!(dst.slice().len().is_power_of_two());
    let log_count = dst.slice().len().ilog2();
    assert_eq!(src.slice().len().ilog2(), log_count + 1);
    assert!(log_count < 32);
    let (grid_dim, block_dim) = get_launch_dims(1 << log_count);
    let challenge = challenge.as_ptr() as *const <ExtensionField as DeviceRepr>::Type;
    let src = src.as_ptr_and_stride();
    let dst = dst.as_mut_ptr_and_stride();
    let config = CudaLaunchConfig::basic(grid_dim, block_dim, stream);
    let args = FoldArguments::new(coset_inverse, challenge, src, dst, log_count);
    FoldFunction::default().launch(&config, &args)
}

cuda_kernel!(
    PartialProductsOfFGCHunk,
    partial_products_f_g_chunk_kernel(
        num: MutPtrAndStride<<EF as DeviceRepr>::Type>,
        denom: MutPtrAndStride<<EF as DeviceRepr>::Type>,
        variable_cols_chunk: PtrAndStride<BF>,
        sigma_cols_chunk: PtrAndStride<BF>,
        omega_values: PtrAndStride<BF>,
        non_residues_by_beta_chunk: *const <ExtensionField as DeviceRepr>::Type,
        beta_c0: *const BF,
        beta_c1: *const BF,
        gamma_c0: *const BF,
        gamma_c1: *const BF,
        num_cols_this_chunk: u32,
        count: u32,
    )
);

#[allow(clippy::too_many_arguments)]
pub fn partial_products_f_g_chunk(
    num: &mut (impl DeviceVectorMutImpl<EF> + ?Sized),
    denom: &mut (impl DeviceVectorMutImpl<EF> + ?Sized),
    variable_cols_chunk: &(impl DeviceMatrixImpl<BF> + ?Sized),
    sigma_cols_chunk: &(impl DeviceMatrixImpl<BF> + ?Sized),
    omega_values: &(impl DeviceVectorImpl<BF> + ?Sized),
    non_residues_by_beta_chunk: &(impl DeviceVectorImpl<ExtensionField> + ?Sized),
    beta_c0: &DeviceVariable<BF>,
    beta_c1: &DeviceVariable<BF>,
    gamma_c0: &DeviceVariable<BF>,
    gamma_c1: &DeviceVariable<BF>,
    num_cols_per_product: usize,
    stream: &CudaStream,
) -> CudaResult<()> {
    let num_cols = variable_cols_chunk.cols();
    assert!(num_cols <= u32::MAX as usize); // a silly check, but why not
    assert!(num_cols <= num_cols_per_product);
    let count = variable_cols_chunk.stride();
    assert!(count <= u32::MAX as usize);
    assert_eq!(num.slice().len(), count);
    assert_eq!(denom.slice().len(), count);
    assert_eq!(sigma_cols_chunk.cols(), num_cols);
    assert_eq!(sigma_cols_chunk.stride(), count);
    assert_eq!(omega_values.slice().len(), count);
    assert_eq!(non_residues_by_beta_chunk.slice().len(), num_cols);
    let num = num.as_mut_ptr_and_stride();
    let denom = denom.as_mut_ptr_and_stride();
    let variable_cols_chunk = variable_cols_chunk.as_ptr_and_stride();
    let sigma_cols_chunk = sigma_cols_chunk.as_ptr_and_stride();
    let omega_values = omega_values.as_ptr_and_stride();
    let non_residues_by_beta_chunk = non_residues_by_beta_chunk.as_ptr();
    let beta_c0 = beta_c0.as_ptr();
    let beta_c1 = beta_c1.as_ptr();
    let gamma_c0 = gamma_c0.as_ptr();
    let gamma_c1 = gamma_c1.as_ptr();
    let num_cols = num_cols as u32;
    let count = count as u32;
    let (grid_dim, block_dim) = get_launch_dims(count);
    let config = CudaLaunchConfig::basic(grid_dim, block_dim, stream);
    let args = PartialProductsOfFGCHunkArguments::new(
        num,
        denom,
        variable_cols_chunk,
        sigma_cols_chunk,
        omega_values,
        non_residues_by_beta_chunk,
        beta_c0,
        beta_c1,
        gamma_c0,
        gamma_c1,
        num_cols,
        count,
    );
    PartialProductsOfFGCHunkFunction::default().launch(&config, &args)
}

cuda_kernel!(
    PartialProductsQuotientTerms,
    partial_products_quotient_terms_kernel(
        partial_products: PtrAndStride<<EF as DeviceRepr>::Type>,
        z_poly: PtrAndStride<<EF as DeviceRepr>::Type>,
        variable_cols: PtrAndStride<BF>,
        sigma_cols: PtrAndStride<BF>,
        omega_values: PtrAndStride<BF>,
        powers_of_alpha: *const <ExtensionField as DeviceRepr>::Type,
        non_residues_by_beta: *const <ExtensionField as DeviceRepr>::Type,
        beta_c0: *const BF,
        beta_c1: *const BF,
        gamma_c0: *const BF,
        gamma_c1: *const BF,
        quotient: MutPtrAndStride<<EF as DeviceRepr>::Type>,
        num_cols: u32,
        num_cols_per_product: u32,
        count: u32,
    )
);

#[allow(clippy::too_many_arguments)]
pub fn partial_products_quotient_terms(
    partial_products: &(impl DeviceMatrixImpl<EF> + ?Sized),
    z_poly: &(impl DeviceVectorImpl<EF> + ?Sized),
    variable_cols: &(impl DeviceMatrixImpl<BF> + ?Sized),
    sigma_cols: &(impl DeviceMatrixImpl<BF> + ?Sized),
    omega_values: &(impl DeviceVectorImpl<BF> + ?Sized),
    powers_of_alpha: &(impl DeviceVectorImpl<ExtensionField> + ?Sized),
    non_residues_by_beta: &(impl DeviceVectorImpl<ExtensionField> + ?Sized),
    beta_c0: &DeviceVariable<BF>,
    beta_c1: &DeviceVariable<BF>,
    gamma_c0: &DeviceVariable<BF>,
    gamma_c1: &DeviceVariable<BF>,
    quotient: &mut (impl DeviceVectorMutImpl<EF> + ?Sized),
    num_cols_per_product: usize,
    stream: &CudaStream,
) -> CudaResult<()> {
    let num_cols = variable_cols.cols();
    assert!(num_cols <= u32::MAX as usize);
    let count = variable_cols.stride();
    assert!(count <= u32::MAX as usize);
    assert_eq!(count.count_ones(), 1);
    let expected_num_partial_products =
        ((num_cols + num_cols_per_product - 1) / num_cols_per_product) - 1;
    // Handling empty partial products would require special-case logic.
    // For now we don't need it. Assert as a reminder.
    assert!(partial_products.cols() > 0);
    assert_eq!(partial_products.cols(), expected_num_partial_products);
    assert_eq!(partial_products.stride(), count);
    assert_eq!(z_poly.slice().len(), count);
    assert_eq!(sigma_cols.cols(), num_cols);
    assert_eq!(sigma_cols.stride(), count);
    assert_eq!(omega_values.slice().len(), count);
    assert_eq!(
        powers_of_alpha.slice().len(),
        expected_num_partial_products + 1
    );
    assert_eq!(non_residues_by_beta.slice().len(), num_cols);
    assert_eq!(quotient.slice().len(), count);
    let partial_products = partial_products.as_ptr_and_stride();
    let z_poly = z_poly.as_ptr_and_stride();
    let variable_cols = variable_cols.as_ptr_and_stride();
    let sigma_cols = sigma_cols.as_ptr_and_stride();
    let omega_values = omega_values.as_ptr_and_stride();
    let powers_of_alpha = powers_of_alpha.as_ptr();
    let non_residues_by_beta = non_residues_by_beta.as_ptr();
    let beta_c0 = beta_c0.as_ptr();
    let beta_c1 = beta_c1.as_ptr();
    let gamma_c0 = gamma_c0.as_ptr();
    let gamma_c1 = gamma_c1.as_ptr();
    let quotient = quotient.as_mut_ptr_and_stride();
    let num_cols = num_cols as u32;
    let num_cols_per_product = num_cols_per_product as u32;
    let log_count: u32 = count.trailing_zeros();
    let (grid_dim, block_dim) = get_launch_dims(count as u32);
    let config = CudaLaunchConfig::basic(grid_dim, block_dim, stream);
    let args = PartialProductsQuotientTermsArguments::new(
        partial_products,
        z_poly,
        variable_cols,
        sigma_cols,
        omega_values,
        powers_of_alpha,
        non_residues_by_beta,
        beta_c0,
        beta_c1,
        gamma_c0,
        gamma_c1,
        quotient,
        num_cols,
        num_cols_per_product,
        log_count,
    );
    PartialProductsQuotientTermsFunction::default().launch(&config, &args)
}

cuda_kernel!(
    LookupAggregatedTableValues,
    lookup_aggregated_table_values_kernel(
        table_cols: PtrAndStride<BF>,
        beta_c0: *const BF,
        beta_c1: *const BF,
        powers_of_gamma: *const <ExtensionField as DeviceRepr>::Type,
        aggregated_table_values: MutPtrAndStride<<EF as DeviceRepr>::Type>,
        num_cols: u32,
        count: u32,
    )
);

pub fn lookup_aggregated_table_values(
    table_cols: &(impl DeviceMatrixImpl<BF> + ?Sized),
    beta_c0: &DeviceVariable<BF>,
    beta_c1: &DeviceVariable<BF>,
    powers_of_gamma: &(impl DeviceVectorImpl<ExtensionField> + ?Sized),
    aggregated_table_values: &mut (impl DeviceVectorMutImpl<EF> + ?Sized),
    num_cols: usize,
    stream: &CudaStream,
) -> CudaResult<()> {
    assert!(num_cols <= u32::MAX as usize);
    let count = table_cols.stride();
    assert!(count <= u32::MAX as usize);
    assert_eq!(table_cols.cols(), num_cols);
    assert_eq!(table_cols.stride(), count);
    assert_eq!(powers_of_gamma.slice().len(), num_cols);
    assert_eq!(aggregated_table_values.slice().len(), count);
    let table_cols = table_cols.as_ptr_and_stride();
    let beta_c0 = beta_c0.as_ptr();
    let beta_c1 = beta_c1.as_ptr();
    let powers_of_gamma = powers_of_gamma.as_ptr();
    let aggregated_table_values = aggregated_table_values.as_mut_ptr_and_stride();
    let num_cols = num_cols as u32;
    let count = count as u32;
    let (grid_dim, block_dim) = get_launch_dims(count);
    let config = CudaLaunchConfig::basic(grid_dim, block_dim, stream);
    let args = LookupAggregatedTableValuesArguments::new(
        table_cols,
        beta_c0,
        beta_c1,
        powers_of_gamma,
        aggregated_table_values,
        num_cols,
        count,
    );
    LookupAggregatedTableValuesFunction::default().launch(&config, &args)
}

cuda_kernel!(
    LookupSubargsAAndB,
    lookup_subargs_a_and_b_kernel(
        variable_cols: PtrAndStride<BF>,
        subargs_a: MutPtrAndStride<<EF as DeviceRepr>::Type>,
        subargs_b: MutPtrAndStride<<EF as DeviceRepr>::Type>,
        beta_c0: *const BF,
        beta_c1: *const BF,
        powers_of_gamma: *const <ExtensionField as DeviceRepr>::Type,
        table_id_col: PtrAndStride<BF>,
        aggregated_table_values_inv: PtrAndStride<<EF as DeviceRepr>::Type>,
        multiplicity_cols: PtrAndStride<BF>,
        num_subargs_a: u32,
        num_subargs_b: u32,
        num_cols_per_subarg: u32,
        count: u32,
    )
);

#[allow(clippy::too_many_arguments)]
pub fn lookup_subargs_a_and_b(
    variable_cols: &(impl DeviceMatrixImpl<BF> + ?Sized),
    subargs_a: &mut (impl DeviceMatrixMutImpl<EF> + ?Sized),
    subargs_b: &mut (impl DeviceMatrixMutImpl<EF> + ?Sized),
    beta_c0: &DeviceVariable<BF>,
    beta_c1: &DeviceVariable<BF>,
    powers_of_gamma: &(impl DeviceVectorImpl<ExtensionField> + ?Sized),
    table_id_col: &(impl DeviceVectorImpl<BF> + ?Sized),
    aggregated_table_values_inv: &(impl DeviceVectorImpl<EF> + ?Sized),
    multiplicity_cols: &(impl DeviceMatrixImpl<BF> + ?Sized),
    num_cols_per_subarg: usize,
    stream: &CudaStream,
) -> CudaResult<()> {
    let num_cols = variable_cols.cols();
    assert!(num_cols <= u32::MAX as usize);
    let count = variable_cols.stride();
    assert!(count <= u32::MAX as usize);
    let num_subargs_a = num_cols / num_cols_per_subarg;
    // num_cols should be an even multiple of num_cols_per_subarg
    assert_eq!(num_subargs_a * num_cols_per_subarg, num_cols);
    let num_subargs_b = subargs_b.cols();
    assert_eq!(num_subargs_b, 1);
    assert_eq!(subargs_a.cols(), num_subargs_a);
    assert_eq!(subargs_a.stride(), count);
    assert_eq!(subargs_b.stride(), count);
    assert_eq!(powers_of_gamma.slice().len(), num_cols_per_subarg + 1);
    assert_eq!(table_id_col.slice().len(), count);
    assert_eq!(aggregated_table_values_inv.slice().len(), count);
    assert_eq!(multiplicity_cols.cols(), num_subargs_b);
    assert_eq!(multiplicity_cols.stride(), count);
    let variable_cols = variable_cols.as_ptr_and_stride();
    let subargs_a = subargs_a.as_mut_ptr_and_stride();
    let subargs_b = subargs_b.as_mut_ptr_and_stride();
    let beta_c0 = beta_c0.as_ptr();
    let beta_c1 = beta_c1.as_ptr();
    let powers_of_gamma = powers_of_gamma.as_ptr();
    let table_id_col = table_id_col.as_ptr_and_stride();
    let aggregated_table_values_inv = aggregated_table_values_inv.as_ptr_and_stride();
    let multiplicity_cols = multiplicity_cols.as_ptr_and_stride();
    let num_subargs_a = num_subargs_a as u32;
    let num_subargs_b = num_subargs_b as u32;
    let num_cols_per_subarg = num_cols_per_subarg as u32;
    let count = count as u32;
    let (grid_dim, block_dim) = get_launch_dims(count);
    let config = CudaLaunchConfig::basic(grid_dim, block_dim, stream);
    let args = LookupSubargsAAndBArguments::new(
        variable_cols,
        subargs_a,
        subargs_b,
        beta_c0,
        beta_c1,
        powers_of_gamma,
        table_id_col,
        aggregated_table_values_inv,
        multiplicity_cols,
        num_subargs_a,
        num_subargs_b,
        num_cols_per_subarg,
        count,
    );
    LookupSubargsAAndBFunction::default().launch(&config, &args)
}

cuda_kernel!(
    LookupQuotientAAndB,
    lookup_quotient_a_and_b_kernel(
        variable_cols: PtrAndStride<BF>,
        table_cols: PtrAndStride<BF>,
        subargs_a: PtrAndStride<<EF as DeviceRepr>::Type>,
        subargs_b: PtrAndStride<<EF as DeviceRepr>::Type>,
        beta_c0: *const BF,
        beta_c1: *const BF,
        powers_of_gamma: *const <ExtensionField as DeviceRepr>::Type,
        powers_of_alpha: *const <ExtensionField as DeviceRepr>::Type,
        table_id_col: PtrAndStride<BF>,
        multiplicity_cols: PtrAndStride<BF>,
        quotient: MutPtrAndStride<<EF as DeviceRepr>::Type>,
        num_subargs_a: u32,
        num_subargs_b: u32,
        num_cols_per_subarg: u32,
        count: u32,
    )
);

#[allow(clippy::too_many_arguments)]
pub fn lookup_quotient_a_and_b(
    variable_cols: &(impl DeviceMatrixImpl<BF> + ?Sized),
    table_cols: &(impl DeviceMatrixImpl<BF> + ?Sized),
    subargs_a: &(impl DeviceMatrixImpl<EF> + ?Sized),
    subargs_b: &(impl DeviceMatrixImpl<EF> + ?Sized),
    beta_c0: &DeviceVariable<BF>,
    beta_c1: &DeviceVariable<BF>,
    powers_of_gamma: &(impl DeviceVectorImpl<ExtensionField> + ?Sized),
    powers_of_alpha: &(impl DeviceVectorImpl<ExtensionField> + ?Sized),
    table_id_col: &(impl DeviceVectorImpl<BF> + ?Sized),
    multiplicity_cols: &(impl DeviceMatrixImpl<BF> + ?Sized),
    quotient: &mut (impl DeviceVectorMutImpl<EF> + ?Sized),
    num_cols_per_subarg: usize,
    stream: &CudaStream,
) -> CudaResult<()> {
    let num_cols = variable_cols.cols();
    assert!(num_cols <= u32::MAX as usize);
    let count = variable_cols.stride();
    assert!(count <= u32::MAX as usize);
    let num_subargs_a = num_cols / num_cols_per_subarg;
    // num_cols should be an even multiple of num_cols_per_subarg
    assert_eq!(num_subargs_a * num_cols_per_subarg, num_cols);
    let num_subargs_b = subargs_b.cols();
    assert_eq!(num_subargs_b, 1);
    assert_eq!(table_cols.cols(), num_cols_per_subarg + 1);
    assert_eq!(table_cols.stride(), count);
    assert_eq!(subargs_a.cols(), num_subargs_a);
    assert_eq!(subargs_a.stride(), count);
    assert_eq!(subargs_b.stride(), count);
    assert_eq!(powers_of_gamma.slice().len(), num_cols_per_subarg + 1);
    assert_eq!(powers_of_alpha.slice().len(), num_subargs_a + num_subargs_b);
    assert_eq!(table_id_col.slice().len(), count);
    assert_eq!(multiplicity_cols.cols(), num_subargs_b);
    assert_eq!(multiplicity_cols.stride(), count);
    assert_eq!(quotient.slice().len(), count);
    let variable_cols = variable_cols.as_ptr_and_stride();
    let table_cols = table_cols.as_ptr_and_stride();
    let subargs_a = subargs_a.as_ptr_and_stride();
    let subargs_b = subargs_b.as_ptr_and_stride();
    let beta_c0 = beta_c0.as_ptr();
    let beta_c1 = beta_c1.as_ptr();
    let powers_of_gamma = powers_of_gamma.as_ptr();
    let powers_of_alpha = powers_of_alpha.as_ptr();
    let table_id_col = table_id_col.as_ptr_and_stride();
    let multiplicity_cols = multiplicity_cols.as_ptr_and_stride();
    let quotient = quotient.as_mut_ptr_and_stride();
    let num_subargs_a = num_subargs_a as u32;
    let num_subargs_b = num_subargs_b as u32;
    let num_cols_per_subarg = num_cols_per_subarg as u32;
    let count = count as u32;
    let (grid_dim, block_dim) = get_launch_dims(count);
    let config = CudaLaunchConfig::basic(grid_dim, block_dim, stream);
    let args = LookupQuotientAAndBArguments::new(
        variable_cols,
        table_cols,
        subargs_a,
        subargs_b,
        beta_c0,
        beta_c1,
        powers_of_gamma,
        powers_of_alpha,
        table_id_col,
        multiplicity_cols,
        quotient,
        num_subargs_a,
        num_subargs_b,
        num_cols_per_subarg,
        count,
    );
    LookupQuotientAAndBFunction::default().launch(&config, &args)
}

cuda_kernel!(
    DeepQuotientExceptPublicInputs,
    deep_quotient_except_public_inputs_kernel(
        variable_cols: PtrAndStride<BF>,
        witness_cols: PtrAndStride<BF>,
        constant_cols: PtrAndStride<BF>,
        permutation_cols: PtrAndStride<BF>,
        z_poly: PtrAndStride<<EF as DeviceRepr>::Type>,
        partial_products: PtrAndStride<<EF as DeviceRepr>::Type>,
        multiplicity_cols: PtrAndStride<BF>,
        lookup_a_polys: PtrAndStride<<EF as DeviceRepr>::Type>,
        lookup_b_polys: PtrAndStride<<EF as DeviceRepr>::Type>,
        table_cols: PtrAndStride<BF>,
        quotient_constraint_polys: PtrAndStride<<EF as DeviceRepr>::Type>,
        evaluations_at_z: *const <ExtensionField as DeviceRepr>::Type,
        evaluations_at_z_omega: *const <ExtensionField as DeviceRepr>::Type,
        evaluations_at_zero: *const <ExtensionField as DeviceRepr>::Type,
        challenges: *const <ExtensionField as DeviceRepr>::Type,
        denom_at_z: PtrAndStride<<EF as DeviceRepr>::Type>,
        denom_at_z_omega: PtrAndStride<<EF as DeviceRepr>::Type>,
        denom_at_zero: PtrAndStride<BF>,
        quotient: MutPtrAndStride<<EF as DeviceRepr>::Type>,
        num_variable_cols: u32,
        num_witness_cols: u32,
        num_constants_cols: u32,
        num_permutation_cols: u32,
        num_partial_products: u32,
        num_multiplicity_cols: u32,
        num_lookup_a_polys: u32,
        num_lookup_b_polys: u32,
        num_table_cols: u32,
        num_quotient_constraint_polys: u32,
        z_omega_challenge_offset: u32,
        zero_challenge_offset: u32,
        count: u32,
    )
);

#[allow(clippy::too_many_arguments)]
pub fn deep_quotient_except_public_inputs(
    variable_cols: &(impl DeviceMatrixImpl<BF> + ?Sized),
    witness_cols: &(impl DeviceMatrixImpl<BF> + ?Sized),
    constant_cols: &(impl DeviceMatrixImpl<BF> + ?Sized),
    permutation_cols: &(impl DeviceMatrixImpl<BF> + ?Sized),
    z_poly: &(impl DeviceVectorImpl<EF> + ?Sized),
    partial_products: &(impl DeviceMatrixImpl<EF> + ?Sized),
    multiplicity_cols: &(impl DeviceMatrixImpl<BF> + ?Sized),
    lookup_a_polys: &(impl DeviceMatrixImpl<EF> + ?Sized),
    lookup_b_polys: &(impl DeviceMatrixImpl<EF> + ?Sized),
    table_cols: &(impl DeviceMatrixImpl<BF> + ?Sized),
    quotient_constraint_polys: &(impl DeviceMatrixImpl<EF> + ?Sized),
    evaluations_at_z: &(impl DeviceVectorImpl<ExtensionField> + ?Sized),
    evaluations_at_z_omega: &(impl DeviceVectorImpl<ExtensionField> + ?Sized),
    evaluations_at_zero: &(impl DeviceVectorImpl<ExtensionField> + ?Sized),
    challenges: &(impl DeviceVectorImpl<ExtensionField> + ?Sized),
    denom_at_z: &(impl DeviceVectorImpl<EF> + ?Sized),
    denom_at_z_omega: &(impl DeviceVectorImpl<EF> + ?Sized),
    denom_at_zero: &(impl DeviceVectorImpl<BF> + ?Sized),
    quotient: &mut (impl DeviceVectorMutImpl<EF> + ?Sized),
    stream: &CudaStream,
) -> CudaResult<()> {
    let count = variable_cols.stride();
    assert!(count <= u32::MAX as usize);
    assert_eq!(witness_cols.stride(), count);
    assert_eq!(constant_cols.stride(), count);
    assert_eq!(permutation_cols.stride(), count);
    assert_eq!(z_poly.slice().len(), count);
    assert_eq!(partial_products.stride(), count);
    assert_eq!(multiplicity_cols.stride(), count);
    assert_eq!(lookup_a_polys.stride(), count);
    assert_eq!(lookup_b_polys.stride(), count);
    assert_eq!(table_cols.stride(), count);
    assert_eq!(quotient_constraint_polys.stride(), count);
    assert_eq!(evaluations_at_z_omega.slice().len(), 1);
    assert_eq!(
        evaluations_at_zero.slice().len(),
        lookup_a_polys.cols() + lookup_b_polys.cols()
    );
    assert_eq!(denom_at_z.slice().len(), count);
    assert_eq!(denom_at_z_omega.slice().len(), count);
    assert_eq!(quotient.slice().len(), count);
    let mut num_terms_at_z = 0;
    num_terms_at_z +=
        variable_cols.cols() + witness_cols.cols() + constant_cols.cols() + permutation_cols.cols();
    num_terms_at_z += 1; // z_poly
    num_terms_at_z += partial_products.cols();
    num_terms_at_z += multiplicity_cols.cols();
    num_terms_at_z += lookup_a_polys.cols();
    num_terms_at_z += lookup_b_polys.cols();
    num_terms_at_z += table_cols.cols();
    num_terms_at_z += quotient_constraint_polys.cols();
    assert_eq!(evaluations_at_z.slice().len(), num_terms_at_z);
    assert_eq!(evaluations_at_z_omega.slice().len(), 1);
    assert_eq!(
        evaluations_at_zero.slice().len(),
        lookup_a_polys.cols() + lookup_b_polys.cols()
    );
    let mut num_terms_from_evals = 0;
    num_terms_from_evals += evaluations_at_z.slice().len();
    num_terms_from_evals += evaluations_at_z_omega.slice().len();
    num_terms_from_evals += evaluations_at_zero.slice().len();
    assert_eq!(challenges.slice().len(), num_terms_from_evals);
    let num_variable_cols = variable_cols.cols() as u32;
    let num_witness_cols = witness_cols.cols() as u32;
    let num_constant_cols = constant_cols.cols() as u32;
    let num_permutation_cols = permutation_cols.cols() as u32;
    let num_partial_products = partial_products.cols() as u32;
    let num_multiplicity_cols = multiplicity_cols.cols() as u32;
    if num_multiplicity_cols > 0 {
        assert_eq!(denom_at_zero.slice().len(), count)
    } else {
        assert_eq!(denom_at_zero.slice().len(), 0)
    }
    let num_lookup_a_polys = lookup_a_polys.cols() as u32;
    let num_lookup_b_polys = lookup_b_polys.cols() as u32;
    let num_table_cols = table_cols.cols() as u32;
    let num_quotient_constraint_polys = quotient_constraint_polys.cols() as u32;
    let variable_cols = variable_cols.as_ptr_and_stride();
    let witness_cols = witness_cols.as_ptr_and_stride();
    let constant_cols = constant_cols.as_ptr_and_stride();
    let permutation_cols = permutation_cols.as_ptr_and_stride();
    let z_poly = z_poly.as_ptr_and_stride();
    let partial_products = partial_products.as_ptr_and_stride();
    let multiplicity_cols = multiplicity_cols.as_ptr_and_stride();
    let lookup_a_polys = lookup_a_polys.as_ptr_and_stride();
    let lookup_b_polys = lookup_b_polys.as_ptr_and_stride();
    let table_cols = table_cols.as_ptr_and_stride();
    let quotient_constraint_polys = quotient_constraint_polys.as_ptr_and_stride();
    let evaluations_at_z = evaluations_at_z.as_ptr();
    let evaluations_at_z_omega = evaluations_at_z_omega.as_ptr();
    let evaluations_at_zero = evaluations_at_zero.as_ptr();
    let challenges = challenges.as_ptr();
    let denom_at_z = denom_at_z.as_ptr_and_stride();
    let denom_at_z_omega = denom_at_z_omega.as_ptr_and_stride();
    let denom_at_zero = denom_at_zero.as_ptr_and_stride();
    let quotient = quotient.as_mut_ptr_and_stride();
    let z_omega_challenge_offset = 1
        + num_partial_products
        + num_multiplicity_cols
        + num_lookup_a_polys
        + num_lookup_b_polys
        + num_table_cols
        + num_quotient_constraint_polys;
    let zero_challenge_offset = num_lookup_a_polys
        + num_lookup_b_polys
        + num_table_cols
        + num_quotient_constraint_polys
        + 1;
    let count = count as u32;
    let (grid_dim, block_dim) = get_launch_dims(count);
    let config = CudaLaunchConfig::basic(grid_dim, block_dim, stream);
    let args = DeepQuotientExceptPublicInputsArguments::new(
        variable_cols,
        witness_cols,
        constant_cols,
        permutation_cols,
        z_poly,
        partial_products,
        multiplicity_cols,
        lookup_a_polys,
        lookup_b_polys,
        table_cols,
        quotient_constraint_polys,
        evaluations_at_z,
        evaluations_at_z_omega,
        evaluations_at_zero,
        challenges,
        denom_at_z,
        denom_at_z_omega,
        denom_at_zero,
        quotient,
        num_variable_cols,
        num_witness_cols,
        num_constant_cols,
        num_permutation_cols,
        num_partial_products,
        num_multiplicity_cols,
        num_lookup_a_polys,
        num_lookup_b_polys,
        num_table_cols,
        num_quotient_constraint_polys,
        z_omega_challenge_offset,
        zero_challenge_offset,
        count,
    );
    DeepQuotientExceptPublicInputsFunction::default().launch(&config, &args)
}

cuda_kernel!(
    DeepQuotientPublicInput,
    deep_quotient_public_input_kernel(
        values: PtrAndStride<BF>,
        expected_value: BF,
        challenge: *const <ExtensionField as DeviceRepr>::Type,
        quotient: MutPtrAndStride<<EF as DeviceRepr>::Type>,
        count: u32,
    )
);

pub fn deep_quotient_public_input(
    values: &(impl DeviceVectorImpl<BF> + ?Sized),
    expected_value: BF,
    challenge: &(impl DeviceVectorImpl<ExtensionField> + ?Sized),
    quotient: &mut (impl DeviceVectorMutImpl<EF> + ?Sized),
    stream: &CudaStream,
) -> CudaResult<()> {
    let count = values.slice().len();
    assert!(count <= u32::MAX as usize);
    assert_eq!(values.slice().len(), count);
    assert_eq!(challenge.slice().len(), 1);
    assert_eq!(quotient.slice().len(), count);
    let values = values.as_ptr_and_stride();
    let challenge = challenge.as_ptr();
    let quotient = quotient.as_mut_ptr_and_stride();
    let count = count as u32;
    let (grid_dim, block_dim) = get_launch_dims(count);
    let config = CudaLaunchConfig::basic(grid_dim, block_dim, stream);
    let args =
        DeepQuotientPublicInputArguments::new(values, expected_value, challenge, quotient, count);
    DeepQuotientPublicInputFunction::default().launch(&config, &args)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::context::{Context, OMEGA_LOG_ORDER};
    use crate::device_structures::{DeviceMatrix, DeviceMatrixMut};
    use crate::extension_field::test_helpers::{transmute_gf_vec, ExtensionFieldTest};
    use crate::extension_field::{convert, ExtensionField};
    use crate::ops_cub::device_run_length_encode::{encode, get_encode_temp_storage_bytes};
    use crate::ops_cub::device_scan::{get_scan_temp_storage_bytes, scan_in_place, ScanOperation};
    use crate::ops_simple::set_to_zero;
    use boojum::cs::implementations::utils::{
        domain_generator_for_size, precompute_twiddles_for_fft,
    };
    use boojum::field::goldilocks::GoldilocksField;
    use boojum::field::{Field, PrimeField};
    use boojum::worker::Worker;
    use cudart::memory::{memory_copy_async, DeviceAllocation};
    use cudart::result::CudaResult;
    use cudart::slice::DeviceSlice;
    use cudart::stream::CudaStream;
    use itertools::Itertools;
    use rand::distributions::{Distribution, Uniform};
    use rand::{thread_rng, Rng};
    use serial_test::serial;
    use std::alloc::Global;
    use std::cmp::max;
    use std::fmt::Debug;
    use std::mem;

    fn assert_equal<T: PartialEq + Debug>((a, b): (T, T)) {
        assert_eq!(a, b);
    }

    type GetPowersOfWOrGDeviceFn =
        fn(u32, u32, bool, bool, &mut DeviceSlice<BF>, &CudaStream) -> CudaResult<()>;

    fn test_get_powers_of_w_or_g(
        log_degree: u32,
        inverse: bool,
        generator: BF,
        device_fn: GetPowersOfWOrGDeviceFn,
    ) {
        let n = 1 << log_degree;
        let context = Context::create(12, 12).unwrap();
        let mut h_result = vec![BF::ZERO; n];
        let mut d_result = DeviceAllocation::alloc(n).unwrap();
        let stream = CudaStream::default();
        device_fn(
            log_degree,
            0,
            inverse,
            false,
            &mut d_result[..n / 2],
            &stream,
        )
        .unwrap();
        device_fn(
            log_degree,
            n as u32 / 2,
            inverse,
            false,
            &mut d_result[n / 2..],
            &stream,
        )
        .unwrap();
        memory_copy_async(&mut h_result, &d_result, &stream).unwrap();
        stream.synchronize().unwrap();
        let mut generator = generator;
        if inverse {
            generator = generator.inverse().unwrap();
        }
        h_result
            .into_iter()
            .enumerate()
            .map(|(i, x)| (generator.pow_u64(i as u64), x))
            .for_each(assert_equal);
        context.destroy().unwrap();
    }

    fn test_get_powers_of_w(inverse: bool) {
        const LOG_DEGREE: u32 = 16;
        let generator = domain_generator_for_size((1 << LOG_DEGREE) as u64);
        test_get_powers_of_w_or_g(LOG_DEGREE, inverse, generator, super::get_powers_of_w);
    }

    #[test]
    #[serial]
    fn get_powers_of_w() {
        test_get_powers_of_w(false);
    }

    #[test]
    #[serial]
    fn get_powers_of_w_inverse() {
        test_get_powers_of_w(true);
    }

    fn test_get_powers_of_g(inverse: bool) {
        const LOG_DEGREE: u32 = 16;
        let generator = BF::multiplicative_generator().pow_u64(1 << (OMEGA_LOG_ORDER - LOG_DEGREE));
        test_get_powers_of_w_or_g(LOG_DEGREE, inverse, generator, super::get_powers_of_g);
    }

    #[test]
    #[serial]
    fn get_powers_of_g() {
        test_get_powers_of_g(false);
    }

    #[test]
    #[serial]
    fn get_powers_of_g_inverse() {
        test_get_powers_of_g(true);
    }

    fn test_get_powers_bf(by_val: bool) {
        const N: usize = 1 << 16;
        let mut h_result = vec![BF::ZERO; N];
        let mut d_result = DeviceAllocation::alloc(N).unwrap();
        let stream = CudaStream::default();
        let base = GoldilocksField(42);
        let mut d_base = DeviceAllocation::alloc(1).unwrap();
        memory_copy_async(&mut d_base, &[base], &stream).unwrap();
        let b = &d_base[0];
        let (d_result_0, d_result_1) = d_result.split_at_mut(N / 2);
        if by_val {
            get_powers_by_val(base, 0, false, d_result_0, &stream).unwrap();
            get_powers_by_val(base, N as u32 / 2, false, d_result_1, &stream).unwrap();
        } else {
            get_powers_by_ref(b, 0, false, d_result_0, &stream).unwrap();
            get_powers_by_ref(b, N as u32 / 2, false, d_result_1, &stream).unwrap();
        }
        memory_copy_async(&mut h_result, &d_result, &stream).unwrap();
        stream.synchronize().unwrap();
        h_result
            .into_iter()
            .enumerate()
            .map(|(i, x)| (base.pow_u64(i as u64), x))
            .for_each(assert_equal);
    }

    #[test]
    fn get_powers_by_val_bf() {
        test_get_powers_bf(true);
    }

    #[test]
    fn get_powers_by_ref_bf() {
        test_get_powers_bf(false);
    }

    fn test_get_powers_ef(by_val: bool) {
        const N: usize = 1 << 16;
        let mut h_result_0 = transmute_gf_vec::<EF>(vec![BF::ZERO; N * 2]);
        let mut h_result_1 = transmute_gf_vec::<EF>(vec![BF::ZERO; N * 2]);
        let mut d_result_0 = DeviceAllocation::alloc(N).unwrap();
        let mut d_result_1 = DeviceAllocation::alloc(N).unwrap();
        let stream = CudaStream::default();
        let base_ef =
            ExtensionField::from_coeff_in_base([GoldilocksField(42), GoldilocksField(42)]);
        let base_vf = unsafe { mem::transmute(base_ef) };
        let mut d_base = DeviceAllocation::alloc(1).unwrap();
        memory_copy_async(&mut d_base, &[base_vf], &stream).unwrap();
        let b = &d_base[0];
        if by_val {
            get_powers_by_val(base_vf, 0, false, &mut d_result_0, &stream).unwrap();
            get_powers_by_val(base_vf, N as u32, false, &mut d_result_1, &stream).unwrap();
        } else {
            get_powers_by_ref(b, 0, false, &mut d_result_0, &stream).unwrap();
            get_powers_by_ref(b, N as u32, false, &mut d_result_1, &stream).unwrap();
        }
        memory_copy_async(&mut h_result_0, &d_result_0, &stream).unwrap();
        memory_copy_async(&mut h_result_1, &d_result_1, &stream).unwrap();
        stream.synchronize().unwrap();
        let i_0 = EF::get_iterator(&h_result_0);
        let i_1 = EF::get_iterator(&h_result_1);
        i_0.chain(i_1)
            .enumerate()
            .map(|(i, x)| (base_ef.pow_u64(i as u64), x))
            .for_each(assert_equal);
    }

    #[test]
    fn get_powers_by_val_ef() {
        test_get_powers_ef(true);
    }

    #[test]
    fn get_powers_by_ref_ef() {
        test_get_powers_ef(false);
    }

    fn test_omega_shift(in_place: bool, inverse: bool) {
        const LOG_DEGREE: u32 = 16;
        const N: usize = 1 << LOG_DEGREE;
        const SHIFT: u32 = 42;
        let context = Context::create(12, 12).unwrap();
        let h_src = Uniform::new(0, BF::ORDER)
            .sample_iter(&mut thread_rng())
            .take(N)
            .map(GoldilocksField)
            .collect_vec();
        let mut h_dst = vec![BF::ZERO; N];
        let stream = CudaStream::default();
        if in_place {
            let mut d_values = DeviceAllocation::alloc(N).unwrap();
            memory_copy_async(&mut d_values, &h_src, &stream).unwrap();
            super::omega_shift_in_place(
                &mut d_values[..N / 2],
                LOG_DEGREE,
                0,
                inverse,
                SHIFT,
                &stream,
            )
            .unwrap();
            super::omega_shift_in_place(
                &mut d_values[N / 2..],
                LOG_DEGREE,
                N as u32 / 2,
                inverse,
                SHIFT,
                &stream,
            )
            .unwrap();
            memory_copy_async(&mut h_dst, &d_values, &stream).unwrap();
        } else {
            let mut d_src = DeviceAllocation::alloc(N).unwrap();
            let mut d_dst = DeviceAllocation::alloc(N).unwrap();
            memory_copy_async(&mut d_src, &h_src, &stream).unwrap();
            super::omega_shift(
                &d_src[..N / 2],
                LOG_DEGREE,
                0,
                inverse,
                SHIFT,
                &mut d_dst[..N / 2],
                &stream,
            )
            .unwrap();
            super::omega_shift(
                &d_src[N / 2..],
                LOG_DEGREE,
                N as u32 / 2,
                inverse,
                SHIFT,
                &mut d_dst[N / 2..],
                &stream,
            )
            .unwrap();
            memory_copy_async(&mut h_dst, &d_dst, &stream).unwrap();
        }
        stream.synchronize().unwrap();
        let mut generator: BF = domain_generator_for_size((1 << LOG_DEGREE) as u64);
        if inverse {
            generator = generator.inverse().unwrap();
        }
        h_src
            .into_iter()
            .enumerate()
            .map(|(i, x)| x * generator.pow_u64(SHIFT as u64 * i as u64))
            .zip(h_dst)
            .for_each(assert_equal);
        context.destroy().unwrap();
    }

    #[test]
    #[serial]
    fn omega_shift() {
        test_omega_shift(false, false);
    }

    #[test]
    #[serial]
    fn omega_shift_inverse() {
        test_omega_shift(false, true);
    }

    #[test]
    #[serial]
    fn omega_shift_in_place() {
        test_omega_shift(true, false);
    }

    #[test]
    #[serial]
    fn omega_shift_in_place_inverse() {
        test_omega_shift(true, true);
    }

    fn test_bit_reverse(in_place: bool) {
        const LOG_ROWS: usize = 16;
        const ROWS: usize = 1 << LOG_ROWS;
        const COLS: usize = 16;
        const N: usize = COLS << LOG_ROWS;
        let h_src = Uniform::new(0, BF::ORDER)
            .sample_iter(&mut thread_rng())
            .take(N)
            .map(GoldilocksField)
            .collect_vec();
        let mut h_dst = vec![BF::ZERO; N];
        let stream = CudaStream::default();
        if in_place {
            let mut d_values = DeviceAllocation::alloc(N).unwrap();
            memory_copy_async(&mut d_values, &h_src, &stream).unwrap();
            let mut matrix = DeviceMatrixMut::new(&mut d_values, ROWS);
            super::bit_reverse_in_place(&mut matrix, &stream).unwrap();
            memory_copy_async(&mut h_dst, &d_values, &stream).unwrap();
        } else {
            let mut d_src = DeviceAllocation::alloc(N).unwrap();
            let mut d_dst = DeviceAllocation::alloc(N).unwrap();
            memory_copy_async(&mut d_src, &h_src, &stream).unwrap();
            let src_matrix = DeviceMatrix::new(&d_src, ROWS);
            let mut dst_matrix = DeviceMatrixMut::new(&mut d_dst, ROWS);
            super::bit_reverse(&src_matrix, &mut dst_matrix, &stream).unwrap();
            memory_copy_async(&mut h_dst, &d_dst, &stream).unwrap();
        }
        stream.synchronize().unwrap();
        h_src
            .into_iter()
            .chunks(ROWS)
            .into_iter()
            .zip(h_dst.chunks(ROWS))
            .for_each(|(s, d)| {
                s.enumerate()
                    .map(|(i, x)| (x, d[i.reverse_bits() >> (usize::BITS - LOG_ROWS as u32)]))
                    .for_each(assert_equal);
            });
    }

    #[test]
    #[serial]
    fn bit_reverse() {
        test_bit_reverse(false);
    }

    #[test]
    #[serial]
    fn bit_reverse_in_place() {
        test_bit_reverse(true);
    }

    fn test_batch_inv_bf(in_place: bool) {
        const LOG_N: usize = 16;
        const N: usize = 1 << LOG_N;
        let h_src = Uniform::new(0, BF::ORDER)
            .sample_iter(&mut thread_rng())
            .take(N)
            .map(GoldilocksField)
            .collect_vec();
        let mut h_dst = vec![BF::ZERO; N];
        let stream = CudaStream::default();
        if in_place {
            let mut d_values = DeviceAllocation::alloc(N).unwrap();
            memory_copy_async(&mut d_values, &h_src, &stream).unwrap();
            batch_inv_in_place::<BF>(&mut d_values, &stream).unwrap();
            memory_copy_async(&mut h_dst, &d_values, &stream).unwrap();
        } else {
            let mut d_src = DeviceAllocation::alloc(N).unwrap();
            let mut d_dst = DeviceAllocation::alloc(N).unwrap();
            memory_copy_async(&mut d_src, &h_src, &stream).unwrap();
            batch_inv::<BF>(&d_src, &mut d_dst, &stream).unwrap();
            memory_copy_async(&mut h_dst, &d_dst, &stream).unwrap();
        }
        stream.synchronize().unwrap();
        h_src
            .into_iter()
            .map(|x| x.inverse().unwrap_or_default())
            .zip(h_dst)
            .for_each(assert_equal);
    }

    #[test]
    fn batch_inv_bf() {
        test_batch_inv_bf(false);
    }

    #[test]
    fn batch_inv_in_place_bf() {
        test_batch_inv_bf(true);
    }

    fn test_batch_inv_ef(in_place: bool) {
        const LOG_N: usize = 16;
        const N: usize = 1 << LOG_N;
        let h_src_bf = Uniform::new(0, BF::ORDER)
            .sample_iter(&mut thread_rng())
            .take(2 * N)
            .map(GoldilocksField)
            .collect_vec();
        let mut h_dst_bf = vec![BF::ZERO; 2 * N];
        let stream = CudaStream::default();
        if in_place {
            let mut d_values_bf = DeviceAllocation::alloc(2 * N).unwrap();
            memory_copy_async(&mut d_values_bf, &h_src_bf, &stream).unwrap();
            let d_values_ef = unsafe { d_values_bf.transmute_mut::<EF>() };
            batch_inv_in_place::<EF>(d_values_ef, &stream).unwrap();
            memory_copy_async(&mut h_dst_bf, &d_values_bf, &stream).unwrap();
        } else {
            let mut d_src_bf = DeviceAllocation::alloc(2 * N).unwrap();
            let mut d_dst_bf = DeviceAllocation::alloc(2 * N).unwrap();
            memory_copy_async(&mut d_src_bf, &h_src_bf, &stream).unwrap();
            let d_src_ef = unsafe { d_src_bf.transmute::<EF>() };
            let d_dst_ef = unsafe { d_dst_bf.transmute_mut::<EF>() };
            batch_inv::<EF>(d_src_ef, d_dst_ef, &stream).unwrap();
            memory_copy_async(&mut h_dst_bf, &d_dst_bf, &stream).unwrap();
        }
        stream.synchronize().unwrap();
        let h_src_c0 = &h_src_bf[0..N];
        let h_src_c1 = &h_src_bf[N..2 * N];
        let h_dst_c0 = &h_dst_bf[0..N];
        let h_dst_c1 = &h_dst_bf[N..2 * N];
        for (((src_c0, src_c1), dst_c0), dst_c1) in
            h_src_c0.iter().zip(h_src_c1).zip(h_dst_c0).zip(h_dst_c1)
        {
            let control = ExtensionField::from_coeff_in_base([*src_c0, *src_c1]);
            let control = control.inverse().unwrap_or_default();
            let result = ExtensionField::from_coeff_in_base([*dst_c0, *dst_c1]);
            assert_eq!(control, result)
        }
    }

    #[test]
    fn batch_inv_ef() {
        test_batch_inv_ef(false);
    }

    #[test]
    fn batch_inv_ef_in_place() {
        test_batch_inv_ef(true);
    }

    #[test]
    fn pack_variable_indexes() {
        const LOG_N: usize = 16;
        const N: usize = 1 << LOG_N;
        let mut h_src = Uniform::new(0, 1u64 << 31)
            .sample_iter(&mut thread_rng())
            .take(N)
            .collect_vec();
        let mut h_dst = vec![0u32; N];
        h_src[0] = 1 << 63;
        h_src[N - 1] = 1 << 63;
        let mut d_src = DeviceAllocation::alloc(N).unwrap();
        let mut d_dst = DeviceAllocation::alloc(N).unwrap();
        let stream = CudaStream::default();
        memory_copy_async(&mut d_src, &h_src, &stream).unwrap();
        super::pack_variable_indexes(&d_src, &mut d_dst, &stream).unwrap();
        memory_copy_async(&mut h_dst, &d_dst, &stream).unwrap();
        stream.synchronize().unwrap();
        h_src
            .into_iter()
            .map(|x| if x == (1 << 63) { 1 << 31 } else { x as u32 })
            .zip(h_dst)
            .for_each(assert_equal);
    }

    #[test]
    fn select() {
        const LOG_N: usize = 16;
        const N: usize = 1 << LOG_N;
        const PLACEHOLDER: u32 = 1 << 31;
        let h_indexes = (0..N)
            .map(|i| {
                if i == 0 {
                    PLACEHOLDER
                } else {
                    (i.reverse_bits() >> (usize::BITS - LOG_N as u32)) as u32
                }
            })
            .collect_vec();
        let h_src = Uniform::new(0, BF::ORDER)
            .sample_iter(&mut thread_rng())
            .take(N)
            .map(GoldilocksField)
            .collect_vec();
        let mut h_dst = vec![BF::ZERO; N];
        let mut d_indexes = DeviceAllocation::alloc(N).unwrap();
        let mut d_src = DeviceAllocation::alloc(N).unwrap();
        let mut d_dst = DeviceAllocation::alloc(N).unwrap();
        let stream = CudaStream::default();
        memory_copy_async(&mut d_indexes, &h_indexes, &stream).unwrap();
        memory_copy_async(&mut d_src, &h_src, &stream).unwrap();
        super::select(&d_indexes, &d_src, &mut d_dst, &stream).unwrap();
        memory_copy_async(&mut h_dst, &d_dst, &stream).unwrap();
        stream.synchronize().unwrap();
        h_indexes
            .into_iter()
            .map(|i| {
                if i == PLACEHOLDER {
                    BF::ZERO
                } else {
                    h_src[i as usize]
                }
            })
            .zip(h_dst)
            .for_each(assert_equal);
    }

    #[test]
    fn mark_ends_of_runs() {
        const LOG_DEGREE: u32 = 16;
        const N: usize = 1 << LOG_DEGREE;
        const RANGE_MAX: u32 = 2;
        let temp_storage_bytes = max(
            get_encode_temp_storage_bytes::<u32>(N as i32).unwrap(),
            get_scan_temp_storage_bytes::<u32>(ScanOperation::Sum, false, false, N as i32).unwrap(),
        );
        let mut d_temp_storage = DeviceAllocation::alloc(temp_storage_bytes).unwrap();
        let mut rng = thread_rng();
        let h_in = Uniform::new(0, RANGE_MAX)
            .sample_iter(&mut rng)
            .take(N)
            .collect_vec();
        let mut h_result = vec![0u32; N];
        let mut d_in = DeviceAllocation::alloc(N).unwrap();
        let mut d_unique_out = DeviceAllocation::alloc(N).unwrap();
        let mut d_counts_out = DeviceAllocation::alloc(N).unwrap();
        let mut d_num_runs_out = DeviceAllocation::alloc(1).unwrap();
        let mut d_offsets = DeviceAllocation::alloc(N).unwrap();
        let mut d_result = DeviceAllocation::alloc(N).unwrap();
        let stream = CudaStream::default();
        memory_copy_async(&mut d_in, &h_in, &stream).unwrap();
        set_to_zero(&mut d_counts_out, &stream).unwrap();
        encode(
            &mut d_temp_storage,
            &d_in,
            &mut d_unique_out,
            &mut d_counts_out,
            &mut d_num_runs_out[0],
            &stream,
        )
        .unwrap();
        memory_copy_async(&mut d_offsets, &d_counts_out, &stream).unwrap();
        scan_in_place(
            ScanOperation::Sum,
            false,
            false,
            &mut d_temp_storage,
            &mut d_offsets,
            &stream,
        )
        .unwrap();
        set_to_zero(&mut d_result, &stream).unwrap();
        super::mark_ends_of_runs(&d_counts_out, &d_offsets, &mut d_result, &stream).unwrap();
        memory_copy_async(&mut h_result, &d_result, &stream).unwrap();
        stream.synchronize().unwrap();
        for i in 0..N {
            let expected = if i == N - 1 || h_in[i] != h_in[i + 1] {
                1
            } else {
                0
            };
            assert_eq!(h_result[i], expected);
        }
    }

    #[test]
    #[serial]
    fn generate_permutation_matrix() {
        const LOG_NUM_ROWS: usize = 8;
        const NUM_ROWS: usize = 1 << LOG_NUM_ROWS;
        const NUM_COLS: usize = 8;
        const NUM_CELLS: usize = NUM_COLS << LOG_NUM_ROWS;
        const RANGE_MAX: usize = NUM_ROWS - 1;
        const PLACEHOLDER: u32 = 1 << 31;
        let context = Context::create(12, 12).unwrap();
        let h_variable_indexes = (0..NUM_CELLS)
            .map(|i| (i % RANGE_MAX) as u32)
            .map(|i| if i == 0 { PLACEHOLDER } else { i })
            .collect_vec();
        let h_scalars = Uniform::new(0, BF::ORDER)
            .sample_iter(&mut thread_rng())
            .take(NUM_COLS)
            .map(GoldilocksField)
            .collect_vec();
        let mut h_permutation_matrix = vec![BF::ZERO; NUM_CELLS];
        let mut h_twiddles = vec![BF::ZERO; NUM_ROWS];
        let mut d_variable_indexes = DeviceAllocation::alloc(NUM_CELLS).unwrap();
        let mut d_scalars = DeviceAllocation::alloc(NUM_COLS).unwrap();
        let mut d_permutation_matrix = DeviceAllocation::alloc(NUM_CELLS).unwrap();
        let mut d_twiddles = DeviceAllocation::alloc(NUM_ROWS).unwrap();
        let temp_storage_bytes =
            get_generate_permutation_matrix_temp_storage_bytes(NUM_CELLS).unwrap();
        let mut d_temp_storage = DeviceAllocation::alloc(temp_storage_bytes).unwrap();
        let stream = CudaStream::default();
        memory_copy_async(&mut d_variable_indexes, &h_variable_indexes, &stream).unwrap();
        memory_copy_async(&mut d_scalars, &h_scalars, &stream).unwrap();
        super::generate_permutation_matrix(
            &mut d_temp_storage,
            &d_variable_indexes,
            &d_scalars,
            &mut d_permutation_matrix,
            &stream,
        )
        .unwrap();
        memory_copy_async(&mut h_permutation_matrix, &d_permutation_matrix, &stream).unwrap();
        super::get_powers_of_w(
            LOG_NUM_ROWS as u32,
            0,
            false,
            false,
            &mut d_twiddles,
            &stream,
        )
        .unwrap();
        memory_copy_async(&mut h_twiddles, &d_twiddles, &stream).unwrap();
        stream.synchronize().unwrap();
        for (col, &scalar) in h_scalars.iter().enumerate() {
            for (row, &twiddle) in h_twiddles.iter().enumerate() {
                let source_index = col * NUM_ROWS + row;
                let target_index = if h_variable_indexes[source_index] == PLACEHOLDER {
                    source_index
                } else {
                    let mut idx = source_index + RANGE_MAX;
                    if idx >= NUM_CELLS {
                        idx = source_index % RANGE_MAX;
                    }
                    idx
                };
                let expected = twiddle * scalar;
                let actual = h_permutation_matrix[target_index];
                assert_eq!(expected, actual);
            }
        }
        context.destroy().unwrap();
    }

    #[test]
    fn set_values_from_packed_bits() {
        const LOG_N: usize = 16;
        const N: usize = 1 << LOG_N;
        const PACKED_N: usize = N / u32::BITS as usize;
        let rng = &mut thread_rng();
        let h_packed_bits = (0..PACKED_N).map(|_| rng.gen()).collect_vec();
        let mut h_result = vec![BF::ZERO; N];
        let mut d_packed_bits = DeviceAllocation::alloc(PACKED_N).unwrap();
        let mut d_result = DeviceAllocation::alloc(N).unwrap();
        let stream = CudaStream::default();
        memory_copy_async(&mut d_packed_bits, &h_packed_bits, &stream).unwrap();
        super::set_values_from_packed_bits(&d_packed_bits, &mut d_result, &stream).unwrap();
        memory_copy_async(&mut h_result, &d_result, &stream).unwrap();
        stream.synchronize().unwrap();
        h_packed_bits
            .into_iter()
            .flat_map(|word| (0..u32::BITS).map(move |i| (word >> i) & 1))
            .map(|bit| if bit == 1 { BF::ONE } else { BF::ZERO })
            .zip(h_result)
            .for_each(assert_equal);
    }

    #[test]
    #[serial]
    fn fold() {
        const LOG_N: usize = 16;
        const N: usize = 1 << LOG_N;
        let context = Context::create(12, 12).unwrap();
        let worker = Worker::new_with_num_threads(1);
        type F = BF;
        let roots = precompute_twiddles_for_fft::<F, F, Global, true>(N * 2, &worker, &mut ());
        let coset_inverse = <F as PrimeField>::multiplicative_generator()
            .inverse()
            .unwrap();
        let uniform = Uniform::new(0, BF::ORDER);
        let mut rng = thread_rng();
        let h_src = uniform
            .sample_iter(&mut rng)
            .map(GoldilocksField)
            .array_chunks()
            .map(ExtensionField::from_coeff_in_base)
            .take(N * 2)
            .collect_vec();
        let h_challenge = uniform
            .sample_iter(&mut rng)
            .map(GoldilocksField)
            .array_chunks()
            .map(ExtensionField::from_coeff_in_base)
            .next()
            .unwrap();
        let mut h_dst = vec![ExtensionField::ZERO; N];
        let stream = CudaStream::default();
        let mut d_challenge = DeviceAllocation::alloc(1).unwrap();
        let mut d_src_ef = DeviceAllocation::alloc(N * 2).unwrap();
        let mut d_src_vf = DeviceAllocation::alloc(N * 2).unwrap();
        let mut d_dst_vf = DeviceAllocation::alloc(N).unwrap();
        let mut d_dst_ef = DeviceAllocation::alloc(N).unwrap();
        memory_copy_async(&mut d_challenge, &[h_challenge], &stream).unwrap();
        memory_copy_async(&mut d_src_ef, &h_src, &stream).unwrap();
        convert(&d_src_ef, &mut d_src_vf, &stream).unwrap();
        super::fold(
            coset_inverse,
            &d_challenge[0],
            &d_src_vf,
            &mut d_dst_vf,
            &stream,
        )
        .unwrap();
        convert(&d_dst_vf, &mut d_dst_ef, &stream).unwrap();
        memory_copy_async(&mut h_dst, &d_dst_ef, &stream).unwrap();
        stream.synchronize().unwrap();
        context.destroy().unwrap();
        h_src
            .into_iter()
            .chunks(2)
            .into_iter()
            .enumerate()
            .map(|(i, chunk)| {
                let v = chunk.collect_vec();
                let mut sum = v[0];
                sum.add_assign(&v[1]);
                let mut diff = v[0];
                diff.sub_assign(&v[1]);
                diff.mul_assign_by_base(&coset_inverse);
                diff.mul_assign_by_base(&roots[i]);
                diff.mul_assign(&h_challenge);
                sum.add_assign(&diff);
                sum
            })
            .zip(h_dst)
            .for_each(assert_equal);
    }
}
