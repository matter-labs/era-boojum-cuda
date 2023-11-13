use std::os::raw::c_void;

use boojum::field::goldilocks::GoldilocksField;

use cudart::error::get_last_error;
use cudart::kernel_args;
use cudart::result::{CudaResult, CudaResultWrap};
use cudart::slice::DeviceSlice;
use cudart::stream::CudaStream;
use cudart_sys::cudaLaunchKernel;

use crate::context::OMEGA_LOG_ORDER;

extern "C" {
    fn n2b_1_stage(
        inputs_matrix: *const GoldilocksField,
        outputs_matrix: *mut GoldilocksField,
        stride_between_input_arrays: u32,
        stride_between_output_arrays: u32,
        start_stage: u32,
        stages_this_launch: u32,
        log_n: u32,
        inverse: bool,
        blocks_per_ntt: u32,
        log_extension_degree: u32,
        coset_index: u32,
    );
    fn n2b_final_7_or_8_stages(
        inputs_matrix: *const GoldilocksField,
        outputs_matrix: *mut GoldilocksField,
        stride_between_input_arrays: u32,
        stride_between_output_arrays: u32,
        start_stage: u32,
        stages_this_launch: u32,
        log_n: u32,
        inverse: bool,
        blocks_per_ntt: u32,
        log_extension_degree: u32,
        coset_index: u32,
    );
    fn n2b_final_9_to_12_stages(
        inputs_matrix: *const GoldilocksField,
        outputs_matrix: *mut GoldilocksField,
        stride_between_input_arrays: u32,
        stride_between_output_arrays: u32,
        start_stage: u32,
        stages_this_launch: u32,
        log_n: u32,
        inverse: bool,
        blocks_per_ntt: u32,
        log_extension_degree: u32,
        coset_index: u32,
    );
    fn n2b_nonfinal_7_or_8_stages(
        inputs_matrix: *const GoldilocksField,
        outputs_matrix: *mut GoldilocksField,
        stride_between_input_arrays: u32,
        stride_between_output_arrays: u32,
        start_stage: u32,
        stages_this_launch: u32,
        log_n: u32,
        inverse: bool,
        blocks_per_ntt: u32,
        log_extension_degree: u32,
        coset_index: u32,
    );
    fn b2n_1_stage(
        inputs_matrix: *const GoldilocksField,
        outputs_matrix: *mut GoldilocksField,
        stride_between_input_arrays: u32,
        stride_between_output_arrays: u32,
        start_stage: u32,
        stages_this_launch: u32,
        log_n: u32,
        inverse: bool,
        blocks_per_ntt: u32,
        log_extension_degree: u32,
        coset_index: u32,
    );
    fn b2n_initial_7_or_8_stages(
        inputs_matrix: *const GoldilocksField,
        outputs_matrix: *mut GoldilocksField,
        stride_between_input_arrays: u32,
        stride_between_output_arrays: u32,
        start_stage: u32,
        stages_this_launch: u32,
        log_n: u32,
        inverse: bool,
        blocks_per_ntt: u32,
        log_extension_degree: u32,
        coset_index: u32,
    );
    fn b2n_initial_9_to_12_stages(
        inputs_matrix: *const GoldilocksField,
        outputs_matrix: *mut GoldilocksField,
        stride_between_input_arrays: u32,
        stride_between_output_arrays: u32,
        start_stage: u32,
        stages_this_launch: u32,
        log_n: u32,
        inverse: bool,
        blocks_per_ntt: u32,
        log_extension_degree: u32,
        coset_index: u32,
    );
    fn b2n_noninitial_7_or_8_stages(
        inputs_matrix: *const GoldilocksField,
        outputs_matrix: *mut GoldilocksField,
        stride_between_input_arrays: u32,
        stride_between_output_arrays: u32,
        start_stage: u32,
        stages_this_launch: u32,
        log_n: u32,
        inverse: bool,
        blocks_per_ntt: u32,
        log_extension_degree: u32,
        coset_index: u32,
    );
}

#[allow(non_camel_case_types)]
#[allow(clippy::upper_case_acronyms)]
enum KERN {
    N2B_FINAL_7_OR_8(u32),
    N2B_FINAL_9_TO_12(u32),
    N2B_NONFINAL_7_OR_8(u32),
    B2N_INITIAL_7_OR_8(u32),
    B2N_INITIAL_9_TO_12(u32),
    B2N_NONINITIAL_7_OR_8(u32),
    SKIP,
}

// Kernel plans for sizes 2^16..24.
// SKIPs are needed to keep arrays the same size.
// I'd rather use a hashmap containing vectors of different sizes instead of a list of fixed-size lists,
// but Rust didn't let me declare hashmaps or vectors const.
const PLANS: [[[KERN; 3]; 9]; 2] = [
    [
        [
            KERN::N2B_NONFINAL_7_OR_8(8),
            KERN::N2B_FINAL_7_OR_8(8),
            KERN::SKIP,
        ],
        [
            KERN::N2B_NONFINAL_7_OR_8(8),
            KERN::N2B_FINAL_9_TO_12(9),
            KERN::SKIP,
        ],
        [
            KERN::N2B_NONFINAL_7_OR_8(8),
            KERN::N2B_FINAL_9_TO_12(10),
            KERN::SKIP,
        ],
        [
            KERN::N2B_NONFINAL_7_OR_8(8),
            KERN::N2B_FINAL_9_TO_12(11),
            KERN::SKIP,
        ],
        [
            KERN::N2B_NONFINAL_7_OR_8(8),
            KERN::N2B_FINAL_9_TO_12(12),
            KERN::SKIP,
        ],
        [
            KERN::N2B_NONFINAL_7_OR_8(7),
            KERN::N2B_NONFINAL_7_OR_8(7),
            KERN::N2B_FINAL_7_OR_8(7),
        ],
        [
            KERN::N2B_NONFINAL_7_OR_8(7),
            KERN::N2B_NONFINAL_7_OR_8(7),
            KERN::N2B_FINAL_7_OR_8(8),
        ],
        [
            KERN::N2B_NONFINAL_7_OR_8(7),
            KERN::N2B_NONFINAL_7_OR_8(8),
            KERN::N2B_FINAL_7_OR_8(8),
        ],
        [
            KERN::N2B_NONFINAL_7_OR_8(8),
            KERN::N2B_NONFINAL_7_OR_8(8),
            KERN::N2B_FINAL_7_OR_8(8),
        ],
    ],
    [
        [
            KERN::B2N_INITIAL_7_OR_8(8),
            KERN::B2N_NONINITIAL_7_OR_8(8),
            KERN::SKIP,
        ],
        [
            KERN::B2N_INITIAL_9_TO_12(9),
            KERN::B2N_NONINITIAL_7_OR_8(8),
            KERN::SKIP,
        ],
        [
            KERN::B2N_INITIAL_9_TO_12(10),
            KERN::B2N_NONINITIAL_7_OR_8(8),
            KERN::SKIP,
        ],
        [
            KERN::B2N_INITIAL_9_TO_12(11),
            KERN::B2N_NONINITIAL_7_OR_8(8),
            KERN::SKIP,
        ],
        [
            KERN::B2N_INITIAL_9_TO_12(12),
            KERN::B2N_NONINITIAL_7_OR_8(8),
            KERN::SKIP,
        ],
        [
            KERN::B2N_INITIAL_7_OR_8(7),
            KERN::B2N_NONINITIAL_7_OR_8(7),
            KERN::B2N_NONINITIAL_7_OR_8(7),
        ],
        [
            KERN::B2N_INITIAL_7_OR_8(8),
            KERN::B2N_NONINITIAL_7_OR_8(7),
            KERN::B2N_NONINITIAL_7_OR_8(7),
        ],
        [
            KERN::B2N_INITIAL_7_OR_8(8),
            KERN::B2N_NONINITIAL_7_OR_8(8),
            KERN::B2N_NONINITIAL_7_OR_8(7),
        ],
        [
            KERN::B2N_INITIAL_7_OR_8(8),
            KERN::B2N_NONINITIAL_7_OR_8(8),
            KERN::B2N_NONINITIAL_7_OR_8(8),
        ],
    ],
];

#[allow(clippy::too_many_arguments)]
fn launch(
    nblocks: u32,
    nthreads: u32,
    smem_bytes: usize,
    stream: &CudaStream,
    kernel: *const c_void,
    inputs_matrix: *const GoldilocksField,
    outputs_matrix: *mut GoldilocksField,
    stride_between_input_arrays: u32,
    stride_between_output_arrays: u32,
    start_stage: u32,
    stages_this_launch: u32,
    log_n: u32,
    inverse: bool,
    blocks_per_ntt: u32,
    log_extension_degree: u32,
    coset_index: u32,
) -> CudaResult<()> {
    unsafe {
        let mut args = kernel_args!(
            &inputs_matrix,
            &outputs_matrix,
            &stride_between_input_arrays,
            &stride_between_output_arrays,
            &start_stage,
            &stages_this_launch,
            &log_n,
            &inverse,
            &blocks_per_ntt,
            &log_extension_degree,
            &coset_index
        );
        cudaLaunchKernel(
            kernel,
            nblocks.into(),
            nthreads.into(),
            args.as_mut_ptr(),
            smem_bytes,
            stream.into(),
        )
        .wrap()
    }
}

// Carries out LDE for all cosets in a single launch, which improves saturation for smaller sizes.
// results must contain 2^log_extension_degree DeviceAllocationSlices, to hold all the output cosets.
#[allow(clippy::too_many_arguments)]
pub fn batch_ntt_internal(
    inputs_ptr_in: *const GoldilocksField,
    outputs_ptr: *mut GoldilocksField,
    log_n: u32,
    num_ntts: u32,
    stride_between_input_arrays_in: u32,
    stride_between_output_arrays: u32,
    bitrev_inputs: bool,
    inverse: bool,
    log_extension_degree: u32,
    coset_index: u32,
    stream: &CudaStream,
) -> CudaResult<()> {
    const PADDED_WARP_SCRATCH_SIZE: usize = (256 / 16) * 17 + 1;
    assert!(log_n >= 1);
    assert!((log_n + log_extension_degree) <= OMEGA_LOG_ORDER);
    let n = 1 << log_n;
    assert!(n <= stride_between_input_arrays_in);
    assert!(n <= stride_between_output_arrays);
    if coset_index > 0 {
        assert!(log_extension_degree > 0);
        assert!(coset_index < (1 << log_extension_degree));
    }
    // The log_n < 16 path isn't performant, and is meant to unblock
    // small proofs for debugging purposes only.
    if log_n < 16 {
        let threads: u32 = 128;
        let n: u32 = 1 << log_n;
        let blocks_per_ntt: u32 = (n + 2 * threads - 1) / (2 * threads);
        let blocks = blocks_per_ntt * num_ntts;
        let kernel = if bitrev_inputs {
            b2n_1_stage
        } else {
            n2b_1_stage
        };
        for stage in 0..log_n {
            let inputs_ptr = if stage == 0 {
                inputs_ptr_in
            } else {
                outputs_ptr
            };
            let stride_between_input_arrays = if stage == 0 {
                stride_between_input_arrays_in
            } else {
                stride_between_output_arrays
            };
            launch(
                blocks,
                threads,
                0,
                stream,
                kernel as *const c_void,
                inputs_ptr,
                outputs_ptr,
                stride_between_input_arrays,
                stride_between_output_arrays,
                stage,
                1,
                log_n,
                inverse,
                blocks_per_ntt,
                log_extension_degree,
                coset_index,
            )?;
        }
        return Ok(());
    }
    let plan = &PLANS[bitrev_inputs as usize][log_n as usize - 16];
    let mut stage: u32 = 0;
    for kernel in plan {
        let start_stage = stage;
        // grid and block size for smem kernels
        let blocks_per_ntt_smem: u32 = n / 4096;
        let nthreads_smem: u32 = 512;
        let smem_bytes: usize = (512 / 32) * PADDED_WARP_SCRATCH_SIZE * 8;
        let total_blocks_smem: u32 = blocks_per_ntt_smem * num_ntts;
        // Raw input pointers
        let inputs_ptr = if stage == 0 {
            inputs_ptr_in
        } else {
            outputs_ptr
        };
        let stride_between_input_arrays = if stage == 0 {
            stride_between_input_arrays_in
        } else {
            stride_between_output_arrays
        };
        match kernel {
            KERN::N2B_FINAL_7_OR_8(stages) => {
                stage += stages;
                launch(
                    total_blocks_smem,
                    nthreads_smem,
                    smem_bytes,
                    stream,
                    n2b_final_7_or_8_stages as *const c_void,
                    inputs_ptr,
                    outputs_ptr,
                    stride_between_input_arrays,
                    stride_between_output_arrays,
                    start_stage,
                    *stages,
                    log_n,
                    inverse,
                    blocks_per_ntt_smem,
                    log_extension_degree,
                    coset_index,
                )
            }
            KERN::N2B_FINAL_9_TO_12(stages) => {
                stage += stages;
                launch(
                    total_blocks_smem,
                    nthreads_smem,
                    smem_bytes,
                    stream,
                    n2b_final_9_to_12_stages as *const c_void,
                    inputs_ptr,
                    outputs_ptr,
                    stride_between_input_arrays,
                    stride_between_output_arrays,
                    start_stage,
                    *stages,
                    log_n,
                    inverse,
                    blocks_per_ntt_smem,
                    log_extension_degree,
                    coset_index,
                )
            }
            KERN::N2B_NONFINAL_7_OR_8(stages) => {
                stage += stages;
                launch(
                    total_blocks_smem,
                    nthreads_smem,
                    smem_bytes,
                    stream,
                    n2b_nonfinal_7_or_8_stages as *const c_void,
                    inputs_ptr,
                    outputs_ptr,
                    stride_between_input_arrays,
                    stride_between_output_arrays,
                    start_stage,
                    *stages,
                    log_n,
                    inverse,
                    blocks_per_ntt_smem,
                    log_extension_degree,
                    coset_index,
                )
            }
            KERN::B2N_INITIAL_7_OR_8(stages) => {
                stage += stages;
                launch(
                    total_blocks_smem,
                    nthreads_smem,
                    smem_bytes,
                    stream,
                    b2n_initial_7_or_8_stages as *const c_void,
                    inputs_ptr,
                    outputs_ptr,
                    stride_between_input_arrays,
                    stride_between_output_arrays,
                    start_stage,
                    *stages,
                    log_n,
                    inverse,
                    blocks_per_ntt_smem,
                    log_extension_degree,
                    coset_index,
                )
            }
            KERN::B2N_INITIAL_9_TO_12(stages) => {
                stage += stages;
                launch(
                    total_blocks_smem,
                    nthreads_smem,
                    smem_bytes,
                    stream,
                    b2n_initial_9_to_12_stages as *const c_void,
                    inputs_ptr,
                    outputs_ptr,
                    stride_between_input_arrays,
                    stride_between_output_arrays,
                    start_stage,
                    *stages,
                    log_n,
                    inverse,
                    blocks_per_ntt_smem,
                    log_extension_degree,
                    coset_index,
                )
            }
            KERN::B2N_NONINITIAL_7_OR_8(stages) => {
                stage += stages;
                launch(
                    total_blocks_smem,
                    nthreads_smem,
                    smem_bytes,
                    stream,
                    b2n_noninitial_7_or_8_stages as *const c_void,
                    inputs_ptr,
                    outputs_ptr,
                    stride_between_input_arrays,
                    stride_between_output_arrays,
                    start_stage,
                    *stages,
                    log_n,
                    inverse,
                    blocks_per_ntt_smem,
                    log_extension_degree,
                    coset_index,
                )
            }
            KERN::SKIP => get_last_error().wrap(),
        }?;
    }
    assert_eq!(stage, log_n);
    get_last_error().wrap()
}

#[allow(clippy::too_many_arguments)]
pub fn batch_ntt_out_of_place(
    inputs_matrix: &DeviceSlice<GoldilocksField>,
    outputs_matrix: &mut DeviceSlice<GoldilocksField>,
    log_n: u32,
    num_ntts: u32,
    inputs_offset: u32,
    outputs_offset: u32,
    stride_between_input_arrays: u32,
    stride_between_output_arrays: u32,
    bitrev_inputs: bool,
    inverse: bool,
    log_extension_degree: u32,
    coset_index: u32,
    stream: &CudaStream,
) -> CudaResult<()> {
    let n = 1u32 << log_n;
    assert!(inputs_offset + n <= stride_between_input_arrays);
    assert!(outputs_offset + n <= stride_between_output_arrays);
    assert!(inputs_matrix.len() >= (num_ntts * stride_between_input_arrays) as usize);
    assert!(outputs_matrix.len() >= (num_ntts * stride_between_output_arrays) as usize);

    let inputs_matrix_ptr = inputs_matrix.as_ptr();
    let outputs_matrix_ptr = outputs_matrix.as_mut_ptr();
    assert!(inputs_matrix_ptr != outputs_matrix_ptr);
    let inputs_matrix_ptr = unsafe { inputs_matrix_ptr.add(inputs_offset as usize) };
    let outputs_matrix_ptr = unsafe { outputs_matrix_ptr.add(outputs_offset as usize) };
    assert!(inputs_matrix_ptr != outputs_matrix_ptr); // might as well recheck after applying offsets

    batch_ntt_internal(
        inputs_matrix_ptr,
        outputs_matrix_ptr,
        log_n,
        num_ntts,
        stride_between_input_arrays,
        stride_between_output_arrays,
        bitrev_inputs,
        inverse,
        log_extension_degree,
        coset_index,
        stream,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn batch_ntt_in_place(
    inputs_matrix: &mut DeviceSlice<GoldilocksField>,
    log_n: u32,
    num_ntts: u32,
    inputs_offset: u32,
    stride_between_input_arrays: u32,
    bitrev_inputs: bool,
    inverse: bool,
    log_extension_degree: u32,
    coset_index: u32,
    stream: &CudaStream,
) -> CudaResult<()> {
    let n = 1u32 << log_n;
    assert!(inputs_offset + n <= stride_between_input_arrays);
    assert!(inputs_matrix.len() >= (num_ntts * stride_between_input_arrays) as usize);

    let inputs_matrix_ptr = unsafe { inputs_matrix.as_mut_ptr().add(inputs_offset as usize) };

    batch_ntt_internal(
        inputs_matrix_ptr,
        inputs_matrix_ptr,
        log_n,
        num_ntts,
        stride_between_input_arrays,
        stride_between_input_arrays,
        bitrev_inputs,
        inverse,
        log_extension_degree,
        coset_index,
        stream,
    )
}

#[cfg(test)]
mod tests {
    use std::alloc::Global;
    use std::ops::Range;

    use boojum::cs::implementations::utils::{
        domain_generator_for_size, precompute_twiddles_for_fft,
    };
    use boojum::fft::{
        bitreverse_enumeration_inplace, fft_natural_to_bitreversed, ifft_natural_to_natural,
    };
    use boojum::field::{Field, PrimeField};
    use boojum::worker::Worker;
    use rand::Rng;
    use serial_test::serial;

    use cudart::memory::{memory_copy_async, CudaHostAllocFlags, DeviceAllocation, HostAllocation};

    use crate::context::Context;

    use super::*;

    fn correctness(
        log_n_range: Range<u32>,
        inverse: bool,
        log_extension_degree: u32,
        coset_index: u32,
        num_ntts: u32,
    ) {
        let ctx = Context::create(12, 12).unwrap();
        let n_max = 1 << (log_n_range.end - 1);
        let worker = Worker::new();
        // The CPU NTT uses bitreved twiddles, so it turns out the twiddles for a size-N/2 NTT are just the first
        // N/2 elements of the twiddles for a size-N NTT. In other words, we can pregenerate just the max-size
        // array, and use slices of it for smaller sizes.
        let twiddles = if inverse {
            precompute_twiddles_for_fft::<GoldilocksField, GoldilocksField, Global, true>(
                n_max,
                &worker,
                &mut (),
            )
        } else {
            precompute_twiddles_for_fft::<GoldilocksField, GoldilocksField, Global, false>(
                n_max,
                &worker,
                &mut (),
            )
        };
        let mut rng = rand::thread_rng();
        const OFFSET_MULTIPLIER: u32 = 1;
        let io_stride: u32 = (n_max as u32) * (OFFSET_MULTIPLIER + 1);
        let io_size = (io_stride * num_ntts) as usize;
        // Using parallel rng generation, as in the benches, does not reduce runtime noticeably
        let mut inputs_host =
            HostAllocation::<GoldilocksField>::alloc(io_size, CudaHostAllocFlags::DEFAULT).unwrap();
        inputs_host.fill_with(|| GoldilocksField(rng.gen()));
        let mut inputs_bitrev_host =
            HostAllocation::<GoldilocksField>::alloc(io_size, CudaHostAllocFlags::DEFAULT).unwrap();
        let mut outputs_n2b_in_place_host =
            HostAllocation::<GoldilocksField>::alloc(io_size, CudaHostAllocFlags::DEFAULT).unwrap();
        let mut outputs_n2b_out_of_place_host =
            HostAllocation::<GoldilocksField>::alloc(io_size, CudaHostAllocFlags::DEFAULT).unwrap();
        let mut outputs_b2n_in_place_host =
            HostAllocation::<GoldilocksField>::alloc(io_size, CudaHostAllocFlags::DEFAULT).unwrap();
        let mut outputs_b2n_out_of_place_host =
            HostAllocation::<GoldilocksField>::alloc(io_size, CudaHostAllocFlags::DEFAULT).unwrap();
        let mut inputs_device = DeviceAllocation::<GoldilocksField>::alloc(io_size).unwrap();
        let mut outputs_device = DeviceAllocation::<GoldilocksField>::alloc(io_size).unwrap();
        let stream = CudaStream::default();
        for log_n in log_n_range {
            let n = 1u32 << log_n;
            let io_offset = n * OFFSET_MULTIPLIER;
            for (chunk_in, chunk_out) in inputs_host
                .chunks(n as usize)
                .zip(inputs_bitrev_host.chunks_mut(n as usize))
            {
                chunk_out.copy_from_slice(chunk_in);
                bitreverse_enumeration_inplace(chunk_out);
            }

            // Nonbitrev to bitrev, in-place
            memory_copy_async(&mut inputs_device, &inputs_host, &stream).unwrap();
            batch_ntt_in_place(
                &mut inputs_device,
                log_n,
                num_ntts,
                io_offset,
                io_stride,
                false,
                inverse,
                log_extension_degree,
                coset_index,
                &stream,
            )
            .unwrap();
            memory_copy_async(&mut outputs_n2b_in_place_host, &inputs_device, &stream).unwrap();

            // Nonbitrev to bitrev, out of place
            memory_copy_async(&mut inputs_device, &inputs_host, &stream).unwrap();
            batch_ntt_out_of_place(
                &inputs_device,
                &mut outputs_device,
                log_n,
                num_ntts,
                io_offset,
                io_offset,
                io_stride,
                io_stride,
                false,
                inverse,
                log_extension_degree,
                coset_index,
                &stream,
            )
            .unwrap();
            memory_copy_async(&mut outputs_n2b_out_of_place_host, &outputs_device, &stream)
                .unwrap();

            // // Bitrev to nonbitrev, in-place
            memory_copy_async(&mut inputs_device, &inputs_bitrev_host, &stream).unwrap();
            batch_ntt_in_place(
                &mut inputs_device,
                log_n,
                num_ntts,
                io_offset,
                io_stride,
                true,
                inverse,
                log_extension_degree,
                coset_index,
                &stream,
            )
            .unwrap();
            memory_copy_async(&mut outputs_b2n_in_place_host, &inputs_device, &stream).unwrap();

            // Bitrev to nonbitrev, out of place
            memory_copy_async(&mut inputs_device, &inputs_bitrev_host, &stream).unwrap();
            batch_ntt_out_of_place(
                &inputs_device,
                &mut outputs_device,
                log_n,
                num_ntts,
                io_offset,
                io_offset,
                io_stride,
                io_stride,
                true,
                inverse,
                log_extension_degree,
                coset_index,
                &stream,
            )
            .unwrap();
            memory_copy_async(&mut outputs_b2n_out_of_place_host, &outputs_device, &stream)
                .unwrap();

            stream.synchronize().unwrap();

            let mut coset = GoldilocksField::ONE;
            if log_extension_degree > 0 {
                let lde_size: u64 = (n << log_extension_degree) as u64;
                let omega = domain_generator_for_size::<GoldilocksField>(lde_size);
                let mut root_for_coset = GoldilocksField::ONE;
                for _ in 0..coset_index {
                    root_for_coset.mul_assign(&omega);
                }
                coset.mul_assign(&GoldilocksField::multiplicative_generator());
                coset.mul_assign(&root_for_coset);
            }

            for ntt in 0..num_ntts {
                let ntt = ntt as usize;
                let stride = io_stride as usize;
                let mut values = vec![GoldilocksField::ZERO; n as usize];
                values.copy_from_slice(
                    inputs_host
                        .chunks(stride)
                        .nth(ntt)
                        .unwrap()
                        .chunks(n as usize)
                        .nth(OFFSET_MULTIPLIER as usize)
                        .unwrap(),
                );
                let twiddles = &twiddles[..(n as usize >> 1)];
                fn get_chunk_mut(
                    array: &mut [GoldilocksField],
                    stride: usize,
                    ntt: usize,
                    n: u32,
                ) -> &mut [GoldilocksField] {
                    array
                        .chunks_mut(stride)
                        .nth(ntt)
                        .unwrap()
                        .chunks_mut(n as usize)
                        .nth(OFFSET_MULTIPLIER as usize)
                        .unwrap()
                }
                let results_n2b_in_place =
                    get_chunk_mut(&mut outputs_n2b_in_place_host, stride, ntt, n);
                let results_n2b_out_of_place =
                    get_chunk_mut(&mut outputs_n2b_out_of_place_host, stride, ntt, n);
                let results_b2n_in_place =
                    get_chunk_mut(&mut outputs_b2n_in_place_host, stride, ntt, n);
                let results_b2n_out_of_place =
                    get_chunk_mut(&mut outputs_b2n_out_of_place_host, stride, ntt, n);
                if inverse {
                    ifft_natural_to_natural(&mut values, coset, twiddles);
                    bitreverse_enumeration_inplace(results_n2b_in_place);
                    bitreverse_enumeration_inplace(results_n2b_out_of_place);
                } else {
                    fft_natural_to_bitreversed(&mut values, coset, twiddles);
                    bitreverse_enumeration_inplace(results_b2n_in_place);
                    bitreverse_enumeration_inplace(results_b2n_out_of_place);
                }

                for i in 0..n as usize {
                    let value = values[i];
                    assert_eq!(
                        value, results_n2b_in_place[i],
                        "Natural to bitrev in-place results incorrect for size 2^{}, ntt {}, coset_index {}, index {}",
                        log_n, ntt, coset_index, i
                    );
                    assert_eq!(
                        value, results_n2b_out_of_place[i],
                        "Natural to bitrev out of place results incorrect for size 2^{}, ntt {}, coset_index {}, index {}",
                        log_n, ntt, coset_index, i
                    );
                    assert_eq!(
                        value, results_b2n_in_place[i],
                        "Bitrev to natural in-place results incorrect for size 2^{}, ntt {}, coset_index {}, index {}",
                        log_n, ntt, coset_index, i
                    );
                    assert_eq!(
                        value, results_b2n_out_of_place[i],
                        "Bitrev to natural out of place results incorrect for size 2^{}, ntt {}, coset_index {}, index {}",
                        log_n, ntt, coset_index, i
                    );
                }
            }
        }
        ctx.destroy().unwrap();
    }

    #[test]
    #[serial]
    fn correctness_batch_ntt_fwd() {
        correctness(1..17, false, 0, 0, 3);
    }

    #[test]
    #[serial]
    #[ignore]
    fn correctness_batch_ntt_fwd_large() {
        correctness(17..25, false, 0, 0, 3);
    }

    #[test]
    #[serial]
    fn correctness_batch_lde_fwd() {
        correctness(1..17, false, 3, 3, 3);
    }

    #[test]
    #[serial]
    #[ignore]
    fn correctness_batch_lde_fwd_large() {
        correctness(17..22, false, 3, 3, 3);
    }

    #[test]
    #[serial]
    fn correctness_batch_ntt_inv() {
        correctness(1..17, true, 0, 0, 1);
    }

    #[test]
    #[serial]
    #[ignore]
    fn correctness_batch_ntt_inv_large() {
        correctness(17..25, true, 0, 0, 1);
    }

    #[test]
    #[serial]
    fn correctness_batch_lde_inv() {
        correctness(1..17, true, 3, 3, 3);
    }

    #[test]
    #[serial]
    #[ignore]
    fn correctness_batch_lde_inv_large() {
        correctness(17..22, true, 3, 3, 3);
    }
}
