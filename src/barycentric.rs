use crate::device_structures::{
    DeviceMatrix, DeviceMatrixImpl, DeviceMatrixMut, DeviceMatrixMutImpl, DeviceRepr,
    DeviceVectorImpl, DeviceVectorMutImpl, MutPtrAndStride, PtrAndStride, Vectorized,
};
use crate::extension_field::ExtensionField;
use crate::ops_complex::bit_reverse_in_place;
use crate::ops_cub::device_reduce::{
    batch_reduce, get_batch_reduce_temp_storage_bytes, ReduceOperation,
};
use crate::utils::WARP_SIZE;
use crate::BaseField;
use boojum::cs::implementations::utils::domain_generator_for_size;
use boojum::field::{Field, PrimeField};
use cudart::execution::{CudaLaunchConfig, Dim3, KernelFunction};
use cudart::paste::paste;
use cudart::result::CudaResult;
use cudart::slice::{DeviceSlice, DeviceVariable};
use cudart::stream::CudaStream;
use cudart::{cuda_kernel_declaration, cuda_kernel_signature_arguments_and_function};
use std::cmp;

type BF = BaseField;
type EF = ExtensionField;

cuda_kernel_signature_arguments_and_function!(
    PrecomputeCommonFactor<T>,
    x: *const T,
    common_factor: *mut T,
    coset: BF,
    count: u32,
);

macro_rules! precompute_common_factor_kernel {
    ($type:ty) => {
        paste! {
            cuda_kernel_declaration!(
                [<barycentric_precompute_common_factor_ $type:lower _kernel>](
                    x: *const $type,
                    common_factor: *mut $type,
                    coset: BF,
                    count: u32,
                )
            );
        }
    };
}

cuda_kernel_signature_arguments_and_function!(
    PrecomputeLagrangeCoeffs<T: Vectorized>,
    x: *const T,
    common_factor: *const T,
    w_inv_step: BF,
    coset: BF,
    lagrange_coeffs: MutPtrAndStride<<<T as Vectorized>::Type as DeviceRepr>::Type>,
    log_count: u32,
);

macro_rules! precompute_lagrange_coeffs_kernel {
    ($type:ty) => {
        paste! {
            cuda_kernel_declaration!(
                [<barycentric_precompute_lagrange_coeffs_ $type:lower _kernel>](
                    x: *const $type,
                    common_factor: *const $type,
                    w_inv_step: BF,
                    coset: BF,
                    lagrange_coeffs: MutPtrAndStride<<<$type as Vectorized>::Type as DeviceRepr>::Type>,
                    log_count: u32,
                )
            );
        }
    };
}

pub trait PrecomputeImpl {
    type X: Vectorized;
    const BF_ELEMS_COUNT: u32;
    const INV_BATCH_SIZE: u32;
    const COMMON_FACTOR_FUNCTION: PrecomputeCommonFactorSignature<Self::X>;
    const LAGRANGE_COEFFS_FUNCTION: PrecomputeLagrangeCoeffsSignature<Self::X>;
}

pub fn precompute_lagrange_coeffs<T: PrecomputeImpl>(
    x: &DeviceVariable<T::X>,
    common_factor_storage: &mut DeviceVariable<T::X>,
    coset: BF,
    lagrange_coeffs: &mut (impl DeviceVectorMutImpl<<T::X as Vectorized>::Type> + ?Sized),
    bit_reverse: bool,
    stream: &CudaStream,
) -> CudaResult<()> {
    let inv_batch: u32 = T::INV_BATCH_SIZE;
    assert!(lagrange_coeffs.slice().len() <= u32::MAX as usize);
    assert!(lagrange_coeffs.slice().len().is_power_of_two());
    let count = lagrange_coeffs.slice().len() as u32;
    let x_arg = x.as_ptr();
    let common_factor = common_factor_storage.as_mut_ptr();
    let config = CudaLaunchConfig::basic(1, 1, stream);
    let args = PrecomputeCommonFactorArguments::new(x_arg, common_factor, coset, count);
    PrecomputeCommonFactorFunction(T::COMMON_FACTOR_FUNCTION).launch(&config, &args)?;
    let log_count: u32 = count.trailing_zeros();
    let block_dim = WARP_SIZE * 4;
    let grid_dim = (count + inv_batch * block_dim - 1) / (inv_batch * block_dim);
    let w = domain_generator_for_size::<BF>((1 << log_count) as u64);
    let w_inv = w.inverse().expect("inverse of omega must exist");
    let w_inv_step = w_inv.pow_u64((block_dim * grid_dim) as u64);
    let common_factor = common_factor_storage.as_ptr();
    let dst = lagrange_coeffs.as_mut_ptr_and_stride();
    let config = CudaLaunchConfig::basic(grid_dim, block_dim, stream);
    let args = PrecomputeLagrangeCoeffsArguments::new(
        x_arg,
        common_factor,
        w_inv_step,
        coset,
        dst,
        log_count,
    );
    PrecomputeLagrangeCoeffsFunction(T::LAGRANGE_COEFFS_FUNCTION).launch(&config, &args)?;
    if bit_reverse {
        let slice = lagrange_coeffs.slice_mut();
        let stride = slice.len();
        let slice = unsafe { slice.transmute_mut::<BF>() };
        let mut matrix = DeviceMatrixMut::new(slice, stride);
        bit_reverse_in_place(&mut matrix, stream)?;
    }
    Ok(())
}

macro_rules! precompute_impl {
    ($name:ident, $type:ty, $ec:expr, $bs:expr) => {
        paste! {
            pub struct $name {}
            precompute_common_factor_kernel!($type);
            precompute_lagrange_coeffs_kernel!($type);
            impl PrecomputeImpl for $name {
                type X = $type;
                const BF_ELEMS_COUNT: u32 = $ec;
                const INV_BATCH_SIZE: u32 = $bs;
                const COMMON_FACTOR_FUNCTION: PrecomputeCommonFactorSignature<Self::X> =
                    [<barycentric_precompute_common_factor_ $type:lower _kernel>];
                const LAGRANGE_COEFFS_FUNCTION: PrecomputeLagrangeCoeffsSignature<Self::X> =
                    [<barycentric_precompute_lagrange_coeffs_ $type:lower _kernel>];
            }
        }
    };
}

precompute_impl!(PrecomputeAtBase, BF, 1, 10);
precompute_impl!(PrecomputeAtExt, EF, 2, 6);

cuda_kernel_signature_arguments_and_function!(
    PartialReduce<X: Vectorized, Y: Vectorized>,
    batch_ys: PtrAndStride<<<Y as Vectorized>::Type as DeviceRepr>::Type>,
    lagrange_coeffs: PtrAndStride<<<X as Vectorized>::Type as DeviceRepr>::Type>,
    partial_sums: MutPtrAndStride<<<X as Vectorized>::Type as DeviceRepr>::Type>,
    log_count: u32,
    num_polys: u32,
);

macro_rules! partial_reduce_kernel {
    ($tx:ty, $ty:ty) => {
        paste! {
            cuda_kernel_declaration!(
                [<batch_barycentric_partial_reduce_ $ty:lower _ $tx:lower _kernel>](
                    batch_ys: PtrAndStride<<<$ty as Vectorized>::Type as DeviceRepr>::Type>,
                    lagrange_coeffs: PtrAndStride<<<$tx as Vectorized>::Type as DeviceRepr>::Type>,
                    partial_sums: MutPtrAndStride<<<$tx as Vectorized>::Type as DeviceRepr>::Type>,
                    log_count: u32,
                    num_polys: u32,
                )
            );
        }
    };
}

pub trait EvalImpl {
    type X: Vectorized;
    type Y: Vectorized;
    const BF_ELEMS_COUNT: u32;
    const PARTIAL_REDUCE_ELEMS_PER_THREAD: u32;
    const PARTIAL_REDUCE_FUNCTION: PartialReduceSignature<Self::X, Self::Y>;
}

fn get_batch_partial_reduce_grid_block<T: EvalImpl>(count: u32) -> (Dim3, u32, u32) {
    let elems_per_thread: u32 = T::PARTIAL_REDUCE_ELEMS_PER_THREAD;
    let block_dim_x = WARP_SIZE;
    let grid_dim = (count + elems_per_thread * block_dim_x - 1) / (elems_per_thread * block_dim_x);
    let mut block_dim: Dim3 = block_dim_x.into();
    block_dim.y = WARP_SIZE;
    let grid_size_x = block_dim_x * grid_dim;
    (block_dim, grid_dim, grid_size_x)
}

pub fn get_batch_eval_temp_storage_sizes<T: EvalImpl>(
    batch_ys: &(impl DeviceMatrixImpl<<T::Y as Vectorized>::Type> + ?Sized),
) -> CudaResult<(usize, usize)> {
    assert!(batch_ys.stride() <= u32::MAX as usize);
    assert_eq!(batch_ys.stride().count_ones(), 1);
    let count = batch_ys.stride() as u32;
    let num_polys = batch_ys.cols() as u32;
    let (_, _, grid_size_x) = get_batch_partial_reduce_grid_block::<T>(count);
    let partial_reduce_temp_elems = (num_polys * cmp::min(count, grid_size_x)) as usize;
    let elems_count: u32 = T::BF_ELEMS_COUNT;
    let final_cub_reduce_temp_bytes = get_batch_reduce_temp_storage_bytes::<BF>(
        ReduceOperation::Sum,
        (elems_count * num_polys) as i32,
        (elems_count * count) as i32,
    )?;
    Ok((partial_reduce_temp_elems, final_cub_reduce_temp_bytes))
}

pub fn batch_eval<T: EvalImpl>(
    batch_ys: &(impl DeviceMatrixImpl<<T::Y as Vectorized>::Type> + ?Sized),
    lagrange_coeffs: &(impl DeviceVectorImpl<<T::X as Vectorized>::Type> + ?Sized),
    temp_storage_partial_reduce: &mut (impl DeviceMatrixMutImpl<<T::X as Vectorized>::Type> + ?Sized),
    temp_storage_final_cub_reduce: &mut DeviceSlice<u8>,
    evals: &mut (impl DeviceVectorMutImpl<T::X> + ?Sized),
    stream: &CudaStream,
) -> CudaResult<()> {
    assert!(batch_ys.stride() <= u32::MAX as usize);
    assert_eq!(batch_ys.stride().count_ones(), 1);
    let count = batch_ys.stride() as u32;
    let num_polys = batch_ys.cols() as u32;
    let log_count = count.trailing_zeros();
    assert_eq!(evals.slice().len() as u32, num_polys);
    let (block_dim, grid_dim, _) = get_batch_partial_reduce_grid_block::<T>(count);
    // double-check
    let (partial_reduce_temp_elems, final_cub_reduce_temp_bytes) =
        get_batch_eval_temp_storage_sizes::<T>(batch_ys)?;
    assert_eq!(
        temp_storage_partial_reduce.slice().len(),
        partial_reduce_temp_elems
    );
    assert_eq!(
        temp_storage_final_cub_reduce.len(),
        final_cub_reduce_temp_bytes
    );
    let src = batch_ys.as_ptr_and_stride();
    let coeffs = lagrange_coeffs.as_ptr_and_stride();
    let dst = temp_storage_partial_reduce.as_mut_ptr_and_stride();
    let config = CudaLaunchConfig::basic(grid_dim, block_dim, stream);
    let args = PartialReduceArguments::<T::X, T::Y>::new(src, coeffs, dst, log_count, num_polys);
    PartialReduceFunction::<T::X, T::Y>(T::PARTIAL_REDUCE_FUNCTION).launch(&config, &args)?;
    let stride = temp_storage_partial_reduce.stride();
    let temp_storage_partial_reduce_view = unsafe {
        temp_storage_partial_reduce
            .slice_mut()
            .transmute_mut::<BF>()
    };
    let temp_storage_partial_reduce_matrix =
        DeviceMatrix::new(temp_storage_partial_reduce_view, stride);
    let evals_view = unsafe { evals.slice_mut().transmute_mut::<BF>() };
    batch_reduce::<BF, _>(
        ReduceOperation::Sum,
        temp_storage_final_cub_reduce,
        &temp_storage_partial_reduce_matrix,
        evals_view,
        stream,
    )
    // maybe "transpose" layout of batch results to vectorized using Convert
}

macro_rules! eval_impl {
    ($name:ident, $tx:ty, $ty:ty, $ec:expr, $ept:expr) => {
        paste! {
            pub struct $name {}
            partial_reduce_kernel!($tx, $ty);
            impl EvalImpl for $name {
                type X = $tx;
                type Y = $ty;
                const BF_ELEMS_COUNT: u32 = $ec;
                const PARTIAL_REDUCE_ELEMS_PER_THREAD: u32 = $ept;
                const PARTIAL_REDUCE_FUNCTION: PartialReduceSignature<Self::X, Self::Y> =
                    [<batch_barycentric_partial_reduce_ $ty:lower _ $tx:lower _kernel>];
            }
        }
    };
}

eval_impl!(EvalBaseAtBase, BF, BF, 1, 12);
eval_impl!(EvalBaseAtExt, EF, BF, 2, 6);
eval_impl!(EvalExtAtExt, EF, EF, 2, 6);

#[cfg(test)]
mod tests {
    use crate::context::Context;
    use crate::device_structures::{DeviceMatrix, DeviceMatrixMut};
    use crate::extension_field::test_helpers::{transmute_gf_vec, ExtensionFieldTest};
    use crate::extension_field::{ExtensionField, VectorizedExtensionField};
    use crate::ntt::batch_ntt_in_place;
    use crate::BaseField;
    use boojum::cs::implementations::utils::{
        precompute_for_barycentric_evaluation, precompute_for_barycentric_evaluation_in_extension,
    };
    use boojum::field::goldilocks::GoldilocksExt2;
    use boojum::field::{rand_from_rng, Field, PrimeField, U64Representable};
    use boojum::worker::Worker;
    use cudart::memory::{memory_copy_async, DeviceAllocation};
    use cudart::stream::CudaStream;
    use rand::{thread_rng, Rng};
    use serial_test::serial;
    use std::alloc::Global;

    type BF = BaseField;
    type EF = ExtensionField;
    type VF = VectorizedExtensionField;

    #[test]
    #[serial]
    fn test_precompute_lagrange_coeffs_at_base() {
        let context = Context::create(12, 12).unwrap();
        let worker = Worker::new();
        for log_count in 0..20 {
            let count = 1usize << log_count;
            let h_x: [BF; 1] = [BF::from_u64_unchecked(thread_rng().gen())];
            let mut h_dst = vec![BF::ZERO; count];
            let coset = BF::multiplicative_generator();
            let mut d_x = DeviceAllocation::<BF>::alloc(1).unwrap();
            let mut d_common_factor_storage = DeviceAllocation::<BF>::alloc(1).unwrap();
            let mut d_dst = DeviceAllocation::<BF>::alloc(count).unwrap();
            let stream = CudaStream::default();
            memory_copy_async(&mut d_x, &h_x, &stream).unwrap();
            super::precompute_lagrange_coeffs::<super::PrecomputeAtBase>(
                &d_x[0],
                &mut d_common_factor_storage[0],
                coset,
                &mut d_dst,
                true,
                &stream,
            )
            .unwrap();
            memory_copy_async(&mut h_dst, &d_dst, &stream).unwrap();
            stream.synchronize().unwrap();
            let precomps: Vec<BF> =
                precompute_for_barycentric_evaluation(count, coset, h_x[0], &worker, &mut ());
            for i in 0..count {
                assert_eq!(
                    h_dst[i], precomps[i],
                    "log_count {} at index {}",
                    log_count, i
                );
            }
        }
        context.destroy().unwrap();
    }

    #[test]
    #[serial]
    fn test_precompute_lagrange_coeffs_at_ext() {
        let context = Context::create(12, 12).unwrap();
        let worker = Worker::new();
        for log_count in 0..20 {
            let count = 1usize << log_count;
            let c0 = BF::from_u64_unchecked(thread_rng().gen());
            let c1 = BF::from_u64_unchecked(thread_rng().gen());
            let h_x: [EF; 1] = [EF::from_coeff_in_base([c0, c1])];
            let h_dst_storage = vec![BF::ZERO; count << 1];
            let mut h_dst = transmute_gf_vec::<VF>(h_dst_storage);
            let coset = BF::multiplicative_generator();
            let mut d_x = DeviceAllocation::<EF>::alloc(1).unwrap();
            let mut d_common_factor_storage = DeviceAllocation::<EF>::alloc(1).unwrap();
            let mut d_dst = DeviceAllocation::<VF>::alloc(count).unwrap();
            let stream = CudaStream::default();
            memory_copy_async(&mut d_x, &h_x, &stream).unwrap();
            super::precompute_lagrange_coeffs::<super::PrecomputeAtExt>(
                &d_x[0],
                &mut d_common_factor_storage[0],
                coset,
                &mut d_dst,
                true,
                &stream,
            )
            .unwrap();
            memory_copy_async(&mut h_dst, &d_dst, &stream).unwrap();
            stream.synchronize().unwrap();
            let [precomps_c0, precomps_c1] = precompute_for_barycentric_evaluation_in_extension::<
                BF,
                GoldilocksExt2,
                BF,
                Global,
            >(count, coset, h_x[0], &worker, &mut ());
            let iterator = VF::get_iterator(&h_dst);
            for (i, val) in iterator.enumerate() {
                assert_eq!(
                    val,
                    EF::from_coeff_in_base([precomps_c0[i], precomps_c1[i]]),
                    "log_count {} at index {}",
                    log_count,
                    i
                );
            }
        }
        context.destroy().unwrap();
    }

    #[test]
    #[serial]
    fn test_batch_eval_base_at_base() {
        let context = Context::create(12, 12).unwrap();
        let coset = BF::multiplicative_generator();
        let stream = CudaStream::default();
        let num_polys_to_test = [1, 3, 32, 33];
        for num_polys in num_polys_to_test.into_iter() {
            for log_count in 0..17 {
                let count = 1 << log_count;
                let mut rng = thread_rng();
                let coeffs: Vec<_> = (0..(num_polys * count))
                    .map(|_| rand_from_rng::<_, BF>(&mut rng))
                    .collect();
                let x = rand_from_rng::<_, BF>(&mut rng);
                let mut naive = vec![BF::ZERO; num_polys];
                for poly in 0..num_polys {
                    let coeffs_this_poly = &coeffs[(poly * count)..((poly + 1) * count)];
                    let mut current = BF::ONE;
                    for el in coeffs_this_poly.iter() {
                        let mut tmp = *el;
                        tmp.mul_assign(&current);
                        naive[poly].add_assign(&tmp);
                        current.mul_assign(&x);
                    }
                }
                let mut d_batch_ys = DeviceAllocation::alloc(num_polys * count).unwrap();
                memory_copy_async(&mut d_batch_ys, &coeffs, &stream).unwrap();
                if log_count > 0 {
                    batch_ntt_in_place(
                        &mut d_batch_ys,
                        log_count,
                        num_polys as u32,
                        0,
                        count as u32,
                        false,
                        false,
                        1,
                        0,
                        &stream,
                    )
                    .unwrap();
                }
                let h_x = [x];
                let mut d_x = DeviceAllocation::<BF>::alloc(1).unwrap();
                let mut d_common_factor_storage = DeviceAllocation::<BF>::alloc(1).unwrap();
                let mut d_lagrange_coeffs = DeviceAllocation::alloc(count).unwrap();
                memory_copy_async(&mut d_x, &h_x, &stream).unwrap();
                super::precompute_lagrange_coeffs::<super::PrecomputeAtBase>(
                    &d_x[0],
                    &mut d_common_factor_storage[0],
                    coset,
                    &mut d_lagrange_coeffs,
                    true,
                    &stream,
                )
                .unwrap();
                let d_batch_ys = DeviceMatrix::new(&d_batch_ys, count);
                let (partial_reduce_temp_elems, final_cub_reduce_temp_bytes) =
                    super::get_batch_eval_temp_storage_sizes::<super::EvalBaseAtBase>(&d_batch_ys)
                        .unwrap();
                let mut temp_storage_partial_reduce =
                    DeviceAllocation::alloc(partial_reduce_temp_elems).unwrap();
                let mut temp_storage_partial_reduce = DeviceMatrixMut::new(
                    &mut temp_storage_partial_reduce,
                    partial_reduce_temp_elems / num_polys,
                );
                let mut temp_storage_final_cub_reduce =
                    DeviceAllocation::alloc(final_cub_reduce_temp_bytes).unwrap();
                let mut d_evals = DeviceAllocation::alloc(num_polys).unwrap();
                super::batch_eval::<super::EvalBaseAtBase>(
                    &d_batch_ys,
                    &d_lagrange_coeffs,
                    &mut temp_storage_partial_reduce,
                    &mut temp_storage_final_cub_reduce,
                    &mut d_evals,
                    &stream,
                )
                .unwrap();
                let mut h_evals = vec![BF::ZERO; num_polys];
                memory_copy_async(&mut h_evals, &d_evals, &stream).unwrap();
                stream.synchronize().unwrap();
                for i in 0..num_polys {
                    assert_eq!(
                        h_evals[i], naive[i],
                        "log_count {}, num_polys {}, poly {}",
                        log_count, num_polys, i
                    );
                }
            }
        }
        context.destroy().unwrap();
    }

    #[test]
    #[serial]
    fn test_batch_eval_base_at_ext() {
        let context = Context::create(12, 12).unwrap();
        let coset = BF::multiplicative_generator();
        let stream = CudaStream::default();
        let num_polys_to_test = [1, 3, 32, 33];
        for num_polys in num_polys_to_test.into_iter() {
            for log_count in 0..17 {
                let count = 1 << log_count;
                let mut rng = thread_rng();
                let coeffs: Vec<_> = (0..(num_polys * count))
                    .map(|_| rand_from_rng::<_, BF>(&mut rng))
                    .collect();
                let x_c0 = rand_from_rng::<_, BF>(&mut rng);
                let x_c1 = rand_from_rng::<_, BF>(&mut rng);
                let x = EF::from_coeff_in_base([x_c0, x_c1]);
                let mut naive = vec![EF::ZERO; num_polys];
                for poly in 0..num_polys {
                    let coeffs_this_poly = &coeffs[(poly * count)..((poly + 1) * count)];
                    let mut current = EF::ONE;
                    for el in coeffs_this_poly.iter() {
                        let mut tmp = EF::from_coeff_in_base([*el, BF::ZERO]);
                        tmp.mul_assign(&current);
                        naive[poly].add_assign(&tmp);
                        current.mul_assign(&x);
                    }
                }
                let mut d_batch_ys = DeviceAllocation::alloc(num_polys * count).unwrap();
                memory_copy_async(&mut d_batch_ys, &coeffs, &stream).unwrap();
                if log_count > 0 {
                    batch_ntt_in_place(
                        &mut d_batch_ys,
                        log_count,
                        num_polys as u32,
                        0,
                        count as u32,
                        false,
                        false,
                        1,
                        0,
                        &stream,
                    )
                    .unwrap();
                }
                let h_x = [x];
                let mut d_x = DeviceAllocation::<EF>::alloc(1).unwrap();
                let mut d_common_factor_storage = DeviceAllocation::<EF>::alloc(1).unwrap();
                let mut d_lagrange_coeffs = DeviceAllocation::<VF>::alloc(count).unwrap();
                memory_copy_async(&mut d_x, &h_x, &stream).unwrap();
                super::precompute_lagrange_coeffs::<super::PrecomputeAtExt>(
                    &d_x[0],
                    &mut d_common_factor_storage[0],
                    coset,
                    &mut d_lagrange_coeffs,
                    true,
                    &stream,
                )
                .unwrap();
                let d_batch_ys = DeviceMatrix::new(&d_batch_ys, count);
                let (partial_reduce_temp_elems, final_cub_reduce_temp_bytes) =
                    super::get_batch_eval_temp_storage_sizes::<super::EvalBaseAtExt>(&d_batch_ys)
                        .unwrap();
                let mut temp_storage_partial_reduce =
                    DeviceAllocation::alloc(partial_reduce_temp_elems).unwrap();
                let mut temp_storage_partial_reduce = DeviceMatrixMut::new(
                    &mut temp_storage_partial_reduce,
                    partial_reduce_temp_elems / num_polys,
                );
                let mut temp_storage_final_cub_reduce =
                    DeviceAllocation::alloc(final_cub_reduce_temp_bytes).unwrap();
                let mut d_evals = DeviceAllocation::alloc(num_polys).unwrap();
                super::batch_eval::<super::EvalBaseAtExt>(
                    &d_batch_ys,
                    &d_lagrange_coeffs,
                    &mut temp_storage_partial_reduce,
                    &mut temp_storage_final_cub_reduce,
                    &mut d_evals,
                    &stream,
                )
                .unwrap();
                let mut h_evals = vec![EF::ZERO; num_polys];
                memory_copy_async(&mut h_evals, &d_evals, &stream).unwrap();
                stream.synchronize().unwrap();
                for i in 0..num_polys {
                    assert_eq!(
                        h_evals[i], naive[i],
                        "log_count {}, num_polys {}, poly {}",
                        log_count, num_polys, i
                    );
                }
            }
        }
        context.destroy().unwrap();
    }

    #[test]
    #[serial]
    fn test_batch_eval_ext_at_ext() {
        let context = Context::create(12, 12).unwrap();
        let coset = BF::multiplicative_generator();
        let stream = CudaStream::default();
        let num_polys_to_test = [1, 3, 32, 33];
        for num_polys in num_polys_to_test.into_iter() {
            for log_count in 0..17 {
                let count = 1 << log_count;
                let mut rng = thread_rng();
                let coeffs: Vec<_> = (0..(2 * num_polys * count))
                    .map(|_| rand_from_rng::<_, BF>(&mut rng))
                    .collect();
                let x_c0 = rand_from_rng::<_, BF>(&mut rng);
                let x_c1 = rand_from_rng::<_, BF>(&mut rng);
                let x = EF::from_coeff_in_base([x_c0, x_c1]);
                let mut naive = vec![EF::ZERO; num_polys];
                for poly in 0..num_polys {
                    let coeffs_c0_this_poly = &coeffs[(2 * poly * count)..((2 * poly + 1) * count)];
                    let coeffs_c1_this_poly =
                        &coeffs[((2 * poly + 1) * count)..((2 * poly + 2) * count)];
                    let mut current = EF::ONE;
                    for (c0, c1) in coeffs_c0_this_poly.iter().zip(coeffs_c1_this_poly.iter()) {
                        let mut tmp = EF::from_coeff_in_base([*c0, *c1]);
                        tmp.mul_assign(&current);
                        naive[poly].add_assign(&tmp);
                        current.mul_assign(&x);
                    }
                }
                let mut d_batch_ys = DeviceAllocation::<BF>::alloc(2 * num_polys * count).unwrap();
                memory_copy_async(&mut d_batch_ys, &coeffs, &stream).unwrap();
                if log_count > 0 {
                    batch_ntt_in_place(
                        &mut d_batch_ys,
                        log_count,
                        2 * num_polys as u32,
                        0,
                        count as u32,
                        false,
                        false,
                        1,
                        0,
                        &stream,
                    )
                    .unwrap();
                }
                let h_x = [x];
                let mut d_x = DeviceAllocation::<EF>::alloc(1).unwrap();
                let mut d_common_factor_storage = DeviceAllocation::<EF>::alloc(1).unwrap();
                let mut d_lagrange_coeffs = DeviceAllocation::<VF>::alloc(count).unwrap();
                memory_copy_async(&mut d_x, &h_x, &stream).unwrap();
                super::precompute_lagrange_coeffs::<super::PrecomputeAtExt>(
                    &d_x[0],
                    &mut d_common_factor_storage[0],
                    coset,
                    &mut d_lagrange_coeffs,
                    true,
                    &stream,
                )
                .unwrap();
                let d_batch_ys = unsafe { d_batch_ys.transmute::<VF>() };
                let d_batch_ys = DeviceMatrix::new(d_batch_ys, count);
                let (partial_reduce_temp_elems, final_cub_reduce_temp_bytes) =
                    super::get_batch_eval_temp_storage_sizes::<super::EvalExtAtExt>(&d_batch_ys)
                        .unwrap();
                let mut temp_storage_partial_reduce =
                    DeviceAllocation::<VF>::alloc(partial_reduce_temp_elems).unwrap();
                let mut temp_storage_partial_reduce = DeviceMatrixMut::new(
                    &mut temp_storage_partial_reduce,
                    partial_reduce_temp_elems / num_polys,
                );
                let mut temp_storage_final_cub_reduce =
                    DeviceAllocation::alloc(final_cub_reduce_temp_bytes).unwrap();
                let mut d_evals = DeviceAllocation::<EF>::alloc(num_polys).unwrap();
                super::batch_eval::<super::EvalExtAtExt>(
                    &d_batch_ys,
                    &d_lagrange_coeffs,
                    &mut temp_storage_partial_reduce,
                    &mut temp_storage_final_cub_reduce,
                    &mut d_evals,
                    &stream,
                )
                .unwrap();
                let mut h_evals = vec![EF::ZERO; num_polys];
                memory_copy_async(&mut h_evals, &d_evals, &stream).unwrap();
                stream.synchronize().unwrap();
                for i in 0..num_polys {
                    assert_eq!(
                        h_evals[i], naive[i],
                        "log_count {}, num_polys {}, poly {}",
                        log_count, num_polys, i
                    );
                }
            }
        }
        context.destroy().unwrap();
    }
}
