use std::cmp;

use boojum::cs::implementations::utils::domain_generator_for_size;
use boojum::field::goldilocks::GoldilocksField;
use boojum::field::{Field, PrimeField};

use cudart::execution::{KernelFiveArgs, KernelFourArgs, KernelLaunch, KernelSixArgs};
use cudart::result::CudaResult;
use cudart::slice::{DeviceSlice, DeviceVariable};
use cudart::stream::CudaStream;
use cudart_sys::dim3;

use crate::device_structures::{
    DeviceMatrix, DeviceMatrixImpl, DeviceMatrixMutImpl, DeviceRepr, DeviceVectorImpl,
    DeviceVectorMutImpl, ExtensionFieldDeviceType, MutPtrAndStride, PtrAndStride,
    VectorizedExtensionFieldDeviceType,
};
use crate::extension_field::{ExtensionField, VectorizedExtensionField};
use crate::ops_complex::bit_reverse_in_place;
use crate::ops_cub::device_reduce::{
    batch_reduce, get_batch_reduce_temp_storage_bytes, ReduceOperation,
};
use crate::utils::WARP_SIZE;

extern "C" {
    fn barycentric_precompute_common_factor_at_base_kernel(
        x: *const GoldilocksField,
        common_factor: *mut GoldilocksField,
        coset: GoldilocksField,
        count: u32,
    );

    fn barycentric_precompute_common_factor_at_ext_kernel(
        x: *const ExtensionFieldDeviceType,
        common_factor: *mut ExtensionFieldDeviceType,
        coset: GoldilocksField,
        count: u32,
    );

    fn barycentric_precompute_lagrange_coeffs_at_base_kernel(
        x: *const GoldilocksField,
        common_factor: *const GoldilocksField,
        w_inv_step: GoldilocksField,
        coset: GoldilocksField,
        lagrange_coeffs: MutPtrAndStride<GoldilocksField>,
        log_count: u32,
    );

    fn barycentric_precompute_lagrange_coeffs_at_ext_kernel(
        x: *const ExtensionFieldDeviceType,
        common_factor: *const ExtensionFieldDeviceType,
        w_inv_step: GoldilocksField,
        coset: GoldilocksField,
        lagrange_coeffs: MutPtrAndStride<VectorizedExtensionFieldDeviceType>,
        log_count: u32,
    );

    fn batch_barycentric_partial_reduce_base_at_base_kernel(
        batch_ys: PtrAndStride<GoldilocksField>,
        lagrange_coeffs: PtrAndStride<GoldilocksField>,
        partial_sums: MutPtrAndStride<GoldilocksField>,
        log_count: u32,
        num_polys: u32,
    );

    fn batch_barycentric_partial_reduce_base_at_ext_kernel(
        batch_ys: PtrAndStride<GoldilocksField>,
        lagrange_coeffs: PtrAndStride<VectorizedExtensionFieldDeviceType>,
        partial_sums: MutPtrAndStride<VectorizedExtensionFieldDeviceType>,
        log_count: u32,
        num_polys: u32,
    );

    fn batch_barycentric_partial_reduce_ext_at_ext_kernel(
        batch_ys: PtrAndStride<VectorizedExtensionFieldDeviceType>,
        lagrange_coeffs: PtrAndStride<VectorizedExtensionFieldDeviceType>,
        partial_sums: MutPtrAndStride<VectorizedExtensionFieldDeviceType>,
        log_count: u32,
        num_polys: u32,
    );
}

pub trait PrecomputeImpl {
    const X_IS_1_OR_2_GF_ELEMS: u32;
    type X: Clone + DeviceRepr;
    type XVec: DeviceRepr;
    fn get_precompute_inv_batch() -> u32;
    fn get_common_factor_kernel() -> unsafe extern "C" fn(
        *const <Self::X as DeviceRepr>::Type,
        *mut <Self::X as DeviceRepr>::Type,
        GoldilocksField,
        u32,
    );
    #[allow(clippy::type_complexity)]
    fn get_precompute_kernel() -> unsafe extern "C" fn(
        *const <Self::X as DeviceRepr>::Type,
        *const <Self::X as DeviceRepr>::Type,
        GoldilocksField,
        GoldilocksField,
        MutPtrAndStride<<Self::XVec as DeviceRepr>::Type>,
        u32,
    );
    fn bit_reverse_coeffs(
        lagrange_coeffs: &mut DeviceSlice<Self::XVec>,
        stream: &CudaStream,
    ) -> CudaResult<()>;
}

pub struct PrecomputeAtBase {}

pub struct PrecomputeAtExt {}

impl PrecomputeImpl for PrecomputeAtBase {
    const X_IS_1_OR_2_GF_ELEMS: u32 = 1;
    type X = GoldilocksField;
    type XVec = GoldilocksField;
    fn get_precompute_inv_batch() -> u32 {
        10
    }
    fn get_common_factor_kernel(
    ) -> unsafe extern "C" fn(*const GoldilocksField, *mut GoldilocksField, GoldilocksField, u32)
    {
        barycentric_precompute_common_factor_at_base_kernel
    }
    fn get_precompute_kernel() -> unsafe extern "C" fn(
        *const GoldilocksField,
        *const GoldilocksField,
        GoldilocksField,
        GoldilocksField,
        MutPtrAndStride<GoldilocksField>,
        u32,
    ) {
        barycentric_precompute_lagrange_coeffs_at_base_kernel
    }
    fn bit_reverse_coeffs(
        lagrange_coeffs: &mut DeviceSlice<GoldilocksField>,
        stream: &CudaStream,
    ) -> CudaResult<()> {
        bit_reverse_in_place(lagrange_coeffs, stream)
    }
}

impl PrecomputeImpl for PrecomputeAtExt {
    const X_IS_1_OR_2_GF_ELEMS: u32 = 2;
    type X = ExtensionField;
    type XVec = VectorizedExtensionField;
    fn get_precompute_inv_batch() -> u32 {
        6
    }
    fn get_common_factor_kernel() -> unsafe extern "C" fn(
        *const ExtensionFieldDeviceType,
        *mut ExtensionFieldDeviceType,
        GoldilocksField,
        u32,
    ) {
        barycentric_precompute_common_factor_at_ext_kernel
    }
    fn get_precompute_kernel() -> unsafe extern "C" fn(
        *const ExtensionFieldDeviceType,
        *const ExtensionFieldDeviceType,
        GoldilocksField,
        GoldilocksField,
        MutPtrAndStride<VectorizedExtensionFieldDeviceType>,
        u32,
    ) {
        barycentric_precompute_lagrange_coeffs_at_ext_kernel
    }
    fn bit_reverse_coeffs(
        lagrange_coeffs: &mut DeviceSlice<VectorizedExtensionField>,
        stream: &CudaStream,
    ) -> CudaResult<()> {
        let count = lagrange_coeffs.len();
        let coeffs_as_gf = unsafe { lagrange_coeffs.transmute_mut::<GoldilocksField>() };
        bit_reverse_in_place(&mut coeffs_as_gf[0..count], stream)?;
        bit_reverse_in_place(&mut coeffs_as_gf[count..], stream)
    }
}

pub trait EvalImpl {
    const X_IS_1_OR_2_GF_ELEMS: u32;
    type X: DeviceRepr;
    type XVec: DeviceRepr;
    type YsVec: DeviceRepr;
    fn get_partial_reduce_elems_per_thread() -> u32;
    #[allow(clippy::type_complexity)]
    fn get_partial_reduce_kernel() -> unsafe extern "C" fn(
        PtrAndStride<<Self::YsVec as DeviceRepr>::Type>,
        PtrAndStride<<Self::XVec as DeviceRepr>::Type>,
        MutPtrAndStride<<Self::XVec as DeviceRepr>::Type>,
        u32,
        u32,
    );
}

pub struct EvalBaseAtBase {}

pub struct EvalBaseAtExt {}

pub struct EvalExtAtExt {}

impl EvalImpl for EvalBaseAtBase {
    const X_IS_1_OR_2_GF_ELEMS: u32 = 1;
    type X = GoldilocksField;
    type XVec = GoldilocksField;
    type YsVec = GoldilocksField;
    fn get_partial_reduce_elems_per_thread() -> u32 {
        12
    }
    fn get_partial_reduce_kernel() -> unsafe extern "C" fn(
        PtrAndStride<GoldilocksField>,
        PtrAndStride<GoldilocksField>,
        MutPtrAndStride<GoldilocksField>,
        u32,
        u32,
    ) {
        batch_barycentric_partial_reduce_base_at_base_kernel
    }
}

impl EvalImpl for EvalBaseAtExt {
    const X_IS_1_OR_2_GF_ELEMS: u32 = 2;
    type X = ExtensionField;
    type XVec = VectorizedExtensionField;
    type YsVec = GoldilocksField;
    fn get_partial_reduce_elems_per_thread() -> u32 {
        6
    }
    fn get_partial_reduce_kernel() -> unsafe extern "C" fn(
        PtrAndStride<GoldilocksField>,
        PtrAndStride<VectorizedExtensionFieldDeviceType>,
        MutPtrAndStride<VectorizedExtensionFieldDeviceType>,
        u32,
        u32,
    ) {
        batch_barycentric_partial_reduce_base_at_ext_kernel
    }
}

impl EvalImpl for EvalExtAtExt {
    const X_IS_1_OR_2_GF_ELEMS: u32 = 2;
    type X = ExtensionField;
    type XVec = VectorizedExtensionField;
    type YsVec = VectorizedExtensionField;
    fn get_partial_reduce_elems_per_thread() -> u32 {
        6
    }
    fn get_partial_reduce_kernel() -> unsafe extern "C" fn(
        PtrAndStride<VectorizedExtensionFieldDeviceType>,
        PtrAndStride<VectorizedExtensionFieldDeviceType>,
        MutPtrAndStride<VectorizedExtensionFieldDeviceType>,
        u32,
        u32,
    ) {
        batch_barycentric_partial_reduce_ext_at_ext_kernel
    }
}

pub fn precompute_lagrange_coeffs<T: PrecomputeImpl>(
    x: &DeviceVariable<T::X>,
    common_factor_storage: &mut DeviceVariable<T::X>,
    coset: GoldilocksField,
    lagrange_coeffs: &mut (impl DeviceVectorMutImpl<T::XVec> + ?Sized),
    bit_reverse: bool,
    stream: &CudaStream,
) -> CudaResult<()> {
    let inv_batch: u32 = T::get_precompute_inv_batch();
    assert!(lagrange_coeffs.slice().len() <= u32::MAX as usize);
    assert_eq!(lagrange_coeffs.slice().len().count_ones(), 1);
    let count = lagrange_coeffs.slice().len() as u32;
    let x_arg = x.as_ptr() as *const <T::X as DeviceRepr>::Type;
    let common_factor = common_factor_storage.as_mut_ptr() as *mut <T::X as DeviceRepr>::Type;
    let args = (&x_arg, &common_factor, &coset, &count);
    unsafe {
        KernelFourArgs::launch(
            T::get_common_factor_kernel(),
            1.into(),
            1.into(),
            args,
            0,
            stream,
        )
    }?;
    let log_count: u32 = count.trailing_zeros();
    let block_dim = WARP_SIZE * 4;
    let grid_dim = (count + inv_batch * block_dim - 1) / (inv_batch * block_dim);
    let w = domain_generator_for_size::<GoldilocksField>((1 << log_count) as u64);
    let w_inv = w.inverse().expect("inverse of omega must exist");
    let w_inv_step = w_inv.pow_u64((block_dim * grid_dim) as u64);
    let block_dim: dim3 = block_dim.into();
    let grid_dim: dim3 = grid_dim.into();
    let common_factor = common_factor_storage.as_ptr() as *const <T::X as DeviceRepr>::Type;
    let dst = lagrange_coeffs.as_mut_ptr_and_stride();
    let args = (
        &x_arg,
        &common_factor,
        &w_inv_step,
        &coset,
        &dst,
        &log_count,
    );
    unsafe {
        KernelSixArgs::launch(
            T::get_precompute_kernel(),
            grid_dim,
            block_dim,
            args,
            0,
            stream,
        )
    }?;
    if bit_reverse {
        T::bit_reverse_coeffs(lagrange_coeffs.slice_mut(), stream)?;
    }
    Ok(())
}

fn get_batch_partial_reduce_grid_block<T: EvalImpl>(count: u32) -> (dim3, u32, u32) {
    let elems_per_thread: u32 = T::get_partial_reduce_elems_per_thread();
    let block_dim_x = WARP_SIZE;
    let grid_dim = (count + elems_per_thread * block_dim_x - 1) / (elems_per_thread * block_dim_x);
    let mut block_dim: dim3 = block_dim_x.into();
    block_dim.y = WARP_SIZE;
    let grid_size_x = block_dim_x * grid_dim;
    (block_dim, grid_dim, grid_size_x)
}

pub fn get_batch_eval_temp_storage_sizes<T: EvalImpl>(
    batch_ys: &(impl DeviceMatrixImpl<T::YsVec> + ?Sized),
) -> CudaResult<(usize, usize)> {
    assert!(batch_ys.stride() <= u32::MAX as usize);
    assert_eq!(batch_ys.stride().count_ones(), 1);
    let count = batch_ys.stride() as u32;
    let num_polys = batch_ys.cols() as u32;
    let (_, _, grid_size_x) = get_batch_partial_reduce_grid_block::<T>(count);
    let partial_reduce_temp_elems = (num_polys * cmp::min(count, grid_size_x)) as usize;
    let final_cub_reduce_temp_bytes = get_batch_reduce_temp_storage_bytes::<GoldilocksField>(
        ReduceOperation::Sum,
        (T::X_IS_1_OR_2_GF_ELEMS * num_polys) as i32,
        (T::X_IS_1_OR_2_GF_ELEMS * count) as i32,
    )?;
    Ok((partial_reduce_temp_elems, final_cub_reduce_temp_bytes))
}

pub fn batch_eval<T: EvalImpl>(
    batch_ys: &(impl DeviceMatrixImpl<T::YsVec> + ?Sized),
    lagrange_coeffs: &(impl DeviceVectorImpl<T::XVec> + ?Sized),
    temp_storage_partial_reduce: &mut (impl DeviceMatrixMutImpl<T::XVec> + ?Sized),
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
    let grid_dim: dim3 = grid_dim.into();
    let src = batch_ys.as_ptr_and_stride();
    let coeffs = lagrange_coeffs.as_ptr_and_stride();
    let dst = temp_storage_partial_reduce.as_mut_ptr_and_stride();
    let args = (&src, &coeffs, &dst, &log_count, &num_polys);
    unsafe {
        KernelFiveArgs::launch(
            T::get_partial_reduce_kernel(),
            grid_dim,
            block_dim,
            args,
            0,
            stream,
        )?;
    }
    let stride = temp_storage_partial_reduce.stride();
    let temp_storage_partial_reduce_view = unsafe {
        temp_storage_partial_reduce
            .slice_mut()
            .transmute_mut::<GoldilocksField>()
    };
    let temp_storage_partial_reduce_matrix =
        DeviceMatrix::new(temp_storage_partial_reduce_view, stride);
    let evals_view = unsafe { evals.slice_mut().transmute_mut::<GoldilocksField>() };
    batch_reduce::<GoldilocksField, _>(
        ReduceOperation::Sum,
        temp_storage_final_cub_reduce,
        &temp_storage_partial_reduce_matrix,
        evals_view,
        stream,
    )
    // maybe "transpose" layout of batch results to vectorized using Convert
}

#[cfg(test)]
mod tests {
    use std::alloc::Global;

    use boojum::cs::implementations::utils::{
        precompute_for_barycentric_evaluation, precompute_for_barycentric_evaluation_in_extension,
    };
    use boojum::field::goldilocks::{GoldilocksExt2, GoldilocksField};
    use boojum::field::{rand_from_rng, Field, PrimeField};
    use boojum::worker::Worker;
    use rand::{thread_rng, Rng};
    use serial_test::serial;

    use cudart::memory::{memory_copy_async, DeviceAllocation};
    use cudart::stream::CudaStream;

    use crate::context::Context;
    use crate::device_structures::{DeviceMatrix, DeviceMatrixMut};
    use crate::extension_field::test_helpers::{transmute_gf_vec, ExtensionFieldTest};
    use crate::extension_field::{ExtensionField, VectorizedExtensionField};
    use crate::ntt::batch_ntt_in_place;

    #[test]
    #[serial]
    fn test_precompute_lagrange_coeffs_at_base() {
        let context = Context::create(12, 12).unwrap();
        let worker = Worker::new();
        for log_count in 0..20 {
            let count = 1usize << log_count;
            let h_x: [GoldilocksField; 1] = [GoldilocksField(thread_rng().gen())];
            let mut h_dst = vec![GoldilocksField::ZERO; count];
            let coset = GoldilocksField::multiplicative_generator();
            let mut d_x = DeviceAllocation::<GoldilocksField>::alloc(1).unwrap();
            let mut d_common_factor_storage =
                DeviceAllocation::<GoldilocksField>::alloc(1).unwrap();
            let mut d_dst = DeviceAllocation::<GoldilocksField>::alloc(count).unwrap();
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
            let precomps: Vec<GoldilocksField> =
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
            let c0 = GoldilocksField(thread_rng().gen());
            let c1 = GoldilocksField(thread_rng().gen());
            let h_x: [ExtensionField; 1] = [ExtensionField::from_coeff_in_base([c0, c1])];
            let h_dst_storage = vec![GoldilocksField::ZERO; count << 1];
            let mut h_dst = transmute_gf_vec::<VectorizedExtensionField>(h_dst_storage);
            let coset = GoldilocksField::multiplicative_generator();
            let mut d_x = DeviceAllocation::<ExtensionField>::alloc(1).unwrap();
            let mut d_common_factor_storage = DeviceAllocation::<ExtensionField>::alloc(1).unwrap();
            let mut d_dst = DeviceAllocation::<VectorizedExtensionField>::alloc(count).unwrap();
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
                GoldilocksField,
                GoldilocksExt2,
                GoldilocksField,
                Global,
            >(count, coset, h_x[0], &worker, &mut ());
            let iterator = VectorizedExtensionField::get_iterator(&h_dst);
            for (i, val) in iterator.enumerate() {
                assert_eq!(
                    val,
                    ExtensionField::from_coeff_in_base([precomps_c0[i], precomps_c1[i]]),
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
        type F = GoldilocksField;
        let context = Context::create(12, 12).unwrap();
        let coset = GoldilocksField::multiplicative_generator();
        let stream = CudaStream::default();
        let num_polys_to_test = [1, 3, 32, 33];
        for num_polys in num_polys_to_test.into_iter() {
            for log_count in 0..17 {
                let count = 1 << log_count;
                let mut rng = thread_rng();
                let coeffs: Vec<_> = (0..(num_polys * count))
                    .map(|_| rand_from_rng::<_, F>(&mut rng))
                    .collect();
                let x = rand_from_rng::<_, F>(&mut rng);
                let mut naive = vec![GoldilocksField::ZERO; num_polys];
                for poly in 0..num_polys {
                    let coeffs_this_poly = &coeffs[(poly * count)..((poly + 1) * count)];
                    let mut current = GoldilocksField::ONE;
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
                let mut d_x = DeviceAllocation::<GoldilocksField>::alloc(1).unwrap();
                let mut d_common_factor_storage =
                    DeviceAllocation::<GoldilocksField>::alloc(1).unwrap();
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
                let mut h_evals = vec![GoldilocksField::ZERO; num_polys];
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
        type F = GoldilocksField;
        let context = Context::create(12, 12).unwrap();
        let coset = GoldilocksField::multiplicative_generator();
        let stream = CudaStream::default();
        let num_polys_to_test = [1, 3, 32, 33];
        for num_polys in num_polys_to_test.into_iter() {
            for log_count in 0..17 {
                let count = 1 << log_count;
                let mut rng = thread_rng();
                let coeffs: Vec<_> = (0..(num_polys * count))
                    .map(|_| rand_from_rng::<_, F>(&mut rng))
                    .collect();
                let x_c0 = rand_from_rng::<_, F>(&mut rng);
                let x_c1 = rand_from_rng::<_, F>(&mut rng);
                let x = ExtensionField::from_coeff_in_base([x_c0, x_c1]);
                let mut naive = vec![ExtensionField::ZERO; num_polys];
                for poly in 0..num_polys {
                    let coeffs_this_poly = &coeffs[(poly * count)..((poly + 1) * count)];
                    let mut current = ExtensionField::ONE;
                    for el in coeffs_this_poly.iter() {
                        let mut tmp =
                            ExtensionField::from_coeff_in_base([*el, GoldilocksField::ZERO]);
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
                let mut d_x = DeviceAllocation::<ExtensionField>::alloc(1).unwrap();
                let mut d_common_factor_storage =
                    DeviceAllocation::<ExtensionField>::alloc(1).unwrap();
                let mut d_lagrange_coeffs =
                    DeviceAllocation::<VectorizedExtensionField>::alloc(count).unwrap();
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
                let mut h_evals = vec![ExtensionField::ZERO; num_polys];
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
        type F = GoldilocksField;
        let context = Context::create(12, 12).unwrap();
        let coset = GoldilocksField::multiplicative_generator();
        let stream = CudaStream::default();
        let num_polys_to_test = [1, 3, 32, 33];
        for num_polys in num_polys_to_test.into_iter() {
            for log_count in 0..17 {
                let count = 1 << log_count;
                let mut rng = thread_rng();
                let coeffs: Vec<_> = (0..(2 * num_polys * count))
                    .map(|_| rand_from_rng::<_, F>(&mut rng))
                    .collect();
                let x_c0 = rand_from_rng::<_, F>(&mut rng);
                let x_c1 = rand_from_rng::<_, F>(&mut rng);
                let x = ExtensionField::from_coeff_in_base([x_c0, x_c1]);
                let mut naive = vec![ExtensionField::ZERO; num_polys];
                for poly in 0..num_polys {
                    let coeffs_c0_this_poly = &coeffs[(2 * poly * count)..((2 * poly + 1) * count)];
                    let coeffs_c1_this_poly =
                        &coeffs[((2 * poly + 1) * count)..((2 * poly + 2) * count)];
                    let mut current = ExtensionField::ONE;
                    for (c0, c1) in coeffs_c0_this_poly.iter().zip(coeffs_c1_this_poly.iter()) {
                        let mut tmp = ExtensionField::from_coeff_in_base([*c0, *c1]);
                        tmp.mul_assign(&current);
                        naive[poly].add_assign(&tmp);
                        current.mul_assign(&x);
                    }
                }
                let mut d_batch_ys =
                    DeviceAllocation::<GoldilocksField>::alloc(2 * num_polys * count).unwrap();
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
                let mut d_x = DeviceAllocation::<ExtensionField>::alloc(1).unwrap();
                let mut d_common_factor_storage =
                    DeviceAllocation::<ExtensionField>::alloc(1).unwrap();
                let mut d_lagrange_coeffs =
                    DeviceAllocation::<VectorizedExtensionField>::alloc(count).unwrap();
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
                let d_batch_ys = unsafe { d_batch_ys.transmute::<VectorizedExtensionField>() };
                let d_batch_ys = DeviceMatrix::new(d_batch_ys, count);
                let (partial_reduce_temp_elems, final_cub_reduce_temp_bytes) =
                    super::get_batch_eval_temp_storage_sizes::<super::EvalExtAtExt>(&d_batch_ys)
                        .unwrap();
                let mut temp_storage_partial_reduce =
                    DeviceAllocation::<VectorizedExtensionField>::alloc(partial_reduce_temp_elems)
                        .unwrap();
                let mut temp_storage_partial_reduce = DeviceMatrixMut::new(
                    &mut temp_storage_partial_reduce,
                    partial_reduce_temp_elems / num_polys,
                );
                let mut temp_storage_final_cub_reduce =
                    DeviceAllocation::alloc(final_cub_reduce_temp_bytes).unwrap();
                let mut d_evals = DeviceAllocation::<ExtensionField>::alloc(num_polys).unwrap();
                super::batch_eval::<super::EvalExtAtExt>(
                    &d_batch_ys,
                    &d_lagrange_coeffs,
                    &mut temp_storage_partial_reduce,
                    &mut temp_storage_final_cub_reduce,
                    &mut d_evals,
                    &stream,
                )
                .unwrap();
                let mut h_evals = vec![ExtensionField::ZERO; num_polys];
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
