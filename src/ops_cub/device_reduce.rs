use std::ptr::{null, null_mut};

use boojum::field::goldilocks::GoldilocksField;

use cudart::result::{CudaResult, CudaResultWrap};
use cudart::slice::{DeviceSlice, DeviceVariable};
use cudart::stream::CudaStream;
use cudart_sys::{cudaError_t, cudaStream_t, cuda_fn_and_stub};

use crate::device_structures::{
    DeviceMatrixChunkImpl, DeviceRepr, DeviceVectorChunkImpl, PtrAndStride,
};
use crate::extension_field::ExtensionField;

cuda_fn_and_stub! {
    fn reduce_add_bf(
        d_temp_storage: *mut u8,
        temp_storage_bytes: &mut usize,
        d_in: *const GoldilocksField,
        d_out: *mut GoldilocksField,
        num_items: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
}

cuda_fn_and_stub! {
    fn reduce_add_ef(
        d_temp_storage: *mut u8,
        temp_storage_bytes: &mut usize,
        d_in: *const ExtensionField,
        d_out: *mut ExtensionField,
        num_items: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
}

cuda_fn_and_stub! {
    fn segmented_reduce_add_bf(
        d_temp_storage: *mut u8,
        temp_storage_bytes: &mut usize,
        d_in: PtrAndStride<GoldilocksField>,
        d_out: *mut GoldilocksField,
        num_segments: i32,
        num_items: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
}

cuda_fn_and_stub! {
    fn segmented_reduce_add_ef(
        d_temp_storage: *mut u8,
        temp_storage_bytes: &mut usize,
        d_in: PtrAndStride<ExtensionField>,
        d_out: *mut ExtensionField,
        num_segments: i32,
        num_items: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
}

cuda_fn_and_stub! {
    fn reduce_mul_bf(
        d_temp_storage: *mut u8,
        temp_storage_bytes: &mut usize,
        d_in: *const GoldilocksField,
        d_out: *mut GoldilocksField,
        num_items: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
}

cuda_fn_and_stub! {
    fn reduce_mul_ef(
        d_temp_storage: *mut u8,
        temp_storage_bytes: &mut usize,
        d_in: *const ExtensionField,
        d_out: *mut ExtensionField,
        num_items: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
}

cuda_fn_and_stub! {
    fn segmented_reduce_mul_bf(
        d_temp_storage: *mut u8,
        temp_storage_bytes: &mut usize,
        d_in: PtrAndStride<GoldilocksField>,
        d_out: *mut GoldilocksField,
        num_segments: i32,
        num_items: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
}

cuda_fn_and_stub! {
    fn segmented_reduce_mul_ef(
        d_temp_storage: *mut u8,
        temp_storage_bytes: &mut usize,
        d_in: PtrAndStride<ExtensionField>,
        d_out: *mut ExtensionField,
        num_segments: i32,
        num_items: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
}

#[derive(Copy, Clone)]
pub enum ReduceOperation {
    Sum,
    Product,
}

type ReduceFunction<T> = unsafe extern "C" fn(
    d_temp_storage: *mut u8,
    temp_storage_bytes: &mut usize,
    d_in: *const T,
    d_out: *mut T,
    num_items: i32,
    stream: cudaStream_t,
) -> cudaError_t;

type SegmentedReduceFunction<T> = unsafe extern "C" fn(
    d_temp_storage: *mut u8,
    temp_storage_bytes: &mut usize,
    d_in: PtrAndStride<T>,
    d_out: *mut T,
    num_segments: i32,
    num_items: i32,
    stream: cudaStream_t,
) -> cudaError_t;

pub trait Reduce: DeviceRepr {
    fn get_reduce_function(operation: ReduceOperation) -> ReduceFunction<Self::Type>;

    fn get_segmented_reduce_function(
        operation: ReduceOperation,
    ) -> SegmentedReduceFunction<Self::Type>;

    fn get_reduce_temp_storage_bytes(
        operation: ReduceOperation,
        num_items: i32,
    ) -> CudaResult<usize> {
        let mut temp_storage_bytes = 0;
        let function = Self::get_reduce_function(operation);
        unsafe {
            function(
                null_mut(),
                &mut temp_storage_bytes,
                null(),
                null_mut(),
                num_items,
                null_mut(),
            )
            .wrap_value(temp_storage_bytes)
        }
    }

    fn get_batch_reduce_temp_storage_bytes(
        operation: ReduceOperation,
        batch_size: i32,
        num_items: i32,
    ) -> CudaResult<usize> {
        let mut temp_storage_bytes = 0;
        let function = Self::get_segmented_reduce_function(operation);
        unsafe {
            function(
                null_mut(),
                &mut temp_storage_bytes,
                PtrAndStride::new(null(), num_items as usize),
                null_mut(),
                batch_size,
                num_items,
                null_mut(),
            )
            .wrap_value(temp_storage_bytes)
        }
    }

    fn reduce<I>(
        operation: ReduceOperation,
        d_temp_storage: &mut DeviceSlice<u8>,
        d_in: &I,
        d_out: &mut DeviceVariable<Self>,
        stream: &CudaStream,
    ) -> CudaResult<()>
    where
        I: DeviceVectorChunkImpl<Self> + ?Sized,
    {
        let mut temp_storage_bytes = d_temp_storage.len();
        assert!(d_in.rows() <= i32::MAX as usize);
        let num_items = d_in.rows() as i32;
        let function = Self::get_reduce_function(operation);
        unsafe {
            function(
                d_temp_storage.as_mut_ptr(),
                &mut temp_storage_bytes,
                d_in.as_ptr(),
                d_out.as_mut_ptr() as *mut _,
                num_items,
                stream.into(),
            )
            .wrap()
        }
    }

    fn batch_reduce<I>(
        operation: ReduceOperation,
        d_temp_storage: &mut DeviceSlice<u8>,
        d_in: &I,
        d_out: &mut DeviceSlice<Self>,
        stream: &CudaStream,
    ) -> CudaResult<()>
    where
        I: DeviceMatrixChunkImpl<Self> + ?Sized,
    {
        let mut temp_storage_bytes = d_temp_storage.len();
        assert_eq!(d_in.cols(), d_out.len());
        let num_segments = d_in.cols() as i32;
        let num_items = d_in.rows() as i32;
        let function = Self::get_segmented_reduce_function(operation);
        unsafe {
            function(
                d_temp_storage.as_mut_ptr(),
                &mut temp_storage_bytes,
                d_in.as_ptr_and_stride(),
                d_out.as_mut_ptr() as *mut _,
                num_segments,
                num_items,
                stream.into(),
            )
            .wrap()
        }
    }
}

impl Reduce for GoldilocksField {
    fn get_reduce_function(operation: ReduceOperation) -> ReduceFunction<Self::Type> {
        match operation {
            ReduceOperation::Sum => reduce_add_bf,
            ReduceOperation::Product => reduce_mul_bf,
        }
    }

    fn get_segmented_reduce_function(
        operation: ReduceOperation,
    ) -> SegmentedReduceFunction<Self::Type> {
        match operation {
            ReduceOperation::Sum => segmented_reduce_add_bf,
            ReduceOperation::Product => segmented_reduce_mul_bf,
        }
    }
}

impl Reduce for ExtensionField {
    fn get_reduce_function(operation: ReduceOperation) -> ReduceFunction<Self::Type> {
        match operation {
            ReduceOperation::Sum => reduce_add_ef,
            ReduceOperation::Product => reduce_mul_ef,
        }
    }

    fn get_segmented_reduce_function(
        operation: ReduceOperation,
    ) -> SegmentedReduceFunction<Self::Type> {
        match operation {
            ReduceOperation::Sum => segmented_reduce_add_ef,
            ReduceOperation::Product => segmented_reduce_mul_ef,
        }
    }
}

pub fn get_reduce_temp_storage_bytes<T: Reduce>(
    operation: ReduceOperation,
    num_items: i32,
) -> CudaResult<usize> {
    T::get_reduce_temp_storage_bytes(operation, num_items)
}

pub fn get_batch_reduce_temp_storage_bytes<T: Reduce>(
    operation: ReduceOperation,
    batch_size: i32,
    num_items: i32,
) -> CudaResult<usize> {
    T::get_batch_reduce_temp_storage_bytes(operation, batch_size, num_items)
}

pub fn reduce<T, I>(
    operation: ReduceOperation,
    d_temp_storage: &mut DeviceSlice<u8>,
    d_in: &I,
    d_out: &mut DeviceVariable<T>,
    stream: &CudaStream,
) -> CudaResult<()>
where
    T: Reduce,
    I: DeviceVectorChunkImpl<T> + ?Sized,
{
    T::reduce(operation, d_temp_storage, d_in, d_out, stream)
}

pub fn batch_reduce<T, I>(
    operation: ReduceOperation,
    d_temp_storage: &mut DeviceSlice<u8>,
    d_in: &I,
    d_out: &mut DeviceSlice<T>,
    stream: &CudaStream,
) -> CudaResult<()>
where
    T: Reduce,
    I: DeviceMatrixChunkImpl<T> + ?Sized,
{
    T::batch_reduce(operation, d_temp_storage, d_in, d_out, stream)
}

#[cfg(test)]
mod tests {
    use boojum::field::goldilocks::GoldilocksField;
    use boojum::field::Field;
    use itertools::Itertools;
    use rand::{thread_rng, Rng};

    use cudart::memory::{memory_copy_async, DeviceAllocation};
    use cudart::stream::CudaStream;

    use crate::device_structures::DeviceMatrix;
    use crate::extension_field::ExtensionField;
    use crate::ops_cub::device_reduce::{Reduce, ReduceOperation};

    type GenerateFunction<F> = fn(usize) -> Vec<F>;

    type HostFunction<F> = fn(F, F) -> F;

    fn reduce<F: Field + Reduce>(
        generate_function: GenerateFunction<F>,
        operation: ReduceOperation,
        init: F,
        host_function: HostFunction<F>,
    ) {
        const NUM_ITEMS: usize = 1 << 16;
        let temp_storage_bytes =
            super::get_reduce_temp_storage_bytes::<F>(operation, NUM_ITEMS as i32).unwrap();
        let mut d_temp_storage = DeviceAllocation::alloc(temp_storage_bytes).unwrap();
        let h_in = generate_function(NUM_ITEMS);
        let mut h_out = [F::default()];
        let mut d_in = DeviceAllocation::alloc(NUM_ITEMS).unwrap();
        let mut d_out = DeviceAllocation::alloc(1).unwrap();
        let stream = CudaStream::default();
        memory_copy_async(&mut d_in, &h_in, &stream).unwrap();
        super::reduce(
            operation,
            &mut d_temp_storage,
            &d_in,
            &mut d_out[0],
            &stream,
        )
        .unwrap();
        memory_copy_async(&mut h_out, &d_out, &stream).unwrap();
        stream.synchronize().unwrap();
        let result = h_in.into_iter().fold(init, host_function);
        assert_eq!(result, h_out[0]);
    }

    fn batch_reduce<F: Field + Reduce>(
        generate_function: GenerateFunction<F>,
        operation: ReduceOperation,
        init: F,
        host_function: HostFunction<F>,
    ) {
        const BATCH_SIZE: usize = 1 << 8;
        const NUM_ITEMS: usize = 1 << 8;
        let temp_storage_bytes = super::get_batch_reduce_temp_storage_bytes::<GoldilocksField>(
            operation,
            BATCH_SIZE as i32,
            NUM_ITEMS as i32,
        )
        .unwrap();
        let mut d_temp_storage = DeviceAllocation::alloc(temp_storage_bytes).unwrap();
        let h_in = generate_function(NUM_ITEMS * BATCH_SIZE);
        let mut h_out = [F::default(); BATCH_SIZE];
        let mut d_in = DeviceAllocation::alloc(NUM_ITEMS * BATCH_SIZE).unwrap();
        let mut d_out = DeviceAllocation::alloc(BATCH_SIZE).unwrap();
        let stream = CudaStream::default();
        memory_copy_async(&mut d_in, &h_in, &stream).unwrap();
        let d_in_matrix = DeviceMatrix::new(&d_in, NUM_ITEMS);
        super::batch_reduce(
            operation,
            &mut d_temp_storage,
            &d_in_matrix,
            &mut d_out,
            &stream,
        )
        .unwrap();
        memory_copy_async(&mut h_out, &d_out, &stream).unwrap();
        stream.synchronize().unwrap();
        let result = h_in
            .into_iter()
            .chunks(NUM_ITEMS)
            .into_iter()
            .map(|c| c.fold(init, host_function))
            .collect_vec();
        assert!(result
            .into_iter()
            .zip(h_out.into_iter())
            .all(|(a, b)| a == b));
    }

    type TestFunction<F> = fn(generate: GenerateFunction<F>, ReduceOperation, F, HostFunction<F>);

    fn test_sum<F: Field>(generate_function: GenerateFunction<F>, test_function: TestFunction<F>) {
        test_function(
            generate_function,
            ReduceOperation::Sum,
            F::ZERO,
            |state, x| {
                let mut result = state;
                result.add_assign(&x);
                result
            },
        )
    }

    fn test_product<F: Field>(
        generate_function: GenerateFunction<F>,
        test_function: TestFunction<F>,
    ) {
        test_function(
            generate_function,
            ReduceOperation::Product,
            F::ONE,
            |state, x| {
                let mut result = state;
                result.mul_assign(&x);
                result
            },
        )
    }

    fn generate_gf(count: usize) -> Vec<GoldilocksField> {
        (0..count)
            .map(|_| GoldilocksField::from_nonreduced_u64(thread_rng().gen()))
            .collect_vec()
    }

    fn generate_ef(count: usize) -> Vec<ExtensionField> {
        (0..count * 2)
            .map(|_| GoldilocksField::from_nonreduced_u64(thread_rng().gen()))
            .chunks(2)
            .into_iter()
            .map(|mut c| {
                let a = c.next().unwrap();
                let b = c.next().unwrap();
                ExtensionField::from_coeff_in_base([a, b])
            })
            .collect_vec()
    }

    #[test]
    fn sum_bf() {
        test_sum(generate_gf, reduce)
    }

    #[test]
    fn batch_sum_bf() {
        test_sum(generate_gf, batch_reduce)
    }

    #[test]
    fn product_bf() {
        test_product(generate_gf, reduce)
    }

    #[test]
    fn batch_product_bf() {
        test_product(generate_gf, batch_reduce)
    }

    #[test]
    fn sum_ef() {
        test_sum(generate_ef, reduce)
    }

    #[test]
    fn batch_sum_ef() {
        test_sum(generate_ef, batch_reduce)
    }

    #[test]
    fn product_ef() {
        test_product(generate_ef, reduce)
    }

    #[test]
    fn batch_product_ef() {
        test_product(generate_ef, batch_reduce)
    }
}
