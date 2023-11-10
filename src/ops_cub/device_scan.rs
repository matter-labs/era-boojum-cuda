use std::ptr::null_mut;

use boojum::field::goldilocks::GoldilocksField;

use cudart::event::{CudaEvent, CudaEventCreateFlags};
use cudart::result::{CudaResult, CudaResultWrap};
use cudart::slice::DeviceSlice;
use cudart::stream::{CudaStream, CudaStreamCreateFlags, CudaStreamWaitEventFlags};
use cudart_sys::{cudaError_t, cudaStream_t};

use crate::extension_field::ExtensionField;

extern "C" {
    fn exclusive_sum_u32(
        d_temp_storage: *mut u8,
        temp_storage_bytes: &mut usize,
        d_in: *const u32,
        d_out: *mut u32,
        num_items: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    fn exclusive_sum_reverse_u32(
        d_temp_storage: *mut u8,
        temp_storage_bytes: &mut usize,
        d_in: *const u32,
        d_out: *mut u32,
        num_items: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    fn inclusive_sum_u32(
        d_temp_storage: *mut u8,
        temp_storage_bytes: &mut usize,
        d_in: *const u32,
        d_out: *mut u32,
        num_items: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    fn inclusive_sum_reverse_u32(
        d_temp_storage: *mut u8,
        temp_storage_bytes: &mut usize,
        d_in: *const u32,
        d_out: *mut u32,
        num_items: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    fn exclusive_scan_add_bf(
        d_temp_storage: *mut u8,
        temp_storage_bytes: &mut usize,
        d_in: *const GoldilocksField,
        d_out: *mut GoldilocksField,
        num_items: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    fn exclusive_scan_reverse_add_bf(
        d_temp_storage: *mut u8,
        temp_storage_bytes: &mut usize,
        d_in: *const GoldilocksField,
        d_out: *mut GoldilocksField,
        num_items: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    fn inclusive_scan_add_bf(
        d_temp_storage: *mut u8,
        temp_storage_bytes: &mut usize,
        d_in: *const GoldilocksField,
        d_out: *mut GoldilocksField,
        num_items: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    fn inclusive_scan_reverse_add_bf(
        d_temp_storage: *mut u8,
        temp_storage_bytes: &mut usize,
        d_in: *const GoldilocksField,
        d_out: *mut GoldilocksField,
        num_items: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    #[allow(improper_ctypes)]
    fn exclusive_scan_add_ef(
        d_temp_storage: *mut u8,
        temp_storage_bytes: &mut usize,
        d_in: *const ExtensionField,
        d_out: *mut ExtensionField,
        num_items: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    #[allow(improper_ctypes)]
    fn exclusive_scan_reverse_add_ef(
        d_temp_storage: *mut u8,
        temp_storage_bytes: &mut usize,
        d_in: *const ExtensionField,
        d_out: *mut ExtensionField,
        num_items: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    #[allow(improper_ctypes)]
    fn inclusive_scan_add_ef(
        d_temp_storage: *mut u8,
        temp_storage_bytes: &mut usize,
        d_in: *const ExtensionField,
        d_out: *mut ExtensionField,
        num_items: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    #[allow(improper_ctypes)]
    fn inclusive_scan_reverse_add_ef(
        d_temp_storage: *mut u8,
        temp_storage_bytes: &mut usize,
        d_in: *const ExtensionField,
        d_out: *mut ExtensionField,
        num_items: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    fn exclusive_scan_mul_bf(
        d_temp_storage: *mut u8,
        temp_storage_bytes: &mut usize,
        d_in: *const GoldilocksField,
        d_out: *mut GoldilocksField,
        num_items: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    fn exclusive_scan_reverse_mul_bf(
        d_temp_storage: *mut u8,
        temp_storage_bytes: &mut usize,
        d_in: *const GoldilocksField,
        d_out: *mut GoldilocksField,
        num_items: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    fn inclusive_scan_mul_bf(
        d_temp_storage: *mut u8,
        temp_storage_bytes: &mut usize,
        d_in: *const GoldilocksField,
        d_out: *mut GoldilocksField,
        num_items: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    fn inclusive_scan_reverse_mul_bf(
        d_temp_storage: *mut u8,
        temp_storage_bytes: &mut usize,
        d_in: *const GoldilocksField,
        d_out: *mut GoldilocksField,
        num_items: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    #[allow(improper_ctypes)]
    fn exclusive_scan_mul_ef(
        d_temp_storage: *mut u8,
        temp_storage_bytes: &mut usize,
        d_in: *const ExtensionField,
        d_out: *mut ExtensionField,
        num_items: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    #[allow(improper_ctypes)]
    fn exclusive_scan_reverse_mul_ef(
        d_temp_storage: *mut u8,
        temp_storage_bytes: &mut usize,
        d_in: *const ExtensionField,
        d_out: *mut ExtensionField,
        num_items: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    #[allow(improper_ctypes)]
    fn inclusive_scan_mul_ef(
        d_temp_storage: *mut u8,
        temp_storage_bytes: &mut usize,
        d_in: *const ExtensionField,
        d_out: *mut ExtensionField,
        num_items: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    #[allow(improper_ctypes)]
    fn inclusive_scan_reverse_mul_ef(
        d_temp_storage: *mut u8,
        temp_storage_bytes: &mut usize,
        d_in: *const ExtensionField,
        d_out: *mut ExtensionField,
        num_items: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
}

#[derive(Copy, Clone)]
pub enum ScanOperation {
    Sum,
    Product,
}

type ScanFunction<T> = unsafe extern "C" fn(
    d_temp_storage: *mut u8,
    temp_storage_bytes: &mut usize,
    d_in: *const T,
    d_out: *mut T,
    num_items: i32,
    stream: cudaStream_t,
) -> cudaError_t;

pub trait Scan: Sized {
    fn get_function(operation: ScanOperation, inclusive: bool, reverse: bool)
        -> ScanFunction<Self>;

    fn get_scan_temp_storage_bytes(
        operation: ScanOperation,
        inclusive: bool,
        reverse: bool,
        num_items: i32,
    ) -> CudaResult<usize> {
        let d_temp_storage = DeviceSlice::empty_mut();
        let mut temp_storage_bytes = 0;
        let d_in = DeviceSlice::empty();
        let d_out = DeviceSlice::empty_mut();
        let function = Self::get_function(operation, inclusive, reverse);
        unsafe {
            function(
                d_temp_storage.as_mut_ptr(),
                &mut temp_storage_bytes,
                d_in.as_ptr(),
                d_out.as_mut_ptr(),
                num_items,
                null_mut(),
            )
            .wrap_value(temp_storage_bytes)
        }
    }

    fn get_batch_scan_temp_storage_bytes(
        operation: ScanOperation,
        inclusive: bool,
        reverse: bool,
        batch_size: i32,
        num_items: i32,
    ) -> CudaResult<usize> {
        get_scan_temp_storage_bytes::<Self>(operation, inclusive, reverse, num_items)
            .map(|x| x * batch_size as usize)
    }

    fn scan(
        operation: ScanOperation,
        inclusive: bool,
        reverse: bool,
        d_temp_storage: &mut DeviceSlice<u8>,
        d_in: &DeviceSlice<Self>,
        d_out: &mut DeviceSlice<Self>,
        stream: &CudaStream,
    ) -> CudaResult<()> {
        let mut temp_storage_bytes = d_temp_storage.len();
        assert_eq!(d_in.len(), d_out.len());
        assert!(d_out.len() <= i32::MAX as usize);
        let num_items = d_out.len() as i32;
        let function = Self::get_function(operation, inclusive, reverse);
        unsafe {
            function(
                d_temp_storage.as_mut_ptr(),
                &mut temp_storage_bytes,
                d_in.as_ptr(),
                d_out.as_mut_ptr(),
                num_items,
                stream.into(),
            )
            .wrap()
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn batch_scan(
        operation: ScanOperation,
        inclusive: bool,
        reverse: bool,
        batch_size: i32,
        d_temp_storage: &mut DeviceSlice<u8>,
        d_in: &DeviceSlice<Self>,
        d_out: &mut DeviceSlice<Self>,
        stream: &CudaStream,
    ) -> CudaResult<()> {
        let num_items = (d_in.len() / batch_size as usize) as i32;
        Self::batch_chunk_scan(
            operation,
            inclusive,
            reverse,
            batch_size,
            0,
            num_items,
            d_temp_storage,
            d_in,
            d_out,
            stream,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn batch_chunk_scan(
        operation: ScanOperation,
        inclusive: bool,
        reverse: bool,
        batch_size: i32,
        chunk_offset: i32,
        num_items: i32,
        d_temp_storage: &mut DeviceSlice<u8>,
        d_in: &DeviceSlice<Self>,
        d_out: &mut DeviceSlice<Self>,
        stream: &CudaStream,
    ) -> CudaResult<()> {
        assert_eq!(d_in.len() % batch_size as usize, 0);
        assert_eq!(d_in.len(), d_out.len());
        let temp_storage_stride = d_temp_storage.len() / batch_size as usize;
        let data_stride = d_in.len() / batch_size as usize;
        assert!(chunk_offset + num_items <= data_stride as i32);
        let parent_ready = CudaEvent::create_with_flags(CudaEventCreateFlags::DISABLE_TIMING)?;
        parent_ready.record(stream)?;
        for i in 0..batch_size as usize {
            let child_stream = CudaStream::create_with_flags(CudaStreamCreateFlags::NON_BLOCKING)?;
            child_stream.wait_event(&parent_ready, CudaStreamWaitEventFlags::DEFAULT)?;
            let d_temp_storage =
                &mut d_temp_storage[i * temp_storage_stride..(i + 1) * temp_storage_stride];
            let data_offset = i * data_stride + chunk_offset as usize;
            let d_in = &d_in[data_offset..data_offset + num_items as usize];
            let d_out = &mut d_out[data_offset..data_offset + num_items as usize];
            scan(
                operation,
                inclusive,
                reverse,
                d_temp_storage,
                d_in,
                d_out,
                &child_stream,
            )?;
            let child_finished =
                CudaEvent::create_with_flags(CudaEventCreateFlags::DISABLE_TIMING)?;
            child_finished.record(&child_stream)?;
            stream.wait_event(&child_finished, CudaStreamWaitEventFlags::DEFAULT)?;
            child_finished.destroy()?;
            child_stream.destroy()?;
        }
        parent_ready.destroy()?;
        Ok(())
    }

    fn scan_in_place(
        operation: ScanOperation,
        inclusive: bool,
        reverse: bool,
        d_temp_storage: &mut DeviceSlice<u8>,
        d_values: &mut DeviceSlice<Self>,
        stream: &CudaStream,
    ) -> CudaResult<()> {
        let mut temp_storage_bytes = d_temp_storage.len();
        assert!(d_values.len() <= i32::MAX as usize);
        let num_items = d_values.len() as i32;
        let function = Self::get_function(operation, inclusive, reverse);
        unsafe {
            function(
                d_temp_storage.as_mut_ptr(),
                &mut temp_storage_bytes,
                d_values.as_ptr(),
                d_values.as_mut_ptr(),
                num_items,
                stream.into(),
            )
            .wrap()
        }
    }

    fn batch_scan_in_place(
        operation: ScanOperation,
        inclusive: bool,
        reverse: bool,
        batch_size: i32,
        d_temp_storage: &mut DeviceSlice<u8>,
        d_values: &mut DeviceSlice<Self>,
        stream: &CudaStream,
    ) -> CudaResult<()> {
        let num_items = (d_values.len() / batch_size as usize) as i32;
        Self::batch_chunk_scan_in_place(
            operation,
            inclusive,
            reverse,
            batch_size,
            0,
            num_items,
            d_temp_storage,
            d_values,
            stream,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn batch_chunk_scan_in_place(
        operation: ScanOperation,
        inclusive: bool,
        reverse: bool,
        batch_size: i32,
        chunk_offset: i32,
        num_items: i32,
        d_temp_storage: &mut DeviceSlice<u8>,
        d_values: &mut DeviceSlice<Self>,
        stream: &CudaStream,
    ) -> CudaResult<()> {
        assert_eq!(d_values.len() % batch_size as usize, 0);
        let temp_storage_stride = d_temp_storage.len() / batch_size as usize;
        let data_stride = d_values.len() / batch_size as usize;
        assert!(chunk_offset + num_items <= data_stride as i32);
        let parent_ready = CudaEvent::create_with_flags(CudaEventCreateFlags::DISABLE_TIMING)?;
        parent_ready.record(stream)?;
        for i in 0..batch_size as usize {
            let child_stream = CudaStream::create_with_flags(CudaStreamCreateFlags::NON_BLOCKING)?;
            child_stream.wait_event(&parent_ready, CudaStreamWaitEventFlags::DEFAULT)?;
            let d_temp_storage =
                &mut d_temp_storage[i * temp_storage_stride..(i + 1) * temp_storage_stride];
            let data_offset = i * data_stride + chunk_offset as usize;
            let d_values = &mut d_values[data_offset..data_offset + num_items as usize];
            scan_in_place(
                operation,
                inclusive,
                reverse,
                d_temp_storage,
                d_values,
                &child_stream,
            )?;
            let child_finished =
                CudaEvent::create_with_flags(CudaEventCreateFlags::DISABLE_TIMING)?;
            child_finished.record(&child_stream)?;
            stream.wait_event(&child_finished, CudaStreamWaitEventFlags::DEFAULT)?;
            child_finished.destroy()?;
            child_stream.destroy()?;
        }
        parent_ready.destroy()?;
        Ok(())
    }
}

impl Scan for u32 {
    fn get_function(
        operation: ScanOperation,
        inclusive: bool,
        reverse: bool,
    ) -> ScanFunction<Self> {
        match (operation, inclusive, reverse) {
            (ScanOperation::Sum, false, false) => exclusive_sum_u32,
            (ScanOperation::Sum, false, true) => exclusive_sum_reverse_u32,
            (ScanOperation::Sum, true, false) => inclusive_sum_u32,
            (ScanOperation::Sum, true, true) => inclusive_sum_reverse_u32,
            (ScanOperation::Product, _, _) => unimplemented!(),
        }
    }
}

impl Scan for GoldilocksField {
    fn get_function(
        operation: ScanOperation,
        inclusive: bool,
        reverse: bool,
    ) -> ScanFunction<Self> {
        match (operation, inclusive, reverse) {
            (ScanOperation::Sum, false, false) => exclusive_scan_add_bf,
            (ScanOperation::Sum, false, true) => exclusive_scan_reverse_add_bf,
            (ScanOperation::Sum, true, false) => inclusive_scan_add_bf,
            (ScanOperation::Sum, true, true) => inclusive_scan_reverse_add_bf,
            (ScanOperation::Product, false, false) => exclusive_scan_mul_bf,
            (ScanOperation::Product, false, true) => exclusive_scan_reverse_mul_bf,
            (ScanOperation::Product, true, false) => inclusive_scan_mul_bf,
            (ScanOperation::Product, true, true) => inclusive_scan_reverse_mul_bf,
        }
    }
}

impl Scan for ExtensionField {
    fn get_function(
        operation: ScanOperation,
        inclusive: bool,
        reverse: bool,
    ) -> ScanFunction<Self> {
        match (operation, inclusive, reverse) {
            (ScanOperation::Sum, false, false) => exclusive_scan_add_ef,
            (ScanOperation::Sum, false, true) => exclusive_scan_reverse_add_ef,
            (ScanOperation::Sum, true, false) => inclusive_scan_add_ef,
            (ScanOperation::Sum, true, true) => inclusive_scan_reverse_add_ef,
            (ScanOperation::Product, false, false) => exclusive_scan_mul_ef,
            (ScanOperation::Product, false, true) => exclusive_scan_reverse_mul_ef,
            (ScanOperation::Product, true, false) => inclusive_scan_mul_ef,
            (ScanOperation::Product, true, true) => inclusive_scan_reverse_mul_ef,
        }
    }
}

pub fn get_scan_temp_storage_bytes<T: Scan>(
    operation: ScanOperation,
    inclusive: bool,
    reverse: bool,
    num_items: i32,
) -> CudaResult<usize> {
    T::get_scan_temp_storage_bytes(operation, inclusive, reverse, num_items)
}

pub fn get_batch_scan_temp_storage_bytes<T: Scan>(
    operation: ScanOperation,
    inclusive: bool,
    reverse: bool,
    batch_size: i32,
    num_items: i32,
) -> CudaResult<usize> {
    T::get_batch_scan_temp_storage_bytes(operation, inclusive, reverse, batch_size, num_items)
}

pub fn scan<T: Scan>(
    operation: ScanOperation,
    inclusive: bool,
    reverse: bool,
    d_temp_storage: &mut DeviceSlice<u8>,
    d_in: &DeviceSlice<T>,
    d_out: &mut DeviceSlice<T>,
    stream: &CudaStream,
) -> CudaResult<()> {
    T::scan(
        operation,
        inclusive,
        reverse,
        d_temp_storage,
        d_in,
        d_out,
        stream,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn batch_scan<T: Scan>(
    operation: ScanOperation,
    inclusive: bool,
    reverse: bool,
    batch_size: i32,
    d_temp_storage: &mut DeviceSlice<u8>,
    d_in: &DeviceSlice<T>,
    d_out: &mut DeviceSlice<T>,
    stream: &CudaStream,
) -> CudaResult<()> {
    T::batch_scan(
        operation,
        inclusive,
        reverse,
        batch_size,
        d_temp_storage,
        d_in,
        d_out,
        stream,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn batch_chunk_scan<T: Scan>(
    operation: ScanOperation,
    inclusive: bool,
    reverse: bool,
    batch_size: i32,
    chunk_offset: i32,
    num_items: i32,
    d_temp_storage: &mut DeviceSlice<u8>,
    d_in: &DeviceSlice<T>,
    d_out: &mut DeviceSlice<T>,
    stream: &CudaStream,
) -> CudaResult<()> {
    T::batch_chunk_scan(
        operation,
        inclusive,
        reverse,
        batch_size,
        chunk_offset,
        num_items,
        d_temp_storage,
        d_in,
        d_out,
        stream,
    )
}

pub fn scan_in_place<T: Scan>(
    operation: ScanOperation,
    inclusive: bool,
    reverse: bool,
    d_temp_storage: &mut DeviceSlice<u8>,
    d_values: &mut DeviceSlice<T>,
    stream: &CudaStream,
) -> CudaResult<()> {
    T::scan_in_place(
        operation,
        inclusive,
        reverse,
        d_temp_storage,
        d_values,
        stream,
    )
}

pub fn batch_scan_in_place<T: Scan>(
    operation: ScanOperation,
    inclusive: bool,
    reverse: bool,
    batch_size: i32,
    d_temp_storage: &mut DeviceSlice<u8>,
    d_values: &mut DeviceSlice<T>,
    stream: &CudaStream,
) -> CudaResult<()> {
    T::batch_scan_in_place(
        operation,
        inclusive,
        reverse,
        batch_size,
        d_temp_storage,
        d_values,
        stream,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn batch_chunk_scan_in_place<T: Scan>(
    operation: ScanOperation,
    inclusive: bool,
    reverse: bool,
    batch_size: i32,
    chunk_offset: i32,
    num_items: i32,
    d_temp_storage: &mut DeviceSlice<u8>,
    d_values: &mut DeviceSlice<T>,
    stream: &CudaStream,
) -> CudaResult<()> {
    T::batch_chunk_scan_in_place(
        operation,
        inclusive,
        reverse,
        batch_size,
        chunk_offset,
        num_items,
        d_temp_storage,
        d_values,
        stream,
    )
}

#[cfg(test)]
mod tests {
    use std::convert::identity;

    use boojum::field::goldilocks::GoldilocksField;
    use boojum::field::Field;
    use itertools::Itertools;
    use rand::distributions::Uniform;
    use rand::{thread_rng, Rng};

    use cudart::memory::{memory_copy_async, DeviceAllocation};
    use cudart::stream::CudaStream;

    use crate::extension_field::ExtensionField;
    use crate::ops_cub::device_scan::{get_scan_temp_storage_bytes, Scan};

    use super::ScanOperation;

    trait ScanTest {
        fn get_initial_state(operation: ScanOperation) -> Self;
        fn scan_operation(operation: ScanOperation, a: Self, b: Self) -> Self;
    }

    impl ScanTest for u32 {
        fn get_initial_state(operation: ScanOperation) -> Self {
            match operation {
                ScanOperation::Sum => 0,
                ScanOperation::Product => 1,
            }
        }

        fn scan_operation(operation: ScanOperation, a: Self, b: Self) -> Self {
            match operation {
                ScanOperation::Sum => a + b,
                ScanOperation::Product => a * b,
            }
        }
    }

    impl ScanTest for GoldilocksField {
        fn get_initial_state(operation: ScanOperation) -> Self {
            match operation {
                ScanOperation::Sum => GoldilocksField::ZERO,
                ScanOperation::Product => GoldilocksField::ONE,
            }
        }

        fn scan_operation(operation: ScanOperation, a: Self, b: Self) -> Self {
            match operation {
                ScanOperation::Sum => a + b,
                ScanOperation::Product => a * b,
            }
        }
    }

    impl ScanTest for ExtensionField {
        fn get_initial_state(operation: ScanOperation) -> Self {
            match operation {
                ScanOperation::Sum => ExtensionField::ZERO,
                ScanOperation::Product => ExtensionField::ONE,
            }
        }

        fn scan_operation(operation: ScanOperation, a: Self, b: Self) -> Self {
            match operation {
                ScanOperation::Sum => {
                    let mut result = a;
                    result.add_assign(&b);
                    result
                }
                ScanOperation::Product => {
                    let mut result = a;
                    result.mul_assign(&b);
                    result
                }
            }
        }
    }

    fn verify<T>(
        operation: ScanOperation,
        inclusive: bool,
        reverse: bool,
        h_in: Vec<T>,
        h_out: Vec<T>,
    ) where
        T: Scan + ScanTest + Default + Copy + Eq,
    {
        let h_in = if reverse {
            h_in.into_iter().rev().collect_vec()
        } else {
            h_in
        };
        let initial_state = T::get_initial_state(operation);
        let state_fn = |state: &mut T, x| {
            let current = *state;
            let next = T::scan_operation(operation, current, x);
            *state = next;
            Some(if inclusive { next } else { current })
        };
        let h_in = h_in.into_iter().scan(initial_state, state_fn).collect_vec();
        let h_in = if reverse {
            h_in.into_iter().rev().collect_vec()
        } else {
            h_in
        };
        assert!(h_in.into_iter().zip(h_out.into_iter()).all(|(x, y)| x == y));
    }

    fn scan<T>(operation: ScanOperation, inclusive: bool, reverse: bool, convert: fn(u32) -> T)
    where
        T: Scan + ScanTest + Default + Copy + Eq,
    {
        const NUM_ITEMS: usize = 1 << 16;
        const RANGE_MAX: u32 = 1 << 16;
        let temp_storage_bytes =
            get_scan_temp_storage_bytes::<T>(operation, inclusive, reverse, NUM_ITEMS as i32)
                .unwrap();
        let mut d_temp_storage = DeviceAllocation::alloc(temp_storage_bytes).unwrap();
        let h_in = thread_rng()
            .sample_iter(Uniform::from(0..RANGE_MAX))
            .map(convert)
            .take(NUM_ITEMS)
            .collect_vec();
        let mut h_out = vec![T::default(); NUM_ITEMS];
        let mut d_in = DeviceAllocation::alloc(NUM_ITEMS).unwrap();
        let mut d_out = DeviceAllocation::alloc(NUM_ITEMS).unwrap();
        let stream = CudaStream::default();
        memory_copy_async(&mut d_in, &h_in, &stream).unwrap();
        super::scan(
            operation,
            inclusive,
            reverse,
            &mut d_temp_storage,
            &d_in,
            &mut d_out,
            &stream,
        )
        .unwrap();
        memory_copy_async(&mut h_out, &d_out, &stream).unwrap();
        stream.synchronize().unwrap();
        verify(operation, inclusive, reverse, h_in, h_out);
    }

    fn batch_scan<T>(
        operation: ScanOperation,
        inclusive: bool,
        reverse: bool,
        convert: fn(u32) -> T,
    ) where
        T: Scan + ScanTest + Default + Copy + Eq,
    {
        const BATCH_SIZE: usize = 1 << 8;
        const NUM_ITEMS: usize = 1 << 8;
        const RANGE_MAX: u32 = 1 << 16;
        let temp_storage_bytes = super::get_batch_scan_temp_storage_bytes::<T>(
            operation,
            inclusive,
            reverse,
            BATCH_SIZE as i32,
            NUM_ITEMS as i32,
        )
        .unwrap();
        let mut d_temp_storage = DeviceAllocation::alloc(temp_storage_bytes).unwrap();
        let h_in = thread_rng()
            .sample_iter(Uniform::from(0..RANGE_MAX))
            .map(convert)
            .take(NUM_ITEMS * BATCH_SIZE)
            .collect_vec();
        let mut h_out = vec![T::default(); NUM_ITEMS * BATCH_SIZE];
        let mut d_in = DeviceAllocation::alloc(NUM_ITEMS * BATCH_SIZE).unwrap();
        let mut d_out = DeviceAllocation::alloc(NUM_ITEMS * BATCH_SIZE).unwrap();
        let stream = CudaStream::default();
        memory_copy_async(&mut d_in, &h_in, &stream).unwrap();
        super::batch_scan(
            operation,
            inclusive,
            reverse,
            BATCH_SIZE as i32,
            &mut d_temp_storage,
            &d_in,
            &mut d_out,
            &stream,
        )
        .unwrap();
        memory_copy_async(&mut h_out, &d_out, &stream).unwrap();
        stream.synchronize().unwrap();
        h_in.into_iter()
            .chunks(NUM_ITEMS)
            .into_iter()
            .zip(h_out.chunks(NUM_ITEMS))
            .for_each(|(h_in, h_out)| {
                let h_in = h_in.collect_vec();
                let h_out = Vec::from(h_out);
                verify(operation, inclusive, reverse, h_in, h_out);
            });
    }

    fn batch_chunk_scan<T>(
        operation: ScanOperation,
        inclusive: bool,
        reverse: bool,
        convert: fn(u32) -> T,
    ) where
        T: Scan + ScanTest + Default + Copy + Eq,
    {
        const BATCH_SIZE: usize = 1 << 8;
        const NUM_ITEMS: usize = 1 << 8;
        const STRIDE: usize = NUM_ITEMS * 2;
        const RANGE_MAX: u32 = 1 << 16;
        let temp_storage_bytes = super::get_batch_scan_temp_storage_bytes::<T>(
            operation,
            inclusive,
            reverse,
            BATCH_SIZE as i32,
            NUM_ITEMS as i32,
        )
        .unwrap();
        let mut d_temp_storage = DeviceAllocation::alloc(temp_storage_bytes).unwrap();
        let h_in = thread_rng()
            .sample_iter(Uniform::from(0..RANGE_MAX))
            .map(convert)
            .take(STRIDE * BATCH_SIZE)
            .collect_vec();
        let mut h_out = vec![T::default(); STRIDE * BATCH_SIZE];
        let mut d_in = DeviceAllocation::alloc(STRIDE * BATCH_SIZE).unwrap();
        let mut d_out = DeviceAllocation::alloc(STRIDE * BATCH_SIZE).unwrap();
        let stream = CudaStream::default();
        memory_copy_async(&mut d_in, &h_in, &stream).unwrap();
        super::batch_chunk_scan(
            operation,
            inclusive,
            reverse,
            BATCH_SIZE as i32,
            NUM_ITEMS as i32,
            NUM_ITEMS as i32,
            &mut d_temp_storage,
            &d_in,
            &mut d_out,
            &stream,
        )
        .unwrap();
        memory_copy_async(&mut h_out, &d_out, &stream).unwrap();
        stream.synchronize().unwrap();
        h_in.into_iter()
            .chunks(STRIDE)
            .into_iter()
            .skip(NUM_ITEMS)
            .zip(
                h_out
                    .into_iter()
                    .chunks(NUM_ITEMS)
                    .into_iter()
                    .skip(NUM_ITEMS),
            )
            .for_each(|(h_in, h_out)| {
                let h_in = h_in.collect_vec();
                let h_out = h_out.collect_vec();
                verify(operation, inclusive, reverse, h_in, h_out);
            });
    }

    fn scan_u32(operation: ScanOperation, inclusive: bool, reverse: bool) {
        scan::<u32>(operation, inclusive, reverse, identity);
    }

    fn batch_scan_u32(operation: ScanOperation, inclusive: bool, reverse: bool) {
        batch_scan::<u32>(operation, inclusive, reverse, identity);
    }

    fn batch_chunk_scan_u32(operation: ScanOperation, inclusive: bool, reverse: bool) {
        batch_chunk_scan::<u32>(operation, inclusive, reverse, identity);
    }

    fn scan_bf(operation: ScanOperation, inclusive: bool, reverse: bool) {
        scan::<GoldilocksField>(operation, inclusive, reverse, |x| {
            GoldilocksField::from_nonreduced_u64(x as u64)
        });
    }

    fn batch_scan_bf(operation: ScanOperation, inclusive: bool, reverse: bool) {
        batch_scan::<GoldilocksField>(operation, inclusive, reverse, |x| {
            GoldilocksField::from_nonreduced_u64(x as u64)
        });
    }

    fn batch_chunk_scan_bf(operation: ScanOperation, inclusive: bool, reverse: bool) {
        batch_chunk_scan::<GoldilocksField>(operation, inclusive, reverse, |x| {
            GoldilocksField::from_nonreduced_u64(x as u64)
        });
    }

    fn scan_ef(operation: ScanOperation, inclusive: bool, reverse: bool) {
        scan::<ExtensionField>(operation, inclusive, reverse, |x| {
            let c0 = GoldilocksField(x as u64);
            let mut c1 = c0;
            c1.mul_assign(&GoldilocksField(42));
            ExtensionField::from_coeff_in_base([c0, c1])
        });
    }

    fn batch_scan_ef(operation: ScanOperation, inclusive: bool, reverse: bool) {
        batch_scan::<ExtensionField>(operation, inclusive, reverse, |x| {
            let c0 = GoldilocksField(x as u64);
            let mut c1 = c0;
            c1.mul_assign(&GoldilocksField(42));
            ExtensionField::from_coeff_in_base([c0, c1])
        });
    }

    fn batch_chunk_scan_ef(operation: ScanOperation, inclusive: bool, reverse: bool) {
        batch_chunk_scan::<ExtensionField>(operation, inclusive, reverse, |x| {
            let c0 = GoldilocksField(x as u64);
            let mut c1 = c0;
            c1.mul_assign(&GoldilocksField(42));
            ExtensionField::from_coeff_in_base([c0, c1])
        });
    }

    #[test]
    fn sum_exclusive_forward_u32() {
        scan_u32(ScanOperation::Sum, false, false);
    }

    #[test]
    fn sum_inclusive_forward_u32() {
        scan_u32(ScanOperation::Sum, true, false);
    }

    #[test]
    fn sum_exclusive_reverse_u32() {
        scan_u32(ScanOperation::Sum, false, true);
    }

    #[test]
    fn sum_inclusive_reverse_u32() {
        scan_u32(ScanOperation::Sum, true, true);
    }

    #[test]
    fn batch_sum_exclusive_forward_u32() {
        batch_scan_u32(ScanOperation::Sum, false, false);
    }

    #[test]
    fn batch_sum_inclusive_forward_u32() {
        batch_scan_u32(ScanOperation::Sum, true, false);
    }

    #[test]
    fn batch_sum_exclusive_reverse_u32() {
        batch_scan_u32(ScanOperation::Sum, false, true);
    }

    #[test]
    fn batch_sum_inclusive_reverse_u32() {
        batch_scan_u32(ScanOperation::Sum, true, true);
    }

    #[test]
    fn batch_chunk_sum_exclusive_forward_u32() {
        batch_chunk_scan_u32(ScanOperation::Sum, false, false);
    }

    #[test]
    fn batch_chunk_sum_inclusive_forward_u32() {
        batch_chunk_scan_u32(ScanOperation::Sum, true, false);
    }

    #[test]
    fn batch_chunk_sum_exclusive_reverse_u32() {
        batch_chunk_scan_u32(ScanOperation::Sum, false, true);
    }

    #[test]
    fn batch_chunk_sum_inclusive_reverse_u32() {
        batch_chunk_scan_u32(ScanOperation::Sum, true, true);
    }

    #[test]
    fn sum_exclusive_forward_bf() {
        scan_bf(ScanOperation::Sum, false, false);
    }

    #[test]
    fn sum_inclusive_forward_bf() {
        scan_bf(ScanOperation::Sum, true, false);
    }

    #[test]
    fn sum_exclusive_reverse_bf() {
        scan_bf(ScanOperation::Sum, false, true);
    }

    #[test]
    fn sum_inclusive_reverse_bf() {
        scan_bf(ScanOperation::Sum, true, true);
    }

    #[test]
    fn batch_sum_exclusive_forward_bf() {
        batch_scan_bf(ScanOperation::Sum, false, false);
    }

    #[test]
    fn batch_sum_inclusive_forward_bf() {
        batch_scan_bf(ScanOperation::Sum, true, false);
    }

    #[test]
    fn batch_sum_exclusive_reverse_bf() {
        batch_scan_bf(ScanOperation::Sum, false, true);
    }

    #[test]
    fn batch_sum_inclusive_reverse_bf() {
        batch_scan_bf(ScanOperation::Sum, true, true);
    }

    #[test]
    fn batch_chunk_sum_exclusive_forward_bf() {
        batch_chunk_scan_bf(ScanOperation::Sum, false, false);
    }

    #[test]
    fn batch_chunk_sum_inclusive_forward_bf() {
        batch_chunk_scan_bf(ScanOperation::Sum, true, false);
    }

    #[test]
    fn batch_chunk_sum_exclusive_reverse_bf() {
        batch_chunk_scan_bf(ScanOperation::Sum, false, true);
    }

    #[test]
    fn batch_chunk_sum_inclusive_reverse_bf() {
        batch_chunk_scan_bf(ScanOperation::Sum, true, true);
    }

    #[test]
    fn product_exclusive_forward_bf() {
        scan_bf(ScanOperation::Product, false, false);
    }

    #[test]
    fn product_inclusive_forward_bf() {
        scan_bf(ScanOperation::Product, true, false);
    }

    #[test]
    fn product_exclusive_reverse_bf() {
        scan_bf(ScanOperation::Product, false, true);
    }

    #[test]
    fn product_inclusive_reverse_bf() {
        scan_bf(ScanOperation::Product, true, true);
    }

    #[test]
    fn batch_product_exclusive_forward_bf() {
        batch_scan_bf(ScanOperation::Product, false, false);
    }

    #[test]
    fn batch_product_inclusive_forward_bf() {
        batch_scan_bf(ScanOperation::Product, true, false);
    }

    #[test]
    fn batch_product_exclusive_reverse_bf() {
        batch_scan_bf(ScanOperation::Product, false, true);
    }

    #[test]
    fn batch_product_inclusive_reverse_bf() {
        batch_scan_bf(ScanOperation::Product, true, true);
    }

    #[test]
    fn batch_chunk_product_exclusive_forward_bf() {
        batch_chunk_scan_bf(ScanOperation::Product, false, false);
    }

    #[test]
    fn batch_chunk_product_inclusive_forward_bf() {
        batch_chunk_scan_bf(ScanOperation::Product, true, false);
    }

    #[test]
    fn batch_chunk_product_exclusive_reverse_bf() {
        batch_chunk_scan_bf(ScanOperation::Product, false, true);
    }

    #[test]
    fn batch_chunk_product_inclusive_reverse_bf() {
        batch_chunk_scan_bf(ScanOperation::Product, true, true);
    }

    #[test]
    fn sum_exclusive_forward_ef() {
        scan_ef(ScanOperation::Sum, false, false);
    }

    #[test]
    fn sum_inclusive_forward_ef() {
        scan_ef(ScanOperation::Sum, true, false);
    }

    #[test]
    fn sum_exclusive_reverse_ef() {
        scan_ef(ScanOperation::Sum, false, true);
    }

    #[test]
    fn sum_inclusive_reverse_ef() {
        scan_ef(ScanOperation::Sum, true, true);
    }

    #[test]
    fn batch_sum_exclusive_forward_ef() {
        batch_scan_ef(ScanOperation::Sum, false, false);
    }

    #[test]
    fn batch_sum_inclusive_forward_ef() {
        batch_scan_ef(ScanOperation::Sum, true, false);
    }

    #[test]
    fn batch_sum_exclusive_reverse_ef() {
        batch_scan_ef(ScanOperation::Sum, false, true);
    }

    #[test]
    fn batch_sum_inclusive_reverse_ef() {
        batch_scan_ef(ScanOperation::Sum, true, true);
    }

    #[test]
    fn batch_chunk_sum_exclusive_forward_ef() {
        batch_chunk_scan_ef(ScanOperation::Sum, false, false);
    }

    #[test]
    fn batch_chunk_sum_inclusive_forward_ef() {
        batch_chunk_scan_ef(ScanOperation::Sum, true, false);
    }

    #[test]
    fn batch_chunk_sum_exclusive_reverse_ef() {
        batch_chunk_scan_ef(ScanOperation::Sum, false, true);
    }

    #[test]
    fn batch_chunk_sum_inclusive_reverse_ef() {
        batch_chunk_scan_ef(ScanOperation::Sum, true, true);
    }

    #[test]
    fn product_exclusive_forward_ef() {
        scan_ef(ScanOperation::Product, false, false);
    }

    #[test]
    fn product_inclusive_forward_ef() {
        scan_ef(ScanOperation::Product, true, false);
    }

    #[test]
    fn product_exclusive_reverse_ef() {
        scan_ef(ScanOperation::Product, false, true);
    }

    #[test]
    fn product_inclusive_reverse_ef() {
        scan_ef(ScanOperation::Product, true, true);
    }

    #[test]
    fn batch_product_exclusive_forward_ef() {
        batch_scan_ef(ScanOperation::Product, false, false);
    }

    #[test]
    fn batch_product_inclusive_forward_ef() {
        batch_scan_ef(ScanOperation::Product, true, false);
    }

    #[test]
    fn batch_product_exclusive_reverse_ef() {
        batch_scan_ef(ScanOperation::Product, false, true);
    }

    #[test]
    fn batch_product_inclusive_reverse_ef() {
        batch_scan_ef(ScanOperation::Product, true, true);
    }

    #[test]
    fn batch_chunk_product_exclusive_forward_ef() {
        batch_chunk_scan_ef(ScanOperation::Product, false, false);
    }

    #[test]
    fn batch_chunk_product_inclusive_forward_ef() {
        batch_chunk_scan_ef(ScanOperation::Product, true, false);
    }

    #[test]
    fn batch_chunk_product_exclusive_reverse_ef() {
        batch_chunk_scan_ef(ScanOperation::Product, false, true);
    }

    #[test]
    fn batch_chunk_product_inclusive_reverse_ef() {
        batch_chunk_scan_ef(ScanOperation::Product, true, true);
    }
}
