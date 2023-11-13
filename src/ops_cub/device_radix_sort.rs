use std::ptr::null_mut;

use boojum::field::goldilocks::GoldilocksField;

use cudart::result::{CudaResult, CudaResultWrap};
use cudart::slice::DeviceSlice;
use cudart::stream::CudaStream;
use cudart_sys::{cudaError_t, cudaStream_t};

extern "C" {
    fn sort_keys_u32(
        d_temp_storage: *mut u8,
        temp_storage_bytes: &mut usize,
        d_keys_in: *const u32,
        d_keys_out: *mut u32,
        num_items: u32,
        begin_bit: i32,
        end_bit: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    fn sort_keys_descending_u32(
        d_temp_storage: *mut u8,
        temp_storage_bytes: &mut usize,
        d_keys_in: *const u32,
        d_keys_out: *mut u32,
        num_items: u32,
        begin_bit: i32,
        end_bit: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    fn sort_keys_u64(
        d_temp_storage: *mut u8,
        temp_storage_bytes: &mut usize,
        d_keys_in: *const u64,
        d_keys_out: *mut u64,
        num_items: u32,
        begin_bit: i32,
        end_bit: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    fn sort_keys_descending_u64(
        d_temp_storage: *mut u8,
        temp_storage_bytes: &mut usize,
        d_keys_in: *const u64,
        d_keys_out: *mut u64,
        num_items: u32,
        begin_bit: i32,
        end_bit: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
}

pub type SortKeysFunction<T> = unsafe extern "C" fn(
    *mut u8,
    &mut usize,
    *const T,
    *mut T,
    num_items: u32,
    begin_bit: i32,
    end_bit: i32,
    stream: cudaStream_t,
) -> cudaError_t;

pub trait SortKeys: Sized {
    fn get_function(descending: bool) -> SortKeysFunction<Self>;

    fn get_sort_keys_temp_storage_bytes(
        descending: bool,
        num_items: u32,
        begin_bit: i32,
        end_bit: i32,
    ) -> CudaResult<usize> {
        let d_temp_storage = DeviceSlice::empty_mut();
        let mut temp_storage_bytes = 0;
        let d_keys_in = DeviceSlice::empty();
        let d_keys_out = DeviceSlice::empty_mut();
        let function = Self::get_function(descending);
        unsafe {
            function(
                d_temp_storage.as_mut_ptr(),
                &mut temp_storage_bytes,
                d_keys_in.as_ptr(),
                d_keys_out.as_mut_ptr(),
                num_items,
                begin_bit,
                end_bit,
                null_mut(),
            )
            .wrap_value(temp_storage_bytes)
        }
    }

    fn sort_keys(
        descending: bool,
        d_temp_storage: &mut DeviceSlice<u8>,
        d_keys_in: &DeviceSlice<Self>,
        d_keys_out: &mut DeviceSlice<Self>,
        begin_bit: i32,
        end_bit: i32,
        stream: &CudaStream,
    ) -> CudaResult<()> {
        let mut temp_storage_bytes = d_temp_storage.len();
        assert_eq!(d_keys_in.len(), d_keys_out.len());
        assert!(d_keys_out.len() <= u32::MAX as usize);
        let num_items = d_keys_out.len() as u32;
        let function = Self::get_function(descending);
        unsafe {
            function(
                d_temp_storage.as_mut_ptr(),
                &mut temp_storage_bytes,
                d_keys_in.as_ptr(),
                d_keys_out.as_mut_ptr(),
                num_items,
                begin_bit,
                end_bit,
                stream.into(),
            )
            .wrap()
        }
    }
}

impl SortKeys for u32 {
    fn get_function(descending: bool) -> SortKeysFunction<Self> {
        if descending {
            sort_keys_descending_u32
        } else {
            sort_keys_u32
        }
    }
}

impl SortKeys for u64 {
    fn get_function(descending: bool) -> SortKeysFunction<Self> {
        if descending {
            sort_keys_descending_u64
        } else {
            sort_keys_u64
        }
    }
}

impl SortKeys for GoldilocksField {
    fn get_function(_descending: bool) -> SortKeysFunction<Self> {
        unimplemented!()
    }

    fn sort_keys(
        descending: bool,
        d_temp_storage: &mut DeviceSlice<u8>,
        d_keys_in: &DeviceSlice<Self>,
        d_keys_out: &mut DeviceSlice<Self>,
        begin_bit: i32,
        end_bit: i32,
        stream: &CudaStream,
    ) -> CudaResult<()> {
        let d_keys_in = unsafe { d_keys_in.transmute() };
        let d_keys_out = unsafe { d_keys_out.transmute_mut() };
        u64::sort_keys(
            descending,
            d_temp_storage,
            d_keys_in,
            d_keys_out,
            begin_bit,
            end_bit,
            stream,
        )
    }
}

pub fn get_sort_keys_temp_storage_bytes<T: SortKeys>(
    descending: bool,
    num_items: u32,
    begin_bit: i32,
    end_bit: i32,
) -> CudaResult<usize> {
    T::get_sort_keys_temp_storage_bytes(descending, num_items, begin_bit, end_bit)
}

pub fn sort_keys<T: SortKeys>(
    descending: bool,
    d_temp_storage: &mut DeviceSlice<u8>,
    d_keys_in: &DeviceSlice<T>,
    d_keys_out: &mut DeviceSlice<T>,
    begin_bit: i32,
    end_bit: i32,
    stream: &CudaStream,
) -> CudaResult<()> {
    T::sort_keys(
        descending,
        d_temp_storage,
        d_keys_in,
        d_keys_out,
        begin_bit,
        end_bit,
        stream,
    )
}

extern "C" {
    fn sort_pairs_u32_by_u32(
        d_temp_storage: *mut u8,
        temp_storage_bytes: &mut usize,
        d_keys_in: *const u32,
        d_keys_out: *mut u32,
        d_values_in: *const u32,
        d_values_out: *mut u32,
        num_items: u32,
        begin_bit: i32,
        end_bit: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    fn sort_pairs_descending_u32_by_u32(
        d_temp_storage: *mut u8,
        temp_storage_bytes: &mut usize,
        d_keys_in: *const u32,
        d_keys_out: *mut u32,
        d_values_in: *const u32,
        d_values_out: *mut u32,
        num_items: u32,
        begin_bit: i32,
        end_bit: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    fn sort_pairs_u32_by_u64(
        d_temp_storage: *mut u8,
        temp_storage_bytes: &mut usize,
        d_keys_in: *const u64,
        d_keys_out: *mut u64,
        d_values_in: *const u32,
        d_values_out: *mut u32,
        num_items: u32,
        begin_bit: i32,
        end_bit: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    fn sort_pairs_descending_u32_by_u64(
        d_temp_storage: *mut u8,
        temp_storage_bytes: &mut usize,
        d_keys_in: *const u64,
        d_keys_out: *mut u64,
        d_values_in: *const u32,
        d_values_out: *mut u32,
        num_items: u32,
        begin_bit: i32,
        end_bit: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    fn sort_pairs_u64_by_u32(
        d_temp_storage: *mut u8,
        temp_storage_bytes: &mut usize,
        d_keys_in: *const u32,
        d_keys_out: *mut u32,
        d_values_in: *const u64,
        d_values_out: *mut u64,
        num_items: u32,
        begin_bit: i32,
        end_bit: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    fn sort_pairs_descending_u64_by_u32(
        d_temp_storage: *mut u8,
        temp_storage_bytes: &mut usize,
        d_keys_in: *const u32,
        d_keys_out: *mut u32,
        d_values_in: *const u64,
        d_values_out: *mut u64,
        num_items: u32,
        begin_bit: i32,
        end_bit: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    fn sort_pairs_u64_by_u64(
        d_temp_storage: *mut u8,
        temp_storage_bytes: &mut usize,
        d_keys_in: *const u64,
        d_keys_out: *mut u64,
        d_values_in: *const u64,
        d_values_out: *mut u64,
        num_items: u32,
        begin_bit: i32,
        end_bit: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;

    fn sort_pairs_descending_u64_by_u64(
        d_temp_storage: *mut u8,
        temp_storage_bytes: &mut usize,
        d_keys_in: *const u64,
        d_keys_out: *mut u64,
        d_values_in: *const u64,
        d_values_out: *mut u64,
        num_items: u32,
        begin_bit: i32,
        end_bit: i32,
        stream: cudaStream_t,
    ) -> cudaError_t;
}

pub type SortPairsFunction<K, V> = unsafe extern "C" fn(
    *mut u8,
    &mut usize,
    *const K,
    *mut K,
    *const V,
    *mut V,
    num_items: u32,
    begin_bit: i32,
    end_bit: i32,
    stream: cudaStream_t,
) -> cudaError_t;

pub trait SortPairs<K, V> {
    fn get_function(descending: bool) -> SortPairsFunction<K, V>;

    fn get_sort_pairs_temp_storage_bytes(
        descending: bool,
        num_items: u32,
        begin_bit: i32,
        end_bit: i32,
    ) -> CudaResult<usize> {
        let d_temp_storage = DeviceSlice::empty_mut();
        let mut temp_storage_bytes = 0;
        let d_keys_in = DeviceSlice::empty();
        let d_keys_out = DeviceSlice::empty_mut();
        let d_values_in = DeviceSlice::empty();
        let d_values_out = DeviceSlice::empty_mut();
        let function = Self::get_function(descending);
        unsafe {
            function(
                d_temp_storage.as_mut_ptr(),
                &mut temp_storage_bytes,
                d_keys_in.as_ptr(),
                d_keys_out.as_mut_ptr(),
                d_values_in.as_ptr(),
                d_values_out.as_mut_ptr(),
                num_items,
                begin_bit,
                end_bit,
                null_mut(),
            )
            .wrap_value(temp_storage_bytes)
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn sort_pairs(
        descending: bool,
        d_temp_storage: &mut DeviceSlice<u8>,
        d_keys_in: &DeviceSlice<K>,
        d_keys_out: &mut DeviceSlice<K>,
        d_values_in: &DeviceSlice<V>,
        d_values_out: &mut DeviceSlice<V>,
        begin_bit: i32,
        end_bit: i32,
        stream: &CudaStream,
    ) -> CudaResult<()> {
        let mut temp_storage_bytes = d_temp_storage.len();
        assert_eq!(d_keys_in.len(), d_values_out.len());
        assert_eq!(d_keys_out.len(), d_values_out.len());
        assert_eq!(d_values_in.len(), d_values_out.len());
        assert!(d_values_out.len() <= u32::MAX as usize);
        let num_items = d_values_out.len() as u32;
        let function = Self::get_function(descending);
        unsafe {
            function(
                d_temp_storage.as_mut_ptr(),
                &mut temp_storage_bytes,
                d_keys_in.as_ptr(),
                d_keys_out.as_mut_ptr(),
                d_values_in.as_ptr(),
                d_values_out.as_mut_ptr(),
                num_items,
                begin_bit,
                end_bit,
                stream.into(),
            )
            .wrap()
        }
    }
}

impl SortPairs<u32, u32> for (u32, u32) {
    fn get_function(descending: bool) -> SortPairsFunction<u32, u32> {
        if descending {
            sort_pairs_descending_u32_by_u32
        } else {
            sort_pairs_u32_by_u32
        }
    }
}

impl SortPairs<u64, u32> for (u64, u32) {
    fn get_function(descending: bool) -> SortPairsFunction<u64, u32> {
        if descending {
            sort_pairs_descending_u32_by_u64
        } else {
            sort_pairs_u32_by_u64
        }
    }
}

impl SortPairs<u32, u64> for (u32, u64) {
    fn get_function(descending: bool) -> SortPairsFunction<u32, u64> {
        if descending {
            sort_pairs_descending_u64_by_u32
        } else {
            sort_pairs_u64_by_u32
        }
    }
}

impl SortPairs<u64, u64> for (u64, u64) {
    fn get_function(descending: bool) -> SortPairsFunction<u64, u64> {
        if descending {
            sort_pairs_descending_u64_by_u64
        } else {
            sort_pairs_u64_by_u64
        }
    }
}

pub fn get_sort_pairs_temp_storage_bytes<K, V>(
    descending: bool,
    num_items: u32,
    begin_bit: i32,
    end_bit: i32,
) -> CudaResult<usize>
where
    (K, V): SortPairs<K, V>,
{
    <(K, V)>::get_sort_pairs_temp_storage_bytes(descending, num_items, begin_bit, end_bit)
}

#[allow(clippy::too_many_arguments)]
pub fn sort_pairs<K, V>(
    descending: bool,
    d_temp_storage: &mut DeviceSlice<u8>,
    d_keys_in: &DeviceSlice<K>,
    d_keys_out: &mut DeviceSlice<K>,
    d_values_in: &DeviceSlice<V>,
    d_values_out: &mut DeviceSlice<V>,
    begin_bit: i32,
    end_bit: i32,
    stream: &CudaStream,
) -> CudaResult<()>
where
    (K, V): SortPairs<K, V>,
{
    <(K, V)>::sort_pairs(
        descending,
        d_temp_storage,
        d_keys_in,
        d_keys_out,
        d_values_in,
        d_values_out,
        begin_bit,
        end_bit,
        stream,
    )
}

#[cfg(test)]
mod tests {
    use std::mem::size_of;
    use std::ops::Not;

    use itertools::Itertools;
    use rand::distributions::{Distribution, Standard};
    use rand::{thread_rng, Rng};

    use cudart::memory::{memory_copy_async, DeviceAllocation};
    use cudart::stream::CudaStream;

    use super::*;

    fn test_sort_keys<T>(descending: bool)
    where
        T: SortKeys + Default + Clone + Ord + Eq,
        Standard: Distribution<T>,
    {
        const NUM_ITEMS: usize = 1 << 16;
        let end_bit = size_of::<T>() as i32 * 8;
        let temp_storage_bytes =
            get_sort_keys_temp_storage_bytes::<T>(descending, NUM_ITEMS as u32, 0, end_bit)
                .unwrap();
        let mut d_temp_storage = DeviceAllocation::alloc(temp_storage_bytes).unwrap();
        let mut h_keys_in = (0..NUM_ITEMS).map(|_| thread_rng().gen()).collect_vec();
        let mut h_keys_out = vec![T::default(); NUM_ITEMS];
        let mut d_keys_in = DeviceAllocation::alloc(NUM_ITEMS).unwrap();
        let mut d_keys_out = DeviceAllocation::alloc(NUM_ITEMS).unwrap();
        let stream = CudaStream::default();
        memory_copy_async(&mut d_keys_in, &h_keys_in, &stream).unwrap();
        sort_keys(
            descending,
            &mut d_temp_storage,
            &d_keys_in,
            &mut d_keys_out,
            0,
            end_bit,
            &stream,
        )
        .unwrap();
        memory_copy_async(&mut h_keys_out, &d_keys_out, &stream).unwrap();
        stream.synchronize().unwrap();
        h_keys_in.sort();
        if descending {
            h_keys_in.reverse()
        };
        assert!(h_keys_in
            .into_iter()
            .zip(h_keys_out.into_iter())
            .all(|(x, y)| x == y));
    }

    trait DeriveValue<K, V> {
        fn derive_value(key: &K) -> V;
    }

    impl DeriveValue<u32, u32> for (u32, u32) {
        fn derive_value(key: &u32) -> u32 {
            (*key).not()
        }
    }

    impl DeriveValue<u32, u64> for (u32, u64) {
        fn derive_value(key: &u32) -> u64 {
            let x = (*key).not() as u64;
            x | x << 32
        }
    }

    impl DeriveValue<u64, u32> for (u64, u32) {
        fn derive_value(key: &u64) -> u32 {
            let x = (*key).not();
            ((x >> 32) ^ (x & 0xffffffff)) as u32
        }
    }

    impl DeriveValue<u64, u64> for (u64, u64) {
        fn derive_value(key: &u64) -> u64 {
            (*key).not()
        }
    }

    fn test_sort_pairs<K, V>(descending: bool)
    where
        (K, V): SortPairs<K, V> + DeriveValue<K, V>,
        K: Default + Clone + Ord + Eq,
        V: Default + Clone + Eq,
        Standard: Distribution<K> + Distribution<V>,
    {
        const NUM_ITEMS: usize = 1 << 16;
        let end_bit = size_of::<K>() as i32 * 8;
        let temp_storage_bytes =
            get_sort_pairs_temp_storage_bytes::<K, V>(descending, NUM_ITEMS as u32, 0, end_bit)
                .unwrap();
        let mut d_temp_storage = DeviceAllocation::alloc(temp_storage_bytes).unwrap();
        let h_keys_in = (0..NUM_ITEMS).map(|_| thread_rng().gen()).collect_vec();
        let h_values_in = h_keys_in.iter().map(<(K, V)>::derive_value).collect_vec();
        let mut h_keys_out = vec![K::default(); NUM_ITEMS];
        let mut h_values_out = vec![V::default(); NUM_ITEMS];
        let mut d_keys_in = DeviceAllocation::alloc(NUM_ITEMS).unwrap();
        let mut d_values_in = DeviceAllocation::alloc(NUM_ITEMS).unwrap();
        let mut d_keys_out = DeviceAllocation::alloc(NUM_ITEMS).unwrap();
        let mut d_values_out = DeviceAllocation::alloc(NUM_ITEMS).unwrap();
        let stream = CudaStream::default();
        memory_copy_async(&mut d_keys_in, &h_keys_in, &stream).unwrap();
        memory_copy_async(&mut d_values_in, &h_values_in, &stream).unwrap();
        sort_pairs(
            descending,
            &mut d_temp_storage,
            &d_keys_in,
            &mut d_keys_out,
            &d_values_in,
            &mut d_values_out,
            0,
            end_bit,
            &stream,
        )
        .unwrap();
        memory_copy_async(&mut h_keys_out, &d_keys_out, &stream).unwrap();
        memory_copy_async(&mut h_values_out, &d_values_out, &stream).unwrap();
        stream.synchronize().unwrap();
        let mut pairs_in = h_keys_in.into_iter().zip(h_values_in).collect_vec();
        let pairs_out = h_keys_out.into_iter().zip(h_values_out).collect_vec();
        pairs_in.sort_by_key(|(k, _)| k.clone());
        if descending {
            pairs_in.reverse()
        };
        assert!(pairs_in
            .into_iter()
            .zip(pairs_out.into_iter())
            .all(|(x, y)| x == y));
    }

    #[test]
    fn sort_keys_ascending_u32() {
        test_sort_keys::<u32>(false);
    }

    #[test]
    fn sort_keys_descending_u32() {
        test_sort_keys::<u32>(true);
    }

    #[test]
    fn sort_keys_ascending_u64() {
        test_sort_keys::<u64>(false);
    }

    #[test]
    fn sort_keys_descending_u64() {
        test_sort_keys::<u64>(true);
    }

    #[test]
    fn sort_pairs_ascending_u32_by_u32() {
        test_sort_pairs::<u32, u32>(false);
    }

    #[test]
    fn sort_pairs_descending_u32_by_u32() {
        test_sort_pairs::<u32, u32>(true);
    }

    #[test]
    fn sort_pairs_ascending_u32_by_u64() {
        test_sort_pairs::<u64, u32>(false);
    }

    #[test]
    fn sort_pairs_descending_u32_by_u64() {
        test_sort_pairs::<u64, u32>(true);
    }

    #[test]
    fn sort_pairs_ascending_u64_by_u32() {
        test_sort_pairs::<u32, u64>(false);
    }

    #[test]
    fn sort_pairs_descending_u64_by_u32() {
        test_sort_pairs::<u32, u64>(true);
    }

    #[test]
    fn sort_pairs_ascending_u64_by_u64() {
        test_sort_pairs::<u64, u64>(false);
    }

    #[test]
    fn sort_pairs_descending_u64_by_u64() {
        test_sort_pairs::<u64, u64>(true);
    }
}
