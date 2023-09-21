// Stream Ordered Memory Allocator
// https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY__POOLS.html

use core::ffi::c_void;
use std::alloc::Layout;
use std::mem;
use std::mem::MaybeUninit;
use std::ops::{Deref, DerefMut};

use cudart_sys::*;

use crate::result::{CudaResult, CudaResultWrap};
use crate::slice::{AllocationData, CudaSlice, CudaSliceMut, DeviceSlice};
use crate::stream::CudaStream;

#[repr(i32)]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum CudaMemPoolAttributeI32 {
    ReuseFollowEventDependencies = CudaMemPoolAttribute::ReuseFollowEventDependencies as i32,
    ReuseAllowOpportunistic = CudaMemPoolAttribute::ReuseAllowOpportunistic as i32,
    ReuseAllowInternalDependencies = CudaMemPoolAttribute::ReuseAllowInternalDependencies as i32,
}

impl From<CudaMemPoolAttributeI32> for i32 {
    fn from(attribute: CudaMemPoolAttributeI32) -> Self {
        attribute as i32
    }
}

#[repr(i32)]
#[derive(Debug, Copy, Clone, Hash, PartialEq, Eq)]
pub enum CudaMemPoolAttributeU64 {
    AttrReleaseThreshold = CudaMemPoolAttribute::AttrReleaseThreshold as i32,
    AttrReservedMemCurrent = CudaMemPoolAttribute::AttrReservedMemCurrent as i32,
    AttrReservedMemHigh = CudaMemPoolAttribute::AttrReservedMemHigh as i32,
    AttrUsedMemCurrent = CudaMemPoolAttribute::AttrUsedMemCurrent as i32,
    AttrUsedMemHigh = CudaMemPoolAttribute::AttrUsedMemHigh as i32,
}

impl From<CudaMemPoolAttributeU64> for i32 {
    fn from(attribute: CudaMemPoolAttributeU64) -> Self {
        attribute as i32
    }
}

#[repr(transparent)]
#[derive(Debug)]
pub struct CudaMemPool {
    handle: cudaMemPool_t,
}

impl CudaMemPool {
    pub(crate) fn from_handle(handle: cudaMemPool_t) -> Self {
        Self { handle }
    }

    pub fn get_access(&self, location: CudaMemLocation) -> CudaResult<CudaMemAccessFlags> {
        let mut result = MaybeUninit::<CudaMemAccessFlags>::uninit();
        unsafe {
            cudaMemPoolGetAccess(
                result.as_mut_ptr(),
                self.handle,
                &location as *const CudaMemLocation as *mut CudaMemLocation,
            )
            .wrap_maybe_uninit(result)
        }
    }

    pub fn get_attribute_value<T: Into<i32>, U>(&self, attribute: T) -> CudaResult<U> {
        let mut value = MaybeUninit::<U>::uninit();
        unsafe {
            cudaMemPoolGetAttribute(
                self.handle,
                mem::transmute(attribute.into()),
                value.as_mut_ptr() as *mut c_void,
            )
            .wrap_maybe_uninit(value)
        }
    }

    pub fn set_access(&self, descriptors: &[CudaMemAccessDesc]) -> CudaResult<()> {
        unsafe { cudaMemPoolSetAccess(self.handle, descriptors.as_ptr(), descriptors.len()).wrap() }
    }

    pub fn set_attribute_value<T: Into<i32>, U>(&self, attribute: T, value: U) -> CudaResult<()> {
        unsafe {
            cudaMemPoolSetAttribute(
                self.handle,
                mem::transmute(attribute.into()),
                &value as *const _ as *mut c_void,
            )
            .wrap()
        }
    }

    pub fn trim_to(&self, min_bytes_to_keep: usize) -> CudaResult<()> {
        unsafe { cudaMemPoolTrimTo(self.handle, min_bytes_to_keep).wrap() }
    }
}

impl From<&CudaMemPool> for cudaMemPool_t {
    fn from(pool: &CudaMemPool) -> Self {
        pool.handle
    }
}

pub trait AttributeHandler<T> {
    type Value;
    fn get_attribute(&self, attribute: T) -> CudaResult<Self::Value>;
    fn set_attribute(&self, attribute: T, value: Self::Value) -> CudaResult<()>;
}

impl AttributeHandler<CudaMemPoolAttributeI32> for CudaMemPool {
    type Value = i32;

    fn get_attribute(&self, attribute: CudaMemPoolAttributeI32) -> CudaResult<Self::Value> {
        self.get_attribute_value(attribute)
    }

    fn set_attribute(
        &self,
        attribute: CudaMemPoolAttributeI32,
        value: Self::Value,
    ) -> CudaResult<()> {
        self.set_attribute_value(attribute, value)
    }
}

impl AttributeHandler<CudaMemPoolAttributeU64> for CudaMemPool {
    type Value = u64;

    fn get_attribute(&self, attribute: CudaMemPoolAttributeU64) -> CudaResult<Self::Value> {
        self.get_attribute_value(attribute)
    }

    fn set_attribute(
        &self,
        attribute: CudaMemPoolAttributeU64,
        value: Self::Value,
    ) -> CudaResult<()> {
        self.set_attribute_value(attribute, value)
    }
}

#[repr(transparent)]
#[derive(Debug)]
pub struct CudaOwnedMemPool {
    pool: CudaMemPool,
}

impl CudaOwnedMemPool {
    fn from_handle(handle: cudaMemPool_t) -> Self {
        Self {
            pool: CudaMemPool::from_handle(handle),
        }
    }

    pub fn create(properties: &CudaMemPoolProperties) -> CudaResult<Self> {
        let mut handle = MaybeUninit::<cudaMemPool_t>::uninit();
        unsafe {
            cudaMemPoolCreate(handle.as_mut_ptr(), properties)
                .wrap_maybe_uninit(handle)
                .map(Self::from_handle)
        }
    }

    pub fn create_for_device(device_id: i32) -> CudaResult<Self> {
        let props = CudaMemPoolProperties {
            allocType: CudaMemAllocationType::Pinned,
            handleTypes: CudaMemAllocationHandleType::None,
            location: CudaMemLocation {
                type_: CudaMemLocationType::Device,
                id: device_id,
            },
            ..Default::default()
        };
        Self::create(&props)
    }

    pub fn destroy(self) -> CudaResult<()> {
        let pool = self.pool.handle;
        mem::forget(self);
        unsafe { cudaMemPoolDestroy(pool).wrap() }
    }
}

impl Drop for CudaOwnedMemPool {
    fn drop(&mut self) {
        unsafe { cudaMemPoolDestroy(self.pool.handle).eprint_error_and_backtrace() };
    }
}

impl Deref for CudaOwnedMemPool {
    type Target = CudaMemPool;

    fn deref(&self) -> &Self::Target {
        &self.pool
    }
}

#[derive(Debug)]
pub struct DevicePoolAllocation<'a, T> {
    data: AllocationData<T>,
    stream: &'a CudaStream,
}

impl<'a, T> DevicePoolAllocation<'a, T> {
    pub fn alloc_async(length: usize, stream: &'a CudaStream) -> CudaResult<Self> {
        let layout = Layout::array::<T>(length).unwrap();
        let mut dev_ptr = MaybeUninit::<*mut c_void>::uninit();
        unsafe {
            cudaMallocAsync(dev_ptr.as_mut_ptr(), layout.size(), stream.into())
                .wrap_maybe_uninit(dev_ptr)
                .map(|ptr| Self {
                    data: AllocationData {
                        ptr: ptr as *mut T,
                        len: length,
                    },
                    stream,
                })
        }
    }

    pub fn alloc_from_pool_async(
        length: usize,
        pool: &CudaMemPool,
        stream: &'a CudaStream,
    ) -> CudaResult<Self> {
        let layout = Layout::array::<T>(length).unwrap();
        let mut dev_ptr = MaybeUninit::<*mut c_void>::uninit();
        unsafe {
            cudaMallocFromPoolAsync(
                dev_ptr.as_mut_ptr(),
                layout.size(),
                pool.handle,
                stream.into(),
            )
            .wrap_maybe_uninit(dev_ptr)
            .map(|ptr| Self {
                data: AllocationData {
                    ptr: ptr as *mut T,
                    len: length,
                },
                stream,
            })
        }
    }

    pub fn free_async(self, stream: &CudaStream) -> CudaResult<()> {
        unsafe {
            let ptr = self.as_c_void_ptr() as *mut c_void;
            mem::forget(self);
            cudaFreeAsync(ptr, stream.into()).wrap()
        }
    }

    pub fn swap_stream(self, stream: &CudaStream) -> DevicePoolAllocation<T> {
        let data = AllocationData {
            ptr: self.data.ptr,
            len: self.data.len,
        };
        mem::forget(self);
        DevicePoolAllocation { data, stream }
    }

    /// # Safety
    ///
    /// The caller must ensure that the inputs are valid.
    pub unsafe fn from_raw_parts(ptr: *mut T, len: usize, stream: &'a CudaStream) -> Self {
        Self {
            data: AllocationData { ptr, len },
            stream,
        }
    }

    pub fn into_raw_parts(self) -> (*mut T, usize, &'a CudaStream) {
        let result = (self.data.ptr, self.data.len, self.stream);
        mem::forget(self);
        result
    }
}

impl<'a, T> Drop for DevicePoolAllocation<'a, T> {
    fn drop(&mut self) {
        unsafe {
            cudaFreeAsync(self.as_mut_c_void_ptr(), self.stream.into()).eprint_error_and_backtrace()
        };
    }
}

impl<'a, T> Deref for DevicePoolAllocation<'a, T> {
    type Target = DeviceSlice<T>;

    fn deref(&self) -> &Self::Target {
        Self::Target::from_allocation_data(&self.data)
    }
}

impl<'a, T> DerefMut for DevicePoolAllocation<'a, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        Self::Target::from_mut_allocation_data(&mut self.data)
    }
}

impl<'a, T> CudaSlice<T> for DevicePoolAllocation<'a, T> {
    unsafe fn as_slice(&self) -> &[T] {
        self.data.as_slice()
    }
}

impl<'a, T> CudaSliceMut<T> for DevicePoolAllocation<'a, T> {
    unsafe fn as_mut_slice(&mut self) -> &mut [T] {
        self.data.as_mut_slice()
    }
}

#[cfg(test)]
mod tests {
    use serial_test::serial;

    use crate::memory::memory_copy_async;

    use super::*;

    const LENGTH: usize = 1024;

    #[test]
    #[serial]
    fn mem_pool_for_device_is_ok() {
        let result = CudaOwnedMemPool::create_for_device(0);
        assert!(result.is_ok());
    }

    #[test]
    #[serial]
    fn mem_pool_destroy_is_ok() {
        let pool = CudaOwnedMemPool::create_for_device(0).unwrap();
        let result = pool.destroy();
        assert!(result.is_ok());
    }

    #[test]
    #[serial]
    fn device_pool_allocation_alloc_async_is_ok() {
        let stream = CudaStream::create().unwrap();
        let result = DevicePoolAllocation::<u32>::alloc_async(LENGTH, &stream);
        assert!(result.is_ok());
    }

    #[test]
    #[serial]
    fn device_pool_allocation_alloc_from_pool_async_is_ok() {
        let pool = CudaOwnedMemPool::create_for_device(0).unwrap();
        let stream = CudaStream::create().unwrap();
        let result = DevicePoolAllocation::<u32>::alloc_from_pool_async(LENGTH, &pool, &stream);
        assert!(result.is_ok());
    }

    #[test]
    #[serial]
    fn device_pool_allocation_free_is_ok() {
        let stream = CudaStream::create().unwrap();
        let allocation = DevicePoolAllocation::<u32>::alloc_async(LENGTH, &stream).unwrap();
        let result = allocation.free_async(&stream);
        stream.synchronize().unwrap();
        assert_eq!(result, Ok(()));
    }

    #[test]
    #[serial]
    fn device_pool_allocation_alloc_async_len_eq_length() {
        let stream = CudaStream::create().unwrap();
        let allocation = DevicePoolAllocation::<u32>::alloc_async(LENGTH, &stream).unwrap();
        assert_eq!(allocation.len(), LENGTH);
    }

    #[test]
    #[serial]
    fn device_pool_allocation_alloc_async_is_empty_is_false() {
        let stream = CudaStream::create().unwrap();
        let allocation = DevicePoolAllocation::<u32>::alloc_async(LENGTH, &stream).unwrap();
        assert!(!allocation.is_empty());
    }

    #[test]
    #[serial]
    fn device_pool_allocation_deref_len_eq_length() {
        let stream = CudaStream::create().unwrap();
        let allocation = DevicePoolAllocation::<u32>::alloc_async(LENGTH, &stream).unwrap();
        let slice = allocation.deref();
        assert_eq!(slice.len(), LENGTH);
    }

    #[test]
    #[serial]
    fn device_pool_allocation_deref_mut_len_eq_length() {
        let stream = CudaStream::create().unwrap();
        let mut allocation = DevicePoolAllocation::<u32>::alloc_async(LENGTH, &stream).unwrap();
        let slice = allocation.deref_mut();
        assert_eq!(slice.len(), LENGTH);
    }

    #[test]
    #[serial]
    fn device_pool_allocation_slice_index_len_eq_length() {
        let stream = CudaStream::create().unwrap();
        let allocation = DevicePoolAllocation::<u32>::alloc_async(LENGTH, &stream).unwrap();
        let slice = &allocation[..];
        assert_eq!(slice.len(), LENGTH);
    }

    #[test]
    #[serial]
    fn device_pool_allocation_mut_slice_index_mut_len_eq_length() {
        let stream = CudaStream::create().unwrap();
        let mut allocation = DevicePoolAllocation::<u32>::alloc_async(LENGTH, &stream).unwrap();
        let slice = &mut allocation[..];
        assert_eq!(slice.len(), LENGTH);
    }

    #[test]
    #[serial]
    fn device_pool_allocation_drop_frees_memory() {
        let pool = CudaOwnedMemPool::create_for_device(0).unwrap();
        let stream = CudaStream::create().unwrap();
        let allocation =
            DevicePoolAllocation::<u32>::alloc_from_pool_async(LENGTH, &pool, &stream).unwrap();
        drop(allocation);
        let used = pool
            .get_attribute(CudaMemPoolAttributeU64::AttrUsedMemCurrent)
            .unwrap() as usize;
        assert_eq!(used, 0);
    }

    #[test]
    #[serial]
    fn device_pool_allocation_swap_stream() {
        let pool = CudaOwnedMemPool::create_for_device(0).unwrap();
        let stream1 = CudaStream::create().unwrap();
        let allocation =
            DevicePoolAllocation::<u32>::alloc_from_pool_async(LENGTH, &pool, &stream1).unwrap();
        let stream2 = CudaStream::create().unwrap();
        let allocation = allocation.swap_stream(&stream2);
        drop(stream1);
        drop(allocation);
        let used = pool
            .get_attribute(CudaMemPoolAttributeU64::AttrUsedMemCurrent)
            .unwrap() as usize;
        assert_eq!(used, 0);
    }

    #[test]
    #[serial]
    fn memory_copy_device_pool_allocation_to_device_pool_allocation() {
        let values1 = [42u32; LENGTH];
        let mut values2 = [0u32; LENGTH];
        let stream = CudaStream::create().unwrap();
        let mut a1 = DevicePoolAllocation::<u32>::alloc_async(LENGTH, &stream).unwrap();
        let mut a2 = DevicePoolAllocation::<u32>::alloc_async(LENGTH, &stream).unwrap();
        memory_copy_async(&mut a1, &values1, &stream).unwrap();
        memory_copy_async(&mut a2, &a1, &stream).unwrap();
        memory_copy_async(&mut values2, &a2, &stream).unwrap();
        stream.synchronize().unwrap();
        assert!(values2.iter().all(|&x| x == 42u32));
    }

    #[test]
    #[serial]
    fn get_attribute_i32() {
        let pool = CudaOwnedMemPool::create_for_device(0).unwrap();
        let result = pool.get_attribute(CudaMemPoolAttributeI32::ReuseAllowOpportunistic);
        assert_eq!(result, Ok(1));
    }

    #[test]
    #[serial]
    fn get_attribute_u64() {
        let pool = CudaOwnedMemPool::create_for_device(0).unwrap();
        let result = pool.get_attribute(CudaMemPoolAttributeU64::AttrReleaseThreshold);
        assert_eq!(result, Ok(0));
    }

    #[test]
    #[serial]
    fn set_attribute_i32() {
        let pool = CudaOwnedMemPool::create_for_device(0).unwrap();
        let attribute = CudaMemPoolAttributeI32::ReuseAllowOpportunistic;
        let result = pool.set_attribute(attribute, 0);
        assert_eq!(result, Ok(()));
        assert_eq!(pool.get_attribute(attribute), Ok(0));
    }

    #[test]
    #[serial]
    fn set_attribute_u64() {
        let pool = CudaOwnedMemPool::create_for_device(0).unwrap();
        let attribute = CudaMemPoolAttributeU64::AttrReleaseThreshold;
        let result = pool.set_attribute(attribute, u64::MAX);
        assert_eq!(result, Ok(()));
        assert_eq!(pool.get_attribute(attribute), Ok(u64::MAX));
    }

    #[test]
    #[serial]
    fn trim_to_works_correctly() {
        let pool = CudaOwnedMemPool::create_for_device(0).unwrap();
        pool.set_attribute(CudaMemPoolAttributeU64::AttrReleaseThreshold, u64::MAX)
            .unwrap();
        let stream = CudaStream::create().unwrap();
        let allocation =
            DevicePoolAllocation::<u32>::alloc_from_pool_async(LENGTH, &pool, &stream).unwrap();
        let size = mem::size_of::<u32>() * LENGTH;
        let used = pool
            .get_attribute(CudaMemPoolAttributeU64::AttrUsedMemCurrent)
            .unwrap() as usize;
        assert_eq!(used, size);
        allocation.free_async(&stream).unwrap();
        stream.synchronize().unwrap();
        let used = pool
            .get_attribute(CudaMemPoolAttributeU64::AttrUsedMemCurrent)
            .unwrap() as usize;
        assert_eq!(used, 0);
        let reserved = pool
            .get_attribute(CudaMemPoolAttributeU64::AttrReservedMemCurrent)
            .unwrap() as usize;
        assert!(reserved >= size);
        pool.trim_to(0).unwrap();
        let reserved = pool
            .get_attribute(CudaMemPoolAttributeU64::AttrReservedMemCurrent)
            .unwrap() as usize;
        assert_eq!(reserved, 0);
    }
}
