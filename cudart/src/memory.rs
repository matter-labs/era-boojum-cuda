// memory management
// https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__MEMORY.html

use core::ffi::c_void;
use std::alloc::Layout;
use std::mem::{self, MaybeUninit};
use std::ops::{Deref, DerefMut};

use bitflags::bitflags;

use cudart_sys::*;

use crate::result::{CudaResult, CudaResultWrap};
use crate::slice::{AllocationData, CudaSlice, CudaSliceMut, DeviceSlice};
use crate::stream::CudaStream;

#[repr(transparent)]
#[derive(Debug)]
pub struct DeviceAllocation<T>(AllocationData<T>);

impl<T> DeviceAllocation<T> {
    pub fn alloc(length: usize) -> CudaResult<Self> {
        let layout = Layout::array::<T>(length).unwrap();
        let mut dev_ptr = MaybeUninit::<*mut c_void>::uninit();
        unsafe {
            cudaMalloc(dev_ptr.as_mut_ptr(), layout.size())
                .wrap_maybe_uninit(dev_ptr)
                .map(|ptr| {
                    Self(AllocationData {
                        ptr: ptr as *mut T,
                        len: length,
                    })
                })
        }
    }

    pub fn free(self) -> CudaResult<()> {
        unsafe {
            let ptr = self.0.ptr as *mut c_void;
            mem::forget(self);
            cudaFree(ptr).wrap()
        }
    }

    /// # Safety
    ///
    /// The caller must ensure that the inputs are valid.
    pub unsafe fn from_raw_parts(ptr: *mut T, len: usize) -> Self {
        Self(AllocationData { ptr, len })
    }

    pub fn into_raw_parts(self) -> (*mut T, usize) {
        let result = (self.0.ptr, self.0.len);
        mem::forget(self);
        result
    }
}

impl<T> Drop for DeviceAllocation<T> {
    fn drop(&mut self) {
        unsafe { cudaFree(self.as_mut_c_void_ptr()).eprint_error_and_backtrace() };
    }
}

impl<T> Deref for DeviceAllocation<T> {
    type Target = DeviceSlice<T>;

    fn deref(&self) -> &Self::Target {
        Self::Target::from_allocation_data(&self.0)
    }
}

impl<T> DerefMut for DeviceAllocation<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        Self::Target::from_mut_allocation_data(&mut self.0)
    }
}

impl<T> AsRef<DeviceSlice<T>> for DeviceAllocation<T> {
    fn as_ref(&self) -> &DeviceSlice<T> {
        self.deref()
    }
}

impl<T> AsMut<DeviceSlice<T>> for DeviceAllocation<T> {
    fn as_mut(&mut self) -> &mut DeviceSlice<T> {
        self.deref_mut()
    }
}

impl<T> CudaSlice<T> for DeviceAllocation<T> {
    unsafe fn as_slice(&self) -> &[T] {
        self.0.as_slice()
    }
}

impl<T> CudaSliceMut<T> for DeviceAllocation<T> {
    unsafe fn as_mut_slice(&mut self) -> &mut [T] {
        self.0.as_mut_slice()
    }
}

bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct CudaHostAllocFlags: u32 {
        const DEFAULT = cudart_sys::cudaHostAllocDefault;
        const PORTABLE = cudart_sys::cudaHostAllocPortable;
        const MAPPED = cudart_sys::cudaHostAllocMapped;
        const WRITE_COMBINED = cudart_sys::cudaHostAllocWriteCombined;
    }
}

impl Default for CudaHostAllocFlags {
    fn default() -> Self {
        Self::DEFAULT
    }
}

#[repr(transparent)]
#[derive(Debug)]
pub struct HostAllocation<T>(AllocationData<T>);

impl<T> HostAllocation<T> {
    pub fn alloc(length: usize, flags: CudaHostAllocFlags) -> CudaResult<Self> {
        let layout = Layout::array::<T>(length).unwrap();
        let mut ptr = MaybeUninit::<*mut c_void>::uninit();
        unsafe {
            cudaHostAlloc(ptr.as_mut_ptr(), layout.size(), flags.bits())
                .wrap_maybe_uninit(ptr)
                .map(|ptr| {
                    Self(AllocationData {
                        ptr: ptr as *mut T,
                        len: length,
                    })
                })
        }
    }

    pub fn free(self) -> CudaResult<()> {
        unsafe {
            let ptr = self.0.ptr as *mut c_void;
            mem::forget(self);
            cudaFreeHost(ptr).wrap()
        }
    }

    /// # Safety
    ///
    /// The caller must ensure that the inputs are valid.
    pub unsafe fn from_raw_parts(ptr: *mut T, len: usize) -> Self {
        Self(AllocationData { ptr, len })
    }

    pub fn into_raw_parts(self) -> (*mut T, usize) {
        let result = (self.0.ptr, self.0.len);
        mem::forget(self);
        result
    }
}

impl<T> Drop for HostAllocation<T> {
    fn drop(&mut self) {
        unsafe { cudaFreeHost(self.0.ptr as *mut c_void).eprint_error_and_backtrace() };
    }
}

impl<T> Deref for HostAllocation<T> {
    type Target = [T];
    fn deref(&self) -> &Self::Target {
        unsafe { self.0.as_slice() }
    }
}

impl<T> DerefMut for HostAllocation<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { self.0.as_mut_slice() }
    }
}

impl<T> AsRef<[T]> for HostAllocation<T> {
    fn as_ref(&self) -> &[T] {
        self.deref()
    }
}

impl<T> AsMut<[T]> for HostAllocation<T> {
    fn as_mut(&mut self) -> &mut [T] {
        self.deref_mut()
    }
}

bitflags! {
    #[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
    pub struct CudaHostRegisterFlags: u32 {
        const DEFAULT = cudart_sys::cudaHostRegisterDefault;
        const PORTABLE = cudart_sys::cudaHostRegisterPortable;
        const MAPPED = cudart_sys::cudaHostRegisterMapped;
        const IO_MEMORY = cudart_sys::cudaHostRegisterIoMemory;
        const READ_ONLY = cudart_sys::cudaHostRegisterReadOnly;
    }
}

impl Default for CudaHostRegisterFlags {
    fn default() -> Self {
        Self::DEFAULT
    }
}

#[repr(transparent)]
#[derive(Debug)]
pub struct HostRegistration<'a, T>(&'a [T]);

impl<'a, T> HostRegistration<'a, T> {
    pub fn register(slice: &'a [T], flags: CudaHostRegisterFlags) -> CudaResult<Self> {
        let length = slice.len();
        let layout = Layout::array::<T>(length).unwrap();
        unsafe {
            cudaHostRegister(
                slice.as_c_void_ptr() as *mut c_void,
                layout.size(),
                flags.bits(),
            )
            .wrap_value(Self(slice))
        }
    }

    pub fn unregister(self) -> CudaResult<()> {
        unsafe { cudaHostUnregister(self.0.as_c_void_ptr() as *mut c_void).wrap() }
    }
}

impl<T> Drop for HostRegistration<'_, T> {
    fn drop(&mut self) {
        unsafe {
            cudaHostUnregister(self.0.as_c_void_ptr() as *mut c_void).eprint_error_and_backtrace()
        };
    }
}

impl<T> Deref for HostRegistration<'_, T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.0
    }
}

impl<T> AsRef<[T]> for HostRegistration<'_, T> {
    fn as_ref(&self) -> &[T] {
        self.0
    }
}

#[repr(transparent)]
#[derive(Debug)]
pub struct HostRegistrationMut<'a, T>(&'a mut [T]);

impl<'a, T> HostRegistrationMut<'a, T> {
    pub fn register(slice: &'a mut [T], flags: CudaHostRegisterFlags) -> CudaResult<Self> {
        let length = slice.len();
        let layout = Layout::array::<T>(length).unwrap();
        unsafe {
            cudaHostRegister(slice.as_mut_c_void_ptr(), layout.size(), flags.bits())
                .wrap_value(Self(slice))
        }
    }

    pub fn unregister(self) -> CudaResult<()> {
        unsafe { cudaHostUnregister(self.0.as_mut_c_void_ptr()).wrap() }
    }
}

impl<T> Drop for HostRegistrationMut<'_, T> {
    fn drop(&mut self) {
        unsafe { cudaHostUnregister(self.0.as_mut_c_void_ptr()).eprint_error_and_backtrace() };
    }
}

impl<T> Deref for HostRegistrationMut<'_, T> {
    type Target = [T];

    fn deref(&self) -> &Self::Target {
        self.0
    }
}

impl<T> DerefMut for HostRegistrationMut<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.0
    }
}

impl<T> AsRef<[T]> for HostRegistrationMut<'_, T> {
    fn as_ref(&self) -> &[T] {
        self.0
    }
}

impl<T> AsMut<[T]> for HostRegistrationMut<'_, T> {
    fn as_mut(&mut self) -> &mut [T] {
        self.0
    }
}

pub fn memory_copy<T>(
    dst: &mut (impl CudaSliceMut<T> + ?Sized),
    src: &(impl CudaSlice<T> + ?Sized),
) -> CudaResult<()> {
    memory_copy_with_kind(dst, src, CudaMemoryCopyKind::Default)
}

pub fn memory_copy_with_kind<T>(
    dst: &mut (impl CudaSliceMut<T> + ?Sized),
    src: &(impl CudaSlice<T> + ?Sized),
    kind: CudaMemoryCopyKind,
) -> CudaResult<()> {
    unsafe {
        assert_eq!(
            dst.len(),
            src.len(),
            "dst length and src length must be equal"
        );
        let layout = Layout::array::<T>(dst.len()).unwrap();
        cudaMemcpy(
            dst.as_mut_c_void_ptr(),
            src.as_c_void_ptr(),
            layout.size(),
            kind,
        )
        .wrap()
    }
}

pub fn memory_copy_async<T>(
    dst: &mut (impl CudaSliceMut<T> + ?Sized),
    src: &(impl CudaSlice<T> + ?Sized),
    stream: &CudaStream,
) -> CudaResult<()> {
    memory_copy_with_kind_async(dst, src, CudaMemoryCopyKind::Default, stream)
}

pub fn memory_copy_with_kind_async<T>(
    dst: &mut (impl CudaSliceMut<T> + ?Sized),
    src: &(impl CudaSlice<T> + ?Sized),
    kind: CudaMemoryCopyKind,
    stream: &CudaStream,
) -> CudaResult<()> {
    unsafe {
        assert_eq!(
            dst.len(),
            src.len(),
            "dst length and src length must be equal"
        );
        let layout = Layout::array::<T>(dst.len()).unwrap();
        cudaMemcpyAsync(
            dst.as_mut_c_void_ptr(),
            src.as_c_void_ptr(),
            layout.size(),
            kind,
            stream.into(),
        )
        .wrap()
    }
}

pub fn memory_set(dst: &mut (impl CudaSliceMut<u8> + ?Sized), value: u8) -> CudaResult<()> {
    unsafe {
        let layout = Layout::array::<u8>(dst.len()).unwrap();
        cudaMemset(dst.as_mut_c_void_ptr(), value as i32, layout.size()).wrap()
    }
}

pub fn memory_set_async(
    dst: &mut (impl CudaSliceMut<u8> + ?Sized),
    value: u8,
    stream: &CudaStream,
) -> CudaResult<()> {
    unsafe {
        let layout = Layout::array::<u8>(dst.len()).unwrap();
        cudaMemsetAsync(
            dst.as_mut_c_void_ptr(),
            value as i32,
            layout.size(),
            stream.into(),
        )
        .wrap()
    }
}

pub fn memory_get_info() -> CudaResult<(usize, usize)> {
    let mut free = MaybeUninit::<usize>::uninit();
    let mut total = MaybeUninit::<usize>::uninit();
    unsafe {
        let error = cudaMemGetInfo(free.as_mut_ptr(), total.as_mut_ptr());
        if error == CudaError::Success {
            Ok((free.assume_init(), total.assume_init()))
        } else {
            Err(error)
        }
    }
}

#[derive(Copy, Clone, Default, Debug, PartialEq, Eq)]
pub struct HostAllocator {
    flags: CudaHostAllocFlags,
}

impl HostAllocator {
    pub fn new(flags: CudaHostAllocFlags) -> Self {
        Self { flags }
    }
}

#[cfg(test)]
mod tests {
    use serial_test::serial;

    use super::*;

    const LENGTH: usize = 1024;

    #[test]
    #[serial]
    fn device_allocation_alloc_is_ok() {
        let result = DeviceAllocation::<u32>::alloc(LENGTH);
        assert!(result.is_ok());
    }

    #[test]
    #[serial]
    fn device_allocation_free_is_ok() {
        let allocation = DeviceAllocation::<u32>::alloc(LENGTH).unwrap();
        let result = allocation.free();
        assert_eq!(result, Ok(()));
    }

    #[test]
    #[serial]
    fn device_allocation_alloc_len_eq_length() {
        let allocation = DeviceAllocation::<u32>::alloc(LENGTH).unwrap();
        assert_eq!(allocation.len(), LENGTH);
    }

    #[test]
    #[serial]
    fn device_allocation_alloc_is_empty_is_false() {
        let allocation = DeviceAllocation::<u32>::alloc(LENGTH).unwrap();
        assert!(!allocation.is_empty());
    }

    #[test]
    #[serial]
    fn device_allocation_deref_len_eq_length() {
        let allocation = DeviceAllocation::<u32>::alloc(LENGTH).unwrap();
        let slice = allocation.deref();
        assert_eq!(slice.len(), LENGTH);
    }

    #[test]
    #[serial]
    fn device_allocation_deref_mut_len_eq_length() {
        let mut allocation = DeviceAllocation::<u32>::alloc(LENGTH).unwrap();
        let slice = allocation.deref_mut();
        assert_eq!(slice.len(), LENGTH);
    }

    #[test]
    #[serial]
    fn device_allocation_slice_index_len_eq_length() {
        let allocation = DeviceAllocation::<u32>::alloc(LENGTH).unwrap();
        let slice = &allocation[..];
        assert_eq!(slice.len(), LENGTH);
    }

    #[test]
    #[serial]
    fn device_allocation_mut_slice_index_mut_len_eq_length() {
        let mut allocation = DeviceAllocation::<u32>::alloc(LENGTH).unwrap();
        let slice = &mut allocation[..];
        assert_eq!(slice.len(), LENGTH);
    }

    #[test]
    #[serial]
    fn host_allocation_alloc_is_ok() {
        let result = HostAllocation::<u32>::alloc(LENGTH, CudaHostAllocFlags::DEFAULT);
        assert!(result.is_ok());
    }

    #[test]
    #[serial]
    fn host_allocation_free_is_ok() {
        let allocation = HostAllocation::<u32>::alloc(LENGTH, CudaHostAllocFlags::DEFAULT).unwrap();
        let result = allocation.free();
        assert_eq!(result, Ok(()));
    }

    #[test]
    #[serial]
    fn host_allocation_alloc_len_eq_length() {
        let allocation = HostAllocation::<u32>::alloc(LENGTH, CudaHostAllocFlags::DEFAULT).unwrap();
        assert_eq!(allocation.len(), LENGTH);
    }

    #[test]
    #[serial]
    fn host_allocation_alloc_is_empty_is_false() {
        let allocation = HostAllocation::<u32>::alloc(LENGTH, CudaHostAllocFlags::DEFAULT).unwrap();
        assert!(!allocation.is_empty());
    }

    #[test]
    #[serial]
    fn host_allocation_deref_len_eq_length() {
        let allocation = HostAllocation::<u32>::alloc(LENGTH, CudaHostAllocFlags::DEFAULT).unwrap();
        let slice = allocation.deref();
        assert_eq!(slice.len(), LENGTH);
    }

    #[test]
    #[serial]
    fn host_allocation_deref_mut_len_eq_length() {
        let mut allocation =
            HostAllocation::<u32>::alloc(LENGTH, CudaHostAllocFlags::DEFAULT).unwrap();
        let slice = allocation.deref_mut();
        assert_eq!(slice.len(), LENGTH);
    }

    #[test]
    #[serial]
    fn host_allocation_index_len_eq_length() {
        let allocation = HostAllocation::<u32>::alloc(LENGTH, CudaHostAllocFlags::DEFAULT).unwrap();
        let slice = &allocation[..];
        assert_eq!(slice.len(), LENGTH);
    }

    #[test]
    #[serial]
    fn host_allocation_index_mut_len_eq_length() {
        let mut allocation =
            HostAllocation::<u32>::alloc(LENGTH, CudaHostAllocFlags::DEFAULT).unwrap();
        let slice = &mut allocation[..];
        assert_eq!(slice.len(), LENGTH);
    }

    #[test]
    #[serial]
    fn host_allocation_deref_ptrs_are_equal() {
        let allocation = HostAllocation::<u32>::alloc(LENGTH, CudaHostAllocFlags::DEFAULT).unwrap();
        let ptr = allocation.deref().as_ptr();
        assert_eq!(allocation.as_ptr(), ptr);
    }

    #[test]
    #[serial]
    fn host_allocation_deref_mut_ptrs_are_equal() {
        let mut allocation =
            HostAllocation::<u32>::alloc(LENGTH, CudaHostAllocFlags::DEFAULT).unwrap();
        let ptr = allocation.deref_mut().as_mut_ptr();
        assert_eq!(allocation.as_mut_ptr(), ptr);
    }

    #[test]
    #[serial]
    fn host_registration_register_is_ok() {
        let values = [0u32; LENGTH];
        let result = HostRegistration::<u32>::register(&values, CudaHostRegisterFlags::DEFAULT);
        assert!(result.is_ok());
    }

    #[test]
    #[serial]
    fn host_registration_register_empty_error_invalid_value() {
        let values = [0u32; 0];
        let result = HostRegistration::<u32>::register(&values, CudaHostRegisterFlags::DEFAULT);
        assert_eq!(result.err(), Some(CudaError::ErrorInvalidValue));
    }

    #[test]
    #[serial]
    fn host_registration_unregister_is_ok() {
        let values = [0u32; LENGTH];
        let registration =
            HostRegistration::<u32>::register(&values, CudaHostRegisterFlags::DEFAULT).unwrap();
        let result = registration.unregister();
        assert_eq!(result, Ok(()));
    }

    #[test]
    #[serial]
    fn host_registration_register_len_eq_length() {
        let values = [0u32; LENGTH];
        let registration =
            HostRegistration::<u32>::register(&values, CudaHostRegisterFlags::DEFAULT).unwrap();
        assert_eq!(registration.len(), LENGTH);
    }

    #[test]
    #[serial]
    fn host_registration_register_is_empty_is_false() {
        let values = [0u32; LENGTH];
        let registration =
            HostRegistration::<u32>::register(&values, CudaHostRegisterFlags::DEFAULT).unwrap();
        assert!(!registration.is_empty());
    }

    #[test]
    #[serial]
    fn host_registration_deref_len_eq_length() {
        let values = [0u32; LENGTH];
        let registration =
            HostRegistration::<u32>::register(&values, CudaHostRegisterFlags::DEFAULT).unwrap();
        let slice = registration.deref();
        assert_eq!(slice.len(), LENGTH);
    }

    #[test]
    #[serial]
    fn host_registration_mut_deref_mut_len_eq_length() {
        let mut values = [0u32; LENGTH];
        let mut registration =
            HostRegistrationMut::<u32>::register(&mut values, CudaHostRegisterFlags::DEFAULT)
                .unwrap();
        let slice = registration.deref_mut();
        assert_eq!(slice.len(), LENGTH);
    }

    #[test]
    #[serial]
    fn host_registration_index_len_eq_length() {
        let values = [0u32; LENGTH];
        let registration =
            HostRegistration::<u32>::register(&values, CudaHostRegisterFlags::DEFAULT).unwrap();
        let slice = &registration[..];
        assert_eq!(slice.len(), LENGTH);
    }

    #[test]
    #[serial]
    fn host_registration_mut_index_mut_len_eq_length() {
        let mut values = [0u32; LENGTH];
        let mut registration =
            HostRegistrationMut::<u32>::register(&mut values, CudaHostRegisterFlags::DEFAULT)
                .unwrap();
        let slice = &mut registration[..];
        assert_eq!(slice.len(), LENGTH);
    }

    #[test]
    #[serial]
    fn host_registration_deref_ptrs_are_equal() {
        let values = [0u32; LENGTH];
        let registration =
            HostRegistration::<u32>::register(&values, CudaHostRegisterFlags::DEFAULT).unwrap();
        let ptr = registration.deref().as_ptr();
        assert_eq!(registration.as_ptr(), ptr);
    }

    #[test]
    #[serial]
    fn host_registration_mut_deref_mut_ptrs_are_equal() {
        let mut values = [0u32; LENGTH];
        let mut registration =
            HostRegistrationMut::<u32>::register(&mut values, CudaHostRegisterFlags::DEFAULT)
                .unwrap();
        let ptr = registration.deref_mut().as_mut_ptr();
        assert_eq!(registration.as_mut_ptr(), ptr);
    }

    #[test]
    #[serial]
    fn memory_copy_device_slice_to_device_slice() {
        let values1 = [42u32; LENGTH];
        let mut values2 = [0u32; LENGTH];
        let mut a1 = DeviceAllocation::<u32>::alloc(LENGTH).unwrap();
        let mut a2 = DeviceAllocation::<u32>::alloc(LENGTH).unwrap();
        let a1_slice = a1.deref_mut();
        let a2_slice = a2.deref_mut();
        memory_copy(a1_slice, &values1).unwrap();
        memory_copy(a2_slice, a1_slice).unwrap();
        memory_copy(&mut values2, a2_slice).unwrap();
        assert!(values2.iter().all(|&x| x == 42u32));
    }

    #[test]
    #[serial]
    fn memory_copy_device_allocation_to_device_allocation() {
        let values1 = [42u32; LENGTH];
        let mut values2 = [0u32; LENGTH];
        let mut a1 = DeviceAllocation::<u32>::alloc(LENGTH).unwrap();
        let mut a2 = DeviceAllocation::<u32>::alloc(LENGTH).unwrap();
        memory_copy(&mut a1, &values1).unwrap();
        memory_copy(&mut a2, &a1).unwrap();
        memory_copy(&mut values2, &a2).unwrap();
        assert!(values2.iter().all(|&x| x == 42u32));
    }

    #[test]
    #[serial]
    fn memory_copy_host_allocation_to_host_allocation() {
        let mut a1 = HostAllocation::<u32>::alloc(LENGTH, CudaHostAllocFlags::DEFAULT).unwrap();
        let mut a2 = HostAllocation::<u32>::alloc(LENGTH, CudaHostAllocFlags::DEFAULT).unwrap();
        a2.iter_mut().for_each(|x| {
            *x = 42u32;
        });
        memory_copy(&mut a1, &a2).unwrap();
        assert!(a1.iter().all(|&x| x == 42u32));
    }

    #[test]
    #[serial]
    fn memory_copy_host_registration_to_host_registration_mut() {
        let mut values1 = [0u32; LENGTH];
        let values2 = [42u32; LENGTH];
        let mut r1 =
            HostRegistrationMut::register(&mut values1, CudaHostRegisterFlags::DEFAULT).unwrap();
        let r2 = HostRegistration::register(&values2, CudaHostRegisterFlags::DEFAULT).unwrap();
        memory_copy(&mut r1, &r2).unwrap();
        assert!(r1.iter().all(|&x| x == 42u32));
    }

    #[test]
    #[serial]
    fn memory_copy_slice_to_slice() {
        let mut values1 = [0u32; LENGTH];
        let values2 = [42u32; LENGTH];
        memory_copy(&mut values1, &values2).unwrap();
        assert!(values1.iter().all(|&x| x == 42u32));
    }

    #[test]
    #[serial]
    fn memory_copy_async_device_allocation_to_device_allocation() {
        let stream = CudaStream::create().unwrap();
        let values1 = [42u32; LENGTH];
        let mut values2 = [0u32; LENGTH];
        let mut a1 = DeviceAllocation::<u32>::alloc(LENGTH).unwrap();
        let mut a2 = DeviceAllocation::<u32>::alloc(LENGTH).unwrap();
        memory_copy_async(&mut a1, &values1, &stream).unwrap();
        memory_copy_async(&mut a2, &a1, &stream).unwrap();
        memory_copy_async(&mut values2, &a2, &stream).unwrap();
        stream.synchronize().unwrap();
        assert!(values2.iter().all(|&x| x == 42u32));
    }

    #[test]
    #[serial]
    fn memory_copy_async_host_allocation_to_host_allocation() {
        let stream = CudaStream::create().unwrap();
        let mut a1 = HostAllocation::<u32>::alloc(LENGTH, CudaHostAllocFlags::DEFAULT).unwrap();
        let mut a2 = HostAllocation::<u32>::alloc(LENGTH, CudaHostAllocFlags::DEFAULT).unwrap();
        a2.iter_mut().for_each(|x| {
            *x = 42u32;
        });
        memory_copy_async(&mut a1, &a2, &stream).unwrap();
        stream.synchronize().unwrap();
        assert!(a1.iter().all(|&x| x == 42u32));
    }

    #[test]
    #[serial]
    fn memory_copy_async_host_registration_to_host_registration_mut() {
        let stream = CudaStream::create().unwrap();
        let mut values1 = [0u32; LENGTH];
        let values2 = [42u32; LENGTH];
        let mut r1 =
            HostRegistrationMut::register(&mut values1, CudaHostRegisterFlags::DEFAULT).unwrap();
        let r2 = HostRegistration::register(&values2, CudaHostRegisterFlags::DEFAULT).unwrap();
        memory_copy_async(&mut r1, &r2, &stream).unwrap();
        stream.synchronize().unwrap();
        assert!(r1.iter().all(|&x| x == 42u32));
    }

    #[test]
    #[serial]
    fn memory_copy_async_slice_to_slice() {
        let stream = CudaStream::create().unwrap();
        let mut values1 = [0u32; LENGTH];
        let values2 = [42u32; LENGTH];
        memory_copy_async(&mut values1, &values2, &stream).unwrap();
        stream.synchronize().unwrap();
        assert!(values1.iter().all(|&x| x == 42u32));
    }

    #[test]
    #[serial]
    fn memory_set_is_correct() {
        let mut h_values =
            HostAllocation::<u8>::alloc(LENGTH, CudaHostAllocFlags::DEFAULT).unwrap();
        let mut d_values = DeviceAllocation::<u8>::alloc(LENGTH).unwrap();
        memory_set(&mut d_values, 42u8).unwrap();
        memory_copy(&mut h_values, &d_values).unwrap();
        assert!(h_values.iter().all(|&x| x == 42u8));
    }

    #[test]
    #[serial]
    fn memory_set_async_is_correct() {
        let stream = CudaStream::create().unwrap();
        let mut h_values =
            HostAllocation::<u8>::alloc(LENGTH, CudaHostAllocFlags::DEFAULT).unwrap();
        let mut d_values = DeviceAllocation::<u8>::alloc(LENGTH).unwrap();
        memory_set_async(&mut d_values, 42u8, &stream).unwrap();
        memory_copy_async(&mut h_values, &d_values, &stream).unwrap();
        stream.synchronize().unwrap();
        assert!(h_values.iter().all(|&x| x == 42u8));
    }

    #[test]
    #[serial]
    fn memory_get_info_is_correct() {
        let result = memory_get_info();
        assert!(result.is_ok());
        let (free, total) = result.unwrap();
        assert!(total > 0);
        assert!(free <= total);
    }
}
