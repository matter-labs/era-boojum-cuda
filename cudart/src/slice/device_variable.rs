use std::fmt::{Debug, Formatter};
use std::ops::{Deref, DerefMut};
use std::ptr;

use crate::slice::{CudaSlice, CudaSliceMut, CudaVariable, CudaVariableMut, DeviceSlice};

#[repr(transparent)]
pub struct DeviceVariable<T>([T]);

impl<T> DeviceVariable<T> {
    /// # Safety
    /// make sure data_address is pointing to memory located on the device
    pub unsafe fn from_raw_parts<'a>(data_address: *const T) -> &'a Self {
        &*ptr::from_raw_parts(data_address as *const (), 1)
    }

    /// # Safety
    /// make sure data_address is pointing to memory located on the device
    pub unsafe fn from_raw_parts_mut<'a>(data_address: *mut T) -> &'a mut Self {
        &mut *ptr::from_raw_parts_mut(data_address as *mut (), 1)
    }

    /// # Safety
    /// make sure the ref is pointing to memory located on the device
    pub unsafe fn from_ref(s: &T) -> &Self {
        Self::from_raw_parts(s.as_ptr())
    }

    /// # Safety
    /// make sure the ref is pointing to memory located on the device
    pub unsafe fn from_mut(s: &mut T) -> &mut Self {
        Self::from_raw_parts_mut(s.as_mut_ptr())
    }

    /// # Safety
    /// make sure the slice is pointing to memory located on the device
    pub unsafe fn from_slice(slice: &[T]) -> &Self {
        assert_eq!(slice.len(), 1);
        Self::from_raw_parts(slice.as_ptr())
    }

    /// # Safety
    /// make sure the slice is pointing to memory located on the device
    pub unsafe fn from_mut_slice(slice: &mut [T]) -> &mut Self {
        assert_eq!(slice.len(), 1);
        Self::from_raw_parts_mut(slice.as_mut_ptr())
    }

    pub fn as_ptr(&self) -> *const T {
        self.0.as_ptr()
    }

    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.0.as_mut_ptr()
    }
}

impl<T> Deref for DeviceVariable<T> {
    type Target = DeviceSlice<T>;

    fn deref(&self) -> &Self::Target {
        unsafe { DeviceSlice::from_slice(&self.0) }
    }
}

impl<T> DerefMut for DeviceVariable<T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        unsafe { DeviceSlice::from_mut_slice(&mut self.0) }
    }
}

impl<T> Debug for DeviceVariable<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let slice = &self.0;
        f.debug_struct("DeviceVariable")
            .field("ptr", &slice.as_ptr())
            .finish()
    }
}

impl<T> CudaSlice<T> for DeviceVariable<T> {
    unsafe fn as_slice(&self) -> &[T] {
        &self.0
    }
}

impl<T> CudaSliceMut<T> for DeviceVariable<T> {
    unsafe fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.0
    }
}

impl<T> CudaVariable<T> for DeviceVariable<T> {
    unsafe fn as_ref(&self) -> &T {
        &self.0[0]
    }
}

impl<T> CudaVariableMut<T> for DeviceVariable<T> {
    unsafe fn as_mut(&mut self) -> &mut T {
        &mut self.0[0]
    }
}
