use std::fmt::{Debug, Formatter};
use std::mem::size_of;
use std::ptr;
use std::ptr::{null, null_mut};

use crate::slice::iter::{Chunks, ChunksMut};
use crate::slice::AllocationData;
use crate::slice::{CudaSlice, CudaSliceMut};

#[repr(transparent)]
pub struct DeviceSlice<T>([T]);

impl<T> DeviceSlice<T> {
    /// # Safety
    /// make sure data_address is pointing to memory located on the device
    pub unsafe fn from_raw_parts<'a>(data_address: *const T, len: usize) -> &'a Self {
        &*ptr::from_raw_parts(data_address as *const (), len)
    }

    /// # Safety
    /// make sure data_address is pointing to memory located on the device
    pub unsafe fn from_raw_parts_mut<'a>(data_address: *mut T, len: usize) -> &'a mut Self {
        &mut *ptr::from_raw_parts_mut(data_address as *mut (), len)
    }

    /// # Safety
    /// make sure the slice is pointing to memory located on the device
    pub unsafe fn from_slice(slice: &[T]) -> &Self {
        Self::from_raw_parts(slice.as_ptr(), slice.len())
    }

    /// # Safety
    /// make sure the slice is pointing to memory located on the device
    pub unsafe fn from_mut_slice(slice: &mut [T]) -> &mut Self {
        Self::from_raw_parts_mut(slice.as_mut_ptr(), slice.len())
    }

    pub(crate) fn from_allocation_data(data: &AllocationData<T>) -> &Self {
        unsafe { Self::from_slice(data.as_slice()) }
    }

    pub(crate) fn from_mut_allocation_data(data: &mut AllocationData<T>) -> &mut Self {
        unsafe { Self::from_mut_slice(data.as_mut_slice()) }
    }

    pub fn empty<'a>() -> &'a Self {
        unsafe { Self::from_raw_parts(null(), 0) }
    }

    pub fn empty_mut<'a>() -> &'a mut Self {
        unsafe { Self::from_raw_parts_mut(null_mut(), 0) }
    }

    pub fn as_ptr(&self) -> *const T {
        self.0.as_ptr()
    }

    pub fn as_mut_ptr(&mut self) -> *mut T {
        self.0.as_mut_ptr()
    }

    pub fn len(&self) -> usize {
        self.0.len()
    }

    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// # Safety
    /// only use if you know that treating the memory as [U] is valid
    pub unsafe fn transmute<U>(&self) -> &DeviceSlice<U> {
        let size_of_t = size_of::<T>();
        let size_of_u = size_of::<U>();
        assert_eq!((self.len() * size_of_t) % size_of_u, 0);
        let len = self.len() * size_of_t / size_of_u;
        DeviceSlice::from_raw_parts(self.as_ptr() as *const U, len)
    }

    /// # Safety
    /// only use if you know that treating the memory as [U] is valid
    pub unsafe fn transmute_mut<U>(&mut self) -> &mut DeviceSlice<U> {
        let size_of_t = size_of::<T>();
        let size_of_u = size_of::<U>();
        assert_eq!((self.len() * size_of_t) % size_of_u, 0);
        let len = self.len() * size_of_t / size_of_u;
        DeviceSlice::from_raw_parts_mut(self.as_mut_ptr() as *mut U, len)
    }

    pub fn split_at(&self, mid: usize) -> (&Self, &Self) {
        unsafe {
            let (left, right) = self.as_slice().split_at(mid);
            (Self::from_slice(left), Self::from_slice(right))
        }
    }

    pub fn split_at_mut(&mut self, mid: usize) -> (&mut Self, &mut Self) {
        unsafe {
            let (left, right) = self.as_mut_slice().split_at_mut(mid);
            (Self::from_mut_slice(left), Self::from_mut_slice(right))
        }
    }

    pub fn chunks(&self, chunk_size: usize) -> Chunks<T> {
        assert_ne!(chunk_size, 0, "chunk size must be non-zero");
        Chunks::new(self, chunk_size)
    }

    pub fn chunks_mut(&mut self, chunk_size: usize) -> ChunksMut<T> {
        assert_ne!(chunk_size, 0, "chunk size must be non-zero");
        ChunksMut::new(self, chunk_size)
    }
}

impl<T> Debug for DeviceSlice<T> {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let slice = &self.0;
        f.debug_struct("DeviceSlice")
            .field("ptr", &slice.as_ptr())
            .field("len", &slice.len())
            .finish()
    }
}

impl<T> CudaSlice<T> for DeviceSlice<T> {
    unsafe fn as_slice(&self) -> &[T] {
        &self.0
    }
}

impl<T> CudaSliceMut<T> for DeviceSlice<T> {
    unsafe fn as_mut_slice(&mut self) -> &mut [T] {
        &mut self.0
    }
}
