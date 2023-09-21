use std::slice;

use crate::slice::{CudaSlice, CudaSliceMut};

#[derive(Debug)]
pub(crate) struct AllocationData<T> {
    pub ptr: *mut T,
    pub len: usize,
}

unsafe impl<T> Send for AllocationData<T> where Vec<T>: Send {}

unsafe impl<T> Sync for AllocationData<T> where Vec<T>: Sync {}

impl<T> CudaSlice<T> for AllocationData<T> {
    unsafe fn as_slice(&self) -> &[T] {
        slice::from_raw_parts(self.ptr, self.len)
    }
}

impl<T> CudaSliceMut<T> for AllocationData<T> {
    unsafe fn as_mut_slice(&mut self) -> &mut [T] {
        slice::from_raw_parts_mut(self.ptr, self.len)
    }
}
