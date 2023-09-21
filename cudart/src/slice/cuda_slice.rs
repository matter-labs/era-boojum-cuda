use std::ffi::c_void;

pub trait CudaSlice<T> {
    /// # Safety
    /// do not dereference if the memory is located on the device
    unsafe fn as_slice(&self) -> &[T];

    fn as_ptr(&self) -> *const T {
        unsafe { self.as_slice().as_ptr() }
    }

    fn as_c_void_ptr(&self) -> *const c_void {
        self.as_ptr() as *const c_void
    }

    fn is_empty(&self) -> bool {
        unsafe { self.as_slice().is_empty() }
    }

    fn len(&self) -> usize {
        unsafe { self.as_slice().len() }
    }
}

pub trait CudaSliceMut<T>: CudaSlice<T> {
    /// # Safety
    /// do not dereference if the memory is located on the device
    unsafe fn as_mut_slice(&mut self) -> &mut [T];

    fn as_mut_ptr(&mut self) -> *mut T {
        unsafe { self.as_mut_slice().as_mut_ptr() }
    }

    fn as_mut_c_void_ptr(&mut self) -> *mut c_void {
        self.as_mut_ptr() as *mut c_void
    }
}

impl<T> CudaSlice<T> for [T] {
    unsafe fn as_slice(&self) -> &[T] {
        self
    }
}

impl<T> CudaSliceMut<T> for [T] {
    unsafe fn as_mut_slice(&mut self) -> &mut [T] {
        self
    }
}

impl<T, U> CudaSlice<T> for U
where
    Self: AsRef<[T]>,
{
    unsafe fn as_slice(&self) -> &[T] {
        self.as_ref()
    }
}

impl<T, U> CudaSliceMut<T> for U
where
    Self: AsMut<[T]> + AsRef<[T]>,
{
    unsafe fn as_mut_slice(&mut self) -> &mut [T] {
        self.as_mut()
    }
}
