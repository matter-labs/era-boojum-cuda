use std::ffi::c_void;

pub trait CudaVariable<T> {
    /// # Safety
    /// do not dereference if the memory is located on the device
    unsafe fn as_ref(&self) -> &T;

    fn as_ptr(&self) -> *const T {
        unsafe { self.as_ref() }
    }

    fn as_c_void_ptr(&self) -> *const c_void {
        self.as_ptr() as *const c_void
    }
}

pub trait CudaVariableMut<T>: CudaVariable<T> {
    /// # Safety
    /// do not dereference if the memory is located on the device
    unsafe fn as_mut(&mut self) -> &mut T;

    fn as_mut_ptr(&mut self) -> *mut T {
        unsafe { self.as_mut() }
    }

    fn as_mut_c_void_ptr(&mut self) -> *mut c_void {
        self.as_mut_ptr() as *mut c_void
    }
}

impl<T> CudaVariable<T> for T {
    unsafe fn as_ref(&self) -> &T {
        self
    }
}

impl<T> CudaVariableMut<T> for T {
    unsafe fn as_mut(&mut self) -> &mut T {
        self
    }
}
