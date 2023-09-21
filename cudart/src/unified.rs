// Unified Addressing
// https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__UNIFIED.html

use core::ffi::c_void;
use std::mem::MaybeUninit;

use cudart_sys::*;

use crate::result::{CudaResult, CudaResultWrap};
use crate::slice::CudaSlice;

pub fn pointer_get_attributes<T>(
    slice: &(impl CudaSlice<T> + ?Sized),
) -> CudaResult<CudaPointerAttributes> {
    let mut attributes = MaybeUninit::<CudaPointerAttributes>::uninit();
    unsafe {
        cudaPointerGetAttributes(
            attributes.as_mut_ptr(),
            slice.as_c_void_ptr() as *mut c_void,
        )
        .wrap_maybe_uninit(attributes)
    }
}

#[cfg(test)]
mod tests {
    use serial_test::serial;

    use crate::memory::*;

    use super::*;

    #[test]
    #[serial]
    fn pointer_is_unregistered() {
        let values = [0u32];
        let attributes = pointer_get_attributes(&values).unwrap();
        assert_eq!(attributes.type_, CudaMemoryType::Unregistered);
    }

    #[test]
    #[serial]
    fn pointer_is_host() {
        let values = [0u32];
        let registration =
            HostRegistration::register(&values, CudaHostRegisterFlags::DEFAULT).unwrap();
        let attributes = pointer_get_attributes(&registration).unwrap();
        assert_eq!(attributes.type_, CudaMemoryType::Host);
    }

    #[test]
    #[serial]
    fn pointer_is_device() {
        let allocation = DeviceAllocation::<u32>::alloc(1).unwrap();
        let attributes = pointer_get_attributes(&allocation).unwrap();
        assert_eq!(attributes.type_, CudaMemoryType::Device);
    }
}
