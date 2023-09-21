pub(crate) use allocation_data::AllocationData;
pub use cuda_slice::CudaSlice;
pub use cuda_slice::CudaSliceMut;
pub use cuda_variable::CudaVariable;
pub use cuda_variable::CudaVariableMut;
pub use device_slice::DeviceSlice;
pub use device_variable::DeviceVariable;

mod allocation_data;
mod cuda_slice;
mod cuda_variable;
mod device_slice;
mod device_variable;
mod index;
mod iter;
