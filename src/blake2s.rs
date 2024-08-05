use era_cudart::cuda_kernel;
use era_cudart::device::{device_get_attribute, get_device};
use era_cudart::execution::{CudaLaunchConfig, KernelFunction};
use era_cudart::memory::memory_set_async;
use era_cudart::occupancy::max_active_blocks_per_multiprocessor;
use era_cudart::result::CudaResult;
use era_cudart::slice::{DeviceSlice, DeviceVariable};
use era_cudart::stream::CudaStream;
use era_cudart_sys::CudaDeviceAttr;

use crate::utils::WARP_SIZE;

cuda_kernel!(Blake2SPow, blake2s_pow_kernel(seed: *const u8, bits_count: u32, max_nonce: u64, result: *mut u64));

pub fn blake2s_pow(
    seed: &DeviceSlice<u8>,
    bits_count: u32,
    max_nonce: u64,
    result: &mut DeviceVariable<u64>,
    stream: &CudaStream,
) -> CudaResult<()> {
    assert_eq!(seed.len(), 32);
    unsafe {
        memory_set_async(result.transmute_mut(), 0xff, stream)?;
    }
    const BLOCK_SIZE: u32 = WARP_SIZE * 4;
    let device_id = get_device()?;
    let mpc = device_get_attribute(CudaDeviceAttr::MultiProcessorCount, device_id).unwrap();
    let kernel_function = Blake2SPowFunction::default();
    let max_blocks = max_active_blocks_per_multiprocessor(&kernel_function, BLOCK_SIZE as i32, 0)?;
    let num_blocks = (mpc * max_blocks) as u32;
    let config = CudaLaunchConfig::basic(num_blocks, BLOCK_SIZE, stream);
    let seed = seed.as_ptr();
    let result = result.as_mut_ptr();
    let args = Blake2SPowArguments {
        seed,
        bits_count,
        max_nonce,
        result,
    };
    kernel_function.launch(&config, &args)
}

#[cfg(test)]
mod tests {
    use blake2::{Blake2s256, Digest};
    use era_cudart::memory::{memory_copy_async, DeviceAllocation};
    use era_cudart::stream::CudaStream;

    #[test]
    fn blake2s_pow() {
        const BITS_COUNT: u32 = 24;
        let h_seed = [42u8; 32];
        let mut h_result = [0u64; 1];
        let mut d_seed = DeviceAllocation::alloc(32).unwrap();
        let mut d_result = DeviceAllocation::alloc(1).unwrap();
        let stream = CudaStream::default();
        memory_copy_async(&mut d_seed, &h_seed, &stream).unwrap();
        super::blake2s_pow(&d_seed, BITS_COUNT, u64::MAX, &mut d_result[0], &stream).unwrap();
        memory_copy_async(&mut h_result, &d_result, &stream).unwrap();
        stream.synchronize().unwrap();
        let mut digest = Blake2s256::new();
        digest.update(h_seed);
        digest.update(h_result[0].to_le_bytes());
        let output = digest.finalize();
        let mut le_bytes = [0u8; 8];
        le_bytes.copy_from_slice(&output[..8]);
        assert!(u64::from_le_bytes(le_bytes).trailing_zeros() >= BITS_COUNT);
    }
}
