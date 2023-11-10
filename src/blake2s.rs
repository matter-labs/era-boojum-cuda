use cudart::device::{device_get_attribute, get_device};
use cudart::execution::{KernelFourArgs, KernelLaunch};
use cudart::memory::memory_set_async;
use cudart::occupancy::max_active_blocks_per_multiprocessor;
use cudart::result::CudaResult;
use cudart::slice::{DeviceSlice, DeviceVariable};
use cudart::stream::CudaStream;
use cudart_sys::CudaDeviceAttr;

use crate::utils::WARP_SIZE;

extern "C" {
    fn blake2s_pow_kernel(seed: *const u8, bits_count: u32, max_nonce: u64, result: *mut u64);
}

pub fn blake2s_pow(
    seed: &DeviceSlice<u8>,
    bits_count: u32,
    max_nonce: u64,
    result: &mut DeviceVariable<u64>,
    stream: &CudaStream,
) -> CudaResult<()> {
    const BLOCK_SIZE: u32 = WARP_SIZE * 4;
    assert_eq!(seed.len(), 32);
    unsafe {
        memory_set_async(result.transmute_mut(), 0xff, stream)?;
    } // set result to 0 (false
    let seed = seed.as_ptr();
    let result = result.as_mut_ptr();
    let args = (&seed, &bits_count, &max_nonce, &result);
    let device_id = get_device()?;
    let mpc = device_get_attribute(CudaDeviceAttr::MultiProcessorCount, device_id).unwrap();
    let max_blocks = max_active_blocks_per_multiprocessor(
        blake2s_pow_kernel as KernelFourArgs<_, _, _, _>,
        BLOCK_SIZE as i32,
        0,
    )?;
    let num_blocks = (mpc * max_blocks) as u32;
    unsafe {
        KernelFourArgs::launch(
            blake2s_pow_kernel,
            num_blocks.into(),
            BLOCK_SIZE.into(),
            args,
            0,
            stream,
        )
    }
}

#[cfg(test)]
mod tests {
    use blake2::{Blake2s256, Digest};
    use cudart::memory::{memory_copy_async, DeviceAllocation};
    use cudart::stream::CudaStream;

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
