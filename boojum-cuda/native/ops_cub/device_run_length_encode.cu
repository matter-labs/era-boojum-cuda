#include "common.cuh"

namespace device_run_length_encode {

EXTERN cudaError_t encode_u32(void *d_temp_storage, size_t &temp_storage_bytes, const uint32_t *d_in, uint32_t *d_unique_out, unsigned *d_counts_out,
                              unsigned *d_num_runs_out, int num_items, cudaStream_t stream) {
  return DeviceRunLengthEncode::Encode(d_temp_storage, temp_storage_bytes, d_in, d_unique_out, d_counts_out, d_num_runs_out, num_items, stream);
}

EXTERN cudaError_t encode_u64(void *d_temp_storage, size_t &temp_storage_bytes, const uint64_t *d_in, uint64_t *d_unique_out, unsigned *d_counts_out,
                              unsigned *d_num_runs_out, int num_items, cudaStream_t stream) {
  return DeviceRunLengthEncode::Encode(d_temp_storage, temp_storage_bytes, d_in, d_unique_out, d_counts_out, d_num_runs_out, num_items, stream);
}

} // namespace device_run_length_encode
