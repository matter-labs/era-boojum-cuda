#include "common.cuh"

namespace device_radix_sort {

EXTERN cudaError_t sort_keys_u32(void *d_temp_storage, size_t &temp_storage_bytes, const uint32_t *d_keys_in, uint32_t *d_keys_out, unsigned num_items,
                                 int begin_bit, int end_bit, cudaStream_t stream) {
  return DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, num_items, begin_bit, end_bit, stream);
}

EXTERN cudaError_t sort_keys_descending_u32(void *d_temp_storage, size_t &temp_storage_bytes, const uint32_t *d_keys_in, uint32_t *d_keys_out,
                                            unsigned num_items, int begin_bit, int end_bit, cudaStream_t stream) {
  return DeviceRadixSort::SortKeysDescending(d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, num_items, begin_bit, end_bit, stream);
}

EXTERN cudaError_t sort_keys_u64(void *d_temp_storage, size_t &temp_storage_bytes, const uint64_t *d_keys_in, uint64_t *d_keys_out, unsigned num_items,
                                 int begin_bit, int end_bit, cudaStream_t stream) {
  return DeviceRadixSort::SortKeys(d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, num_items, begin_bit, end_bit, stream);
}

EXTERN cudaError_t sort_keys_descending_u64(void *d_temp_storage, size_t &temp_storage_bytes, const uint64_t *d_keys_in, uint64_t *d_keys_out,
                                            unsigned num_items, int begin_bit, int end_bit, cudaStream_t stream) {
  return DeviceRadixSort::SortKeysDescending(d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, num_items, begin_bit, end_bit, stream);
}

EXTERN cudaError_t sort_pairs_u32_by_u32(void *d_temp_storage, size_t &temp_storage_bytes, const uint32_t *d_keys_in, uint32_t *d_keys_out,
                                         const uint32_t *d_values_in, uint32_t *d_values_out, unsigned num_items, int begin_bit, int end_bit,
                                         cudaStream_t stream) {
  return DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out, num_items, begin_bit, end_bit,
                                    stream);
}

EXTERN cudaError_t sort_pairs_descending_u32_by_u32(void *d_temp_storage, size_t &temp_storage_bytes, const uint32_t *d_keys_in, uint32_t *d_keys_out,
                                                    const uint32_t *d_values_in, uint32_t *d_values_out, unsigned num_items, int begin_bit, int end_bit,
                                                    cudaStream_t stream) {
  return DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out, num_items, begin_bit,
                                              end_bit, stream);
}

EXTERN cudaError_t sort_pairs_u32_by_u64(void *d_temp_storage, size_t &temp_storage_bytes, const uint64_t *d_keys_in, uint64_t *d_keys_out,
                                         const uint32_t *d_values_in, uint32_t *d_values_out, unsigned num_items, int begin_bit, int end_bit,
                                         cudaStream_t stream) {
  return DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out, num_items, begin_bit, end_bit,
                                    stream);
}

EXTERN cudaError_t sort_pairs_descending_u32_by_u64(void *d_temp_storage, size_t &temp_storage_bytes, const uint64_t *d_keys_in, uint64_t *d_keys_out,
                                                    const uint32_t *d_values_in, uint32_t *d_values_out, unsigned num_items, int begin_bit, int end_bit,
                                                    cudaStream_t stream) {
  return DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out, num_items, begin_bit,
                                              end_bit, stream);
}

EXTERN cudaError_t sort_pairs_u64_by_u32(void *d_temp_storage, size_t &temp_storage_bytes, const uint32_t *d_keys_in, uint32_t *d_keys_out,
                                         const uint64_t *d_values_in, uint64_t *d_values_out, unsigned num_items, int begin_bit, int end_bit,
                                         cudaStream_t stream) {
  return DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out, num_items, begin_bit, end_bit,
                                    stream);
}

EXTERN cudaError_t sort_pairs_descending_u64_by_u32(void *d_temp_storage, size_t &temp_storage_bytes, const uint32_t *d_keys_in, uint32_t *d_keys_out,
                                                    const uint64_t *d_values_in, uint64_t *d_values_out, unsigned num_items, int begin_bit, int end_bit,
                                                    cudaStream_t stream) {
  return DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out, num_items, begin_bit,
                                              end_bit, stream);
}

EXTERN cudaError_t sort_pairs_u64_by_u64(void *d_temp_storage, size_t &temp_storage_bytes, const uint64_t *d_keys_in, uint64_t *d_keys_out,
                                         const uint64_t *d_values_in, uint64_t *d_values_out, unsigned num_items, int begin_bit, int end_bit,
                                         cudaStream_t stream) {
  return DeviceRadixSort::SortPairs(d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out, num_items, begin_bit, end_bit,
                                    stream);
}

EXTERN cudaError_t sort_pairs_descending_u64_by_u64(void *d_temp_storage, size_t &temp_storage_bytes, const uint64_t *d_keys_in, uint64_t *d_keys_out,
                                                    const uint64_t *d_values_in, uint64_t *d_values_out, unsigned num_items, int begin_bit, int end_bit,
                                                    cudaStream_t stream) {
  return DeviceRadixSort::SortPairsDescending(d_temp_storage, temp_storage_bytes, d_keys_in, d_keys_out, d_values_in, d_values_out, num_items, begin_bit,
                                              end_bit, stream);
}

} // namespace device_radix_sort