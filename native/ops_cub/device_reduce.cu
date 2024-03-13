#include "common.cuh"

namespace device_reduce {

struct offset_iterator {
#if CUB_VERSION >= 200300
  using iterator_category = cuda::std::random_access_iterator_tag;
  using value_type = int;
  using difference_type = int;
  using pointer = int *;
  using reference = int &;
#endif

  const int offset;
  const int stride;

  DEVICE_FORCEINLINE int operator[](const int idx) const { return offset + idx * stride; }
};

using namespace goldilocks;
using namespace memory;

EXTERN cudaError_t reduce_add_bf(void *d_temp_storage, size_t &temp_storage_bytes, const base_field *d_in, base_field *d_out, int num_items,
                                 cudaStream_t stream) {
  return DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, base_field_add(), base_field{}, stream);
}

EXTERN cudaError_t reduce_add_ef(void *d_temp_storage, size_t &temp_storage_bytes, const extension_field *d_in, extension_field *d_out, int num_items,
                                 cudaStream_t stream) {
  return DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, extension_field_add(), extension_field{}, stream);
}

EXTERN cudaError_t segmented_reduce_add_bf(void *d_temp_storage, size_t &temp_storage_bytes, const matrix_accessor<base_field> d_in, base_field *d_out,
                                           int num_segments, int num_items, cudaStream_t stream) {
  int stride = int(d_in.stride);
  return DeviceSegmentedReduce::Reduce(d_temp_storage, temp_storage_bytes, d_in.ptr, d_out, num_segments, offset_iterator{0, stride},
                                       offset_iterator{num_items, stride}, base_field_add(), base_field{0}, stream);
}

EXTERN cudaError_t segmented_reduce_add_ef(void *d_temp_storage, size_t &temp_storage_bytes, const matrix_accessor<extension_field> d_in,
                                           extension_field *d_out, int num_segments, int num_items, cudaStream_t stream) {
  int stride = int(d_in.stride);
  return DeviceSegmentedReduce::Reduce(d_temp_storage, temp_storage_bytes, d_in.ptr, d_out, num_segments, offset_iterator{0, stride},
                                       offset_iterator{num_items, stride}, extension_field_add(), extension_field{}, stream);
}

EXTERN cudaError_t reduce_mul_bf(void *d_temp_storage, size_t &temp_storage_bytes, const base_field *d_in, base_field *d_out, int num_items,
                                 cudaStream_t stream) {
  return DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, base_field_mul(), base_field{1}, stream);
}

EXTERN cudaError_t reduce_mul_ef(void *d_temp_storage, size_t &temp_storage_bytes, const extension_field *d_in, extension_field *d_out, int num_items,
                                 cudaStream_t stream) {
  return DeviceReduce::Reduce(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, extension_field_mul(), extension_field{1}, stream);
}

EXTERN cudaError_t segmented_reduce_mul_bf(void *d_temp_storage, size_t &temp_storage_bytes, const matrix_accessor<base_field> d_in, base_field *d_out,
                                           int num_segments, int num_items, cudaStream_t stream) {
  int stride = int(d_in.stride);
  return DeviceSegmentedReduce::Reduce(d_temp_storage, temp_storage_bytes, d_in.ptr, d_out, num_segments, offset_iterator{0, stride},
                                       offset_iterator{num_items, stride}, base_field_mul(), base_field{1}, stream);
}

EXTERN cudaError_t segmented_reduce_mul_ef(void *d_temp_storage, size_t &temp_storage_bytes, const matrix_accessor<extension_field> d_in,
                                           extension_field *d_out, int num_segments, int num_items, cudaStream_t stream) {
  int stride = int(d_in.stride);
  return DeviceSegmentedReduce::Reduce(d_temp_storage, temp_storage_bytes, d_in.ptr, d_out, num_segments, offset_iterator{0, stride},
                                       offset_iterator{num_items, stride}, extension_field_mul(), extension_field{1}, stream);
}

} // namespace device_reduce