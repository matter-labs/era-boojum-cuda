#include "common.cuh"

namespace device_scan {

using namespace goldilocks;

EXTERN cudaError_t exclusive_sum_u32(void *d_temp_storage, size_t &temp_storage_bytes, const uint32_t *d_in, uint32_t *d_out, int num_items,
                                     cudaStream_t stream) {
  return DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, stream);
}

EXTERN cudaError_t exclusive_sum_reverse_u32(void *d_temp_storage, size_t &temp_storage_bytes, const uint32_t *d_in, uint32_t *d_out, int num_items,
                                             cudaStream_t stream) {
  auto i_in = std::reverse_iterator(d_in + num_items);
  auto i_out = std::reverse_iterator(d_out + num_items);
  return DeviceScan::ExclusiveSum(d_temp_storage, temp_storage_bytes, i_in, i_out, num_items, stream);
}

EXTERN cudaError_t inclusive_sum_u32(void *d_temp_storage, size_t &temp_storage_bytes, const uint32_t *d_in, uint32_t *d_out, int num_items,
                                     cudaStream_t stream) {
  return DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, d_in, d_out, num_items, stream);
}

EXTERN cudaError_t inclusive_sum_reverse_u32(void *d_temp_storage, size_t &temp_storage_bytes, const uint32_t *d_in, uint32_t *d_out, int num_items,
                                             cudaStream_t stream) {
  auto i_in = std::reverse_iterator(d_in + num_items);
  auto i_out = std::reverse_iterator(d_out + num_items);
  return DeviceScan::InclusiveSum(d_temp_storage, temp_storage_bytes, i_in, i_out, num_items, stream);
}

EXTERN cudaError_t inclusive_scan_add_bf(void *d_temp_storage, size_t &temp_storage_bytes, const base_field *d_in, base_field *d_out, int num_items,
                                         cudaStream_t stream) {
  return DeviceScan::InclusiveScan(d_temp_storage, temp_storage_bytes, d_in, d_out, base_field_add(), num_items, stream);
}

EXTERN cudaError_t inclusive_scan_reverse_add_bf(void *d_temp_storage, size_t &temp_storage_bytes, const base_field *d_in, base_field *d_out, int num_items,
                                                 cudaStream_t stream) {
  auto i_in = std::reverse_iterator(d_in + num_items);
  auto i_out = std::reverse_iterator(d_out + num_items);
  return DeviceScan::InclusiveScan(d_temp_storage, temp_storage_bytes, i_in, i_out, base_field_add(), num_items, stream);
}

EXTERN cudaError_t exclusive_scan_add_bf(void *d_temp_storage, size_t &temp_storage_bytes, const base_field *d_in, base_field *d_out, int num_items,
                                         cudaStream_t stream) {
  return DeviceScan::ExclusiveScan(d_temp_storage, temp_storage_bytes, d_in, d_out, base_field_add(), base_field::zero(), num_items, stream);
}

EXTERN cudaError_t exclusive_scan_reverse_add_bf(void *d_temp_storage, size_t &temp_storage_bytes, const base_field *d_in, base_field *d_out, int num_items,
                                                 cudaStream_t stream) {
  auto i_in = std::reverse_iterator(d_in + num_items);
  auto i_out = std::reverse_iterator(d_out + num_items);
  return DeviceScan::ExclusiveScan(d_temp_storage, temp_storage_bytes, i_in, i_out, base_field_add(), base_field::zero(), num_items, stream);
}

EXTERN cudaError_t inclusive_scan_add_ef(void *d_temp_storage, size_t &temp_storage_bytes, const extension_field *d_in, extension_field *d_out, int num_items,
                                         cudaStream_t stream) {
  return DeviceScan::InclusiveScan(d_temp_storage, temp_storage_bytes, d_in, d_out, extension_field_add(), num_items, stream);
}

EXTERN cudaError_t inclusive_scan_reverse_add_ef(void *d_temp_storage, size_t &temp_storage_bytes, const extension_field *d_in, extension_field *d_out,
                                                 int num_items, cudaStream_t stream) {
  auto i_in = std::reverse_iterator(d_in + num_items);
  auto i_out = std::reverse_iterator(d_out + num_items);
  return DeviceScan::InclusiveScan(d_temp_storage, temp_storage_bytes, i_in, i_out, extension_field_add(), num_items, stream);
}

EXTERN cudaError_t exclusive_scan_add_ef(void *d_temp_storage, size_t &temp_storage_bytes, const extension_field *d_in, extension_field *d_out, int num_items,
                                         cudaStream_t stream) {
  return DeviceScan::ExclusiveScan(d_temp_storage, temp_storage_bytes, d_in, d_out, extension_field_add(), extension_field::zero(), num_items, stream);
}

EXTERN cudaError_t exclusive_scan_reverse_add_ef(void *d_temp_storage, size_t &temp_storage_bytes, const extension_field *d_in, extension_field *d_out,
                                                 int num_items, cudaStream_t stream) {
  auto i_in = std::reverse_iterator(d_in + num_items);
  auto i_out = std::reverse_iterator(d_out + num_items);
  return DeviceScan::ExclusiveScan(d_temp_storage, temp_storage_bytes, i_in, i_out, extension_field_add(), extension_field::zero(), num_items, stream);
}

EXTERN cudaError_t inclusive_scan_mul_bf(void *d_temp_storage, size_t &temp_storage_bytes, const base_field *d_in, base_field *d_out, int num_items,
                                         cudaStream_t stream) {
  return DeviceScan::InclusiveScan(d_temp_storage, temp_storage_bytes, d_in, d_out, base_field_mul(), num_items, stream);
}

EXTERN cudaError_t inclusive_scan_reverse_mul_bf(void *d_temp_storage, size_t &temp_storage_bytes, const base_field *d_in, base_field *d_out, int num_items,
                                                 cudaStream_t stream) {
  auto i_in = std::reverse_iterator(d_in + num_items);
  auto i_out = std::reverse_iterator(d_out + num_items);
  return DeviceScan::InclusiveScan(d_temp_storage, temp_storage_bytes, i_in, i_out, base_field_mul(), num_items, stream);
}

EXTERN cudaError_t exclusive_scan_mul_bf(void *d_temp_storage, size_t &temp_storage_bytes, const base_field *d_in, base_field *d_out, int num_items,
                                         cudaStream_t stream) {
  return DeviceScan::ExclusiveScan(d_temp_storage, temp_storage_bytes, d_in, d_out, base_field_mul(), base_field::one(), num_items, stream);
}

EXTERN cudaError_t exclusive_scan_reverse_mul_bf(void *d_temp_storage, size_t &temp_storage_bytes, const base_field *d_in, base_field *d_out, int num_items,
                                                 cudaStream_t stream) {
  auto i_in = std::reverse_iterator(d_in + num_items);
  auto i_out = std::reverse_iterator(d_out + num_items);
  return DeviceScan::ExclusiveScan(d_temp_storage, temp_storage_bytes, i_in, i_out, base_field_mul(), base_field::one(), num_items, stream);
}

EXTERN cudaError_t inclusive_scan_mul_ef(void *d_temp_storage, size_t &temp_storage_bytes, const extension_field *d_in, extension_field *d_out, int num_items,
                                         cudaStream_t stream) {
  return DeviceScan::InclusiveScan(d_temp_storage, temp_storage_bytes, d_in, d_out, extension_field_mul(), num_items, stream);
}

EXTERN cudaError_t inclusive_scan_reverse_mul_ef(void *d_temp_storage, size_t &temp_storage_bytes, const extension_field *d_in, extension_field *d_out,
                                                 int num_items, cudaStream_t stream) {
  auto i_in = std::reverse_iterator(d_in + num_items);
  auto i_out = std::reverse_iterator(d_out + num_items);
  return DeviceScan::InclusiveScan(d_temp_storage, temp_storage_bytes, i_in, i_out, extension_field_mul(), num_items, stream);
}

EXTERN cudaError_t exclusive_scan_mul_ef(void *d_temp_storage, size_t &temp_storage_bytes, const extension_field *d_in, extension_field *d_out, int num_items,
                                         cudaStream_t stream) {
  return DeviceScan::ExclusiveScan(d_temp_storage, temp_storage_bytes, d_in, d_out, extension_field_mul(), extension_field::one(), num_items, stream);
}

EXTERN cudaError_t exclusive_scan_reverse_mul_ef(void *d_temp_storage, size_t &temp_storage_bytes, const extension_field *d_in, extension_field *d_out,
                                                 int num_items, cudaStream_t stream) {
  auto i_in = std::reverse_iterator(d_in + num_items);
  auto i_out = std::reverse_iterator(d_out + num_items);
  return DeviceScan::ExclusiveScan(d_temp_storage, temp_storage_bytes, i_in, i_out, extension_field_mul(), extension_field::one(), num_items, stream);
}

} // namespace device_scan
