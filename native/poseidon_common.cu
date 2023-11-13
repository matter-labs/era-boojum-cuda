#include "goldilocks.cuh"
#include "poseidon_constants.cuh"

namespace poseidon {

using namespace goldilocks;
using namespace memory;
using namespace poseidon_common;

EXTERN __global__ void gather_rows_kernel(const unsigned *indexes, const unsigned indexes_count, const matrix_getter<base_field, ld_modifier::cs> values,
                                          matrix_setter<base_field, st_modifier::cs> results) {
  const unsigned idx = threadIdx.y + blockIdx.x * blockDim.y;
  if (idx >= indexes_count)
    return;
  const unsigned index = indexes[idx];
  const unsigned src_row = index * blockDim.x + threadIdx.x;
  const unsigned dst_row = idx * blockDim.x + threadIdx.x;
  const unsigned col = blockIdx.y;
  results.set(dst_row, col, values.get(src_row, col));
}

EXTERN __global__ void gather_merkle_paths_kernel(const unsigned *indexes, const unsigned indexes_count, const base_field *values,
                                                  const unsigned log_leaves_count, base_field *results) {
  const unsigned idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= indexes_count)
    return;
  const unsigned col = blockIdx.y;
  const unsigned layer_index = blockIdx.z;
  const unsigned layer_offset = (CAPACITY << (log_leaves_count + 1)) - (CAPACITY << (log_leaves_count + 1 - layer_index));
  const unsigned col_offset = col << (log_leaves_count - layer_index);
  const unsigned leaf_index = indexes[idx];
  const unsigned hash_index = (leaf_index >> layer_index) ^ 1;
  const unsigned src_index = layer_offset + col_offset + hash_index;
  const unsigned dst_index = layer_index * indexes_count * CAPACITY + indexes_count * col + idx;
  results[dst_index] = values[src_index];
}

} // namespace poseidon