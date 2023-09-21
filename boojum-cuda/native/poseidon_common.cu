#include "goldilocks.cuh"

namespace poseidon {

using namespace goldilocks;
using namespace memory;

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

EXTERN __global__ void gather_merkle_paths_kernel(const unsigned *indexes, const unsigned indexes_count,
                                                  const matrix_getter<base_field, ld_modifier::cs> values, matrix_setter<base_field, st_modifier::cs> results) {
  const unsigned idx = threadIdx.x + blockIdx.x * blockDim.x;
  if (idx >= indexes_count)
    return;
  const unsigned col = threadIdx.y;
  const unsigned layers_count = gridDim.y;
  const unsigned layer_from_leaves = blockIdx.y;
  const unsigned leaf_index = indexes[idx];
  const unsigned layer_offset = (1 << (layers_count + 1)) - (1 << (layers_count + 1 - layer_from_leaves));
  const unsigned hash_index = (leaf_index >> layer_from_leaves) ^ 1;
  const unsigned src_row = layer_offset + hash_index;
  const unsigned dst_row = layer_from_leaves * indexes_count + idx;
  results.set(dst_row, col, values.get(src_row, col));
}

} // namespace poseidon