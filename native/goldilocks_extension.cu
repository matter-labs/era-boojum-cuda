#include "goldilocks_extension.cuh"
#include "memory.cuh"

namespace goldilocks {

using namespace memory;

EXTERN __global__ void vectorized_to_tuples_kernel(ef_double_matrix_getter<ld_modifier::cs> src, matrix_setter<extension_field, st_modifier::cs> dst,
                                                   const unsigned rows, const unsigned cols) {
  const unsigned gid = threadIdx.x + blockIdx.x * blockDim.x;
  const unsigned count = rows * cols;
  if (gid >= count)
    return;
  const unsigned row = gid % rows;
  const unsigned col = gid / rows;
  const extension_field value = src.get(row, col);
  dst.set(row, col, value);
}

EXTERN __global__ void tuples_to_vectorized_kernel(matrix_getter<extension_field, ld_modifier::cs> src, ef_double_matrix_setter<st_modifier::cs> dst,
                                                   const unsigned rows, const unsigned cols) {
  const unsigned gid = threadIdx.x + blockIdx.x * blockDim.x;
  const unsigned count = rows * cols;
  if (gid >= count)
    return;
  const unsigned row = gid % rows;
  const unsigned col = gid / rows;
  const extension_field value = src.get(row, col);
  dst.set(row, col, value);
}

} // namespace goldilocks