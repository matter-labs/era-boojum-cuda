#include "ops_complex.cuh"

namespace goldilocks {

using namespace memory;

// Helper functions to compute common_factor for precompute_lagrange_coeffs
template <typename COEF_T>
DEVICE_FORCEINLINE void barycentric_precompute_common_factor_impl(const COEF_T *x_ref, COEF_T *common_factor_ref, const base_field coset,
                                                                  const unsigned count) {
  // common_factor = coset * (X^N - coset^N) / (N * coset^N)
  // some math could be done on the CPU, but this is a 1-thread kernel so hopefully negligible
  const auto x = *x_ref;
  const auto cosetN = base_field::pow(coset, count);
  const auto xN = COEF_T::pow(x, count);
  const auto num = COEF_T::mul(COEF_T::sub(xN, cosetN), coset);
  const auto denom = base_field::mul({count, 0}, cosetN);
  const auto common_factor = COEF_T::mul(num, base_field::inv(denom));
  *common_factor_ref = common_factor;
}

EXTERN __global__ void barycentric_precompute_common_factor_at_base_kernel(const base_field *x_ref, base_field *common_factor_ref, const base_field coset,
                                                                           const unsigned count) {
  barycentric_precompute_common_factor_impl(x_ref, common_factor_ref, coset, count);
}

EXTERN __global__ void barycentric_precompute_common_factor_at_ext_kernel(const extension_field *x_ref, extension_field *common_factor_ref,
                                                                          const base_field coset, const unsigned count) {
  barycentric_precompute_common_factor_impl(x_ref, common_factor_ref, coset, count);
}

template <typename T> struct InvBatch {};
template <> struct InvBatch<base_field> {
  static constexpr unsigned INV_BATCH = 10;
};
template <> struct InvBatch<extension_field> {
  static constexpr unsigned INV_BATCH = 6;
};

template <typename COEF_T, typename COEF_SETTER_T>
DEVICE_FORCEINLINE void barycentric_precompute_lagrange_coeffs_impl(const COEF_T *x_ref, const COEF_T *common_factor_ref, const base_field w_inv_step,
                                                                    const base_field coset, COEF_SETTER_T lagrange_coeffs, const unsigned log_count) {
  constexpr unsigned INV_BATCH = InvBatch<COEF_T>::INV_BATCH;

  // per_elem_factor = w^i / (X - coset * w^i)
  //                 = 1 / ((X / w^i) - coset)
  // lagrange_coeff = common_factor * per_elem_factor
  // In per_elem_factor, we can get 1 / w_i "for free" by passing inverse=true to get_power_of_w_device.

  const auto x = *x_ref;
  const auto common_factor = *common_factor_ref;

  const unsigned count = 1 << log_count;
  const auto gid = unsigned(blockIdx.x * blockDim.x + threadIdx.x);
  if (gid >= count)
    return;

  const auto grid_size = unsigned(blockDim.x * gridDim.x);

  COEF_T per_elem_factor_invs[INV_BATCH];
  COEF_T per_elem_factors[INV_BATCH];

  unsigned runtime_batch_size = 0;
  const unsigned shift = OMEGA_LOG_ORDER - log_count;
  auto w_inv = get_power_of_w(gid << shift, true);
#pragma unroll
  for (unsigned i{0}, g{gid}; i < INV_BATCH; i++, g += grid_size)
    if (g < count) {
      per_elem_factor_invs[i] = COEF_T::sub(COEF_T::mul(x, w_inv), coset);
      if (g + grid_size < count)
        w_inv = base_field::mul(w_inv, w_inv_step);
      runtime_batch_size++;
    }

  if (runtime_batch_size < INV_BATCH) {
    batch_inv_registers<COEF_T, INV_BATCH, false>(per_elem_factor_invs, per_elem_factors, runtime_batch_size);
  } else {
    batch_inv_registers<COEF_T, INV_BATCH, true>(per_elem_factor_invs, per_elem_factors, runtime_batch_size);
  }

#pragma unroll
  for (unsigned i{0}, g{gid}; i < INV_BATCH; i++, g += grid_size)
    if (g < count)
      lagrange_coeffs.set(g, COEF_T::mul(per_elem_factors[i], common_factor));
}

EXTERN __global__ void barycentric_precompute_lagrange_coeffs_at_base_kernel(const base_field *x_ref, const base_field *common_factor_ref,
                                                                             const base_field w_inv_step, const base_field coset,
                                                                             vector_setter<base_field, st_modifier::cs> lagrange_coeffs,
                                                                             const unsigned log_count) {
  barycentric_precompute_lagrange_coeffs_impl<base_field>(x_ref, common_factor_ref, w_inv_step, coset, lagrange_coeffs, log_count);
}

EXTERN __global__ void barycentric_precompute_lagrange_coeffs_at_ext_kernel(const extension_field *x_ref, const extension_field *common_factor_ref,
                                                                            const base_field w_inv_step, const base_field coset,
                                                                            ef_double_vector_setter<st_modifier::cs> lagrange_coeffs,
                                                                            const unsigned log_count) {
  barycentric_precompute_lagrange_coeffs_impl<extension_field>(x_ref, common_factor_ref, w_inv_step, coset, lagrange_coeffs, log_count);
}

template <typename T> struct ElemsPerThread {};
template <> struct ElemsPerThread<base_field> {
  static constexpr unsigned ELEMS_PER_THREAD = 12;
};
template <> struct ElemsPerThread<extension_field> {
  static constexpr unsigned ELEMS_PER_THREAD = 6;
};

template <typename COEF_T, typename EVAL_GETTER_T, typename COEF_GETTER_T, typename COEF_SETTER_T>
DEVICE_FORCEINLINE void batch_barycentric_partial_reduce_impl(const EVAL_GETTER_T batch_ys, const COEF_GETTER_T lagrange_coeffs, COEF_SETTER_T partial_sums,
                                                              const unsigned log_count, const unsigned num_polys) {
  constexpr unsigned ELEMS_PER_THREAD = ElemsPerThread<COEF_T>::ELEMS_PER_THREAD;

  const unsigned count = 1 << log_count;
  const auto gidx = unsigned(blockIdx.x * blockDim.x + threadIdx.x);
  const unsigned start_poly = threadIdx.y;
  if ((gidx >= count) || (start_poly >= num_polys))
    return;

  const auto grid_size_x = unsigned(blockDim.x * gridDim.x);

  // Threads with the same threadIdx.x in a block load the same factors,
  // so they should often hit in cache
  COEF_T per_elem_factors[ELEMS_PER_THREAD];
#pragma unroll
  for (unsigned i{0}, row{gidx}; i < ELEMS_PER_THREAD; i++, row += grid_size_x)
    if (row < count)
      per_elem_factors[i] = lagrange_coeffs.get(row);

  for (unsigned col = start_poly; col < num_polys; col += blockDim.y) {
    COEF_T thread_sum = {0};
#pragma unroll
    for (unsigned i{0}, row{gidx}; i < ELEMS_PER_THREAD; i++, row += grid_size_x)
      if (row < count) {
        const auto y = batch_ys.get(row, col);
        thread_sum = COEF_T::add(thread_sum, COEF_T::mul(per_elem_factors[i], y));
      }
    partial_sums.set(gidx, col, thread_sum);
  }
}

// We could also potentially do these with some custom functors passed to cub segmented reduce
EXTERN __launch_bounds__(1024, 1) __global__
    void batch_barycentric_partial_reduce_base_at_base_kernel(matrix_getter<base_field, ld_modifier::cs> batch_ys,
                                                              vector_getter<base_field, ld_modifier::ca> lagrange_coeffs,
                                                              matrix_setter<base_field, st_modifier::cs> partial_sums, const unsigned log_count,
                                                              const unsigned num_polys) {
  batch_barycentric_partial_reduce_impl<base_field>(batch_ys, lagrange_coeffs, partial_sums, log_count, num_polys);
}

EXTERN __launch_bounds__(1024, 1) __global__ void batch_barycentric_partial_reduce_base_at_ext_kernel(matrix_getter<base_field, ld_modifier::cs> batch_ys,
                                                                                                      ef_double_vector_getter<ld_modifier::ca> lagrange_coeffs,
                                                                                                      ef_double_matrix_setter<st_modifier::cs> partial_sums,
                                                                                                      const unsigned log_count, const unsigned num_polys) {
  batch_barycentric_partial_reduce_impl<extension_field>(batch_ys, lagrange_coeffs, partial_sums, log_count, num_polys);
}

EXTERN __launch_bounds__(1024, 1) __global__ void batch_barycentric_partial_reduce_ext_at_ext_kernel(ef_double_matrix_getter<ld_modifier::cs> batch_ys,
                                                                                                     ef_double_vector_getter<ld_modifier::ca> lagrange_coeffs,
                                                                                                     ef_double_matrix_setter<st_modifier::cs> partial_sums,
                                                                                                     const unsigned log_count, const unsigned num_polys) {
  batch_barycentric_partial_reduce_impl<extension_field>(batch_ys, lagrange_coeffs, partial_sums, log_count, num_polys);
}

} // namespace goldilocks