#include "goldilocks.cuh"
#include "ops_complex.cuh"
#include <cstdint>

namespace goldilocks {

using namespace memory;

EXTERN __global__ void get_powers_of_w_kernel(const unsigned log_degree, const unsigned offset, const bool inverse, const bool bit_reverse,
                                              vector_setter<base_field, st_modifier::cs> result, const unsigned count) {
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count)
    return;
  const unsigned shift = OMEGA_LOG_ORDER - log_degree;
  const unsigned raw_index = gid + offset;
  const unsigned index = bit_reverse ? __brev(raw_index) >> (32 - log_degree) : raw_index;
  const unsigned shifted_index = index << shift;
  const base_field value = get_power_of_w(shifted_index, inverse);
  result.set(gid, value);
}

EXTERN __global__ void get_powers_of_g_kernel(const unsigned log_degree, const unsigned offset, const bool inverse, const bool bit_reverse,
                                              vector_setter<base_field, st_modifier::cs> result, const unsigned count) {
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count)
    return;
  const unsigned shift = OMEGA_LOG_ORDER - log_degree;
  const unsigned raw_index = gid + offset;
  const unsigned index = bit_reverse ? __brev(raw_index) >> (32 - log_degree) : raw_index;
  const unsigned shifted_index = index << shift;
  const base_field value = get_power_of_g(shifted_index, inverse);
  result.set(gid, value);
}

DEVICE_FORCEINLINE void get_powers_bf(const base_field &base, const unsigned offset, const bool bit_reverse, base_field *result, const unsigned count) {
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count)
    return;
  const unsigned power = (bit_reverse ? __brev(gid) : gid) + offset;
  const base_field value = base_field::pow(base, power);
  memory::store_cs(result + gid, value);
}

EXTERN __global__ void get_powers_by_val_bf_kernel(const base_field base, const unsigned offset, const bool bit_reverse, base_field *result,
                                                   const unsigned count) {
  get_powers_bf(base, offset, bit_reverse, result, count);
}

EXTERN __global__ void get_powers_by_ref_bf_kernel(const base_field *base, const unsigned offset, const bool bit_reverse, base_field *result,
                                                   const unsigned count) {
  get_powers_bf(*base, offset, bit_reverse, result, count);
}

DEVICE_FORCEINLINE void get_powers_ef(const extension_field &base, const unsigned offset, const bool bit_reverse, extension_field *result,
                                      const unsigned count) {
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count)
    return;
  const unsigned power = (bit_reverse ? __brev(gid) : gid) + offset;
  const extension_field value = extension_field::pow(base, power);
  auto result_bf = reinterpret_cast<base_field *>(result);
  memory::store_cs(result_bf + gid, value[0]);
  memory::store_cs(result_bf + gid + count, value[1]);
}

EXTERN __global__ void get_powers_by_val_ef_kernel(const extension_field base, const unsigned offset, const bool bit_reverse, extension_field *result,
                                                   const unsigned count) {
  get_powers_ef(base, offset, bit_reverse, result, count);
}

EXTERN __global__ void get_powers_by_ref_ef_kernel(const extension_field *base_ptr, const unsigned offset, const bool bit_reverse, extension_field *result,
                                                   const unsigned count) {
  get_powers_ef(*base_ptr, offset, bit_reverse, result, count);
}

EXTERN __global__ void omega_shift_kernel(const base_field *src, const unsigned log_degree, const unsigned offset, const bool inverse, const unsigned shift,
                                          base_field *dst, const unsigned count) {
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count)
    return;
  const base_field value = memory::load_cs(src + gid);
  const unsigned degree_shift = OMEGA_LOG_ORDER - log_degree;
  const unsigned index = shift * (gid + offset) << degree_shift;
  const base_field power = get_power_of_w(index, inverse);
  const base_field result = base_field::mul(value, power);
  memory::store_cs(dst + gid, result);
}

EXTERN __global__ void bit_reverse_naive_kernel(const matrix_getter<base_field, ld_modifier::cs> src, const matrix_setter<base_field, st_modifier::cs> dst,
                                                const unsigned log_count) {
  const unsigned row = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned threads_count = 1 << log_count;
  if (row >= threads_count)
    return;
  const unsigned col = blockIdx.y;
  const unsigned l_index = row;
  const unsigned r_index = __brev(l_index) >> (32 - log_count);
  if (l_index > r_index)
    return;
  const base_field l_value = src.get(l_index, col);
  const base_field r_value = src.get(r_index, col);
  dst.set(l_index, col, r_value);
  dst.set(r_index, col, l_value);
}

DEVICE_FORCEINLINE uint2 triangular_index_flat_to_two_dim(const unsigned index, const unsigned m) {
  const unsigned ii = m * (m + 1) / 2 - 1 - index;
  const unsigned k = floor((sqrtf(float(8 * ii + 1)) - 1) / 2);
  const unsigned jj = ii - k * (k + 1) / 2;
  const unsigned x = m - 1 - jj;
  const unsigned y = m - 1 - k;
  return {x, y};
}

EXTERN __launch_bounds__(128) __global__ void bit_reverse_kernel(const matrix_getter<base_field, ld_modifier::cs> src,
                                                                 const matrix_setter<base_field, st_modifier::cs> dst, const unsigned log_count) {
  static constexpr unsigned LOG_TILE_DIM = 5;
  static constexpr unsigned TILE_DIM = 1u << LOG_TILE_DIM;
  static constexpr unsigned BLOCK_ROWS = 2;
  __shared__ uint64_t tile[2][TILE_DIM][TILE_DIM + 1];
  const unsigned tid_x = threadIdx.x;
  const unsigned tid_y = threadIdx.y;
  const unsigned col = blockIdx.z;
  const unsigned half_log_count = log_count >> 1;
  const unsigned shift = 32 - half_log_count;
  const unsigned stride = gridDim.y << half_log_count;
  const unsigned x_offset = (blockIdx.y << half_log_count) + tid_x;
  const unsigned m = 1u << (half_log_count - LOG_TILE_DIM);
  const uint2 tile_xy = triangular_index_flat_to_two_dim(blockIdx.x, m);
  const bool is_diagonal = tile_xy.x == tile_xy.y;
  const unsigned is_reverse = threadIdx.z;
  if (is_diagonal && is_reverse)
    return;
  const unsigned tile_x = is_reverse ? tile_xy.y : tile_xy.x;
  const unsigned tile_y = is_reverse ? tile_xy.x : tile_xy.y;
  const unsigned tile_x_offset = tile_x * TILE_DIM;
  const unsigned tile_y_offset = tile_y * TILE_DIM;
  const unsigned x_src = tile_x_offset + x_offset;
  const unsigned y_src = tile_y_offset + tid_y;
  const unsigned x_dst = tile_y_offset + x_offset;
  const unsigned y_dst = tile_x_offset + tid_y;
#pragma unroll
  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
    const unsigned idx = tid_y + j;
    const unsigned ry = __brev(y_src + j) >> shift;
    const base_field value = src.get(ry * stride + x_src, col);
    tile[is_reverse][idx][tid_x] = base_field::to_u64(value);
  }
  __syncthreads();
#pragma unroll
  for (int j = 0; j < TILE_DIM; j += BLOCK_ROWS) {
    const unsigned idx = tid_y + j;
    const unsigned ry = __brev(y_dst + j) >> shift;
    base_field value = base_field::from_u64(tile[is_reverse][tid_x][idx]);
    dst.set(ry * stride + x_dst, col, value);
  }
}

EXTERN __global__ void select_kernel(const unsigned *indexes, const base_field *src, base_field *dst, const unsigned count) {
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count)
    return;
  const unsigned index = indexes[gid];
  const base_field value = index & (1u << 31) ? base_field::zero() : memory::load_cs(src + index);
  memory::store_cs(dst + gid, value);
}

template <typename T> struct InvBatch {};
template <> struct InvBatch<base_field> {
  static constexpr unsigned INV_BATCH = 10;
};
template <> struct InvBatch<extension_field> {
  static constexpr unsigned INV_BATCH = 6;
};

template <typename T, typename GETTER, typename SETTER> DEVICE_FORCEINLINE void batch_inv_impl(GETTER src, SETTER dst, const unsigned count) {
  constexpr unsigned INV_BATCH = InvBatch<T>::INV_BATCH;

  // ints for indexing because some bounds checks count down and check if an index drops below 0
  const int gid = int(blockIdx.x * blockDim.x + threadIdx.x);
  if (gid >= count)
    return;

  const int grid_size = int(blockDim.x * gridDim.x);

  T inputs[INV_BATCH];
  T outputs[INV_BATCH];

  // If count < grid size, the kernel is inefficient no matter what (because each thread processes just one element)
  // but we should still bail out if a thread has no assigned elems at all.
  int i = 0;
  int runtime_batch_size = 0;
  int g = gid;
#pragma unroll
  for (; i < INV_BATCH; i++, g += grid_size)
    if (g < count) {
      inputs[i] = src.get(g);
      runtime_batch_size++;
    }

  if (runtime_batch_size < INV_BATCH) {
    batch_inv_registers<T, INV_BATCH, false>(inputs, outputs, runtime_batch_size);
  } else {
    batch_inv_registers<T, INV_BATCH, true>(inputs, outputs, runtime_batch_size);
  }

#pragma unroll
  for (; i >= 0; i--, g -= grid_size)
    if (i < runtime_batch_size)
      dst.set(g, outputs[i]);
}

EXTERN __global__ void batch_inv_bf_kernel(vector_getter<base_field, ld_modifier::cs> src, vector_setter<base_field, st_modifier::cs> dst,
                                           const unsigned count) {
  batch_inv_impl<base_field>(src, dst, count);
}

EXTERN __global__ void batch_inv_ef_kernel(ef_double_vector_getter<ld_modifier::cs> src, ef_double_vector_setter<st_modifier::cs> dst, const unsigned count) {
  batch_inv_impl<extension_field>(src, dst, count);
}

EXTERN __global__ void pack_variable_indexes_kernel(const uint64_t *src, uint32_t *dst, const unsigned count) {
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count)
    return;
  const auto u64 = memory::load_cs(src + gid);
  const auto tuple = ptx::u64::unpack(u64);
  const uint32_t MASK = 1U << 31;
  const auto lo = std::get<0>(tuple) & ~MASK;
  const auto hi = std::get<1>(tuple) & MASK;
  auto u32 = lo | hi;
  memory::store_cs(dst + gid, u32);
}

EXTERN __global__ void mark_ends_of_runs_kernel(const unsigned *num_runs_out, const unsigned *run_offsets, unsigned *result, const unsigned count) {
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned num_runs = *num_runs_out;
  if (gid >= num_runs)
    return;
  const unsigned run_offset = run_offsets[gid];
  const unsigned run_offset_next = gid == num_runs - 1 ? count : run_offsets[gid + 1];
  const unsigned run_length = run_offset_next - run_offset;
  result[run_offset + run_length - 1] = 1;
}

EXTERN __global__ void generate_permutation_matrix_kernel(const unsigned *unique_variable_indexes, const unsigned *run_indexes, const unsigned *num_runs_out,
                                                          const unsigned *run_offsets, const unsigned *cell_indexes, const base_field *scalars,
                                                          base_field *result, const unsigned columns_count, const unsigned log_rows_count) {
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned count = columns_count << log_rows_count;
  if (gid >= count)
    return;
  const unsigned last_run_index = run_indexes[count - 1];
  const unsigned last_run_variable_index = unique_variable_indexes[last_run_index];
  const unsigned run_index = run_indexes[gid];
  const unsigned run_offset = run_offsets[run_index];
  const unsigned num_runs = *num_runs_out;
  const unsigned run_offset_next = run_index == num_runs - 1 ? count : run_offsets[run_index + 1];
  const unsigned run_length = run_offset_next - run_offset;
  const unsigned src_in_run_index = gid - run_offset;
  const bool is_placeholder = run_index == last_run_index && last_run_variable_index == (1U << 31);
  const unsigned dst_in_run_index = is_placeholder ? src_in_run_index : (src_in_run_index + 1) % run_length;
  const unsigned src_cell_index = cell_indexes[run_offset + src_in_run_index];
  const unsigned dst_cell_index = cell_indexes[run_offset + dst_in_run_index];
  const unsigned src_row_index = src_cell_index & ((1 << log_rows_count) - 1);
  const unsigned src_col_index = src_cell_index >> log_rows_count;
  const unsigned shift = OMEGA_LOG_ORDER - log_rows_count;
  const base_field twiddle = get_power_of_w(src_row_index << shift, false);
  const base_field scalar = scalars[src_col_index];
  const base_field value = base_field::mul(twiddle, scalar);
  memory::store_cs(result + dst_cell_index, value);
}

EXTERN __global__ void set_values_from_packed_bits_kernel(const uint32_t *packed_bits, base_field *result, const unsigned count) {
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned word_count = (count - 1) / 32 + 1;
  if (gid >= word_count)
    return;
  const unsigned offset = gid * 32;
  uint32_t word = packed_bits[gid];
  for (unsigned i = 0; i < 32 && offset + i < count; i++) {
    result[offset + i] = (word & 1) ? base_field::one() : base_field::zero();
    word >>= 1;
  }
}

EXTERN __global__ void fold_kernel(const base_field coset_inverse, const extension_field *challenge, const ef_double_vector_getter<ld_modifier::cs> src,
                                   ef_double_vector_setter<st_modifier::cs> dst, const unsigned log_count) {
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= 1 << log_count)
    return;

  const extension_field even = src.get(2 * gid);
  const extension_field odd = src.get(2 * gid + 1);
  const extension_field sum = extension_field::add(even, odd);
  extension_field diff = extension_field::sub(even, odd);
  const unsigned root_index = __brev(gid) >> (32 - OMEGA_LOG_ORDER + 1);
  const base_field root = get_power_of_w(root_index, true);
  diff = extension_field::mul(diff, base_field::mul(root, coset_inverse));
  diff = extension_field::mul(diff, *challenge);
  const extension_field result = extension_field::add(sum, diff);
  dst.set(gid, result);
}

EXTERN __global__ void partial_products_f_g_chunk_kernel(
    ef_double_vector_getter_setter<ld_modifier::cs, st_modifier::cs> num, ef_double_vector_getter_setter<ld_modifier::cs, st_modifier::cs> denom,
    matrix_getter<base_field, ld_modifier::cs> variable_cols_chunk, matrix_getter<base_field, ld_modifier::cs> sigma_cols_chunk,
    vector_getter<base_field, ld_modifier::cs> omega_values, const extension_field *non_residues_by_beta_chunk, const base_field *beta_c0,
    const base_field *beta_c1, const base_field *gamma_c0, const base_field *gamma_c1, const unsigned num_cols_this_chunk, const unsigned count) {
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count)
    return;

  using ef = extension_field;

  variable_cols_chunk.add_row(gid);
  sigma_cols_chunk.add_row(gid);

  // Alternatively, if the kernel turns out to be bandwidth bound, we could accept
  // log_extension_degree and coset_index as args, then use get_power_of_w
  // and get_power_of_g to grab each thread's coset domain value.
  // I doubt it'll make a big difference.
  const auto omega_i = omega_values.get(gid);

  const ef beta{*beta_c0, *beta_c1};
  const ef gamma{*gamma_c0, *gamma_c1};

  ef num_result{ef::one()};
  ef denom_result{ef::one()};
  for (unsigned col = 0; col < num_cols_this_chunk; col++) {
    // Redundant across threads, but should hit in cache
    const auto non_residue_by_beta = non_residues_by_beta_chunk[col];

    // numerator w + beta * non_res * x + gamma
    auto current_num = ef::mul(non_residue_by_beta, omega_i);
    const auto variable = variable_cols_chunk.get_at_col(col);
    current_num = ef::add(current_num, variable);
    current_num = ef::add(gamma, current_num);

    // denominator w + beta * sigma(x) + gamma
    const auto sigma = sigma_cols_chunk.get_at_col(col);
    auto current_denom = ef::mul(beta, sigma);
    current_denom = ef::add(current_denom, variable);
    current_denom = ef::add(current_denom, gamma);

    num_result = ef::mul(current_num, num_result);
    denom_result = ef::mul(current_denom, denom_result);
  }

  num.set(gid, num_result);
  denom.set(gid, denom_result);
}

EXTERN __global__ void partial_products_quotient_terms_kernel(ef_double_matrix_getter<ld_modifier::cs> partial_products,
                                                              ef_double_vector_getter<ld_modifier::cs> z_poly,
                                                              matrix_getter<base_field, ld_modifier::cs> variable_cols,
                                                              matrix_getter<base_field, ld_modifier::cs> sigma_cols,
                                                              vector_getter<base_field, ld_modifier::cs> omega_values, const extension_field *powers_of_alpha,
                                                              const extension_field *non_residues_by_beta, const base_field *beta_c0, const base_field *beta_c1,
                                                              const base_field *gamma_c0, const base_field *gamma_c1,
                                                              ef_double_vector_getter_setter<ld_modifier::cs, st_modifier::cs> quotient,
                                                              const unsigned num_cols, const unsigned num_cols_per_product, const unsigned log_count) {
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  const unsigned count = 1 << log_count;
  if (gid >= count)
    return;

  using ef = extension_field;

  partial_products.add_row(gid);
  variable_cols.add_row(gid);
  sigma_cols.add_row(gid);

  // value at omega * x should be next (rotated) val in the same coset in non-bitreversed order
  const unsigned i = __brev(gid) >> (32 - log_count);
  const unsigned i_shifted = (i == count - 1) ? 0 : i + 1;
  const unsigned gid_shifted = __brev(i_shifted) >> (32 - log_count);

  ef quotient_contribution{ef::zero()};

  // Alternatively, if the kernel turns out to be bandwidth bound, we could accept
  // log_extension_degree and coset_index as args, then use get_power_of_w
  // and get_power_of_g to grab each thread's coset domain value.
  // I doubt it'll make a big difference.
  const auto omega_i = omega_values.get(gid);

  const ef beta{*beta_c0, *beta_c1};
  const ef gamma{*gamma_c0, *gamma_c1};

  const auto num_col_chunks = (num_cols + num_cols_per_product - 1) / num_cols_per_product;
  ef lhs{};
  ef rhs{};
  for (unsigned col_chunk = 0; col_chunk < num_col_chunks; col_chunk++) {
    ef num{ef::one()};
    ef denom{ef::one()};

    unsigned col_chunk_start = col_chunk * num_cols_per_product;
    for (unsigned col = col_chunk_start; (col < col_chunk_start + num_cols_per_product) && (col < num_cols); col++) {
      // Redundant across threads, but should hit in cache
      const auto non_residue_by_beta = non_residues_by_beta[col];

      // numerator w + beta * non_res * x + gamma
      auto current_num = ef::mul(non_residue_by_beta, omega_i);
      auto variable = variable_cols.get_at_col(col);
      current_num = ef::add(current_num, variable);
      current_num = ef::add(gamma, current_num);

      // denominator w + beta * sigma(x) + gamma
      const auto sigma = sigma_cols.get_at_col(col);
      auto current_denom = ef::mul(beta, sigma);
      current_denom = ef::add(current_denom, variable);
      current_denom = ef::add(current_denom, gamma);

      num = ef::mul(num, current_num);
      denom = ef::mul(denom, current_denom);
    }

    // Redundant across threads, but should hit in cache
    const auto alpha = powers_of_alpha[col_chunk];

    rhs = (col_chunk == 0) ? z_poly.get(gid) : lhs;
    // z_poly.get(gid_shifted) accesses might not be coalesced, but this should be negligible
    lhs = (col_chunk < (num_col_chunks - 1)) ? partial_products.get_at_col(col_chunk) : z_poly.get(gid_shifted);
    denom = ef::mul(denom, lhs);
    num = ef::mul(num, rhs);
    denom = ef::sub(denom, num);
    denom = ef::mul(denom, alpha);
    quotient_contribution = ef::add(quotient_contribution, denom);
  }

  const auto existing_quotient = quotient.get(gid);
  quotient.set(gid, ef::add(existing_quotient, quotient_contribution));
}

EXTERN __global__ void lookup_aggregated_table_values_kernel(matrix_getter<base_field, ld_modifier::cs> table_cols, const base_field *beta_c0,
                                                             const base_field *beta_c1, const extension_field *powers_of_gamma,
                                                             ef_double_vector_setter<st_modifier::cs> aggregated_table_values, const unsigned num_cols,
                                                             const unsigned count) {
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count)
    return;

  using ef = extension_field;

  table_cols.add_row(gid);

  const ef beta{*beta_c0, *beta_c1};

  ef tmp{ef::zero()};
  for (unsigned col = 0; col < num_cols; col++) {
    const auto table_val = table_cols.get_at_col(col);
    const auto current_gamma = powers_of_gamma[col];
    tmp = ef::add(tmp, ef::mul(current_gamma, table_val));
  }
  tmp = ef::add(tmp, beta);
  aggregated_table_values.set(gid, tmp);
}

EXTERN __global__ void lookup_subargs_a_and_b_kernel(matrix_getter<base_field, ld_modifier::cs> variable_cols,
                                                     ef_double_matrix_setter<st_modifier::cs> subargs_a, ef_double_matrix_setter<st_modifier::cs> subargs_b,
                                                     const base_field *beta_c0, const base_field *beta_c1, const extension_field *powers_of_gamma,
                                                     vector_getter<base_field, ld_modifier::cs> table_id_col,
                                                     ef_double_vector_getter<ld_modifier::cs> aggregated_table_values_inv,
                                                     matrix_getter<base_field, ld_modifier::cs> multiplicity_cols, const unsigned num_subargs_a,
                                                     const unsigned num_subargs_b, const unsigned num_cols_per_subarg, const unsigned count) {
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count)
    return;

  using ef = extension_field;

  variable_cols.add_row(gid);
  subargs_a.add_row(gid);
  subargs_b.add_row(gid);
  multiplicity_cols.add_row(gid);

  // subargs_a
  const ef beta{*beta_c0, *beta_c1};
  for (unsigned subarg = 0; subarg < num_subargs_a; subarg++) {
    ef w_plus_beta{ef::zero()};
    for (unsigned col_in_subarg = 0; col_in_subarg < num_cols_per_subarg; col_in_subarg++) {
      const unsigned col = col_in_subarg + subarg * num_cols_per_subarg;
      // Redundant across threads, but should hit in cache
      const auto current_gamma = powers_of_gamma[col_in_subarg];
      const auto val = variable_cols.get_at_col(col);
      w_plus_beta = ef::add(w_plus_beta, ef::mul(current_gamma, val));
    }
    const auto table_id = table_id_col.get(gid);
    const auto current_gamma = powers_of_gamma[num_cols_per_subarg];
    w_plus_beta = ef::add(w_plus_beta, ef::mul(current_gamma, table_id));
    w_plus_beta = ef::add(w_plus_beta, beta);

    subargs_a.set_at_col(subarg, w_plus_beta);
  }

  // subargs_b (could be a separate kernel, but convenient to keep together)
  for (unsigned subarg = 0; subarg < num_subargs_b; subarg++) {
    const auto m = multiplicity_cols.get_at_col(subarg);
    const auto t_plus_beta_inv = aggregated_table_values_inv.get(gid);
    subargs_b.set_at_col(subarg, ef::mul(t_plus_beta_inv, m));
  }
}

EXTERN __global__ void lookup_quotient_a_and_b_kernel(matrix_getter<base_field, ld_modifier::cs> variable_cols,
                                                      matrix_getter<base_field, ld_modifier::cs> table_cols, ef_double_matrix_getter<ld_modifier::cs> subargs_a,
                                                      ef_double_matrix_getter<ld_modifier::cs> subargs_b, const base_field *beta_c0, const base_field *beta_c1,
                                                      const extension_field *powers_of_gamma, const extension_field *powers_of_alpha,
                                                      vector_getter<base_field, ld_modifier::cs> table_id_col,
                                                      matrix_getter<base_field, ld_modifier::cs> multiplicity_cols,
                                                      ef_double_vector_getter_setter<ld_modifier::cs, st_modifier::cs> quotient, const unsigned num_subargs_a,
                                                      const unsigned num_subargs_b, const unsigned num_cols_per_subarg, const unsigned count) {
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count)
    return;

  using ef = extension_field;

  variable_cols.add_row(gid);
  table_cols.add_row(gid);
  subargs_a.add_row(gid);
  subargs_b.add_row(gid);
  multiplicity_cols.add_row(gid);

  const ef beta{*beta_c0, *beta_c1};

  // powers_of_gamma and powers_of_alpha loads are redundant across threads and across
  // subargs_a and subargs_b, but they're cache friendly: small and broadcasted.

  ef quotient_contribution{ef::zero()};

  // subargs_a
  for (unsigned subarg = 0; subarg < num_subargs_a; subarg++) {
    ef w_plus_beta{ef::zero()};
    for (unsigned col_in_subarg = 0; col_in_subarg < num_cols_per_subarg; col_in_subarg++) {
      const unsigned col = col_in_subarg + subarg * num_cols_per_subarg;
      const auto current_gamma = powers_of_gamma[col_in_subarg];
      const auto val = variable_cols.get_at_col(col);
      w_plus_beta = ef::add(w_plus_beta, ef::mul(current_gamma, val));
    }
    const auto table_id = table_id_col.get(gid);
    const auto current_gamma = powers_of_gamma[num_cols_per_subarg];
    w_plus_beta = ef::add(w_plus_beta, ef::mul(current_gamma, table_id));
    w_plus_beta = ef::add(w_plus_beta, beta);

    auto a = subargs_a.get_at_col(subarg);
    a = ef::mul(a, w_plus_beta);
    a = ef::sub(a, base_field::one());
    const auto current_alpha = powers_of_alpha[subarg];
    a = ef::mul(a, current_alpha);
    quotient_contribution = ef::add(quotient_contribution, a);
  }

  // subargs_b (could be a separate kernel, but convenient to keep together)
  const auto num_cols_per_subarg_b = num_cols_per_subarg + 1;
  for (unsigned subarg = 0; subarg < num_subargs_b; subarg++) {
    ef t_plus_beta{ef::zero()};
    for (unsigned col_in_subarg = 0; col_in_subarg < num_cols_per_subarg_b; col_in_subarg++) {
      const unsigned col = col_in_subarg + subarg * num_cols_per_subarg_b;
      const auto current_gamma = powers_of_gamma[col_in_subarg];
      const auto val = table_cols.get_at_col(col);
      t_plus_beta = ef::add(t_plus_beta, ef::mul(current_gamma, val));
    }
    t_plus_beta = ef::add(t_plus_beta, beta);

    auto b = subargs_b.get_at_col(subarg);
    b = ef::mul(b, t_plus_beta);
    const auto m = multiplicity_cols.get_at_col(subarg);
    b = ef::sub(b, m);
    const auto current_alpha = powers_of_alpha[num_subargs_a + subarg];
    b = ef::mul(b, current_alpha);
    quotient_contribution = ef::add(quotient_contribution, b);
  }

  const auto existing_quotient = quotient.get(gid);
  quotient.set(gid, ef::add(existing_quotient, quotient_contribution));
}

#define ACCUMULATE(SUM, EVALS, COLS, NUM_COLS)                                                                                                                 \
  {                                                                                                                                                            \
    for (unsigned i = 0; i < (NUM_COLS); i++, challenges++, (EVALS)++) {                                                                                       \
      const auto term = ef::mul(*challenges, ef::sub((COLS).get_at_col(i), *(EVALS)));                                                                         \
      (SUM) = ef::add(SUM, term);                                                                                                                              \
    }                                                                                                                                                          \
  }

EXTERN __global__ void deep_quotient_except_public_inputs_kernel(
    matrix_getter<base_field, ld_modifier::cs> variable_cols, matrix_getter<base_field, ld_modifier::cs> witness_cols,
    matrix_getter<base_field, ld_modifier::cs> constant_cols, matrix_getter<base_field, ld_modifier::cs> permutation_cols,
    ef_double_vector_getter<ld_modifier::cs> z_poly, ef_double_matrix_getter<ld_modifier::cs> partial_products,
    matrix_getter<base_field, ld_modifier::cs> multiplicity_cols, ef_double_matrix_getter<ld_modifier::cs> lookup_a_polys,
    ef_double_matrix_getter<ld_modifier::cs> lookup_b_polys, matrix_getter<base_field, ld_modifier::cs> table_cols,
    ef_double_matrix_getter<ld_modifier::cs> quotient_constraint_polys, const extension_field *evals_at_z, const extension_field *evals_at_z_omega,
    const extension_field *evals_at_zero, const extension_field *challenges, ef_double_vector_getter<ld_modifier::cs> denom_at_z,
    ef_double_vector_getter<ld_modifier::cs> denom_at_z_omega, vector_getter<base_field, ld_modifier::cs> denom_at_zero,
    ef_double_vector_setter<st_modifier::cs> quotient, const unsigned num_variable_cols, const unsigned num_witness_cols, const unsigned num_constant_cols,
    const unsigned num_permutation_cols, const unsigned num_partial_products, const unsigned num_multiplicity_cols, const unsigned num_lookup_a_polys,
    const unsigned num_lookup_b_polys, const unsigned num_table_cols, const unsigned num_quotient_constraint_polys, const unsigned z_omega_challenge_offset,
    const unsigned zero_challenge_offset, const unsigned count) {
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count)
    return;

  using ef = extension_field;

  variable_cols.add_row(gid);
  if (num_witness_cols > 0) {
    witness_cols.add_row(gid);
  }
  constant_cols.add_row(gid);
  permutation_cols.add_row(gid);
  partial_products.add_row(gid);
  if (num_multiplicity_cols > 0) {
    multiplicity_cols.add_row(gid);
    lookup_a_polys.add_row(gid);
    lookup_b_polys.add_row(gid);
    table_cols.add_row(gid);
  }
  quotient_constraint_polys.add_row(gid);

  // Evaluations and challenges loads are redundant across threads, but small and broadcasted

  ef terms_at_z{ef::zero()};
  ACCUMULATE(terms_at_z, evals_at_z, variable_cols, num_variable_cols)
  if (num_witness_cols > 0) {
    ACCUMULATE(terms_at_z, evals_at_z, witness_cols, num_witness_cols)
  }
  ACCUMULATE(terms_at_z, evals_at_z, constant_cols, num_constant_cols)
  ACCUMULATE(terms_at_z, evals_at_z, permutation_cols, num_permutation_cols)
  // handles z_poly terms at z and z_omega adjacently to avoid loading z_poly twice
  const auto z_poly_val = z_poly.get(gid);
  // z_poly term at z
  const auto tmp0 = ef::mul(*challenges, ef::sub(z_poly_val, *evals_at_z));
  terms_at_z = ef::add(terms_at_z, tmp0);
  // z_poly term at z_omega
  const auto tmp1 = ef::mul(*(challenges + z_omega_challenge_offset), ef::sub(z_poly_val, *evals_at_z_omega));
  // might as well initialize overall running sum here
  ef terms_sum = ef::mul(tmp1, denom_at_z_omega.get(gid));
  challenges++;
  evals_at_z++;
  ACCUMULATE(terms_at_z, evals_at_z, partial_products, num_partial_products)
  if (num_multiplicity_cols > 0) {
    ACCUMULATE(terms_at_z, evals_at_z, multiplicity_cols, num_multiplicity_cols)
    // handles lookup_a_polys terms at z and zero adjacently to avoid loading lookup_a_polys twice
    ef terms_at_zero{ef::zero()};
    for (unsigned i = 0; i < num_lookup_a_polys; i++, challenges++, evals_at_z++, evals_at_zero++) {
      const auto a_poly_val = lookup_a_polys.get_at_col(i);
      // a_poly term at z
      const auto tmp0 = ef::mul(*challenges, ef::sub(a_poly_val, *evals_at_z));
      terms_at_z = ef::add(terms_at_z, tmp0);
      // a_poly term at zero
      const auto tmp1 = ef::mul(*(challenges + zero_challenge_offset), ef::sub(a_poly_val, *evals_at_zero));
      terms_at_zero = ef::add(terms_at_zero, tmp1);
    }
    // handles lookup_b_polys terms at z and zero adjacently to avoid loading lookup_a_polys twice
    for (unsigned i = 0; i < num_lookup_b_polys; i++, challenges++, evals_at_z++, evals_at_zero++) {
      const auto b_poly_val = lookup_b_polys.get_at_col(i);
      // b_poly term at z
      const auto tmp0 = ef::mul(*challenges, ef::sub(b_poly_val, *evals_at_z));
      terms_at_z = ef::add(terms_at_z, tmp0);
      // b_poly term at zero
      const auto tmp1 = ef::mul(*(challenges + zero_challenge_offset), ef::sub(b_poly_val, *evals_at_zero));
      terms_at_zero = ef::add(terms_at_zero, tmp1);
    }
    terms_at_zero = ef::mul(terms_at_zero, denom_at_zero.get(gid));
    terms_sum = ef::add(terms_sum, terms_at_zero);
    ACCUMULATE(terms_at_z, evals_at_z, table_cols, num_table_cols)
  }
  ACCUMULATE(terms_at_z, evals_at_z, quotient_constraint_polys, num_quotient_constraint_polys)
  terms_at_z = ef::mul(terms_at_z, denom_at_z.get(gid));
  terms_sum = ef::add(terms_sum, terms_at_z);

  quotient.set(gid, terms_sum);
}

EXTERN __global__ void deep_quotient_public_input_kernel(vector_getter<base_field, ld_modifier::cs> values, base_field expected_value,
                                                         const extension_field *challenge,
                                                         ef_double_vector_getter_setter<ld_modifier::cs, st_modifier::cs> quotient, const unsigned count) {
  const unsigned gid = blockIdx.x * blockDim.x + threadIdx.x;
  if (gid >= count)
    return;

  using bf = base_field;
  using ef = extension_field;

  auto val = values.get(gid);
  val = bf::sub(val, expected_value);
  const auto num = ef::mul(*challenge, val);
  const auto existing_quotient = quotient.get(gid);
  quotient.set(gid, ef::add(num, existing_quotient));
}

} // namespace goldilocks
