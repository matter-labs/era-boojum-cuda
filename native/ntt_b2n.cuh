#pragma once // also, this file should only be compiled in one compile unit because it has __global__ definitions

template <unsigned LOG_VALS_PER_THREAD> DEVICE_FORCEINLINE
void b2n_initial_stages_warp(const base_field *gmem_inputs_matrix, base_field *gmem_outputs_matrix, const unsigned stride_between_input_arrays,
                             const unsigned stride_between_output_arrays, const unsigned start_stage, const unsigned stages_this_launch,
                             const unsigned log_n, const bool inverse, const unsigned num_ntts, const unsigned log_extension_degree,
                             const unsigned coset_idx) {
  constexpr unsigned VALS_PER_THREAD = 1 << LOG_VALS_PER_THREAD;
  constexpr unsigned PAIRS_PER_THREAD = VALS_PER_THREAD >> 1;
  constexpr unsigned VALS_PER_WARP = 32 * VALS_PER_THREAD;
  constexpr unsigned LOG_VALS_PER_BLOCK = 5 + LOG_VALS_PER_THREAD + 2;
  constexpr unsigned VALS_PER_BLOCK = 1 << LOG_VALS_PER_BLOCK;

  __shared__ base_field smem[VALS_PER_BLOCK];

  const unsigned lane_id{threadIdx.x & 31};
  const unsigned warp_id{threadIdx.x >> 5};
  const unsigned gmem_offset = VALS_PER_BLOCK * blockIdx.x + VALS_PER_WARP * warp_id;
  const base_field *gmem_in = gmem_inputs_matrix + gmem_offset + NTTS_PER_BLOCK * stride_between_input_arrays * blockIdx.y;
  base_field *gmem_out = gmem_outputs_matrix + gmem_offset + NTTS_PER_BLOCK * stride_between_output_arrays * blockIdx.y;

  auto twiddle_cache = smem + VALS_PER_WARP * warp_id;

  base_field vals[VALS_PER_THREAD];

  load_initial_twiddles_warp<VALS_PER_WARP, LOG_VALS_PER_THREAD>(twiddle_cache, lane_id, gmem_offset, inverse);

  const unsigned bound = std::min(NTTS_PER_BLOCK, num_ntts - NTTS_PER_BLOCK * blockIdx.y);
  for (unsigned ntt_idx = 0; ntt_idx < bound;
       ntt_idx++, gmem_in += stride_between_input_arrays, gmem_out += stride_between_output_arrays) {
#pragma unroll
    for (unsigned i = 0; i < PAIRS_PER_THREAD; i++) {
      const auto in = memory::load_cs(reinterpret_cast<const uint4*>(gmem_in + 64 * i + 2 * lane_id));
      vals[2 * i][0] = in.x;
      vals[2 * i][1] = in.y;
      vals[2 * i + 1][0] = in.z;
      vals[2 * i + 1][1] = in.w;
    }

    if (log_extension_degree && !inverse) {
      __syncwarp();
      base_field tmp[VALS_PER_THREAD];
      base_field *scratch = twiddle_cache + VALS_PER_THREAD * lane_id;
#pragma unroll
      for (unsigned i = 0; i < VALS_PER_THREAD; i++) {
        tmp[i] = scratch[i];
        scratch[i] = vals[i];
      }
      apply_lde_factors<VALS_PER_THREAD, false>(scratch, gmem_offset, lane_id, log_n, log_extension_degree, coset_idx);
#pragma unroll
      for (unsigned i = 0; i < VALS_PER_THREAD; i++) {
        vals[i] = scratch[i];
        scratch[i] = tmp[i];
      }
      __syncwarp();
    }

    unsigned lane_mask = 1;
    base_field *twiddles_this_stage = twiddle_cache;
    unsigned num_twiddles_this_stage = VALS_PER_WARP >> 1;
    for (unsigned stage = 0; stage < 6; stage++) {
#pragma unroll
      for (unsigned i = 0; i < PAIRS_PER_THREAD; i++) {
        const auto twiddle = twiddles_this_stage[(32 * i + lane_id) >> stage];
        exchg_dif(vals[2 * i], vals[2 * i + 1], twiddle);
        if (stage < 5)
          shfl_xor_bf(vals, i, lane_id, lane_mask);
      }
      lane_mask <<= 1;
      twiddles_this_stage += num_twiddles_this_stage;
      num_twiddles_this_stage >>= 1;
    }

    for (unsigned i = 1; i < LOG_VALS_PER_THREAD; i++) {
#pragma unroll
      for (unsigned j = 0; j < PAIRS_PER_THREAD >> i; j++) {
        const unsigned exchg_tile_sz = 2 << i;
        const unsigned half_exchg_tile_sz = 1 << i;
        const auto twiddle = twiddles_this_stage[j];
#pragma unroll
        for (unsigned k = 0; k < half_exchg_tile_sz; k++) {
          exchg_dif(vals[exchg_tile_sz * j + k], vals[exchg_tile_sz * j + k + half_exchg_tile_sz], twiddle);
        }
      }
      twiddles_this_stage += num_twiddles_this_stage;
      num_twiddles_this_stage >>= 1;
    }

#pragma unroll
    for (unsigned i = 0; i < PAIRS_PER_THREAD; i++) {
      // This output pattern (resulting from the above shfls) is nice, but not obvious.
      // To see why it works, sketch the shfl stages on paper.
      memory::store_cs(gmem_out + 64 * i + lane_id, vals[2 * i]);
      memory::store_cs(gmem_out + 64 * i + lane_id + 32, vals[2 * i + 1]);
    }
  }
}

extern "C" __launch_bounds__(128, 8) __global__
void b2n_initial_8_stages_warp(const base_field *gmem_inputs_matrix, base_field *gmem_outputs_matrix, const unsigned stride_between_input_arrays,
                               const unsigned stride_between_output_arrays, const unsigned start_stage, const unsigned stages_this_launch,
                               const unsigned log_n, const bool inverse, const unsigned num_ntts, const unsigned log_extension_degree,
                               const unsigned coset_idx) {
  b2n_initial_stages_warp<3>(gmem_inputs_matrix, gmem_outputs_matrix, stride_between_input_arrays, stride_between_output_arrays, start_stage,
                             stages_this_launch, log_n, inverse, num_ntts, log_extension_degree, coset_idx);
}

extern "C" __launch_bounds__(128, 8) __global__
void b2n_initial_7_stages_warp(const base_field *gmem_inputs_matrix, base_field *gmem_outputs_matrix, const unsigned stride_between_input_arrays,
                               const unsigned stride_between_output_arrays, const unsigned start_stage, const unsigned stages_this_launch,
                               const unsigned log_n, const bool inverse, const unsigned num_ntts, const unsigned log_extension_degree,
                               const unsigned coset_idx) {
  b2n_initial_stages_warp<2>(gmem_inputs_matrix, gmem_outputs_matrix, stride_between_input_arrays, stride_between_output_arrays, start_stage,
                             stages_this_launch, log_n, inverse, num_ntts, log_extension_degree, coset_idx);
}

template <unsigned LOG_VALS_PER_THREAD> DEVICE_FORCEINLINE
void b2n_initial_stages_block(const base_field *gmem_inputs_matrix, base_field *gmem_outputs_matrix, const unsigned stride_between_input_arrays,
                              const unsigned stride_between_output_arrays, const unsigned start_stage, const unsigned stages_this_launch,
                              const unsigned log_n, const bool inverse, const unsigned num_ntts, const unsigned log_extension_degree,
                              const unsigned coset_idx) {
  constexpr unsigned VALS_PER_THREAD = 1 << LOG_VALS_PER_THREAD;
  constexpr unsigned PAIRS_PER_THREAD = VALS_PER_THREAD >> 1;
  constexpr unsigned VALS_PER_WARP = 32 * VALS_PER_THREAD;
  constexpr unsigned WARPS_PER_BLOCK = VALS_PER_WARP >> 4;
  constexpr unsigned LOG_VALS_PER_BLOCK = 2 * (LOG_VALS_PER_THREAD + 5) - 4;
  constexpr unsigned VALS_PER_BLOCK = 1 << LOG_VALS_PER_BLOCK;

  __shared__ base_field smem[VALS_PER_BLOCK];

  const unsigned lane_id{threadIdx.x & 31};
  const unsigned warp_id{threadIdx.x >> 5};
  const unsigned gmem_block_offset = VALS_PER_BLOCK * blockIdx.x;
  const unsigned gmem_offset = gmem_block_offset + VALS_PER_WARP * warp_id;
  const base_field *gmem_in = gmem_inputs_matrix + gmem_offset +
                              NTTS_PER_BLOCK * stride_between_input_arrays * blockIdx.y;
  // annoyingly scrambled, but should be coalesced overall
  const unsigned gmem_out_thread_offset = 16 * warp_id + VALS_PER_WARP * (lane_id >> 4) + 2 * (lane_id & 7) + ((lane_id >> 3) & 1);
  base_field *gmem_out = gmem_outputs_matrix + gmem_block_offset + gmem_out_thread_offset +
                         NTTS_PER_BLOCK * stride_between_output_arrays * blockIdx.y;

  auto twiddle_cache = smem + VALS_PER_WARP * warp_id;

  base_field vals[VALS_PER_THREAD];

  load_initial_twiddles_warp<VALS_PER_WARP, LOG_VALS_PER_THREAD>(twiddle_cache, lane_id, gmem_offset, inverse);

  const unsigned bound = std::min(NTTS_PER_BLOCK, num_ntts - NTTS_PER_BLOCK * blockIdx.y);
  for (unsigned ntt_idx = 0; ntt_idx < bound;
       ntt_idx++, gmem_in += stride_between_input_arrays, gmem_out += stride_between_output_arrays) {
#pragma unroll
    for (unsigned i = 0; i < PAIRS_PER_THREAD; i++) {
      const auto pair = memory::load_cs(reinterpret_cast<const uint4*>(gmem_in + 64 * i + 2 * lane_id));
      vals[2 * i][0] = pair.x;
      vals[2 * i][1] = pair.y;
      vals[2 * i + 1][0] = pair.z;
      vals[2 * i + 1][1] = pair.w;
    }

    if (log_extension_degree && !inverse) {
      __syncwarp();
      base_field tmp[VALS_PER_THREAD];
      base_field *scratch = twiddle_cache + VALS_PER_THREAD * lane_id;
#pragma unroll
      for (unsigned i = 0; i < VALS_PER_THREAD; i++) {
        tmp[i] = scratch[i];
        scratch[i] = vals[i];
      }
      apply_lde_factors<VALS_PER_THREAD, false>(scratch, gmem_offset, lane_id, log_n, log_extension_degree, coset_idx);
#pragma unroll
      for (unsigned i = 0; i < VALS_PER_THREAD; i++) {
        vals[i] = scratch[i];
        scratch[i] = tmp[i];
      }
      __syncwarp();
    }

    unsigned lane_mask = 1;
    base_field *twiddles_this_stage = twiddle_cache;
    unsigned num_twiddles_this_stage = VALS_PER_WARP >> 1;
    for (unsigned stage = 0; stage < 6; stage++) {
#pragma unroll
      for (unsigned i = 0; i < PAIRS_PER_THREAD; i++) {
        const auto twiddle = twiddles_this_stage[(32 * i + lane_id) >> stage];
        exchg_dif(vals[2 * i], vals[2 * i + 1], twiddle);
        if (stage < 5)
          shfl_xor_bf(vals, i, lane_id, lane_mask);
      }
      lane_mask <<= 1;
      twiddles_this_stage += num_twiddles_this_stage;
      num_twiddles_this_stage >>= 1;
    }

    for (unsigned i = 1; i < LOG_VALS_PER_THREAD; i++) {
#pragma unroll
      for (unsigned j = 0; j < PAIRS_PER_THREAD >> i; j++) {
        const unsigned exchg_tile_sz = 2 << i;
        const unsigned half_exchg_tile_sz = 1 << i;
        const auto twiddle = twiddles_this_stage[j];
#pragma unroll
        for (unsigned k = 0; k < half_exchg_tile_sz; k++)
          exchg_dif(vals[exchg_tile_sz * j + k], vals[exchg_tile_sz * j + k + half_exchg_tile_sz], twiddle);
      }
      twiddles_this_stage += num_twiddles_this_stage;
      num_twiddles_this_stage >>= 1;
    }

    __syncwarp();

    if (ntt_idx < num_ntts - 1) {
#pragma unroll
      for (unsigned i = 0; i < PAIRS_PER_THREAD; i++) {
        // juggle twiddles in registers while we use smem to communicate values
        const auto tmp0 = twiddle_cache[64 * i + lane_id];
        const auto tmp1 = twiddle_cache[64 * i + lane_id + 32];
        twiddle_cache[64 * i + lane_id] = vals[2 * i];
        twiddle_cache[64 * i + lane_id + 32] = vals[2 * i + 1];
        vals[2 * i] = tmp0;
        vals[2 * i + 1] = tmp1;
      }
    } else {
#pragma unroll
      for (unsigned i = 0; i < PAIRS_PER_THREAD; i++) {
        twiddle_cache[64 * i + lane_id] = vals[2 * i];
        twiddle_cache[64 * i + lane_id + 32] = vals[2 * i + 1];
      }
    }

    __syncthreads();

    auto pair_addr = smem + 16 * warp_id + VALS_PER_WARP * (lane_id >> 3) + 2 * (threadIdx.x & 7);
    if (ntt_idx < num_ntts - 1) {
      // juggle twiddles back into smem
      // In theory, we could avoid the full-size stashing and extra syncthreads by
      // "switching" each warp's twiddle region from contiguous to strided-chunks each iteration,
      // but that's a lot of trouble. Let's try the simple approach first.
      base_field tmp[VALS_PER_THREAD];
#pragma unroll
      for (unsigned i = 0; i < PAIRS_PER_THREAD; i++) {
        tmp[2 * i] = vals[2 * i];
        tmp[2 * i + 1] = vals[2 * i + 1];
      }

#pragma unroll
      for (unsigned i = 0; i < PAIRS_PER_THREAD; i++, pair_addr += 4 * VALS_PER_WARP) {
        const auto pair = *reinterpret_cast<const uint4*>(pair_addr);
        vals[2 * i][0] = pair.x;
        vals[2 * i][1] = pair.y;
        vals[2 * i + 1][0] = pair.z;
        vals[2 * i + 1][1] = pair.w;
      }

      __syncthreads();

#pragma unroll
      for (unsigned i = 0; i < PAIRS_PER_THREAD; i++) {
        twiddle_cache[64 * i + lane_id] = tmp[2 * i];
        twiddle_cache[64 * i + lane_id + 32] = tmp[2 * i + 1];
      }

      __syncwarp(); // maybe unnecessary due to shfls below
      // __syncthreads();
    } else {
#pragma unroll
      for (unsigned i = 0; i < PAIRS_PER_THREAD; i++, pair_addr += 4 * VALS_PER_WARP) {
        const auto pair = *reinterpret_cast<const uint4*>(pair_addr);
        vals[2 * i][0] = pair.x;
        vals[2 * i][1] = pair.y;
        vals[2 * i + 1][0] = pair.z;
        vals[2 * i + 1][1] = pair.w;
      }
    }

    const unsigned stages_so_far = 6 + LOG_VALS_PER_THREAD - 1;
    lane_mask = 8;
    unsigned exchg_region_offset = blockIdx.x * (WARPS_PER_BLOCK >> 1) + (lane_id >> 4);
    for (unsigned s = 0; s < 2; s++) {
      if (s + stages_so_far < stages_this_launch) {
#pragma unroll
        for (unsigned i = 0; i < PAIRS_PER_THREAD; i++) {
          // TODO: Handle these cooperatively?
          const auto twiddle = get_twiddle(inverse, exchg_region_offset + ((2 * i) >> s));
          shfl_xor_bf(vals, i, lane_id, lane_mask);
          exchg_dif(vals[2 * i], vals[2 * i + 1], twiddle);
        }
      } else {
#pragma unroll
        for (unsigned i = 0; i < PAIRS_PER_THREAD; i++)
          shfl_xor_bf(vals, i, lane_id, lane_mask);
      }
      lane_mask <<= 1;
      exchg_region_offset >>= 1;
    }

    exchg_region_offset = blockIdx.x * (PAIRS_PER_THREAD >> 1);
    for (unsigned i = 1; i < LOG_VALS_PER_THREAD; i++) {
      if (i + 2 + stages_so_far <= stages_this_launch) {
#pragma unroll
        for (unsigned j = 0; j < PAIRS_PER_THREAD >> i; j++) {
          const unsigned exchg_tile_sz = 2 << i;
          const unsigned half_exchg_tile_sz = 1 << i;
          const auto twiddle = get_twiddle(inverse, exchg_region_offset + (j >> (i - 1)));
#pragma unroll
          for (unsigned k = 0; k < half_exchg_tile_sz; k++)
            exchg_dif(vals[exchg_tile_sz * j + k], vals[exchg_tile_sz * j + k + half_exchg_tile_sz], twiddle);
        }
      }
      exchg_region_offset >>= 1;
    }

#pragma unroll
    for (unsigned i = 0; i < PAIRS_PER_THREAD; i++) {
      memory::store_cs(gmem_out + 4 * i * VALS_PER_WARP, vals[2 * i]);
      memory::store_cs(gmem_out + (4 * i + 2) * VALS_PER_WARP, vals[2 * i + 1]);
    }
  }
}

// extern "C" __launch_bounds__(512, 2) __global__
extern "C" __global__
void b2n_initial_9_to_12_stages_block(const base_field *gmem_inputs_matrix, base_field *gmem_outputs_matrix, const unsigned stride_between_input_arrays,
                                      const unsigned stride_between_output_arrays, const unsigned start_stage, const unsigned stages_this_launch,
                                      const unsigned log_n, const bool inverse, const unsigned num_ntts, const unsigned log_extension_degree,
                                      const unsigned coset_idx) {
  b2n_initial_stages_block<3>(gmem_inputs_matrix, gmem_outputs_matrix, stride_between_input_arrays, stride_between_output_arrays, start_stage,
                              stages_this_launch, log_n, inverse, num_ntts, log_extension_degree, coset_idx);
}

template <unsigned LOG_VALS_PER_THREAD> DEVICE_FORCEINLINE
void b2n_noninitial_stages_block(const base_field *gmem_inputs_matrix, base_field *gmem_outputs_matrix, const unsigned stride_between_input_arrays,
                                 const unsigned stride_between_output_arrays, const unsigned start_stage, const bool skip_first_stage,
                                 const unsigned log_n, const bool inverse, const unsigned num_ntts, const unsigned log_extension_degree,
                                 const unsigned coset_idx) {
  constexpr unsigned VALS_PER_THREAD = 1 << LOG_VALS_PER_THREAD;
  constexpr unsigned PAIRS_PER_THREAD = VALS_PER_THREAD >> 1;
  constexpr unsigned VALS_PER_WARP = 32 * VALS_PER_THREAD;
  constexpr unsigned TILES_PER_WARP = VALS_PER_WARP >> 4;
  constexpr unsigned WARPS_PER_BLOCK = VALS_PER_WARP >> 4;
  constexpr unsigned LOG_VALS_PER_BLOCK = 2 * (LOG_VALS_PER_THREAD + 5) - 4;
  constexpr unsigned VALS_PER_BLOCK = 1 << LOG_VALS_PER_BLOCK;
  constexpr unsigned TILES_PER_BLOCK = VALS_PER_BLOCK >> 4;
  constexpr unsigned EXCHG_REGIONS_PER_BLOCK = TILES_PER_BLOCK >> 1;
  constexpr unsigned MAX_STAGES_THIS_LAUNCH = LOG_VALS_PER_BLOCK - 4;

  __shared__ base_field smem[VALS_PER_BLOCK];

  const unsigned lane_id{threadIdx.x & 31};
  const unsigned warp_id{threadIdx.x >> 5};
  const unsigned log_tile_stride = skip_first_stage ? start_stage - 1 : start_stage;
  const unsigned tile_stride = 1 << log_tile_stride;
  const unsigned log_blocks_per_region = log_tile_stride - 4; // tile size is always 16
  const unsigned block_bfly_region_size = TILES_PER_BLOCK * tile_stride;
  const unsigned block_bfly_region = blockIdx.x >> log_blocks_per_region;
  const unsigned block_exchg_region_offset = block_bfly_region * EXCHG_REGIONS_PER_BLOCK;
  const unsigned block_bfly_region_start = block_bfly_region * block_bfly_region_size;
  const unsigned block_start_in_bfly_region = 16 * (blockIdx.x & ((1 << log_blocks_per_region) - 1));
  const base_field *gmem_in = gmem_inputs_matrix + block_bfly_region_start + block_start_in_bfly_region +
                              NTTS_PER_BLOCK * stride_between_input_arrays * blockIdx.y;
  // annoyingly scrambled, but should be coalesced overall
  const unsigned gmem_out_thread_offset = tile_stride * warp_id + tile_stride * WARPS_PER_BLOCK * (lane_id >> 4)
                                        + 2 * (lane_id & 7) + ((lane_id >> 3) & 1);
  const unsigned gmem_out_offset = block_bfly_region_start + block_start_in_bfly_region + gmem_out_thread_offset;
  base_field *gmem_out = gmem_outputs_matrix + gmem_out_offset + NTTS_PER_BLOCK * stride_between_output_arrays * blockIdx.y;

  auto twiddle_cache = smem + VALS_PER_WARP * warp_id;

  base_field vals[VALS_PER_THREAD];

  load_noninitial_twiddles_warp<LOG_VALS_PER_THREAD>(twiddle_cache, lane_id, warp_id, block_exchg_region_offset, inverse);

  const unsigned bound = std::min(NTTS_PER_BLOCK, num_ntts - NTTS_PER_BLOCK * blockIdx.y);
  for (unsigned ntt_idx = 0; ntt_idx < bound;
       ntt_idx++, gmem_in += stride_between_input_arrays, gmem_out += stride_between_output_arrays) {
    if (skip_first_stage) {
      auto val0_addr = gmem_in + TILES_PER_WARP * tile_stride * warp_id + 2 * tile_stride * (lane_id >> 4) + 2 * (threadIdx.x & 7) + (lane_id >> 3 & 1);
#pragma unroll
      for (unsigned i = 0; i < PAIRS_PER_THREAD; i++) {
        vals[2 * i] = memory::load_cs(val0_addr);
        vals[2 * i + 1] = memory::load_cs(val0_addr + tile_stride);
        val0_addr += 4 * tile_stride;
      }
    } else {
      auto pair_addr = gmem_in + TILES_PER_WARP * tile_stride * warp_id + tile_stride * (lane_id >> 3) + 2 * (threadIdx.x & 7);
#pragma unroll
      for (unsigned i = 0; i < PAIRS_PER_THREAD; i++) {
        const auto pair = memory::load_cs(reinterpret_cast<const uint4*>(pair_addr));
        vals[2 * i][0] = pair.x;
        vals[2 * i][1] = pair.y;
        vals[2 * i + 1][0] = pair.z;
        vals[2 * i + 1][1] = pair.w;
        pair_addr += 4 * tile_stride;
      }
    }

    unsigned lane_mask = 8;
    base_field *twiddles_this_stage = twiddle_cache;
    unsigned num_twiddles_this_stage = 1 << LOG_VALS_PER_THREAD;
    for (unsigned s = 4; s < LOG_VALS_PER_THREAD + 3; s++) {
      if (!skip_first_stage || s > 4) {
#pragma unroll
        for (unsigned i = 0; i < PAIRS_PER_THREAD; i++) {
          const auto twiddle = twiddles_this_stage[(32 * i + lane_id) >> s];
          shfl_xor_bf(vals, i, lane_id, lane_mask);
          exchg_dif(vals[2 * i], vals[2 * i + 1], twiddle);
        }
      }
      lane_mask <<= 1;
      twiddles_this_stage += num_twiddles_this_stage;
      num_twiddles_this_stage >>= 1;
    }

    for (unsigned i = 1; i < LOG_VALS_PER_THREAD; i++) {
#pragma unroll
      for (unsigned j = 0; j < PAIRS_PER_THREAD >> i; j++) {
        const unsigned exchg_tile_sz = 2 << i;
        const unsigned half_exchg_tile_sz = 1 << i;
        const auto twiddle = twiddles_this_stage[j];
#pragma unroll
        for (unsigned k = 0; k < half_exchg_tile_sz; k++)
          exchg_dif(vals[exchg_tile_sz * j + k], vals[exchg_tile_sz * j + k + half_exchg_tile_sz], twiddle);
      }
      twiddles_this_stage += num_twiddles_this_stage;
      num_twiddles_this_stage >>= 1;
    }

    __syncwarp();

    // there are at most 31 per-warp twiddles, so we only need 1 temporary per thread to stash them
    base_field tmp{};
    if (ntt_idx < num_ntts - 1)
      tmp = twiddle_cache[lane_id];

    // annoyingly scrambled but should be bank-conflict-free
    const unsigned smem_thread_offset = 16 * (lane_id >> 4) + 2 * (lane_id & 7) + ((lane_id >> 3) & 1);
#pragma unroll
    for (unsigned i = 0; i < PAIRS_PER_THREAD; i++) {
      twiddle_cache[64 * i + smem_thread_offset] = vals[2 * i];
      twiddle_cache[64 * i + smem_thread_offset + 32] = vals[2 * i + 1];
    }

    __syncthreads();

    auto smem_pair_addr = smem + 16 * warp_id + VALS_PER_WARP * (lane_id >> 3) + 2 * (threadIdx.x & 7);
#pragma unroll
    for (unsigned i = 0; i < PAIRS_PER_THREAD; i++, smem_pair_addr += 4 * VALS_PER_WARP) {
      const auto pair = *reinterpret_cast<const uint4*>(smem_pair_addr);
      vals[2 * i][0] = pair.x;
      vals[2 * i][1] = pair.y;
      vals[2 * i + 1][0] = pair.z;
      vals[2 * i + 1][1] = pair.w;
    }

    const bool is_last_kernel = (start_stage + MAX_STAGES_THIS_LAUNCH - skip_first_stage == log_n);
    if ((ntt_idx < num_ntts - 1) || (is_last_kernel && log_extension_degree && inverse)) {
      __syncthreads();
      twiddle_cache[lane_id] = tmp;
      __syncwarp(); // maybe unnecessary due to shfls below
      // __syncthreads();
    }


    lane_mask = 8;
    unsigned exchg_region_offset = (block_exchg_region_offset >> (LOG_VALS_PER_THREAD + 1)) + (lane_id >> 4);
    for (unsigned s = 0; s < 2; s++) {
#pragma unroll
      for (unsigned i = 0; i < PAIRS_PER_THREAD; i++) {
        // TODO: Handle these cooperatively?
        const auto twiddle = get_twiddle(inverse, exchg_region_offset + ((2 * i) >> s));
        shfl_xor_bf(vals, i, lane_id, lane_mask);
        exchg_dif(vals[2 * i], vals[2 * i + 1], twiddle);
      }
      lane_mask <<= 1;
      exchg_region_offset >>= 1;
    }

    for (unsigned i = 1; i < LOG_VALS_PER_THREAD; i++) {
#pragma unroll
      for (unsigned j = 0; j < PAIRS_PER_THREAD >> i; j++) {
        const unsigned exchg_tile_sz = 2 << i;
        const unsigned half_exchg_tile_sz = 1 << i;
        const auto twiddle = get_twiddle(inverse, exchg_region_offset + (j >> (i - 1)));
#pragma unroll
        for (unsigned k = 0; k < half_exchg_tile_sz; k++)
          exchg_dif(vals[exchg_tile_sz * j + k], vals[exchg_tile_sz * j + k + half_exchg_tile_sz], twiddle);
      }
      exchg_region_offset >>= 1;
    }

    if (inverse && is_last_kernel) {
#pragma unroll
      for (unsigned i = 0; i < VALS_PER_THREAD; i++)
        vals[i] = base_field::mul(vals[i], inv_sizes[log_n]);
    }

    if (inverse && is_last_kernel && log_extension_degree) {
      __syncwarp();
      base_field *scratch = twiddle_cache + lane_id;
      base_field tmp = *scratch;
#pragma unroll
      for (unsigned i = 0; i < VALS_PER_THREAD; i++)
        scratch[32 * i] = vals[i];
#pragma unroll 1
      for (unsigned i = 0; i < PAIRS_PER_THREAD; i++) {
        base_field val0 = scratch[64 * i];
        base_field val1 = scratch[64 * i + 32];
        const unsigned idx0 = gmem_out_offset + 4 * i * tile_stride * WARPS_PER_BLOCK;
        const unsigned idx1 = gmem_out_offset + (4 * i  + 2) * tile_stride * WARPS_PER_BLOCK;
        if (coset_idx) {
          const unsigned shift = OMEGA_LOG_ORDER - log_n - log_extension_degree;
          const unsigned offset = coset_idx << shift;
          auto power_of_w0 = get_power_of_w(idx0 * offset, true);
          auto power_of_w1 = get_power_of_w(idx1 * offset, true);
          val0 = base_field::mul(val0, power_of_w0);
          val1 = base_field::mul(val1, power_of_w1);
        }
        auto power_of_g0 = get_power_of_g(idx0, true);
        auto power_of_g1 = get_power_of_g(idx1, true);
        memory::store_cs(gmem_out - gmem_out_offset + idx0, base_field::mul(val0, power_of_g0));
        memory::store_cs(gmem_out - gmem_out_offset + idx1, base_field::mul(val1, power_of_g1));
      }
      *scratch = tmp;
      __syncwarp();
    } else {
#pragma unroll
      for (unsigned i = 0; i < PAIRS_PER_THREAD; i++) {
        memory::store_cs(gmem_out + 4 * i * tile_stride * WARPS_PER_BLOCK, vals[2 * i]);
        memory::store_cs(gmem_out + (4 * i + 2) * tile_stride * WARPS_PER_BLOCK, vals[2 * i + 1]);
      }
    }
  }
}

extern "C" __launch_bounds__(512, 2) __global__
void b2n_noninitial_7_or_8_stages_block(const base_field *gmem_inputs_matrix, base_field *gmem_outputs_matrix, const unsigned stride_between_input_arrays,
                                   const unsigned stride_between_output_arrays, const unsigned start_stage, const unsigned stages_this_launch,
                                   const unsigned log_n, const bool inverse, const unsigned num_ntts, const unsigned log_extension_degree,
                                   const unsigned coset_idx) {
  b2n_noninitial_stages_block<3>(gmem_inputs_matrix, gmem_outputs_matrix, stride_between_input_arrays, stride_between_output_arrays, start_stage,
                                 stages_this_launch == 7, log_n, inverse, num_ntts, log_extension_degree, coset_idx);
}

// Simple, non-optimized kernel used for log_n < 16, to unblock debugging small proofs.
extern "C" __launch_bounds__(512, 2) __global__
    void b2n_1_stage(const base_field *gmem_inputs_matrix, base_field *gmem_outputs_matrix, const unsigned stride_between_input_arrays,
                     const unsigned stride_between_output_arrays, const unsigned start_stage, const unsigned stages_this_launch, const unsigned log_n,
                     const bool inverse, const unsigned blocks_per_ntt, const unsigned log_extension_degree, const unsigned coset_idx) {
  const unsigned ntt_idx = blockIdx.x / blocks_per_ntt;
  const unsigned bid_in_ntt = blockIdx.x - ntt_idx * blocks_per_ntt;
  const unsigned tid_in_ntt = threadIdx.x + bid_in_ntt * blockDim.x;
  if (tid_in_ntt >= (1 << (log_n - 1)))
    return;
  const unsigned log_exchg_region_sz = start_stage + 1;
  const unsigned exchg_region = tid_in_ntt >> (log_exchg_region_sz - 1);
  const unsigned tid_in_exchg_region = tid_in_ntt - (exchg_region << (log_exchg_region_sz - 1));
  const unsigned exchg_stride = 1 << (log_exchg_region_sz - 1);
  const unsigned a_idx = tid_in_exchg_region + exchg_region * (1 << log_exchg_region_sz);
  const unsigned b_idx = a_idx + exchg_stride;
  const base_field *gmem_input = gmem_inputs_matrix + ntt_idx * stride_between_input_arrays;
  base_field *gmem_output = gmem_outputs_matrix + ntt_idx * stride_between_output_arrays;

  const auto twiddle = get_twiddle(inverse, exchg_region);
  auto a = memory::load_cs(gmem_input + a_idx);
  auto b = memory::load_cs(gmem_input + b_idx);

  if ((start_stage == 0) && log_extension_degree && !inverse) {
    const unsigned a_idx_brev = __brev(a_idx) >> (32 - log_n);
    const unsigned b_idx_brev = __brev(b_idx) >> (32 - log_n);
    if (coset_idx) {
      const unsigned shift = OMEGA_LOG_ORDER - log_n - log_extension_degree;
      const unsigned offset = coset_idx << shift;
      a = base_field::mul(a, get_power_of_w(a_idx_brev * offset, false));
      b = base_field::mul(b, get_power_of_w(b_idx_brev * offset, false));
    }
    a = base_field::mul(a, get_power_of_g(a_idx_brev, false));
    b = base_field::mul(b, get_power_of_g(b_idx_brev, false));
  }

  exchg_dif(a, b, twiddle);

  if (inverse && (start_stage + stages_this_launch == log_n)) {
    a = base_field::mul(a, inv_sizes[log_n]);
    b = base_field::mul(b, inv_sizes[log_n]);
    if (log_extension_degree) {
      if (coset_idx) {
        const unsigned shift = OMEGA_LOG_ORDER - log_n - log_extension_degree;
        const unsigned offset = coset_idx << shift;
        a = base_field::mul(a, get_power_of_w(a_idx * offset, true));
        b = base_field::mul(b, get_power_of_w(b_idx * offset, true));
      }
      a = base_field::mul(a, get_power_of_g(a_idx, true));
      b = base_field::mul(b, get_power_of_g(b_idx, true));
    }
  }

  memory::store_cs(gmem_output + a_idx, a);
  memory::store_cs(gmem_output + b_idx, b);
}
