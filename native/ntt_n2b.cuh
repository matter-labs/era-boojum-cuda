#pragma once // also, this file should only be compiled in one compile unit because it has __global__ definitions

// This kernel basically reverses the pattern of the b2n_initial_stages_warp kernel.
template <unsigned LOG_VALS_PER_THREAD> DEVICE_FORCEINLINE
void n2b_final_stages_warp(const base_field *gmem_inputs_matrix, base_field *gmem_outputs_matrix, const unsigned stride_between_input_arrays,
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
      vals[2 * i] = memory::load_cs(gmem_in + 64 * i + lane_id);
      vals[2 * i + 1] = memory::load_cs(gmem_in + 64 * i + lane_id + 32);
    }

    base_field *twiddles_this_stage = twiddle_cache + VALS_PER_WARP - 2;
    unsigned num_twiddles_this_stage = 1;
    for (unsigned i = 0; i < LOG_VALS_PER_THREAD - 1; i++) {
#pragma unroll
      for (unsigned j = 0; j < (1 << i); j++) {
        const unsigned exchg_tile_sz = VALS_PER_THREAD >> i;
        const unsigned half_exchg_tile_sz = exchg_tile_sz >> 1;
        const auto twiddle = twiddles_this_stage[j];
#pragma unroll
        for (unsigned k = 0; k < half_exchg_tile_sz; k++) {
          exchg_dit(vals[exchg_tile_sz * j + k], vals[exchg_tile_sz * j + k + half_exchg_tile_sz], twiddle);
        }
      }
      num_twiddles_this_stage <<= 1;
      twiddles_this_stage -= num_twiddles_this_stage;
    }

    unsigned lane_mask = 16;
    for (unsigned stage = 0, s = 5; stage < 6; stage++, s--) {
#pragma unroll
      for (unsigned i = 0; i < PAIRS_PER_THREAD; i++) {
        const auto twiddle = twiddles_this_stage[(32 * i + lane_id) >> s];
        exchg_dit(vals[2 * i], vals[2 * i + 1], twiddle);
        if (stage < 5)
          shfl_xor_bf(vals, i, lane_id, lane_mask);
      }
      lane_mask >>= 1;
      num_twiddles_this_stage <<= 1;
      twiddles_this_stage -= num_twiddles_this_stage;
    }

    if (inverse) {
#pragma unroll
      for (unsigned i = 0; i < VALS_PER_THREAD; i++)
        vals[i] = base_field::mul(vals[i], inv_sizes[log_n]);
    }

    if (inverse && log_extension_degree) {
      __syncwarp();
      base_field tmp[VALS_PER_THREAD];
      base_field *scratch = twiddle_cache + VALS_PER_THREAD * lane_id;
#pragma unroll
      for (unsigned i = 0; i < VALS_PER_THREAD; i++) {
        tmp[i] = scratch[i];
        scratch[i] = vals[i];
      }
      apply_lde_factors<VALS_PER_THREAD, true>(scratch, gmem_offset, lane_id, log_n, log_extension_degree, coset_idx);
#pragma unroll
      for (unsigned i = 0; i < PAIRS_PER_THREAD; i++) {
        const uint4 out{scratch[2 * i][0], scratch[2 * i][1], scratch[2 * i + 1][0], scratch[2 * i + 1][1]};
        memory::store_cs(reinterpret_cast<uint4*>(gmem_out + 64 * i + 2 * lane_id), out);
      }
#pragma unroll
      for (unsigned i = 0; i < VALS_PER_THREAD; i++)
        scratch[i] = tmp[i];
      __syncwarp();
    } else {
#pragma unroll
      for (unsigned i = 0; i < PAIRS_PER_THREAD; i++) {
        const uint4 out{vals[2 * i][0], vals[2 * i][1], vals[2 * i + 1][0], vals[2 * i + 1][1]};
        memory::store_cs(reinterpret_cast<uint4*>(gmem_out + 64 * i + 2 * lane_id), out);
      }
    }
  }
}

extern "C" __launch_bounds__(128, 8) __global__
void n2b_final_8_stages_warp(const base_field *gmem_inputs_matrix, base_field *gmem_outputs_matrix, const unsigned stride_between_input_arrays,
                             const unsigned stride_between_output_arrays, const unsigned start_stage, const unsigned stages_this_launch,
                             const unsigned log_n, const bool inverse, const unsigned num_ntts, const unsigned log_extension_degree,
                             const unsigned coset_idx) {
  n2b_final_stages_warp<3>(gmem_inputs_matrix, gmem_outputs_matrix, stride_between_input_arrays, stride_between_output_arrays, start_stage,
                           stages_this_launch, log_n, inverse, num_ntts, log_extension_degree, coset_idx);
}

extern "C" __launch_bounds__(128, 8) __global__
void n2b_final_7_stages_warp(const base_field *gmem_inputs_matrix, base_field *gmem_outputs_matrix, const unsigned stride_between_input_arrays,
                             const unsigned stride_between_output_arrays, const unsigned start_stage, const unsigned stages_this_launch,
                             const unsigned log_n, const bool inverse, const unsigned num_ntts, const unsigned log_extension_degree,
                             const unsigned coset_idx) {
  n2b_final_stages_warp<2>(gmem_inputs_matrix, gmem_outputs_matrix, stride_between_input_arrays, stride_between_output_arrays, start_stage,
                           stages_this_launch, log_n, inverse, num_ntts, log_extension_degree, coset_idx);
}

// This kernel basically reverses the pattern of the b2n_initial_stages_block kernel.
template <unsigned LOG_VALS_PER_THREAD> DEVICE_FORCEINLINE
void n2b_final_stages_block(const base_field *gmem_inputs_matrix, base_field *gmem_outputs_matrix, const unsigned stride_between_input_arrays,
                            const unsigned stride_between_output_arrays, const unsigned start_stage, const unsigned stages_this_launch,
                            const unsigned log_n, const bool inverse, const unsigned num_ntts, const unsigned log_extension_degree,
                            const unsigned coset_idx) {
  constexpr unsigned VALS_PER_THREAD = 1 << LOG_VALS_PER_THREAD;
  constexpr unsigned PAIRS_PER_THREAD = VALS_PER_THREAD >> 1;
  constexpr unsigned VALS_PER_WARP = 32 * VALS_PER_THREAD;
  constexpr unsigned WARPS_PER_BLOCK = VALS_PER_WARP >> 4;
  constexpr unsigned VALS_PER_BLOCK = 32 * VALS_PER_THREAD * WARPS_PER_BLOCK;
  constexpr unsigned MAX_STAGES_THIS_LAUNCH = 2 * (LOG_VALS_PER_THREAD + 5) - 4;

  __shared__ base_field smem[VALS_PER_BLOCK];

  const unsigned lane_id{threadIdx.x & 31};
  const unsigned warp_id{threadIdx.x >> 5};
  const unsigned gmem_block_offset = VALS_PER_BLOCK * blockIdx.x;
  const unsigned gmem_offset = gmem_block_offset + VALS_PER_WARP * warp_id;
  // annoyingly scrambled, but should be coalesced overall
  const unsigned gmem_in_thread_offset = 16 * warp_id + VALS_PER_WARP * (lane_id >> 4) + 2 * (lane_id & 7) + ((lane_id >> 3) & 1);
  const base_field *gmem_in = gmem_inputs_matrix + gmem_block_offset + gmem_in_thread_offset +
                              NTTS_PER_BLOCK * stride_between_input_arrays * blockIdx.y;
  base_field *gmem_out = gmem_outputs_matrix + gmem_offset + NTTS_PER_BLOCK * stride_between_output_arrays * blockIdx.y;

  auto twiddle_cache = smem + VALS_PER_WARP * warp_id;

  base_field vals[VALS_PER_THREAD];

  const unsigned bound = std::min(NTTS_PER_BLOCK, num_ntts - NTTS_PER_BLOCK * blockIdx.y);
  for (unsigned ntt_idx = 0; ntt_idx < bound;
       ntt_idx++, gmem_in += stride_between_input_arrays, gmem_out += stride_between_output_arrays) {
#pragma unroll
    for (unsigned i = 0; i < PAIRS_PER_THREAD; i++) {
      vals[2 * i] = memory::load_cs(gmem_in + 4 * i * VALS_PER_WARP);
      vals[2 * i + 1] = memory::load_cs(gmem_in + (4 * i + 2) * VALS_PER_WARP);
    }

    const unsigned stages_to_skip = MAX_STAGES_THIS_LAUNCH - stages_this_launch;
    unsigned exchg_region_offset = blockIdx.x;
    for (unsigned i = 0; i < LOG_VALS_PER_THREAD - 1; i++) {
      if (i >= stages_to_skip) {
#pragma unroll
        for (unsigned j = 0; j < (1 << i); j++) {
          const unsigned exchg_tile_sz = VALS_PER_THREAD >> i;
          const unsigned half_exchg_tile_sz = exchg_tile_sz >> 1;
          const auto twiddle = get_twiddle(inverse, exchg_region_offset + j);
#pragma unroll
          for (unsigned k = 0; k < half_exchg_tile_sz; k++)
            exchg_dit(vals[exchg_tile_sz * j + k], vals[exchg_tile_sz * j + k + half_exchg_tile_sz], twiddle);
        }
      }
      exchg_region_offset <<= 1;
    }

    unsigned lane_mask = 16;
    unsigned halfwarp_id = lane_id >> 4;
    for (unsigned s = 0; s < 2; s++) {
      if ((s + LOG_VALS_PER_THREAD - 1) >= stages_to_skip) {
#pragma unroll
        for (unsigned i = 0; i < PAIRS_PER_THREAD; i++) {
          // TODO: Handle these cooperatively?
          const auto twiddle = get_twiddle(inverse, exchg_region_offset + ((2 * i + halfwarp_id) >> (1 - s)));
          exchg_dit(vals[2 * i], vals[2 * i + 1], twiddle);
          shfl_xor_bf(vals, i, lane_id, lane_mask);
        }
      } else {
#pragma unroll
        for (unsigned i = 0; i < PAIRS_PER_THREAD; i++)
          shfl_xor_bf(vals, i, lane_id, lane_mask);
      }
      lane_mask >>= 1;
      exchg_region_offset <<= 1;
    }

    __syncwarp(); // maybe unnecessary but can't hurt

    {
      base_field tmp[VALS_PER_THREAD];
      auto pair_addr = smem + 16 * warp_id + VALS_PER_WARP * (lane_id >> 3) + 2 * (threadIdx.x & 7);
      if (ntt_idx > 0) {
#pragma unroll
        for (unsigned i = 0; i < PAIRS_PER_THREAD; i++) {
          tmp[2 * i] = twiddle_cache[64 * i + lane_id];
          tmp[2 * i + 1] = twiddle_cache[64 * i + lane_id + 32];
        }

        __syncthreads();

#pragma unroll
        for (unsigned i = 0; i < PAIRS_PER_THREAD; i++, pair_addr += 4 * VALS_PER_WARP) {
          uint4* pair = reinterpret_cast<uint4*>(pair_addr);
          const uint4 out{vals[2 * i][0], vals[2 * i][1], vals[2 * i + 1][0], vals[2 * i + 1][1]};
          *pair = out;
        }
      } else {
#pragma unroll
        for (unsigned i = 0; i < PAIRS_PER_THREAD; i++, pair_addr += 4 * VALS_PER_WARP) {
          uint4* pair = reinterpret_cast<uint4*>(pair_addr);
          const uint4 out{vals[2 * i][0], vals[2 * i][1], vals[2 * i + 1][0], vals[2 * i + 1][1]};
          *pair = out;
        }
      }

      __syncthreads();

      if (ntt_idx > 0) {
#pragma unroll
        for (unsigned i = 0; i < PAIRS_PER_THREAD; i++) {
          vals[2 * i] = twiddle_cache[64 * i + lane_id];
          vals[2 * i + 1] = twiddle_cache[64 * i + lane_id + 32];
          twiddle_cache[64 * i + lane_id] = tmp[2 * i];
          twiddle_cache[64 * i + lane_id + 32] = tmp[2 * i + 1];
        }

        __syncwarp();
      } else {
#pragma unroll
        for (unsigned i = 0; i < PAIRS_PER_THREAD; i++) {
          vals[2 * i] = twiddle_cache[64 * i + lane_id];
          vals[2 * i + 1] = twiddle_cache[64 * i + lane_id + 32];
        }

         __syncwarp();

        load_initial_twiddles_warp<VALS_PER_WARP, LOG_VALS_PER_THREAD>(twiddle_cache, lane_id, gmem_offset, inverse);
      }
    }

    base_field *twiddles_this_stage = twiddle_cache + VALS_PER_WARP - 2;
    unsigned num_twiddles_this_stage = 1;
    for (unsigned i = 0; i < LOG_VALS_PER_THREAD - 1; i++) {
#pragma unroll
      for (unsigned j = 0; j < (1 << i); j++) {
        const unsigned exchg_tile_sz = VALS_PER_THREAD >> i;
        const unsigned half_exchg_tile_sz = exchg_tile_sz >> 1;
        const auto twiddle = twiddles_this_stage[j];
#pragma unroll
        for (unsigned k = 0; k < half_exchg_tile_sz; k++) {
          exchg_dit(vals[exchg_tile_sz * j + k], vals[exchg_tile_sz * j + k + half_exchg_tile_sz], twiddle);
        }
      }
      num_twiddles_this_stage <<= 1;
      twiddles_this_stage -= num_twiddles_this_stage;
    }

    lane_mask = 16;
    for (unsigned stage = 0, s = 5; stage < 6; stage++, s--) {
#pragma unroll
      for (unsigned i = 0; i < PAIRS_PER_THREAD; i++) {
        const auto twiddle = twiddles_this_stage[(32 * i + lane_id) >> s];
        exchg_dit(vals[2 * i], vals[2 * i + 1], twiddle);
        if (stage < 5)
          shfl_xor_bf(vals, i, lane_id, lane_mask);
      }
      lane_mask >>= 1;
      num_twiddles_this_stage <<= 1;
      twiddles_this_stage -= num_twiddles_this_stage;
    }

    if (inverse) {
#pragma unroll
      for (unsigned i = 0; i < VALS_PER_THREAD; i++)
        vals[i] = base_field::mul(vals[i], inv_sizes[log_n]);
    }

    if (inverse && log_extension_degree) {
      __syncwarp();
      base_field tmp[VALS_PER_THREAD];
      base_field *scratch = twiddle_cache + VALS_PER_THREAD * lane_id;
#pragma unroll
      for (unsigned i = 0; i < VALS_PER_THREAD; i++) {
        tmp[i] = scratch[i];
        scratch[i] = vals[i];
      }
      apply_lde_factors<VALS_PER_THREAD, true>(scratch, gmem_offset, lane_id, log_n, log_extension_degree, coset_idx);
#pragma unroll
      for (unsigned i = 0; i < PAIRS_PER_THREAD; i++) {
        const uint4 out{scratch[2 * i][0], scratch[2 * i][1], scratch[2 * i + 1][0], scratch[2 * i + 1][1]};
        memory::store_cs(reinterpret_cast<uint4*>(gmem_out + 64 * i + 2 * lane_id), out);
      }
#pragma unroll
      for (unsigned i = 0; i < VALS_PER_THREAD; i++)
        scratch[i] = tmp[i];
      __syncwarp();
    } else {
#pragma unroll
      for (unsigned i = 0; i < PAIRS_PER_THREAD; i++) {
        const uint4 out{vals[2 * i][0], vals[2 * i][1], vals[2 * i + 1][0], vals[2 * i + 1][1]};
        memory::store_cs(reinterpret_cast<uint4*>(gmem_out + 64 * i + 2 * lane_id), out);
      }
    }
  }
}

extern "C" __launch_bounds__(512, 2) __global__
void n2b_final_9_to_12_stages_block(const base_field *gmem_inputs_matrix, base_field *gmem_outputs_matrix, const unsigned stride_between_input_arrays,
                                    const unsigned stride_between_output_arrays, const unsigned start_stage, const unsigned stages_this_launch,
                                    const unsigned log_n, const bool inverse, const unsigned num_ntts, const unsigned log_extension_degree,
                                    const unsigned coset_idx) {
  n2b_final_stages_block<3>(gmem_inputs_matrix, gmem_outputs_matrix, stride_between_input_arrays, stride_between_output_arrays, start_stage,
                            stages_this_launch, log_n, inverse, num_ntts, log_extension_degree, coset_idx);
}

// This kernel basically reverses the pattern of the b2n_noninitial_stages_block kernel.
template <unsigned LOG_VALS_PER_THREAD> DEVICE_FORCEINLINE
void n2b_nonfinal_stages_block(const base_field *gmem_inputs_matrix, base_field *gmem_outputs_matrix, const unsigned stride_between_input_arrays,
                            const unsigned stride_between_output_arrays, const unsigned start_stage, const bool skip_last_stage,
                            const unsigned log_n, const bool inverse, const unsigned num_ntts, const unsigned log_extension_degree,
                            const unsigned coset_idx) {
  constexpr unsigned VALS_PER_THREAD = 1 << LOG_VALS_PER_THREAD;
  constexpr unsigned PAIRS_PER_THREAD = VALS_PER_THREAD >> 1;
  constexpr unsigned VALS_PER_WARP = 32 * VALS_PER_THREAD;
  constexpr unsigned TILES_PER_WARP = VALS_PER_WARP >> 4;
  constexpr unsigned WARPS_PER_BLOCK = VALS_PER_WARP >> 4;
  constexpr unsigned VALS_PER_BLOCK = VALS_PER_WARP * WARPS_PER_BLOCK;
  constexpr unsigned TILES_PER_BLOCK = VALS_PER_BLOCK >> 4;
  constexpr unsigned EXCHG_REGIONS_PER_BLOCK = TILES_PER_BLOCK >> 1;
  constexpr unsigned MAX_STAGES_THIS_LAUNCH = 2 * (LOG_VALS_PER_THREAD + 5) - 8;

  __shared__ base_field smem[VALS_PER_BLOCK];

  const unsigned lane_id{threadIdx.x & 31};
  const unsigned warp_id{threadIdx.x >> 5};
  const unsigned log_tile_stride = log_n - start_stage - MAX_STAGES_THIS_LAUNCH;
  const unsigned tile_stride = 1 << log_tile_stride;
  const unsigned log_blocks_per_region = log_tile_stride - 4; // tile size is always 16
  const unsigned block_bfly_region_size = TILES_PER_BLOCK * tile_stride;
  const unsigned block_bfly_region = blockIdx.x >> log_blocks_per_region;
  const unsigned block_bfly_region_start = block_bfly_region * block_bfly_region_size;
  const unsigned block_start_in_bfly_region = 16 * (blockIdx.x & ((1 << log_blocks_per_region) - 1));
  // annoyingly scrambled, but should be coalesced overall
  const unsigned gmem_in_thread_offset = tile_stride * warp_id + tile_stride * WARPS_PER_BLOCK * (lane_id >> 4)
                                       + 2 * (lane_id & 7) + ((lane_id >> 3) & 1);
  const unsigned gmem_in_offset = block_bfly_region_start + block_start_in_bfly_region + gmem_in_thread_offset;
  const base_field *gmem_in = gmem_inputs_matrix + gmem_in_offset + NTTS_PER_BLOCK * stride_between_input_arrays * blockIdx.y;
  base_field *gmem_out = gmem_outputs_matrix + block_bfly_region_start + block_start_in_bfly_region +
                         NTTS_PER_BLOCK * stride_between_output_arrays * blockIdx.y;

  auto twiddle_cache = smem + VALS_PER_WARP * warp_id;

  base_field vals[VALS_PER_THREAD];

  const unsigned bound = std::min(NTTS_PER_BLOCK, num_ntts - NTTS_PER_BLOCK * blockIdx.y);
  for (unsigned ntt_idx = 0; ntt_idx < bound;
       ntt_idx++, gmem_in += stride_between_input_arrays, gmem_out += stride_between_output_arrays) {
#pragma unroll
    for (unsigned i = 0; i < PAIRS_PER_THREAD; i++) {
      vals[2 * i] = memory::load_cs(gmem_in + 4 * i * tile_stride * WARPS_PER_BLOCK);
      vals[2 * i + 1] = memory::load_cs(gmem_in + (4 * i + 2) * tile_stride * WARPS_PER_BLOCK);
    }

    if ((start_stage == 0) && log_extension_degree && !inverse) {
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
        const unsigned idx0 = gmem_in_offset + 4 * i * tile_stride * WARPS_PER_BLOCK;
        const unsigned idx1 = gmem_in_offset + (4 * i  + 2) * tile_stride * WARPS_PER_BLOCK;
        if (coset_idx) {
          const unsigned shift = OMEGA_LOG_ORDER - log_n - log_extension_degree;
          const unsigned offset = coset_idx << shift;
          auto power_of_w0 = get_power_of_w(idx0 * offset, false);
          auto power_of_w1 = get_power_of_w(idx1 * offset, false);
          val0 = base_field::mul(val0, power_of_w0);
          val1 = base_field::mul(val1, power_of_w1);
        }
        auto power_of_g0 = get_power_of_g(idx0, false);
        auto power_of_g1 = get_power_of_g(idx1, false);
        scratch[64 * i] = base_field::mul(val0, power_of_g0);
        scratch[64 * i + 32] = base_field::mul(val1, power_of_g1);
      }
#pragma unroll
      for (unsigned i = 0; i < VALS_PER_THREAD; i++)
        vals[i] = scratch[32 * i];
      *scratch = tmp;
      __syncwarp();
    }

    unsigned block_exchg_region_offset = block_bfly_region;
    for (unsigned i = 0; i < LOG_VALS_PER_THREAD - 1; i++) {
#pragma unroll
      for (unsigned j = 0; j < (1 << i); j++) {
        const unsigned exchg_tile_sz = VALS_PER_THREAD >> i;
        const unsigned half_exchg_tile_sz = exchg_tile_sz >> 1;
        const auto twiddle = get_twiddle(inverse, block_exchg_region_offset + j);
#pragma unroll
        for (unsigned k = 0; k < half_exchg_tile_sz; k++)
          exchg_dit(vals[exchg_tile_sz * j + k], vals[exchg_tile_sz * j + k + half_exchg_tile_sz], twiddle);
      }
      block_exchg_region_offset <<= 1;
    }

    unsigned lane_mask = 16;
    unsigned halfwarp_id = lane_id >> 4;
    for (unsigned s = 0; s < 2; s++) {
#pragma unroll
      for (unsigned i = 0; i < PAIRS_PER_THREAD; i++) {
        // TODO: Handle these cooperatively?
        const auto twiddle = get_twiddle(inverse, block_exchg_region_offset + ((2 * i + halfwarp_id) >> (1 - s)));
        exchg_dit(vals[2 * i], vals[2 * i + 1], twiddle);
        shfl_xor_bf(vals, i, lane_id, lane_mask);
      }
      lane_mask >>= 1;
      block_exchg_region_offset <<= 1;
    }

    __syncwarp(); // maybe unnecessary but can't hurt

    // there are at most 31 per-warp twiddles, so we only need 1 temporary per thread to stash them
    base_field tmp;
    if ((ntt_idx > 0) || ((start_stage == 0) && log_extension_degree && !inverse)) {
      tmp = twiddle_cache[lane_id];
      __syncthreads();
    }

    auto smem_pair_addr = smem + 16 * warp_id + VALS_PER_WARP * (lane_id >> 3) + 2 * (threadIdx.x & 7);
#pragma unroll
    for (unsigned i = 0; i < PAIRS_PER_THREAD; i++, smem_pair_addr += 4 * VALS_PER_WARP) {
      uint4* pair = reinterpret_cast<uint4*>(smem_pair_addr);
      const uint4 out{vals[2 * i][0], vals[2 * i][1], vals[2 * i + 1][0], vals[2 * i + 1][1]};
      *pair = out;
    }

    __syncthreads();

    // annoyingly scrambled but should be bank-conflict-free
    const unsigned smem_thread_offset = 16 * (lane_id >> 4) + 2 * (lane_id & 7) + ((lane_id >> 3) & 1);
#pragma unroll
    for (unsigned i = 0; i < PAIRS_PER_THREAD; i++) {
      vals[2 * i] = twiddle_cache[64 * i + smem_thread_offset];
      vals[2 * i + 1] = twiddle_cache[64 * i + smem_thread_offset + 32];
    }

    __syncwarp();

    if (ntt_idx > 0) {
      twiddle_cache[lane_id] = tmp;
      __syncwarp();
    } else {
      load_noninitial_twiddles_warp<LOG_VALS_PER_THREAD>(twiddle_cache, lane_id, warp_id,
                                                         block_bfly_region * EXCHG_REGIONS_PER_BLOCK, inverse);
    }

    base_field *twiddles_this_stage = twiddle_cache + 2 * VALS_PER_THREAD - 2;
    unsigned num_twiddles_this_stage = 1;
    for (unsigned i = 0; i < LOG_VALS_PER_THREAD - 1; i++) {
#pragma unroll
      for (unsigned j = 0; j < (1 << i); j++) {
        const unsigned exchg_tile_sz = VALS_PER_THREAD >> i;
        const unsigned half_exchg_tile_sz = exchg_tile_sz >> 1;
        const auto twiddle = twiddles_this_stage[j];
#pragma unroll
        for (unsigned k = 0; k < half_exchg_tile_sz; k++) {
          exchg_dit(vals[exchg_tile_sz * j + k], vals[exchg_tile_sz * j + k + half_exchg_tile_sz], twiddle);
        }
      }
      num_twiddles_this_stage <<= 1;
      twiddles_this_stage -= num_twiddles_this_stage;
    }

    lane_mask = 16;
    for (unsigned s = 0; s < 2; s++) {
      if (!skip_last_stage || s < 1) {
#pragma unroll
        for (unsigned i = 0; i < PAIRS_PER_THREAD; i++) {
          // TODO: Handle these cooperatively?
          const auto twiddle = twiddles_this_stage[(2 * i + halfwarp_id) >> (1 - s)];
          exchg_dit(vals[2 * i], vals[2 * i + 1], twiddle);
          shfl_xor_bf(vals, i, lane_id, lane_mask);
        }
        lane_mask >>= 1;
        num_twiddles_this_stage <<= 1;
        twiddles_this_stage -= num_twiddles_this_stage;
      }
    }

    if (skip_last_stage) {
      auto val0_addr = gmem_out + TILES_PER_WARP * tile_stride * warp_id + 2 * tile_stride * (lane_id >> 4) + 2 * (threadIdx.x & 7) + (lane_id >> 3 & 1);
#pragma unroll
      for (unsigned i = 0; i < PAIRS_PER_THREAD; i++) {
        memory::store_cs(val0_addr, vals[2 * i]);
        memory::store_cs(val0_addr + tile_stride, vals[2 * i + 1]);
        val0_addr += 4 * tile_stride;
      }
    } else {
      auto pair_addr = gmem_out + TILES_PER_WARP * tile_stride * warp_id + tile_stride * (lane_id >> 3) + 2 * (threadIdx.x & 7);
#pragma unroll
      for (unsigned i = 0; i < PAIRS_PER_THREAD; i++) {
        const uint4 out{vals[2 * i][0], vals[2 * i][1], vals[2 * i + 1][0], vals[2 * i + 1][1]};
        memory::store_cs(reinterpret_cast<uint4*>(pair_addr), out);
        pair_addr += 4 * tile_stride;
      }
    }
  }
}

extern "C" __launch_bounds__(512, 2) __global__
void n2b_nonfinal_7_or_8_stages_block(const base_field *gmem_inputs_matrix, base_field *gmem_outputs_matrix, const unsigned stride_between_input_arrays,
                                      const unsigned stride_between_output_arrays, const unsigned start_stage, const unsigned stages_this_launch,
                                      const unsigned log_n, const bool inverse, const unsigned num_ntts, const unsigned log_extension_degree,
                                      const unsigned coset_idx) {
  n2b_nonfinal_stages_block<3>(gmem_inputs_matrix, gmem_outputs_matrix, stride_between_input_arrays, stride_between_output_arrays, start_stage,
                               stages_this_launch == 7, log_n, inverse, num_ntts, log_extension_degree, coset_idx);
}

// Simple, non-optimized kernel used for log_n < 16, to unblock debugging small proofs.
extern "C" __launch_bounds__(512, 2) __global__
    void n2b_1_stage(const base_field *gmem_inputs_matrix, base_field *gmem_outputs_matrix, const unsigned stride_between_input_arrays,
                     const unsigned stride_between_output_arrays, const unsigned start_stage, const unsigned stages_this_launch, const unsigned log_n,
                     const bool inverse, const unsigned blocks_per_ntt, const unsigned log_extension_degree, const unsigned coset_idx) {
  const unsigned ntt_idx = blockIdx.x / blocks_per_ntt;
  const unsigned bid_in_ntt = blockIdx.x - ntt_idx * blocks_per_ntt;
  const unsigned tid_in_ntt = threadIdx.x + bid_in_ntt * blockDim.x;
  if (tid_in_ntt >= (1 << (log_n - 1)))
    return;
  const unsigned log_exchg_region_sz = log_n - start_stage;
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
    if (coset_idx) {
      const unsigned shift = OMEGA_LOG_ORDER - log_n - log_extension_degree;
      const unsigned offset = coset_idx << shift;
      a = base_field::mul(a, get_power_of_w(a_idx * offset, false));
      b = base_field::mul(b, get_power_of_w(b_idx * offset, false));
    }
    a = base_field::mul(a, get_power_of_g(a_idx, false));
    b = base_field::mul(b, get_power_of_g(b_idx, false));
  }

  exchg_dit(a, b, twiddle);

  if (inverse && (start_stage + stages_this_launch == log_n)) {
    a = base_field::mul(a, inv_sizes[log_n]);
    b = base_field::mul(b, inv_sizes[log_n]);
    if (log_extension_degree) {
      const unsigned a_idx_brev = __brev(a_idx) >> (32 - log_n);
      const unsigned b_idx_brev = __brev(b_idx) >> (32 - log_n);
      if (coset_idx) {
        const unsigned shift = OMEGA_LOG_ORDER - log_n - log_extension_degree;
        const unsigned offset = coset_idx << shift;
        a = base_field::mul(a, get_power_of_w(a_idx_brev * offset, true));
        b = base_field::mul(b, get_power_of_w(b_idx_brev * offset, true));
      }
      a = base_field::mul(a, get_power_of_g(a_idx_brev, true));
      b = base_field::mul(b, get_power_of_g(b_idx_brev, true));
    }
  }

  memory::store_cs(gmem_output + a_idx, a);
  memory::store_cs(gmem_output + b_idx, b);
}
