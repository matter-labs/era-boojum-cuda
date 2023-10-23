#pragma once // also, this file should only be compiled in one compile unit because it has __global__ definitions

DEVICE_FORCEINLINE void shfl_xor_bf(base_field *vals, const unsigned i, const unsigned lane_id, const unsigned lane_mask) {
  // Some threads need to post vals[2 * i], others need to post vals[2 * i + 1].
  // We use a temporary to avoid calling shfls divergently, which is unsafe on pre-Volta.
  base_field tmp{};
  if (lane_id & lane_mask)
    tmp = vals[2 * i];
  else
    tmp = vals[2 * i + 1];
  tmp[0] = __shfl_xor_sync(0xffffffff, tmp[0], lane_mask);
  tmp[1] = __shfl_xor_sync(0xffffffff, tmp[1], lane_mask);
  if (lane_id & lane_mask)
    vals[2 * i] = tmp;
  else
    vals[2 * i + 1] = tmp;
}

template <unsigned LOG_VALS_PER_THREAD> DEVICE_FORCEINLINE
void b2n_initial_stages_warp(const base_field *gmem_inputs_matrix, base_field *gmem_outputs_matrix, const unsigned stride_between_input_arrays,
                             const unsigned stride_between_output_arrays, const unsigned start_stage, const unsigned stages_this_launch,
                             const unsigned log_n, const bool inverse, const unsigned blocks_per_ntt, const unsigned log_extension_degree,
                             const unsigned coset_idx) {
  constexpr unsigned VALS_PER_THREAD = 1 << LOG_VALS_PER_THREAD;
  constexpr unsigned PAIRS_PER_THREAD = VALS_PER_THREAD >> 1;
  constexpr unsigned WARPS_PER_BLOCK = 4;
  constexpr unsigned VALS_PER_WARP = 32 * VALS_PER_THREAD;
  constexpr unsigned VALS_PER_BLOCK = 32 * VALS_PER_THREAD * WARPS_PER_BLOCK;

  __shared__ base_field smem[VALS_PER_BLOCK];

  const unsigned lane_id{threadIdx.x & 31};
  const unsigned warp_id{threadIdx.x >> 5};
  const unsigned ntt_idx = 0; // blockIdx.x / blocks_per_ntt;
  const unsigned block_idx_in_ntt = blockIdx.x - ntt_idx * blocks_per_ntt;
  const unsigned gmem_offset = ntt_idx * stride_between_input_arrays + VALS_PER_BLOCK * block_idx_in_ntt + VALS_PER_WARP * warp_id;
  const base_field *gmem_in = gmem_inputs_matrix + gmem_offset;
  base_field *gmem_out = gmem_outputs_matrix + gmem_offset;

  auto twiddle_cache = smem + VALS_PER_WARP * warp_id;

  base_field vals[VALS_PER_THREAD];

#pragma unroll
  for (unsigned i = 0; i < PAIRS_PER_THREAD; i++) {
    const auto in = memory::load_cs(reinterpret_cast<const uint4*>(gmem_in + 64 * i + 2 * lane_id));
    vals[2 * i][0] = in.x;
    vals[2 * i][1] = in.y;
    vals[2 * i + 1][0] = in.z;
    vals[2 * i + 1][1] = in.w;
  }

  // cooperatively loads all the twiddles this warp needs
  base_field *twiddles_this_stage = twiddle_cache;
  unsigned num_twiddles_this_stage = VALS_PER_WARP >> 1;
  unsigned exchg_region_offset = gmem_offset >> 1;
  for (unsigned stage = 0; stage < stages_this_launch; stage++) {
#pragma unroll
    for (unsigned i = lane_id; i < num_twiddles_this_stage; i += 32) {
      twiddles_this_stage[i] = get_twiddle(inverse, i + exchg_region_offset);
    }
    twiddles_this_stage += num_twiddles_this_stage;
    num_twiddles_this_stage >>= 1;
    exchg_region_offset >>= 1;
  }

  __syncwarp();

  unsigned lane_mask = 1;
  twiddles_this_stage = twiddle_cache;
  num_twiddles_this_stage = VALS_PER_WARP >> 1;
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

#pragma unroll
  for (unsigned i = 1, stage = 6; i < LOG_VALS_PER_THREAD; i++, stage++) {
    if (stage < stages_this_launch) {
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
  }

#pragma unroll
  for (unsigned i = 0; i < PAIRS_PER_THREAD; i++) {
    // This output pattern (resulting from the above shfls) is nice, but not obvious.
    // To see why it works, sketch the shfl stages on paper.
    memory::store_cs(gmem_out + 64 * i + lane_id, vals[2 * i]);
    memory::store_cs(gmem_out + 64 * i + lane_id + 32, vals[2 * i + 1]);
  }
}

extern "C" __global__
void b2n_initial_up_to_8_stages_warp(const base_field *gmem_inputs_matrix, base_field *gmem_outputs_matrix, const unsigned stride_between_input_arrays,
                                     const unsigned stride_between_output_arrays, const unsigned start_stage, const unsigned stages_this_launch,
                                     const unsigned log_n, const bool inverse, const unsigned blocks_per_ntt, const unsigned log_extension_degree,
                                     const unsigned coset_idx) {
  b2n_initial_stages_warp<3>(gmem_inputs_matrix, gmem_outputs_matrix, stride_between_input_arrays, stride_between_output_arrays, start_stage,
                             stages_this_launch, log_n, inverse, blocks_per_ntt, log_extension_degree, coset_idx);
}

template <unsigned LOG_VALS_PER_THREAD, unsigned LOG_INTERWARP_STAGES> DEVICE_FORCEINLINE
void b2n_initial_stages_block(const base_field *gmem_inputs_matrix, base_field *gmem_outputs_matrix, const unsigned stride_between_input_arrays,
                              const unsigned stride_between_output_arrays, const unsigned start_stage, const unsigned stages_this_launch,
                              const unsigned log_n, const bool inverse, const unsigned blocks_per_ntt, const unsigned log_extension_degree,
                              const unsigned coset_idx) {
  constexpr unsigned VALS_PER_THREAD = 1 << LOG_VALS_PER_THREAD;
  constexpr unsigned PAIRS_PER_THREAD = VALS_PER_THREAD >> 1;
  constexpr unsigned VALS_PER_WARP = 32 * VALS_PER_THREAD;
  constexpr unsigned WARPS_PER_BLOCK = VALS_PER_WARP >> 4;
  constexpr unsigned VALS_PER_BLOCK = 32 * VALS_PER_THREAD * WARPS_PER_BLOCK;

  __shared__ base_field smem[VALS_PER_BLOCK];

  const unsigned lane_id{threadIdx.x & 31};
  const unsigned warp_id{threadIdx.x >> 5};
  const unsigned ntt_idx = 0; // blockIdx.x / blocks_per_ntt;
  const unsigned block_idx_in_ntt = blockIdx.x - ntt_idx * blocks_per_ntt;
  const unsigned gmem_offset = ntt_idx * stride_between_input_arrays + VALS_PER_BLOCK * block_idx_in_ntt + VALS_PER_WARP * warp_id;
  const base_field *gmem_in = gmem_inputs_matrix + gmem_offset;
  base_field *gmem_out = gmem_outputs_matrix + gmem_offset;

  auto twiddle_cache = smem + VALS_PER_WARP * warp_id;

  base_field vals[VALS_PER_THREAD];

#pragma unroll
  for (unsigned i = 0; i < PAIRS_PER_THREAD; i++) {
    const auto pair = memory::load_cs(reinterpret_cast<const uint4*>(gmem_in + 64 * i + 2 * lane_id));
    vals[2 * i][0] = pair.x;
    vals[2 * i][1] = pair.y;
    vals[2 * i + 1][0] = pair.z;
    vals[2 * i + 1][1] = pair.w;
  }

  // cooperatively loads all the twiddles this warp needs
  base_field *twiddles_this_stage = twiddle_cache;
  unsigned num_twiddles_this_stage = VALS_PER_WARP >> 1;
  unsigned exchg_region_offset = gmem_offset >> 1;
  for (unsigned stage = 0; stage < 6 + LOG_VALS_PER_THREAD - 1; stage++) {
#pragma unroll
    for (unsigned i = lane_id; i < num_twiddles_this_stage; i += 32) {
      twiddles_this_stage[i] = get_twiddle(inverse, i + exchg_region_offset);
    }
    twiddles_this_stage += num_twiddles_this_stage;
    num_twiddles_this_stage >>= 1;
    exchg_region_offset >>= 1;
  }

  __syncwarp();

  unsigned lane_mask = 1;
  twiddles_this_stage = twiddle_cache;
  num_twiddles_this_stage = VALS_PER_WARP >> 1;
  for (unsigned stage = 0; stage < 6; stage++) {
#pragma unroll
    for (unsigned i = 0; i < PAIRS_PER_THREAD; i++) {
      const auto twiddle = twiddles_this_stage[(32 * i + lane_id) >> stage];
      exchg_dif(vals[2 * i], vals[2 * i + 1], twiddle);
      shfl_xor_bf(vals, i, lane_id, lane_mask);
    }
    lane_mask <<= 1;
    twiddles_this_stage += num_twiddles_this_stage;
    num_twiddles_this_stage >>= 1;
  }

#pragma unroll
  for (unsigned i = 1, stage = 6; i < LOG_VALS_PER_THREAD; i++, stage++) {
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

#pragma unroll
  for (unsigned i = 0; i < PAIRS_PER_THREAD; i++) {
    // The output pattern (resulting from the above shfls) is nice, but not obvious.
    // To see why it works, sketch the shfl stages on paper.
    // TODO: Stash twiddles in registers while using smem to communicate data values
    //if (ntt_idx != num_ntts - 1) {
    twiddle_cache[64 * i + lane_id] = vals[2 * i];
    twiddle_cache[64 * i + lane_id + 32] = vals[2 * i + 1];
  }

  __syncthreads();

  auto pair_addr = smem + 16 * warp_id + VALS_PER_WARP * (lane_id >> 3) + 2 * (threadIdx.x & 7);
#pragma unroll
  for (unsigned i = 0; i < PAIRS_PER_THREAD; i++, pair_addr += 4 * VALS_PER_WARP) {
    // TODO: Juggle twiddles here as needed
    const auto pair = *reinterpret_cast<const uint4*>(pair_addr);
    vals[2 * i][0] = pair.x;
    vals[2 * i][1] = pair.y;
    vals[2 * i + 1][0] = pair.z;
    vals[2 * i + 1][1] = pair.w;
  }

  // if (ntt_idx != num_ntts - 1)
  //   __syncthreads();

  lane_mask = 8;
  exchg_region_offset = ((blockIdx.x * WARPS_PER_BLOCK) >> 1) + (lane_id & 16);
  unsigned first_interwarp_stage = 6 + LOG_VALS_PER_THREAD - 1;
  for (unsigned s = 0; s < first_interwarp_stage + LOG_INTERWARP_STAGES; s++) {
#pragma unroll
    for (unsigned i = 0; i < PAIRS_PER_THREAD; i++) {
      // TODO: Handle these cooperatively
      const auto twiddle = get_twiddle(inverse, (exchg_region_offset + 2 * i) >> s);
      shfl_xor_bf(vals, i, lane_id, lane_mask);
      exchg_dif(vals[2 * i], vals[2 * i + 1], twiddle);
    }
    lane_mask <<= 1;
  }

#pragma unroll
  for (unsigned i = 1, stage = 6; i < LOG_VALS_PER_THREAD; i++, stage++) {
#pragma unroll
    for (unsigned j = 0; j < PAIRS_PER_THREAD >> i; j++) {
      const unsigned exchg_tile_sz = 2 << i;
      const unsigned half_exchg_tile_sz = 1 << i;
      const auto twiddle = twiddles_this_stage[j];
#pragma unroll
      for (unsigned k = 0; k < half_exchg_tile_sz; k++)
        exchg_dif(vals[exchg_tile_sz * j + k], vals[exchg_tile_sz * j + k + half_exchg_tile_sz], twiddle);
    }
  }

#pragma unroll
  for (unsigned i = 0; i < PAIRS_PER_THREAD; i++) {
    // This output pattern (resulting from the above shfls) is nice, but not obvious.
    // To see why it works, sketch the shfl stages on paper.
    memory::store_cs(gmem_out + 64 * i + lane_id, vals[2 * i]);
    memory::store_cs(gmem_out + 64 * i + lane_id + 32, vals[2 * i + 1]);
  }
}

extern "C" __global__
void b2n_initial_up_to_12_stages_block(const base_field *gmem_inputs_matrix, base_field *gmem_outputs_matrix, const unsigned stride_between_input_arrays,
                                      const unsigned stride_between_output_arrays, const unsigned start_stage, const unsigned stages_this_launch,
                                      const unsigned log_n, const bool inverse, const unsigned blocks_per_ntt, const unsigned log_extension_degree,
                                      const unsigned coset_idx) {
  b2n_initial_stages_block<3, 4>(gmem_inputs_matrix, gmem_outputs_matrix, stride_between_input_arrays, stride_between_output_arrays, start_stage,
                                 stages_this_launch, log_n, inverse, blocks_per_ntt, log_extension_degree, coset_idx);
}

extern "C" __global__
void b2n_noninitial_up_to_8_stages_block(const base_field *gmem_inputs_matrix, base_field *gmem_outputs_matrix, const unsigned stride_between_input_arrays,
                                         const unsigned stride_between_output_arrays, const unsigned start_stage, const unsigned stages_this_launch,
                                         const unsigned log_n, const bool inverse, const unsigned blocks_per_ntt, const unsigned log_extension_degree,
                                         const unsigned coset_idx) {
  // b2n_noninitial_stages_block<3>(gmem_inputs_matrix, gmem_outputs_matrix, stride_between_input_arrays, stride_between_output_arrays, start_stage,
  //                                stages_this_launch, log_n, inverse, blocks_per_ntt, log_extension_degree, coset_idx);
}

// I bet there are ways to write these macros more concisely,
// but the structure is fairly readable and easy to edit.
#define THREE_REGISTER_STAGES_B2N(SKIP_FIRST)                                                                                                                  \
  {                                                                                                                                                            \
    if (!(SKIP_FIRST)) {                                                                                                                                       \
      /* first stage of this set-of-3 stages */                                                                                                                \
      const auto t3 = get_twiddle(inverse, thread_exchg_region);                                                                                               \
      const auto t4 = get_twiddle(inverse, thread_exchg_region + 1);                                                                                           \
      const auto t5 = get_twiddle(inverse, thread_exchg_region + 2);                                                                                           \
      const auto t6 = get_twiddle(inverse, thread_exchg_region + 3);                                                                                           \
      exchg_dif(reg_vals[0], reg_vals[1], t3);                                                                                                                 \
      exchg_dif(reg_vals[2], reg_vals[3], t4);                                                                                                                 \
      exchg_dif(reg_vals[4], reg_vals[5], t5);                                                                                                                 \
      exchg_dif(reg_vals[6], reg_vals[7], t6);                                                                                                                 \
    }                                                                                                                                                          \
    /* second stage of this set-of-3 stages */                                                                                                                 \
    thread_exchg_region >>= 1;                                                                                                                                 \
    const auto t1 = get_twiddle(inverse, thread_exchg_region);                                                                                                 \
    const auto t2 = get_twiddle(inverse, thread_exchg_region + 1);                                                                                             \
    exchg_dif(reg_vals[0], reg_vals[2], t1);                                                                                                                   \
    exchg_dif(reg_vals[1], reg_vals[3], t1);                                                                                                                   \
    exchg_dif(reg_vals[4], reg_vals[6], t2);                                                                                                                   \
    exchg_dif(reg_vals[5], reg_vals[7], t2);                                                                                                                   \
    /* third stage of this set-of-3 stages */                                                                                                                  \
    thread_exchg_region >>= 1;                                                                                                                                 \
    const auto t0 = get_twiddle(inverse, thread_exchg_region);                                                                                                 \
    exchg_dif(reg_vals[0], reg_vals[4], t0);                                                                                                                   \
    exchg_dif(reg_vals[1], reg_vals[5], t0);                                                                                                                   \
    exchg_dif(reg_vals[2], reg_vals[6], t0);                                                                                                                   \
    exchg_dif(reg_vals[3], reg_vals[7], t0);                                                                                                                   \
  }

#define TWO_REGISTER_STAGES_B2N(SKIP_SECOND)                                                                                                                   \
  {                                                                                                                                                            \
    unsigned tmp_thread_exchg_region = thread_exchg_region;                                                                                                    \
    /* first stage of this set-of-2 stages */                                                                                                                  \
    const auto t1 = get_twiddle(inverse, tmp_thread_exchg_region);                                                                                             \
    const auto t2 = get_twiddle(inverse, tmp_thread_exchg_region + 1);                                                                                         \
    exchg_dif(reg_vals[0], reg_vals[1], t1);                                                                                                                   \
    exchg_dif(reg_vals[2], reg_vals[3], t2);                                                                                                                   \
    if (!(SKIP_SECOND)) {                                                                                                                                      \
      /* second stage of this set-of-2 stages */                                                                                                               \
      tmp_thread_exchg_region >>= 1;                                                                                                                           \
      const auto t0 = get_twiddle(inverse, tmp_thread_exchg_region);                                                                                           \
      exchg_dif(reg_vals[0], reg_vals[2], t0);                                                                                                                 \
      exchg_dif(reg_vals[1], reg_vals[3], t0);                                                                                                                 \
    }                                                                                                                                                          \
  }

#define ONE_EXTRA_STAGE_B2N                                                                                                                                    \
  {                                                                                                                                                            \
    const unsigned intrablock_exchg_region = (warp_id >> 1);                                                                                                   \
    const unsigned smem_logical_offset = (warp_id & 1) * 32 + lane_id;                                                                                         \
    const unsigned offset_padded = intrablock_exchg_region * 2 * PADDED_WARP_SCRATCH_SIZE;                                                                     \
    for (int i = 0; i < 4; i++) {                                                                                                                              \
      const unsigned idx = offset_padded + PAD(smem_logical_offset + i * 64);                                                                                  \
      reg_vals[i] = smem[idx];                                                                                                                                 \
      reg_vals[i + 4] = smem[idx + PADDED_WARP_SCRATCH_SIZE];                                                                                                  \
    }                                                                                                                                                          \
    const auto t0 = get_twiddle(inverse, 8 * block_idx_in_ntt + intrablock_exchg_region);                                                                      \
    exchg_dif(reg_vals[0], reg_vals[4], t0);                                                                                                                   \
    exchg_dif(reg_vals[1], reg_vals[5], t0);                                                                                                                   \
    exchg_dif(reg_vals[2], reg_vals[6], t0);                                                                                                                   \
    exchg_dif(reg_vals[3], reg_vals[7], t0);                                                                                                                   \
    const unsigned offset = intrablock_exchg_region * 512 + smem_logical_offset;                                                                               \
    for (int i = 0; i < 4; i++) {                                                                                                                              \
      memory::store_cs(gmem_output + offset + i * 64, reg_vals[i]);                                                                                            \
      memory::store_cs(gmem_output + offset + i * 64 + 256, reg_vals[i + 4]);                                                                                  \
    }                                                                                                                                                          \
    return;                                                                                                                                                    \
  }

#define TWO_EXTRA_STAGES_B2N                                                                                                                                   \
  {                                                                                                                                                            \
    const unsigned intrablock_exchg_region = (warp_id >> 2);                                                                                                   \
    const unsigned smem_logical_offset = (warp_id & 3) * 32 + lane_id;                                                                                         \
    const unsigned offset_padded = intrablock_exchg_region * 4 * PADDED_WARP_SCRATCH_SIZE;                                                                     \
    for (int i = 0; i < 2; i++) {                                                                                                                              \
      for (int j = 0; j < 2; j++) {                                                                                                                            \
        const unsigned idx = offset_padded + PAD(smem_logical_offset + j * 128) + i * PADDED_WARP_SCRATCH_SIZE;                                                \
        reg_vals[2 * i + j] = smem[idx];                                                                                                                       \
        reg_vals[2 * i + j + 4] = smem[idx + 2 * PADDED_WARP_SCRATCH_SIZE];                                                                                    \
      }                                                                                                                                                        \
    }                                                                                                                                                          \
    unsigned global_exchg_region = 8 * block_idx_in_ntt + intrablock_exchg_region * 2;                                                                         \
    const auto t1 = get_twiddle(inverse, global_exchg_region);                                                                                                 \
    const auto t2 = get_twiddle(inverse, global_exchg_region + 1);                                                                                             \
    global_exchg_region >>= 1;                                                                                                                                 \
    const auto t0 = get_twiddle(inverse, global_exchg_region);                                                                                                 \
    exchg_dif(reg_vals[0], reg_vals[2], t1);                                                                                                                   \
    exchg_dif(reg_vals[1], reg_vals[3], t1);                                                                                                                   \
    exchg_dif(reg_vals[4], reg_vals[6], t2);                                                                                                                   \
    exchg_dif(reg_vals[5], reg_vals[7], t2);                                                                                                                   \
    exchg_dif(reg_vals[0], reg_vals[4], t0);                                                                                                                   \
    exchg_dif(reg_vals[1], reg_vals[5], t0);                                                                                                                   \
    exchg_dif(reg_vals[2], reg_vals[6], t0);                                                                                                                   \
    exchg_dif(reg_vals[3], reg_vals[7], t0);                                                                                                                   \
    const unsigned offset = intrablock_exchg_region * 1024 + smem_logical_offset;                                                                              \
    for (int i = 0; i < 4; i++) {                                                                                                                              \
      memory::store_cs(gmem_output + offset + i * 128, reg_vals[i]);                                                                                           \
      memory::store_cs(gmem_output + offset + i * 128 + 512, reg_vals[i + 4]);                                                                                 \
    }                                                                                                                                                          \
    return;                                                                                                                                                    \
  }

#define THREE_OR_FOUR_EXTRA_STAGES_B2N(THREE_STAGES)                                                                                                           \
  {                                                                                                                                                            \
    const unsigned intrablock_exchg_region = (warp_id >> 3);                                                                                                   \
    const unsigned smem_logical_offset = (warp_id & 7) * 32 + lane_id;                                                                                         \
    const unsigned offset_padded = intrablock_exchg_region * 8 * PADDED_WARP_SCRATCH_SIZE + PAD(smem_logical_offset);                                          \
    for (int i = 0; i < 4; i++) {                                                                                                                              \
      const unsigned idx = offset_padded + i * PADDED_WARP_SCRATCH_SIZE;                                                                                       \
      reg_vals[i] = smem[idx];                                                                                                                                 \
      reg_vals[i + 4] = smem[idx + 4 * PADDED_WARP_SCRATCH_SIZE];                                                                                              \
    }                                                                                                                                                          \
    unsigned global_exchg_region = 8 * block_idx_in_ntt + intrablock_exchg_region * 4;                                                                         \
    const auto t3 = get_twiddle(inverse, global_exchg_region);                                                                                                 \
    const auto t4 = get_twiddle(inverse, global_exchg_region + 1);                                                                                             \
    const auto t5 = get_twiddle(inverse, global_exchg_region + 2);                                                                                             \
    const auto t6 = get_twiddle(inverse, global_exchg_region + 3);                                                                                             \
    global_exchg_region >>= 1;                                                                                                                                 \
    const auto t1 = get_twiddle(inverse, global_exchg_region);                                                                                                 \
    const auto t2 = get_twiddle(inverse, global_exchg_region + 1);                                                                                             \
    global_exchg_region >>= 1;                                                                                                                                 \
    const auto t0 = get_twiddle(inverse, global_exchg_region);                                                                                                 \
    exchg_dif(reg_vals[0], reg_vals[1], t3);                                                                                                                   \
    exchg_dif(reg_vals[2], reg_vals[3], t4);                                                                                                                   \
    exchg_dif(reg_vals[4], reg_vals[5], t5);                                                                                                                   \
    exchg_dif(reg_vals[6], reg_vals[7], t6);                                                                                                                   \
    exchg_dif(reg_vals[0], reg_vals[2], t1);                                                                                                                   \
    exchg_dif(reg_vals[1], reg_vals[3], t1);                                                                                                                   \
    exchg_dif(reg_vals[4], reg_vals[6], t2);                                                                                                                   \
    exchg_dif(reg_vals[5], reg_vals[7], t2);                                                                                                                   \
    exchg_dif(reg_vals[0], reg_vals[4], t0);                                                                                                                   \
    exchg_dif(reg_vals[1], reg_vals[5], t0);                                                                                                                   \
    exchg_dif(reg_vals[2], reg_vals[6], t0);                                                                                                                   \
    exchg_dif(reg_vals[3], reg_vals[7], t0);                                                                                                                   \
    if ((THREE_STAGES)) {                                                                                                                                      \
      const unsigned offset = intrablock_exchg_region * 2048 + smem_logical_offset;                                                                            \
      for (int i = 0; i < 4; i++) {                                                                                                                            \
        memory::store_cs(gmem_output + offset + i * 256, reg_vals[i]);                                                                                         \
        memory::store_cs(gmem_output + offset + i * 256 + 1024, reg_vals[i + 4]);                                                                              \
      }                                                                                                                                                        \
    } else {                                                                                                                                                   \
      for (int i = 0; i < 4; i++) {                                                                                                                            \
        const unsigned idx = offset_padded + i * PADDED_WARP_SCRATCH_SIZE;                                                                                     \
        smem[idx] = reg_vals[i];                                                                                                                               \
        smem[idx + 4 * PADDED_WARP_SCRATCH_SIZE] = reg_vals[i + 4];                                                                                            \
      }                                                                                                                                                        \
      /* in theory, we could avoid full __syncthreads by splitting each warp into two half-warps of size 16, */                                                \
      /* assigning first-halves to first 2048 elems and second-halves to second 2048 elems, then */                                                            \
      /* combining results from first and second halves with intrawarp syncs, but that doesn't seem worth the trouble */                                       \
      __syncthreads();                                                                                                                                         \
      const auto t0 = get_twiddle(inverse, block_idx_in_ntt);                                                                                                  \
      int i = threadIdx.x;                                                                                                                                     \
      int i_padded = (threadIdx.x >> 8) * PADDED_WARP_SCRATCH_SIZE + PAD(threadIdx.x & 255);                                                                   \
      for (; i < 2048; i += 512, i_padded += 2 * PADDED_WARP_SCRATCH_SIZE) {                                                                                   \
        reg_vals[0] = smem[i_padded];                                                                                                                          \
        reg_vals[1] = smem[i_padded + 8 * PADDED_WARP_SCRATCH_SIZE];                                                                                           \
        exchg_dif(reg_vals[0], reg_vals[1], t0);                                                                                                               \
        memory::store_cs(gmem_output + i, reg_vals[0]);                                                                                                        \
        memory::store_cs(gmem_output + i + 2048, reg_vals[1]);                                                                                                 \
      }                                                                                                                                                        \
    }                                                                                                                                                          \
    return;                                                                                                                                                    \
  }

extern "C" __launch_bounds__(512, 2) __global__
    void b2n_initial_7_or_8_stages(const base_field *gmem_inputs_matrix, base_field *gmem_outputs_matrix, const unsigned stride_between_input_arrays,
                                   const unsigned stride_between_output_arrays, const unsigned start_stage, const unsigned stages_this_launch,
                                   const unsigned log_n, const bool inverse, const unsigned blocks_per_ntt, const unsigned log_extension_degree,
                                   const unsigned coset_idx) {
  extern __shared__ base_field smem[]; // 4096 elems

  const unsigned tile_stride{16};
  const unsigned lane_in_tile = threadIdx.x & 15;
  const unsigned lane_id{threadIdx.x & 31};
  const unsigned warp_id{threadIdx.x >> 5};
  const unsigned ntt_idx = blockIdx.x / blocks_per_ntt;
  const unsigned block_idx_in_ntt = blockIdx.x - ntt_idx * blocks_per_ntt;
  const base_field *gmem_input = gmem_inputs_matrix + ntt_idx * stride_between_input_arrays + 4096 * block_idx_in_ntt;
  base_field *gmem_output = gmem_outputs_matrix + ntt_idx * stride_between_output_arrays + 4096 * block_idx_in_ntt;

  {
    // maybe some memcpy_asyncs could micro-optimize this
    // maybe an arrive-wait barrier could micro-optimize the start_stage > 0 case
    base_field reg_vals[8];
#pragma unroll 8
    for (unsigned i = 0, t = warp_id * 256 + lane_id; i < 8; i++, t += 32) {
      const unsigned tile = t >> 4;
      const unsigned g = tile * tile_stride + lane_in_tile;
      reg_vals[i] = memory::load_cs(gmem_input + g);
    }
    if (log_extension_degree && !inverse) {
      const unsigned shift = OMEGA_LOG_ORDER - log_n - log_extension_degree;
      const unsigned offset = coset_idx << shift;
#pragma unroll 8
      for (unsigned i = 0, t = warp_id * 256 + lane_id; i < 8; i++, t += 32) {
        const unsigned tile = t >> 4;
        const unsigned idx = __brev(tile * tile_stride + lane_in_tile + 4096 * block_idx_in_ntt) >> (32 - log_n);
        if (coset_idx) {
          auto power_of_w = get_power_of_w(idx * offset, false);
          reg_vals[i] = base_field::mul(reg_vals[i], power_of_w);
        }
        auto power_of_g = get_power_of_g(idx, false);
        reg_vals[i] = base_field::mul(reg_vals[i], power_of_g);
      }
    }
#pragma unroll 8
    for (unsigned i = 0, t = warp_id * 256 + lane_id; i < 8; i++, t += 32) {
      // puts each warp's data in its assigned smem region
      const unsigned s = (t >> 8) * PADDED_WARP_SCRATCH_SIZE + PAD(t & 255);
      smem[s] = reg_vals[i];
    }
  }

  __syncwarp();

  unsigned warp_exchg_region = block_idx_in_ntt * 2048 + warp_id * 128;
  base_field reg_vals[8];
  base_field *warp_scratch = smem + PADDED_WARP_SCRATCH_SIZE * warp_id;

  unsigned thread_exchg_region = warp_exchg_region + lane_id * 4;
  for (int i = 0; i < 8; i++)
    reg_vals[i] = warp_scratch[PAD(lane_id * 8 + i)];
  THREE_REGISTER_STAGES_B2N(false)
  for (int i = 0; i < 8; i++)
    warp_scratch[PAD(lane_id * 8 + i)] = reg_vals[i];
  warp_exchg_region >>= 3;

  __syncwarp();

  thread_exchg_region = warp_exchg_region + (lane_id >> 3) * 4;
  const unsigned vals_start = 64 * (lane_id >> 3) + (lane_id & 7);
  for (int i = 0; i < 8; i++)
    reg_vals[i] = warp_scratch[PAD(vals_start + 8 * i)];
  THREE_REGISTER_STAGES_B2N(false)
  for (int i = 0; i < 8; i++)
    warp_scratch[PAD(vals_start + 8 * i)] = reg_vals[i];
  warp_exchg_region >>= 3;

  __syncwarp();

  thread_exchg_region = warp_exchg_region;
  for (int j = 0; j < 64; j += 32) {
    for (int i = 0; i < 4; i++)
      reg_vals[i] = warp_scratch[PAD(lane_id + 64 * i + j)];
    TWO_REGISTER_STAGES_B2N((stages_this_launch == 7))
    for (int i = 0; i < 4; i++)
      warp_scratch[PAD(lane_id + 64 * i + j)] = reg_vals[i];
  }

  __syncwarp();

// unroll 2 and unroll 4 give comparable perf. Stores don't incur a stall so it's not as important to ILP them.
#pragma unroll 1
  for (unsigned i = 0, t = warp_id * 256 + lane_id; i < 8; i++, t += 32) {
    const unsigned tile = t >> 4;
    const unsigned s = (t >> 8) * PADDED_WARP_SCRATCH_SIZE + PAD(t & 255);
    const auto val = smem[s];
    const unsigned g = tile * tile_stride + lane_in_tile;
    memory::store_cs(gmem_output + g, val);
  }
}

extern "C" __launch_bounds__(512, 2) __global__
    void b2n_initial_9_to_12_stages(const base_field *gmem_inputs_matrix, base_field *gmem_outputs_matrix, const unsigned stride_between_input_arrays,
                                    const unsigned stride_between_output_arrays, const unsigned start_stage, const unsigned stages_this_launch,
                                    const unsigned log_n, const bool inverse, const unsigned blocks_per_ntt, const unsigned log_extension_degree,
                                    const unsigned coset_idx) {
  extern __shared__ base_field smem[]; // 4096 elems

  const unsigned tile_stride{16};
  const unsigned lane_in_tile = threadIdx.x & 15;
  const unsigned lane_id{threadIdx.x & 31};
  const unsigned warp_id{threadIdx.x >> 5};
  const unsigned ntt_idx = blockIdx.x / blocks_per_ntt;
  const unsigned block_idx_in_ntt = blockIdx.x - ntt_idx * blocks_per_ntt;
  const base_field *gmem_input = gmem_inputs_matrix + ntt_idx * stride_between_input_arrays + 4096 * block_idx_in_ntt;
  base_field *gmem_output = gmem_outputs_matrix + ntt_idx * stride_between_output_arrays + 4096 * block_idx_in_ntt;

  {
    // maybe some memcpy_asyncs could further micro-optimize this
    // maybe an arrive-wait barrier could further micro-optimize the start_stage > 0 case
    base_field reg_vals[8];
#pragma unroll 8
    for (unsigned i = 0, t = warp_id * 256 + lane_id; i < 8; i++, t += 32) {
      const unsigned tile = t >> 4;
      const unsigned g = tile * tile_stride + lane_in_tile;
      reg_vals[i] = memory::load_cs(gmem_input + g);
    }
    if (log_extension_degree && !inverse) {
      const unsigned shift = OMEGA_LOG_ORDER - log_n - log_extension_degree;
      const unsigned offset = coset_idx << shift;
#pragma unroll 8
      for (unsigned i = 0, t = warp_id * 256 + lane_id; i < 8; i++, t += 32) {
        const unsigned tile = t >> 4;
        const unsigned idx = __brev(tile * tile_stride + lane_in_tile + 4096 * block_idx_in_ntt) >> (32 - log_n);
        if (coset_idx) {
          auto power_of_w = get_power_of_w(idx * offset, false);
          reg_vals[i] = base_field::mul(reg_vals[i], power_of_w);
        }
        auto power_of_g = get_power_of_g(idx, false);
        reg_vals[i] = base_field::mul(reg_vals[i], power_of_g);
      }
    }
#pragma unroll 8
    for (unsigned i = 0, t = warp_id * 256 + lane_id; i < 8; i++, t += 32) {
      // puts each warp's data in its assigned smem region
      const unsigned s = (t >> 8) * PADDED_WARP_SCRATCH_SIZE + PAD(t & 255);
      smem[s] = reg_vals[i];
    }
  }

  __syncwarp();

  unsigned warp_exchg_region = block_idx_in_ntt * 2048 + warp_id * 128;
  base_field reg_vals[8];
  base_field *warp_scratch = smem + PADDED_WARP_SCRATCH_SIZE * warp_id;

  unsigned thread_exchg_region = warp_exchg_region + lane_id * 4;
  for (int i = 0; i < 8; i++)
    reg_vals[i] = warp_scratch[PAD(lane_id * 8 + i)];
  THREE_REGISTER_STAGES_B2N(false)
  for (int i = 0; i < 8; i++)
    warp_scratch[PAD(lane_id * 8 + i)] = reg_vals[i];
  warp_exchg_region >>= 3;

  __syncwarp();

  thread_exchg_region = warp_exchg_region + (lane_id >> 3) * 4;
  const unsigned vals_start = 64 * (lane_id >> 3) + (lane_id & 7);
  for (int i = 0; i < 8; i++)
    reg_vals[i] = warp_scratch[PAD(vals_start + 8 * i)];
  THREE_REGISTER_STAGES_B2N(false)
  for (int i = 0; i < 8; i++)
    warp_scratch[PAD(vals_start + 8 * i)] = reg_vals[i];
  warp_exchg_region >>= 3;

  __syncwarp();

  thread_exchg_region = warp_exchg_region;
  for (int j = 0; j < 64; j += 32) {
    for (int i = 0; i < 4; i++)
      reg_vals[i] = warp_scratch[PAD(lane_id + 64 * i + j)];
    TWO_REGISTER_STAGES_B2N(false)
    for (int i = 0; i < 4; i++)
      warp_scratch[PAD(lane_id + 64 * i + j)] = reg_vals[i];
  }

  // This start_stage == 0 kernel can handle up to 11 stages if needed by the overall NTT.
  __syncthreads();
  const unsigned stages_remaining = stages_this_launch - 8;
  switch (stages_remaining) {
  case 1:
    ONE_EXTRA_STAGE_B2N
  case 2:
    TWO_EXTRA_STAGES_B2N
  case 3:
    THREE_OR_FOUR_EXTRA_STAGES_B2N(true)
  case 4:
    THREE_OR_FOUR_EXTRA_STAGES_B2N(false)
  }
}

extern "C" __launch_bounds__(512, 2) __global__
    void b2n_noninitial_7_or_8_stages(const base_field *gmem_inputs_matrix, base_field *gmem_outputs_matrix, const unsigned stride_between_input_arrays,
                                      const unsigned stride_between_output_arrays, const unsigned start_stage, const unsigned stages_this_launch,
                                      const unsigned log_n, const bool inverse, const unsigned blocks_per_ntt, const unsigned log_extension_degree,
                                      const unsigned coset_idx) {
  extern __shared__ base_field smem[]; // 4096 elems

  // If we're only doing 7 stages, skip one stage by loading tiles with half the stride,
  // such that the first exchange (between nearest-neighbor tiles) has already happened
  const unsigned log_stride{(stages_this_launch == 7) ? start_stage - 1 : start_stage};
  const unsigned tile_stride{1u << log_stride};
  const unsigned lane_in_tile{threadIdx.x & 15};
  const unsigned lane_id{threadIdx.x & 31};
  const unsigned warp_id{threadIdx.x >> 5};
  const unsigned exchg_region_sz{tile_stride << 8};
  const unsigned log_blocks_per_region{log_stride - 4}; // tile_stride / 16
  const unsigned ntt_idx{blockIdx.x / blocks_per_ntt};
  const unsigned block_idx_in_ntt{blockIdx.x - ntt_idx * blocks_per_ntt};
  unsigned block_exchg_region{block_idx_in_ntt >> log_blocks_per_region};
  const unsigned block_exchg_region_start{block_exchg_region * exchg_region_sz};
  const unsigned block_start_in_exchg_region{16 * (block_idx_in_ntt & ((1 << log_blocks_per_region) - 1))};
  const base_field *gmem_input = gmem_inputs_matrix + ntt_idx * stride_between_input_arrays + block_exchg_region_start + block_start_in_exchg_region;
  base_field *gmem_output = gmem_outputs_matrix + ntt_idx * stride_between_output_arrays + block_exchg_region_start + block_start_in_exchg_region;

  {
    // maybe some memcpy_asyncs could further micro-optimize this
    // maybe an arrive-wait barrier could further micro-optimize the start_stage > 0 case
    base_field reg_vals[8];
#pragma unroll 8
    for (unsigned i = 0, t = warp_id * 256 + lane_id; i < 8; i++, t += 32) {
      const unsigned tile = t >> 4;
      const unsigned g = tile * tile_stride + lane_in_tile;
      reg_vals[i] = memory::load_cs(gmem_input + g);
    }
#pragma unroll 8
    for (unsigned i = 0, t = warp_id * 256 + lane_id; i < 8; i++, t += 32) {
      // puts each warp's data in its assigned smem region
      const unsigned tile = t >> 4;
      const unsigned s = lane_in_tile * PADDED_WARP_SCRATCH_SIZE + PAD(tile);
      smem[s] = reg_vals[i];
    }
  }

  __syncthreads();

  base_field reg_vals[8];
  base_field *warp_scratch = smem + PADDED_WARP_SCRATCH_SIZE * warp_id;

  block_exchg_region *= 128;
  unsigned thread_exchg_region = block_exchg_region + lane_id * 4;
  for (int i = 0; i < 8; i++)
    reg_vals[i] = warp_scratch[PAD(lane_id * 8 + i)];
  THREE_REGISTER_STAGES_B2N((stages_this_launch == 7))
  for (int i = 0; i < 8; i++)
    warp_scratch[PAD(lane_id * 8 + i)] = reg_vals[i];
  block_exchg_region >>= 3;

  __syncwarp();

  thread_exchg_region = block_exchg_region + (lane_id >> 3) * 4;
  const unsigned vals_start = 64 * (lane_id >> 3) + (lane_id & 7);
  for (int i = 0; i < 8; i++)
    reg_vals[i] = warp_scratch[PAD(vals_start + 8 * i)];
  THREE_REGISTER_STAGES_B2N(false)
  for (int i = 0; i < 8; i++)
    warp_scratch[PAD(vals_start + 8 * i)] = reg_vals[i];
  block_exchg_region >>= 3;

  __syncwarp();

  thread_exchg_region = block_exchg_region;
  for (int j = 0; j < 64; j += 32) {
    for (int i = 0; i < 4; i++)
      reg_vals[i] = warp_scratch[PAD(lane_id + 64 * i + j)];
    TWO_REGISTER_STAGES_B2N(false)
    for (int i = 0; i < 4; i++)
      warp_scratch[PAD(lane_id + 64 * i + j)] = reg_vals[i];
  }

  __syncthreads();

// unroll 2 and unroll 4 give comparable perf. Stores don't incur a stall so it's not as important to ILP them.
#pragma unroll 1
  for (unsigned i = 0, t = warp_id * 256 + lane_id; i < 8; i++, t += 32) {
    const unsigned tile = t >> 4;
    const unsigned s = lane_in_tile * PADDED_WARP_SCRATCH_SIZE + PAD(tile);
    auto val = smem[s];
    const unsigned g = tile * tile_stride + lane_in_tile;
    if (inverse && (start_stage + stages_this_launch == log_n)) {
      val = base_field::mul(val, inv_sizes[log_n]);
      if (log_extension_degree) {
        const unsigned idx = g + block_exchg_region_start + block_start_in_exchg_region;
        if (coset_idx) {
          const unsigned shift = OMEGA_LOG_ORDER - log_n - log_extension_degree;
          const unsigned offset = coset_idx << shift;
          auto power_of_w = get_power_of_w(idx * offset, true);
          val = base_field::mul(val, power_of_w);
        }
        auto power_of_g = get_power_of_g(idx, true);
        val = base_field::mul(val, power_of_g);
      }
    }
    memory::store_cs(gmem_output + g, val);
  }
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
