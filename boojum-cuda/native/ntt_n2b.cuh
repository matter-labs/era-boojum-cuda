#pragma once // also, this file should only be compiled in one compile unit because it has __global__ definitions

#define THREE_REGISTER_STAGES_N2B(SKIP_THIRD)                                                                                                                  \
  {                                                                                                                                                            \
    /* first stage of this set-of-3 stages */                                                                                                                  \
    const auto t0 = get_twiddle(inverse, thread_exchg_region);                                                                                                 \
    exchg_dit(reg_vals[0], reg_vals[4], t0);                                                                                                                   \
    exchg_dit(reg_vals[1], reg_vals[5], t0);                                                                                                                   \
    exchg_dit(reg_vals[2], reg_vals[6], t0);                                                                                                                   \
    exchg_dit(reg_vals[3], reg_vals[7], t0);                                                                                                                   \
    /* second stage of this set-of-3 stages */                                                                                                                 \
    thread_exchg_region *= 2;                                                                                                                                  \
    const auto t1 = get_twiddle(inverse, thread_exchg_region);                                                                                                 \
    const auto t2 = get_twiddle(inverse, thread_exchg_region + 1);                                                                                             \
    exchg_dit(reg_vals[0], reg_vals[2], t1);                                                                                                                   \
    exchg_dit(reg_vals[1], reg_vals[3], t1);                                                                                                                   \
    exchg_dit(reg_vals[4], reg_vals[6], t2);                                                                                                                   \
    exchg_dit(reg_vals[5], reg_vals[7], t2);                                                                                                                   \
    if (!(SKIP_THIRD)) {                                                                                                                                       \
      /* third stage of this set-of-3 stages */                                                                                                                \
      thread_exchg_region *= 2;                                                                                                                                \
      const auto t3 = get_twiddle(inverse, thread_exchg_region);                                                                                               \
      const auto t4 = get_twiddle(inverse, thread_exchg_region + 1);                                                                                           \
      const auto t5 = get_twiddle(inverse, thread_exchg_region + 2);                                                                                           \
      const auto t6 = get_twiddle(inverse, thread_exchg_region + 3);                                                                                           \
      exchg_dit(reg_vals[0], reg_vals[1], t3);                                                                                                                 \
      exchg_dit(reg_vals[2], reg_vals[3], t4);                                                                                                                 \
      exchg_dit(reg_vals[4], reg_vals[5], t5);                                                                                                                 \
      exchg_dit(reg_vals[6], reg_vals[7], t6);                                                                                                                 \
    }                                                                                                                                                          \
  }

#define TWO_REGISTER_STAGES_N2B(SKIP_FIRST)                                                                                                                    \
  {                                                                                                                                                            \
    unsigned tmp_thread_exchg_region = thread_exchg_region;                                                                                                    \
    if (!(SKIP_FIRST)) {                                                                                                                                       \
      /* first stage of this set-of-2 stages */                                                                                                                \
      const auto t0 = get_twiddle(inverse, tmp_thread_exchg_region);                                                                                           \
      exchg_dit(reg_vals[0], reg_vals[2], t0);                                                                                                                 \
      exchg_dit(reg_vals[1], reg_vals[3], t0);                                                                                                                 \
    }                                                                                                                                                          \
    /* second stage of this set-of-2 stages */                                                                                                                 \
    tmp_thread_exchg_region *= 2;                                                                                                                              \
    const auto t1 = get_twiddle(inverse, tmp_thread_exchg_region);                                                                                             \
    const auto t2 = get_twiddle(inverse, tmp_thread_exchg_region + 1);                                                                                         \
    exchg_dit(reg_vals[0], reg_vals[1], t1);                                                                                                                   \
    exchg_dit(reg_vals[2], reg_vals[3], t2);                                                                                                                   \
  }

#define ONE_EXTRA_STAGE_N2B                                                                                                                                    \
  {                                                                                                                                                            \
    const unsigned intrablock_exchg_region = (warp_id >> 1);                                                                                                   \
    const unsigned smem_logical_offset = (warp_id & 1) * 32 + lane_id;                                                                                         \
    const unsigned offset = intrablock_exchg_region * 512 + smem_logical_offset;                                                                               \
    for (int i = 0; i < 4; i++) {                                                                                                                              \
      reg_vals[i] = memory::load_cs(gmem_input + offset + i * 64);                                                                                             \
      reg_vals[i + 4] = memory::load_cs(gmem_input + offset + i * 64 + 256);                                                                                   \
    }                                                                                                                                                          \
    const auto t0 = get_twiddle(inverse, 8 * block_idx_in_ntt + intrablock_exchg_region);                                                                      \
    exchg_dit(reg_vals[0], reg_vals[4], t0);                                                                                                                   \
    exchg_dit(reg_vals[1], reg_vals[5], t0);                                                                                                                   \
    exchg_dit(reg_vals[2], reg_vals[6], t0);                                                                                                                   \
    exchg_dit(reg_vals[3], reg_vals[7], t0);                                                                                                                   \
    const unsigned offset_padded = intrablock_exchg_region * 2 * PADDED_WARP_SCRATCH_SIZE;                                                                     \
    for (int i = 0; i < 4; i++) {                                                                                                                              \
      const unsigned idx = offset_padded + PAD(smem_logical_offset + i * 64);                                                                                  \
      smem[idx] = reg_vals[i];                                                                                                                                 \
      smem[idx + PADDED_WARP_SCRATCH_SIZE] = reg_vals[i + 4];                                                                                                  \
    }                                                                                                                                                          \
  }

#define TWO_EXTRA_STAGES_N2B                                                                                                                                   \
  {                                                                                                                                                            \
    const unsigned intrablock_exchg_region = (warp_id >> 2);                                                                                                   \
    const unsigned smem_logical_offset = (warp_id & 3) * 32 + lane_id;                                                                                         \
    const unsigned offset = intrablock_exchg_region * 1024 + smem_logical_offset;                                                                              \
    for (int i = 0; i < 4; i++) {                                                                                                                              \
      reg_vals[i] = memory::load_cs(gmem_input + offset + i * 128);                                                                                            \
      reg_vals[i + 4] = memory::load_cs(gmem_input + offset + i * 128 + 512);                                                                                  \
    }                                                                                                                                                          \
    unsigned global_exchg_region = 4 * block_idx_in_ntt + intrablock_exchg_region;                                                                             \
    const auto t0 = get_twiddle(inverse, global_exchg_region);                                                                                                 \
    global_exchg_region *= 2;                                                                                                                                  \
    const auto t1 = get_twiddle(inverse, global_exchg_region);                                                                                                 \
    const auto t2 = get_twiddle(inverse, global_exchg_region + 1);                                                                                             \
    exchg_dit(reg_vals[0], reg_vals[4], t0);                                                                                                                   \
    exchg_dit(reg_vals[1], reg_vals[5], t0);                                                                                                                   \
    exchg_dit(reg_vals[2], reg_vals[6], t0);                                                                                                                   \
    exchg_dit(reg_vals[3], reg_vals[7], t0);                                                                                                                   \
    exchg_dit(reg_vals[0], reg_vals[2], t1);                                                                                                                   \
    exchg_dit(reg_vals[1], reg_vals[3], t1);                                                                                                                   \
    exchg_dit(reg_vals[4], reg_vals[6], t2);                                                                                                                   \
    exchg_dit(reg_vals[5], reg_vals[7], t2);                                                                                                                   \
    const unsigned offset_padded = intrablock_exchg_region * 4 * PADDED_WARP_SCRATCH_SIZE;                                                                     \
    for (int i = 0; i < 2; i++) {                                                                                                                              \
      for (int j = 0; j < 2; j++) {                                                                                                                            \
        const unsigned idx = offset_padded + PAD(smem_logical_offset + j * 128) + i * PADDED_WARP_SCRATCH_SIZE;                                                \
        smem[idx] = reg_vals[2 * i + j];                                                                                                                       \
        smem[idx + 2 * PADDED_WARP_SCRATCH_SIZE] = reg_vals[2 * i + j + 4];                                                                                    \
      }                                                                                                                                                        \
    }                                                                                                                                                          \
  }

#define THREE_OR_FOUR_EXTRA_STAGES_N2B(THREE_STAGES)                                                                                                           \
  {                                                                                                                                                            \
    const unsigned intrablock_exchg_region = (warp_id >> 3);                                                                                                   \
    const unsigned smem_logical_offset = (warp_id & 7) * 32 + lane_id;                                                                                         \
    const unsigned offset_padded = intrablock_exchg_region * 8 * PADDED_WARP_SCRATCH_SIZE + PAD(smem_logical_offset);                                          \
    if ((THREE_STAGES)) {                                                                                                                                      \
      const unsigned offset = intrablock_exchg_region * 2048 + smem_logical_offset;                                                                            \
      for (int i = 0; i < 4; i++) {                                                                                                                            \
        reg_vals[i] = memory::load_cs(gmem_input + offset + i * 256);                                                                                          \
        reg_vals[i + 4] = memory::load_cs(gmem_input + offset + i * 256 + 1024);                                                                               \
      }                                                                                                                                                        \
    } else {                                                                                                                                                   \
      const auto t0 = get_twiddle(inverse, block_idx_in_ntt);                                                                                                  \
      int i = threadIdx.x;                                                                                                                                     \
      int i_padded = (threadIdx.x >> 8) * PADDED_WARP_SCRATCH_SIZE + PAD(threadIdx.x & 255);                                                                   \
      for (; i < 2048; i += 512, i_padded += 2 * PADDED_WARP_SCRATCH_SIZE) {                                                                                   \
        reg_vals[0] = memory::load_cs(gmem_output + i);                                                                                                        \
        reg_vals[1] = memory::load_cs(gmem_output + i + 2048);                                                                                                 \
        exchg_dit(reg_vals[0], reg_vals[1], t0);                                                                                                               \
        smem[i_padded] = reg_vals[0];                                                                                                                          \
        smem[i_padded + 8 * PADDED_WARP_SCRATCH_SIZE] = reg_vals[1];                                                                                           \
      }                                                                                                                                                        \
      /* in theory it's possible to avoid full __syncthreads() here, see THREE_OR_FOUR_EXTRA_STAGES_B2N */                                                     \
      __syncthreads();                                                                                                                                         \
      for (int i = 0; i < 4; i++) {                                                                                                                            \
        const unsigned idx = offset_padded + i * PADDED_WARP_SCRATCH_SIZE;                                                                                     \
        reg_vals[i] = smem[idx];                                                                                                                               \
        reg_vals[i + 4] = smem[idx + 4 * PADDED_WARP_SCRATCH_SIZE];                                                                                            \
      }                                                                                                                                                        \
    }                                                                                                                                                          \
    unsigned global_exchg_region = 2 * block_idx_in_ntt + intrablock_exchg_region;                                                                             \
    const auto t0 = get_twiddle(inverse, global_exchg_region);                                                                                                 \
    global_exchg_region *= 2;                                                                                                                                  \
    const auto t1 = get_twiddle(inverse, global_exchg_region);                                                                                                 \
    const auto t2 = get_twiddle(inverse, global_exchg_region + 1);                                                                                             \
    global_exchg_region *= 2;                                                                                                                                  \
    const auto t3 = get_twiddle(inverse, global_exchg_region);                                                                                                 \
    const auto t4 = get_twiddle(inverse, global_exchg_region + 1);                                                                                             \
    const auto t5 = get_twiddle(inverse, global_exchg_region + 2);                                                                                             \
    const auto t6 = get_twiddle(inverse, global_exchg_region + 3);                                                                                             \
    exchg_dit(reg_vals[0], reg_vals[4], t0);                                                                                                                   \
    exchg_dit(reg_vals[1], reg_vals[5], t0);                                                                                                                   \
    exchg_dit(reg_vals[2], reg_vals[6], t0);                                                                                                                   \
    exchg_dit(reg_vals[3], reg_vals[7], t0);                                                                                                                   \
    exchg_dit(reg_vals[0], reg_vals[2], t1);                                                                                                                   \
    exchg_dit(reg_vals[1], reg_vals[3], t1);                                                                                                                   \
    exchg_dit(reg_vals[4], reg_vals[6], t2);                                                                                                                   \
    exchg_dit(reg_vals[5], reg_vals[7], t2);                                                                                                                   \
    exchg_dit(reg_vals[0], reg_vals[1], t3);                                                                                                                   \
    exchg_dit(reg_vals[2], reg_vals[3], t4);                                                                                                                   \
    exchg_dit(reg_vals[4], reg_vals[5], t5);                                                                                                                   \
    exchg_dit(reg_vals[6], reg_vals[7], t6);                                                                                                                   \
    for (int i = 0; i < 4; i++) {                                                                                                                              \
      const unsigned idx = offset_padded + i * PADDED_WARP_SCRATCH_SIZE;                                                                                       \
      smem[idx] = reg_vals[i];                                                                                                                                 \
      smem[idx + 4 * PADDED_WARP_SCRATCH_SIZE] = reg_vals[i + 4];                                                                                              \
    }                                                                                                                                                          \
  }

extern "C" __launch_bounds__(512, 2) __global__
    void n2b_final_7_or_8_stages(const base_field *gmem_inputs_matrix, base_field *gmem_outputs_matrix, const unsigned stride_between_input_arrays,
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
#pragma unroll 8
    for (unsigned i = 0, t = warp_id * 256 + lane_id; i < 8; i++, t += 32) {
      // puts each warp's data in its assigned smem region
      const unsigned s = (t >> 8) * PADDED_WARP_SCRATCH_SIZE + PAD(t & 255);
      memory::store(smem + s, reg_vals[i]);
    }
  }

  __syncwarp();

  unsigned warp_exchg_region = block_idx_in_ntt * 16 + warp_id;
  base_field reg_vals[8];
  base_field *warp_scratch = smem + PADDED_WARP_SCRATCH_SIZE * warp_id;

  unsigned thread_exchg_region = warp_exchg_region;
  for (int j = 0; j < 64; j += 32) {
    for (int i = 0; i < 4; i++)
      reg_vals[i] = warp_scratch[PAD(lane_id + 64 * i + j)];
    TWO_REGISTER_STAGES_N2B((stages_this_launch == 7))
    for (int i = 0; i < 4; i++)
      warp_scratch[PAD(lane_id + 64 * i + j)] = reg_vals[i];
  }
  warp_exchg_region *= 4;

  __syncwarp();

  thread_exchg_region = warp_exchg_region + (lane_id >> 3);
  const unsigned vals_start = 64 * (lane_id >> 3) + (lane_id & 7);
  for (int i = 0; i < 8; i++)
    reg_vals[i] = warp_scratch[PAD(vals_start + 8 * i)];
  THREE_REGISTER_STAGES_N2B(false)
  for (int i = 0; i < 8; i++)
    warp_scratch[PAD(vals_start + 8 * i)] = reg_vals[i];
  warp_exchg_region *= 8;

  __syncwarp();

  thread_exchg_region = warp_exchg_region + lane_id;
  for (int i = 0; i < 8; i++)
    reg_vals[i] = warp_scratch[PAD(lane_id * 8 + i)];
  THREE_REGISTER_STAGES_N2B(false)
  for (int i = 0; i < 8; i++)
    warp_scratch[PAD(lane_id * 8 + i)] = reg_vals[i];

  __syncwarp();

// unroll 2 and unroll 4 give comparable perf. Stores don't incur a stall so it's not as important to ILP them.
#pragma unroll 1
  for (unsigned i = 0, t = warp_id * 256 + lane_id; i < 8; i++, t += 32) {
    const unsigned tile = t >> 4;
    const unsigned s = (t >> 8) * PADDED_WARP_SCRATCH_SIZE + PAD(t & 255);
    auto val = smem[s];
    const unsigned g = tile * tile_stride + lane_in_tile;
    if (inverse) {
      val = base_field::mul(val, inv_sizes[log_n]);
      if (log_extension_degree) {
        const unsigned idx = __brev(g + 4096 * block_idx_in_ntt) >> (32 - log_n);
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

extern "C" __launch_bounds__(512, 2) __global__
    void n2b_final_9_to_12_stages(const base_field *gmem_inputs_matrix, base_field *gmem_outputs_matrix, const unsigned stride_between_input_arrays,
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

  base_field reg_vals[8];
  base_field *warp_scratch = smem + PADDED_WARP_SCRATCH_SIZE * warp_id;

  // This kernel can handle up to 11 stages if needed by the overall NTT.
  const unsigned extra_stages = stages_this_launch - 8;
  switch (extra_stages) {
  case 1:
    ONE_EXTRA_STAGE_N2B
    break;
  case 2:
    TWO_EXTRA_STAGES_N2B
    break;
  case 3:
    THREE_OR_FOUR_EXTRA_STAGES_N2B(true)
    break;
  case 4:
    THREE_OR_FOUR_EXTRA_STAGES_N2B(false)
    break;
  }

  __syncthreads();

  unsigned warp_exchg_region = block_idx_in_ntt * 16 + warp_id;
  unsigned thread_exchg_region = warp_exchg_region;
  for (int j = 0; j < 64; j += 32) {
    for (int i = 0; i < 4; i++)
      reg_vals[i] = warp_scratch[PAD(lane_id + 64 * i + j)];
    TWO_REGISTER_STAGES_N2B(false)
    for (int i = 0; i < 4; i++)
      warp_scratch[PAD(lane_id + 64 * i + j)] = reg_vals[i];
  }
  warp_exchg_region *= 4;

  __syncwarp();

  thread_exchg_region = warp_exchg_region + (lane_id >> 3);
  const unsigned vals_start = 64 * (lane_id >> 3) + (lane_id & 7);
  for (int i = 0; i < 8; i++)
    reg_vals[i] = warp_scratch[PAD(vals_start + 8 * i)];
  THREE_REGISTER_STAGES_N2B(false)
  for (int i = 0; i < 8; i++)
    warp_scratch[PAD(vals_start + 8 * i)] = reg_vals[i];
  warp_exchg_region *= 8;

  __syncwarp();

  thread_exchg_region = warp_exchg_region + lane_id;
  for (int i = 0; i < 8; i++)
    reg_vals[i] = warp_scratch[PAD(lane_id * 8 + i)];
  THREE_REGISTER_STAGES_N2B(false)
  for (int i = 0; i < 8; i++)
    warp_scratch[PAD(lane_id * 8 + i)] = reg_vals[i];

  __syncwarp();

// unroll 2 and unroll 4 give comparable perf. Stores don't incur a stall so it's not as important to ILP them.
#pragma unroll 1
  for (unsigned i = 0, t = warp_id * 256 + lane_id; i < 8; i++, t += 32) {
    const unsigned tile = t >> 4;
    const unsigned s = (t >> 8) * PADDED_WARP_SCRATCH_SIZE + PAD(t & 255);
    auto val = smem[s];
    const unsigned g = tile * tile_stride + lane_in_tile;
    if (inverse) {
      val = base_field::mul(val, inv_sizes[log_n]);
      if (log_extension_degree) {
        const unsigned idx = __brev(g + 4096 * block_idx_in_ntt) >> (32 - log_n);
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

extern "C" __launch_bounds__(512, 2) __global__
    void n2b_nonfinal_7_or_8_stages(const base_field *gmem_inputs_matrix, base_field *gmem_outputs_matrix, const unsigned stride_between_input_arrays,
                                    const unsigned stride_between_output_arrays, const unsigned start_stage, const unsigned stages_this_launch,
                                    const unsigned log_n, const bool inverse, const unsigned blocks_per_ntt, const unsigned log_extension_degree,
                                    const unsigned coset_idx) {
  extern __shared__ base_field smem[]; // 4096 elems

  const unsigned log_stride{log_n - start_stage - 1};
  const unsigned tile_stride{1u << (log_stride - 7)};
  const unsigned lane_in_tile = threadIdx.x & 15;
  const unsigned lane_id{threadIdx.x & 31};
  const unsigned warp_id{threadIdx.x >> 5};
  const unsigned exchg_region_sz{1u << (log_stride + 1)};
  const unsigned log_blocks_per_region = log_stride - 11; // tile_stride / 16
  const unsigned ntt_idx = blockIdx.x / blocks_per_ntt;
  const unsigned block_idx_in_ntt = blockIdx.x - ntt_idx * blocks_per_ntt;
  unsigned block_exchg_region = block_idx_in_ntt >> log_blocks_per_region;
  const unsigned block_exchg_region_start = block_exchg_region * exchg_region_sz;
  const unsigned block_start_in_exchg_region = 16 * (block_idx_in_ntt & ((1 << log_blocks_per_region) - 1));
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
    if ((start_stage == 0) && log_extension_degree && !inverse) {
      const unsigned shift = OMEGA_LOG_ORDER - log_n - log_extension_degree;
      const unsigned offset = coset_idx << shift;
#pragma unroll 8
      for (unsigned i = 0, t = warp_id * 256 + lane_id; i < 8; i++, t += 32) {
        const unsigned tile = t >> 4;
        const unsigned idx = tile * tile_stride + lane_in_tile + block_exchg_region_start + block_start_in_exchg_region;
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
      const unsigned tile = t >> 4;
      const unsigned s = lane_in_tile * PADDED_WARP_SCRATCH_SIZE + PAD(tile);
      memory::store(smem + s, reg_vals[i]);
    }
  }

  __syncthreads();

  base_field reg_vals[8];
  base_field *warp_scratch = smem + PADDED_WARP_SCRATCH_SIZE * warp_id;

  unsigned thread_exchg_region = block_exchg_region;
  for (int j = 0; j < 64; j += 32) {
    for (int i = 0; i < 4; i++)
      reg_vals[i] = warp_scratch[PAD(lane_id + 64 * i + j)];
    TWO_REGISTER_STAGES_N2B(false)
    for (int i = 0; i < 4; i++)
      warp_scratch[PAD(lane_id + 64 * i + j)] = reg_vals[i];
  }
  block_exchg_region *= 4;

  __syncwarp();

  thread_exchg_region = block_exchg_region + (lane_id >> 3);
  const unsigned vals_start = 64 * (lane_id >> 3) + (lane_id & 7);
  for (int i = 0; i < 8; i++)
    reg_vals[i] = warp_scratch[PAD(vals_start + 8 * i)];
  THREE_REGISTER_STAGES_N2B(false)
  for (int i = 0; i < 8; i++)
    warp_scratch[PAD(vals_start + 8 * i)] = reg_vals[i];
  block_exchg_region *= 8;

  __syncwarp();

  thread_exchg_region = block_exchg_region + lane_id;
  for (int i = 0; i < 8; i++)
    reg_vals[i] = warp_scratch[PAD(lane_id * 8 + i)];
  THREE_REGISTER_STAGES_N2B((stages_this_launch == 7))
  for (int i = 0; i < 8; i++)
    warp_scratch[PAD(lane_id * 8 + i)] = reg_vals[i];

  __syncthreads();

// unroll 2 and unroll 4 give comparable perf. Stores don't incur a stall so it's not as important to ILP them.
#pragma unroll 1
  for (unsigned i = 0, t = warp_id * 256 + lane_id; i < 8; i++, t += 32) {
    const unsigned tile = t >> 4;
    const unsigned s = lane_in_tile * PADDED_WARP_SCRATCH_SIZE + PAD(tile);
    const auto val = smem[s];
    const unsigned g = tile * tile_stride + lane_in_tile;
    memory::store_cs(gmem_output + g, val);
  }
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
