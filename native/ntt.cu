#include "context.cuh"
#include "goldilocks.cuh"
#include "memory.cuh"

using namespace goldilocks;

namespace ntt {

__device__ __forceinline__ void exchg_dit(base_field &a, base_field &b, const base_field &twiddle) {
  b = base_field::mul(b, twiddle);
  const auto a_tmp = a;
  a = base_field::add(a_tmp, b);
  b = base_field::sub(a_tmp, b);
}

__device__ __forceinline__ void exchg_dif(base_field &a, base_field &b, const base_field &twiddle) {
  const auto a_tmp = a;
  a = base_field::add(a_tmp, b);
  b = base_field::sub(a_tmp, b);
  b = base_field::mul(b, twiddle);
}

// This is a little tricky:
// it assumes "i" NEEDS to be bitreved and accounts for that by assuming "fine" and "coarse"
// arrays are already bitreved.
__device__ __forceinline__ base_field get_twiddle(const bool inverse, const unsigned i) {
  const powers_data &data = inverse ? powers_data_w_inv_bitrev_for_ntt : powers_data_w_bitrev_for_ntt;
  unsigned fine_idx = (i >> data.coarse.log_count) & data.fine.mask;
  unsigned coarse_idx = i & data.coarse.mask;
  auto coarse = memory::load_ca(data.coarse.values + coarse_idx);
  if (fine_idx == 0)
    return coarse;
  auto fine = memory::load_ca(data.fine.values + fine_idx);
  return base_field::mul(fine, coarse);
}

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

template <unsigned VALS_PER_WARP, unsigned LOG_VALS_PER_THREAD>
DEVICE_FORCEINLINE void load_initial_twiddles_warp(base_field *twiddle_cache, const unsigned lane_id, const unsigned gmem_offset, const bool inverse) {
  // cooperatively loads all the twiddles this warp needs for intrawarp stages
  base_field *twiddles_this_stage = twiddle_cache;
  unsigned num_twiddles_this_stage = VALS_PER_WARP >> 1;
  unsigned exchg_region_offset = gmem_offset >> 1;
#pragma unroll
  for (unsigned stage = 0; stage < LOG_VALS_PER_THREAD; stage++) {
#pragma unroll
    for (unsigned i = lane_id; i < num_twiddles_this_stage; i += 32) {
      twiddles_this_stage[i] = get_twiddle(inverse, i + exchg_region_offset);
    }
    twiddles_this_stage += num_twiddles_this_stage;
    num_twiddles_this_stage >>= 1;
    exchg_region_offset >>= 1;
  }

  // loads final 31 twiddles with minimal divergence. pain.
  const unsigned lz = __clz(lane_id);
  const unsigned stage_offset = 5 - (32 - lz);
  const unsigned mask = (1 << (32 - lz)) - 1;
  if (lane_id > 0) {
    exchg_region_offset >>= stage_offset;
    twiddles_this_stage[lane_id ^ 31] = get_twiddle(inverse, (lane_id ^ mask) + exchg_region_offset);
  }

  __syncwarp();
}

template <unsigned LOG_VALS_PER_THREAD>
DEVICE_FORCEINLINE void load_noninitial_twiddles_warp(base_field *twiddle_cache, const unsigned lane_id, const unsigned warp_id,
                                                      const unsigned block_exchg_region_offset, const bool inverse) {
  // cooperatively loads all the twiddles this warp needs for intrawarp stages
  static_assert(LOG_VALS_PER_THREAD <= 4);
  constexpr unsigned NUM_INTRAWARP_STAGES = LOG_VALS_PER_THREAD + 1;

  // tile size 16: num twiddles = vals per warp / 2 / 16 == vals per thread
  unsigned num_twiddles_first_stage = 1 << LOG_VALS_PER_THREAD;
  unsigned exchg_region_offset = block_exchg_region_offset + warp_id * num_twiddles_first_stage;

  // loads 2 * num_twiddles_first_stage - 1 twiddles with minimal divergence. pain.
  if (lane_id > 0 && lane_id < 2 * num_twiddles_first_stage) {
    const unsigned lz = __clz(lane_id);
    const unsigned stage_offset = NUM_INTRAWARP_STAGES - (32 - lz);
    const unsigned mask = (1 << (32 - lz)) - 1;
    exchg_region_offset >>= stage_offset;
    twiddle_cache[lane_id ^ (2 * num_twiddles_first_stage - 1)] = get_twiddle(inverse, (lane_id ^ mask) + exchg_region_offset);
  }

  __syncwarp();
}

// this is a common pattern that happened to arise in several kernels
template <unsigned VALS_PER_THREAD, bool inverse>
DEVICE_FORCEINLINE void apply_lde_factors(base_field *scratch, const unsigned gmem_offset, const unsigned lane_id, const unsigned log_n,
                                          const unsigned log_extension_degree, const unsigned coset_idx) {
#pragma unroll 1
  for (unsigned i = 0; i < VALS_PER_THREAD; i++) {
    base_field val = scratch[i];
    const unsigned idx = __brev(gmem_offset + 64 * (i >> 1) + 2 * lane_id + (i & 1)) >> (32 - log_n);
    if (coset_idx) {
      const unsigned shift = OMEGA_LOG_ORDER - log_n - log_extension_degree;
      const unsigned offset = coset_idx << shift;
      auto power_of_w = get_power_of_w(idx * offset, inverse);
      val = base_field::mul(val, power_of_w);
    }
    auto power_of_g = get_power_of_g(idx, inverse);
    scratch[i] = base_field::mul(val, power_of_g);
  }
}

static __device__ constexpr unsigned NTTS_PER_BLOCK = 8;

#include "ntt_b2n.cuh"
#include "ntt_n2b.cuh"

} // namespace ntt
