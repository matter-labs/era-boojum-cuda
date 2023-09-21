#include "context.cuh"
#include "goldilocks.cuh"
#include "memory.cuh"

using namespace goldilocks;

namespace ntt {

#define PAD(X) (((X) >> 4) * 17 + ((X)&15))
static constexpr unsigned PADDED_WARP_SCRATCH_SIZE = (256 / 16) * 17 + 1;
// for debugging:
// #define PAD(X) (X)
// static constexpr unsigned PADDED_WARP_SCRATCH_SIZE = 256;

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

#include "ntt_b2n.cuh"
#include "ntt_n2b.cuh"

} // namespace ntt
