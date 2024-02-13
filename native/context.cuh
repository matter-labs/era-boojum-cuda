#pragma once
#include "goldilocks.cuh"
#include "memory.cuh"

using namespace goldilocks;

namespace goldilocks {

static constexpr unsigned OMEGA_LOG_ORDER = 24;

struct powers_layer_data {
  const base_field *values;
  unsigned mask;
  unsigned log_count;
};

struct powers_data {
  powers_layer_data fine;
  powers_layer_data coarse;
};

} // namespace goldilocks

EXTERN __device__ __constant__ powers_data powers_data_w;
EXTERN __device__ __constant__ powers_data powers_data_w_bitrev_for_ntt;
EXTERN __device__ __constant__ powers_data powers_data_w_inv_bitrev_for_ntt;
EXTERN __device__ __constant__ powers_data powers_data_g_f;
EXTERN __device__ __constant__ powers_data powers_data_g_i;
EXTERN __device__ __constant__ base_field inv_sizes[OMEGA_LOG_ORDER + 1];

constexpr unsigned FINEST_LOG_COUNT = 7;
constexpr unsigned COARSER_LOG_COUNT = 8;
constexpr unsigned COARSEST_LOG_COUNT = 6;
EXTERN __device__ __constant__ base_field ntt_w_powers_bitrev_finest[1 << FINEST_LOG_COUNT];
EXTERN __device__ __constant__ base_field ntt_w_powers_bitrev_coarser[1 << COARSER_LOG_COUNT];
EXTERN __device__ __constant__ base_field ntt_w_powers_bitrev_coarsest[1 << COARSEST_LOG_COUNT];
EXTERN __device__ __constant__ base_field ntt_w_inv_powers_bitrev_finest[1 << FINEST_LOG_COUNT];
EXTERN __device__ __constant__ base_field ntt_w_inv_powers_bitrev_coarser[1 << COARSER_LOG_COUNT];
EXTERN __device__ __constant__ base_field ntt_w_inv_powers_bitrev_coarsest[1 << COARSEST_LOG_COUNT];

namespace goldilocks {

DEVICE_FORCEINLINE base_field get_power(const powers_data &data, const unsigned index, const bool inverse) {
  const unsigned idx = inverse ? (1u << OMEGA_LOG_ORDER) - index : index;
  const unsigned coarse_idx = (idx >> data.fine.log_count) & data.coarse.mask;
  const base_field coarse = memory::load_ca(data.coarse.values + coarse_idx);
  const unsigned fine_idx = idx & data.fine.mask;
  if (fine_idx == 0)
    return coarse;
  const base_field fine = memory::load_ca(data.fine.values + fine_idx);
  return base_field::mul(fine, coarse);
}

DEVICE_FORCEINLINE base_field get_power_of_w(const unsigned index, const bool inverse) { return get_power(powers_data_w, index, inverse); }

DEVICE_FORCEINLINE base_field get_power_of_g(const unsigned index, const bool inverse) {
  return inverse ? get_power(powers_data_g_i, index, false) : get_power(powers_data_g_f, index, false);
}

} // namespace goldilocks
