#include "context.cuh"

__device__ __constant__ powers_data powers_data_w;
__device__ __constant__ powers_data powers_data_w_bitrev_for_ntt;
__device__ __constant__ powers_data powers_data_w_inv_bitrev_for_ntt;
__device__ __constant__ powers_data powers_data_g_f;
__device__ __constant__ powers_data powers_data_g_i;
__device__ __constant__ base_field inv_sizes[OMEGA_LOG_ORDER + 1];

__device__ __constant__ base_field ntt_w_powers_bitrev_finest[1 << FINEST_LOG_COUNT];
__device__ __constant__ base_field ntt_w_powers_bitrev_coarser[1 << COARSER_LOG_COUNT];
__device__ __constant__ base_field ntt_w_powers_bitrev_coarsest[1 << COARSEST_LOG_COUNT];
__device__ __constant__ base_field ntt_w_inv_powers_bitrev_finest[1 << FINEST_LOG_COUNT];
__device__ __constant__ base_field ntt_w_inv_powers_bitrev_coarser[1 << COARSER_LOG_COUNT];
__device__ __constant__ base_field ntt_w_inv_powers_bitrev_coarsest[1 << COARSEST_LOG_COUNT];
