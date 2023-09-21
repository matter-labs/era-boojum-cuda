#include "context.cuh"

__device__ __constant__ powers_data powers_data_w;
__device__ __constant__ powers_data powers_data_w_bitrev_for_ntt;
__device__ __constant__ powers_data powers_data_w_inv_bitrev_for_ntt;
__device__ __constant__ powers_data powers_data_g_f;
__device__ __constant__ powers_data powers_data_g_i;
__device__ __constant__ base_field inv_sizes[OMEGA_LOG_ORDER + 1];
