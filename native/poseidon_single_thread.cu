#include "goldilocks.cuh"

#if __CUDA_ARCH__ == 800
#define USE_SHARED_MEMORY
#endif
#include "poseidon_single_thread.cuh"
#if __CUDA_ARCH__ == 800
#undef USE_SHARED_MEMORY
#endif

namespace poseidon {

using namespace goldilocks;

static DEVICE_FORCEINLINE void permutation(poseidon_state &state) {
#pragma unroll 1
  for (unsigned round = 0; round < TOTAL_NUM_ROUNDS; round++) {
    if (round < HALF_NUM_FULL_ROUNDS || round >= HALF_NUM_FULL_ROUNDS + NUM_PARTIAL_ROUNDS) {
      if (round != HALF_NUM_FULL_ROUNDS + NUM_PARTIAL_ROUNDS)
        apply_round_constants(state, round);
      apply_non_linearity(state);
      if (round == HALF_NUM_FULL_ROUNDS - 1) {
        apply_fused_round_constants(state);
        full_and_partial_round_fused_mul(state);
      } else {
        apply_mds_matrix(state);
      }
    } else {
      partial_round_optimized(state, round);
    }
  }
}

#if __CUDA_ARCH__ == 800
#define MIN_BLOCKS_COUNT 10
#else
#define MIN_BLOCKS_COUNT 12
#endif
EXTERN __launch_bounds__(64, MIN_BLOCKS_COUNT) __global__
    void poseidon_single_thread_leaves_kernel(const base_field *values, base_field *results, const unsigned rows_count, const unsigned cols_count,
                                              const unsigned count, bool load_intermediate, bool store_intermediate) {
  single_thread_leaves_impl<permutation>(values, results, rows_count, cols_count, count, load_intermediate, store_intermediate);
}
#undef MIN_BLOCKS_COUNT

EXTERN __launch_bounds__(64, 12) __global__ void poseidon_single_thread_nodes_kernel(const field<4> *values, base_field *results, const unsigned count) {
  static_assert(RATE == 2 * CAPACITY);
  single_thread_nodes_impl<permutation>(values, results, count);
}

} // namespace poseidon
