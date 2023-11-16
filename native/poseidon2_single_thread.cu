#include "poseidon2_single_thread.cuh"

namespace poseidon2 {

using namespace goldilocks;

// https://eprint.iacr.org/2023/323.pdf Fig. 1
static DEVICE_FORCEINLINE void permutation(poseidon_state &state) {
  apply_M_eps_matrix(state);
#pragma unroll
  for (unsigned round = 0; round < TOTAL_NUM_ROUNDS; round++) {
    const bool is_full_round = round < HALF_NUM_FULL_ROUNDS || round >= HALF_NUM_FULL_ROUNDS + NUM_PARTIAL_ROUNDS;
    if (is_full_round) {
      apply_round_constants<true>(state, round);
      apply_non_linearity<true>(state);
      apply_M_eps_matrix(state);
    } else {
      apply_round_constants<false>(state, round);
      apply_non_linearity<false>(state);
      apply_M_I_matrix(state);
    }
  }
}

EXTERN __global__ void poseidon2_single_thread_leaves_kernel(const base_field *values, base_field *results, const unsigned rows_count,
                                                             const unsigned cols_count, const unsigned count, bool load_intermediate, bool store_intermediate) {
  static_assert(RATE == 8);
  static_assert(CAPACITY == 4);
  single_thread_leaves_impl<permutation>(values, results, rows_count, cols_count, count, load_intermediate, store_intermediate);
}

EXTERN __global__ void poseidon2_single_thread_nodes_kernel(const field<4> *values, base_field *results, const unsigned count) {
  static_assert(RATE == 8);
  static_assert(CAPACITY == 4);
  single_thread_nodes_impl<permutation>(values, results, count);
}

} // namespace poseidon2
