#pragma once

#include "goldilocks.cuh"
#include "memory.cuh"
#include "poseidon_constants.cuh"
#include "poseidon_utils.cuh"

namespace poseidon2 {

using namespace goldilocks;

template <bool IS_FULL_ROUND> static DEVICE_FORCEINLINE void apply_round_constants(poseidon_state &state, const unsigned round) {
  const auto rc = ALL_ROUND_CONSTANTS[round];
  if (IS_FULL_ROUND) {
#pragma unroll
    for (unsigned i = 0; i < STATE_WIDTH; i++)
      state[i] = field<3>::add_limbs(state[i], base_field::into<3>(rc[i]));
  } else
    state[0] = field<3>::add_limbs(state[0], base_field::into<3>(rc[0]));
}

template <bool IS_FULL_ROUND> static DEVICE_FORCEINLINE void apply_non_linearity(poseidon_state &state) {
#pragma unroll
  for (unsigned i = 0; i < STATE_WIDTH; i++) {
    if (IS_FULL_ROUND || (i == 0)) {
      const base_field f1 = base_field::field3_to_field2(state[i]);
      const base_field f2 = base_field::sqr(f1);
      const base_field f3 = base_field::mul(f1, f2);
      const base_field f4 = base_field::sqr(f2);
      state[i] = field<4>::field4_to_field3(base_field::mul_wide(f3, f4));
    }
    state[i] = field<3>::field3_to_field2_and_carry(state[i]);
  }
}

static DEVICE_FORCEINLINE void apply_M_eps_matrix(poseidon_state &state) {
#pragma unroll
  for (unsigned i = 0; i < STATE_WIDTH; i += TILE) {
    m4_times_tile(&state[i]);
  }
  field<3> acc_tile[TILE] = {0};
#pragma unroll
  for (unsigned i = 0; i < STATE_WIDTH; i += TILE) {
#pragma unroll
    for (unsigned j = 0; j < TILE; j++) {
      acc_tile[j] = field<3>::add_limbs(acc_tile[j], state[i + j]);
    }
  }
#pragma unroll
  for (unsigned i = 0; i < STATE_WIDTH; i += TILE) {
#pragma unroll
    for (unsigned j = 0; j < TILE; j++) {
      state[i + j] = field<3>::add_limbs(acc_tile[j], state[i + j]);
    }
  }
}

static DEVICE_FORCEINLINE void apply_M_I_matrix(poseidon_state &state) {
  field<3> sum{};
#pragma unroll
  for (unsigned i = 0; i < STATE_WIDTH; i++)
    sum = field<3>::add_limbs(sum, state[i]);
#pragma unroll
  for (unsigned i = 0; i < STATE_WIDTH; i++)
    state[i] = field<3>::add_limbs(sum, field<3>::shl(state[i], LOG_MU_MINUS_ONE[i]));
}

} // namespace poseidon2
