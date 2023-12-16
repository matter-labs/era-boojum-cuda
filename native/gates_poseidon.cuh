#pragma once

#include "gates.cuh"
#include "poseidon_single_thread.cuh"
// do not swap the order
#include "poseidon2_single_thread.cuh"

namespace gates {

template <unsigned WITNESSES_COUNT>
DEVICE_FORCEINLINE void reset(const base_field *variables, const base_field *witnesses, const extension_field *challenge_bases,
                              extension_field *challenge_powers, extension_field *quotient_sums, const unsigned challenges_count, const unsigned inputs_stride,
                              goldilocks::field<3> &value, unsigned &witnesses_index, unsigned &variables_index) {
  const base_field src = goldilocks::field<3>::field3_to_field2(value);
  const base_field dst = WITNESSES_COUNT != 0 && witnesses_index < WITNESSES_COUNT ? load(witnesses, witnesses_index++, inputs_stride)
                                                                                   : load(variables, variables_index++, inputs_stride);
  value = base_field::into<3>(dst);
  const base_field contribution = base_field::sub(src, dst);
  GATE_PUSH(contribution)
}

using namespace poseidon_common;

template <unsigned WITNESSES_COUNT>
DEVICE_FORCEINLINE void poseidon_repetition(const base_field *variables, const base_field *witnesses, const extension_field *challenge_bases,
                                            extension_field *challenge_powers, extension_field *quotient_sums, const unsigned challenges_count,
                                            const unsigned inputs_stride) {
  poseidon_state state{};
#pragma unroll
  for (unsigned i = 0; i < STATE_WIDTH; i++)
    state[i] = base_field::into<3>(load(variables, i, inputs_stride));
  unsigned witnesses_index = 0;
  unsigned variables_index = STATE_WIDTH * 2;
#pragma unroll 1
  for (unsigned round = 0; round < TOTAL_NUM_ROUNDS; round++) {
    if (round < HALF_NUM_FULL_ROUNDS || round >= HALF_NUM_FULL_ROUNDS + NUM_PARTIAL_ROUNDS) {
      if (round != 0)
#pragma unroll
        for (auto &value : state)
          reset<WITNESSES_COUNT>(variables, witnesses, challenge_bases, challenge_powers, quotient_sums, challenges_count, inputs_stride, value,
                                 witnesses_index, variables_index);
      if (round != HALF_NUM_FULL_ROUNDS + NUM_PARTIAL_ROUNDS)
        poseidon::apply_round_constants(state, round);
      poseidon::apply_non_linearity(state);
      if (round == HALF_NUM_FULL_ROUNDS - 1) {
        poseidon::apply_fused_round_constants(state);
        poseidon::full_and_partial_round_fused_mul(state);
      } else {
        poseidon::apply_mds_matrix(state);
      }
    } else {
      reset<WITNESSES_COUNT>(variables, witnesses, challenge_bases, challenge_powers, quotient_sums, challenges_count, inputs_stride, state[0], witnesses_index,
                             variables_index);
      poseidon::partial_round_optimized(state, round);
    }
  }
#pragma unroll
  for (unsigned i = 0; i < STATE_WIDTH; i++) {
    const base_field dst = goldilocks::field<3>::field3_to_field2(state[i]);
    const base_field src = load(variables, i + STATE_WIDTH, inputs_stride);
    const base_field contribution = base_field::sub(src, dst);
    GATE_PUSH(contribution)
  }
}

template <unsigned WITNESSES_COUNT>
DEVICE_FORCEINLINE void poseidon2_repetition(const base_field *variables, const base_field *witnesses, const extension_field *challenge_bases,
                                             extension_field *challenge_powers, extension_field *quotient_sums, const unsigned challenges_count,
                                             const unsigned inputs_stride) {
  poseidon_state state{};
#pragma unroll
  for (unsigned i = 0; i < STATE_WIDTH; i++)
    state[i] = base_field::into<3>(load(variables, i, inputs_stride));
  unsigned witnesses_index = 0;
  unsigned variables_index = STATE_WIDTH * 2;
  poseidon2::apply_M_eps_matrix(state);
#pragma unroll 1
  for (unsigned round = 0; round < TOTAL_NUM_ROUNDS; round++) {
    const bool is_full_round = round < HALF_NUM_FULL_ROUNDS || round >= HALF_NUM_FULL_ROUNDS + NUM_PARTIAL_ROUNDS;
    if (is_full_round) {
      if (round != 0)
#pragma unroll
        for (auto &value : state)
          reset<WITNESSES_COUNT>(variables, witnesses, challenge_bases, challenge_powers, quotient_sums, challenges_count, inputs_stride, value,
                                 witnesses_index, variables_index);
      poseidon2::apply_round_constants<true>(state, round);
      poseidon2::apply_non_linearity<true>(state);
      poseidon2::apply_M_eps_matrix(state);
    } else {
      poseidon2::apply_round_constants<false>(state, round);
      reset<WITNESSES_COUNT>(variables, witnesses, challenge_bases, challenge_powers, quotient_sums, challenges_count, inputs_stride, state[0], witnesses_index,
                             variables_index);
      poseidon2::apply_non_linearity<false>(state);
      poseidon2::apply_M_I_matrix(state);
    }
  }
#pragma unroll
  for (unsigned i = 0; i < STATE_WIDTH; i++) {
    const base_field dst = goldilocks::field<3>::field3_to_field2(state[i]);
    const base_field src = load(variables, i + STATE_WIDTH, inputs_stride);
    const base_field contribution = base_field::sub(src, dst);
    GATE_PUSH(contribution)
  }
}

DEVICE_FORCEINLINE void poseidon2_external_matrix(const base_field *variables, const extension_field *challenge_bases, extension_field *challenge_powers,
                                                  extension_field *quotient_sums, const unsigned challenges_count, const unsigned inputs_stride) {
  poseidon_state state{};
#pragma unroll
  for (unsigned i = 0; i < STATE_WIDTH; i++)
    state[i] = base_field::into<3>(load(variables, i, inputs_stride));
  poseidon2::apply_M_eps_matrix(state);
#pragma unroll
  for (unsigned i = 0; i < STATE_WIDTH; i++) {
    const base_field term = goldilocks::field<3>::field3_to_field2(state[i]);
    const base_field result = load(variables, i + STATE_WIDTH, inputs_stride);
    const base_field contribution = base_field::sub(term, result);
    GATE_PUSH(contribution)
  }
}

DEVICE_FORCEINLINE void poseidon2_internal_matrix(const base_field *variables, const extension_field *challenge_bases, extension_field *challenge_powers,
                                                  extension_field *quotient_sums, const unsigned challenges_count, const unsigned inputs_stride) {
  poseidon_state state{};
#pragma unroll
  for (unsigned i = 0; i < STATE_WIDTH; i++)
    state[i] = base_field::into<3>(load(variables, i, inputs_stride));
  poseidon2::apply_M_I_matrix(state);
#pragma unroll
  for (unsigned i = 0; i < STATE_WIDTH; i++) {
    const base_field term = goldilocks::field<3>::field3_to_field2(state[i]);
    const base_field result = load(variables, i + STATE_WIDTH, inputs_stride);
    const base_field contribution = base_field::sub(term, result);
    GATE_PUSH(contribution)
  }
}

#define GATE_POSEIDON(variables_offset, witnesses_offset)                                                                                                      \
  {                                                                                                                                                            \
    poseidon_repetition<witnesses_offset>(variables, witnesses, challenge_bases, challenge_powers, quotient_sums, challenges_count, inputs_stride);            \
    variables += (variables_offset) * inputs_stride;                                                                                                           \
    witnesses += (witnesses_offset) * inputs_stride;                                                                                                           \
  }

#define GATE_POSEIDON2(variables_offset, witnesses_offset)                                                                                                     \
  {                                                                                                                                                            \
    poseidon2_repetition<witnesses_offset>(variables, witnesses, challenge_bases, challenge_powers, quotient_sums, challenges_count, inputs_stride);           \
    variables += (variables_offset) * inputs_stride;                                                                                                           \
    witnesses += (witnesses_offset) * inputs_stride;                                                                                                           \
  }

#define GATE_POSEIDON2_EXTERNAL_MATRIX                                                                                                                         \
  {                                                                                                                                                            \
    poseidon2_external_matrix(variables, challenge_bases, challenge_powers, quotient_sums, challenges_count, inputs_stride);                                   \
    variables += (2 * STATE_WIDTH) * inputs_stride;                                                                                                            \
  }

#define GATE_POSEIDON2_INTERNAL_MATRIX                                                                                                                         \
  {                                                                                                                                                            \
    poseidon2_internal_matrix(variables, challenge_bases, challenge_powers, quotient_sums, challenges_count, inputs_stride);                                   \
    variables += (2 * STATE_WIDTH) * inputs_stride;                                                                                                            \
  }

} // namespace gates
