#pragma once

#include "goldilocks.cuh"

#define POSEIDON_OPTIMIZED

#include "poseidon_constants.cuh"

#undef POSEIDON_OPTIMIZED

#include "poseidon_utils.cuh"

namespace poseidon {

using namespace goldilocks;

static DEVICE_FORCEINLINE void apply_round_constants(poseidon_state &state, const unsigned round) {
  const auto rc = ALL_ROUND_CONSTANTS[round];
#pragma unroll
  for (unsigned i = 0; i < STATE_WIDTH; i++)
    state[i] = field<3>::add_limbs(state[i], base_field::into<3>(rc[i]));
}

static DEVICE_FORCEINLINE void apply_non_linearity(field<3> &state) {
  const base_field f1 = base_field::field3_to_field2(state);
  const base_field f2 = base_field::sqr(f1);
  const base_field f3 = base_field::mul(f1, f2);
  const base_field f4 = base_field::sqr(f2);
  state = field<4>::field4_to_field3(base_field::mul_wide(f3, f4));
  state = field<3>::field3_to_field2_and_carry(state);
}

#ifdef USE_SHARED_MEMORY

#define BLOCK_SIZE 64

static DEVICE_FORCEINLINE field<3> load_shared(const limb state_shared[STATE_WIDTH][3][64], const unsigned index, const unsigned tid) {
  field<3> result;
#pragma unroll
  for (unsigned i = 0; i < 3; i++)
    result[i] = state_shared[index][i][tid];
  return result;
}

static DEVICE_FORCEINLINE void store_shared(const field<3> value, limb state_shared[STATE_WIDTH][3][64], const unsigned index, const unsigned tid) {
#pragma unroll
  for (unsigned i = 0; i < 3; i++)
    state_shared[index][i][tid] = value[i];
}

static DEVICE_FORCEINLINE void apply_non_linearity(poseidon_state &state) {
  __shared__ limb state_shared[STATE_WIDTH][3][BLOCK_SIZE];
  const unsigned tid = threadIdx.x;
#pragma unroll
  for (unsigned i = 0; i < STATE_WIDTH; i++) {
    store_shared(state[i], state_shared, i, tid);
  }
#pragma unroll 1
  for (unsigned i = 0; i < STATE_WIDTH; i++) {
    field<3> s = load_shared(state_shared, i, tid);
    apply_non_linearity(s);
    store_shared(s, state_shared, i, tid);
  }
#pragma unroll
  for (unsigned i = 0; i < STATE_WIDTH; i++) {
    state[i] = load_shared(state_shared, i, tid);
  }
}

#undef BLOCK_SIZE

#else

static DEVICE_FORCEINLINE void apply_non_linearity(poseidon_state &state) {
#pragma unroll
  for (auto &s : state)
    apply_non_linearity(s);
}

#endif

static DEVICE_FORCEINLINE void apply_mds_matrix(poseidon_state &state) {
  poseidon_state result{};
#pragma unroll
  for (unsigned row = 0; row < STATE_WIDTH; row++) {
    field<3> acc;
#pragma unroll
    for (unsigned i = 0; i < STATE_WIDTH; i++) {
      const unsigned index = MDS_MATRIX_EXPS_ORDER[i];
      const unsigned col = (index + row) % STATE_WIDTH;
      field<3> value = state[col];
      acc = i ? field<3>::add_limbs(acc, value) : value;
      const unsigned shift = MDS_MATRIX_SHIFTS[i];
      if (shift)
        acc = field<3>::shl(acc, shift);
    }
    result[row] = acc;
  }
#pragma unroll
  for (unsigned row = 0; row < STATE_WIDTH; row++)
    state[row] = result[row];
}

static DEVICE_FORCEINLINE void apply_fused_round_constants(poseidon_state &state) {
#pragma unroll
  for (unsigned i = 0; i < STATE_WIDTH; i++)
    state[i] = field<3>::add_limbs(state[i], base_field::into<3>(ROUND_CONSTANTS_FUSED_LAST_FULL_AND_FIRST_PARTIAL[i]));
}

static DEVICE_FORCEINLINE void full_and_partial_round_fused_mul(poseidon_state &state) {
  base_field values[STATE_WIDTH];
#pragma unroll
  for (unsigned i = 0; i < STATE_WIDTH; i++)
    values[i] = field<3>::field3_to_field2(state[i]);
#pragma unroll
  for (unsigned row = 0; row < STATE_WIDTH; row++) {
    const base_field *matrix_row = FUSED_DENSE_MATRIX_LAST_FULL_AND_FIRST_PARTIAL[row];
    field<5> acc{};
#pragma unroll
    for (unsigned col = 0; col < STATE_WIDTH; col++)
      acc = field<5>::add_limbs(acc, field<4>::into<5>(base_field::mul_wide(values[col], matrix_row[col])));
    state[row] = field<3>::field3_to_field2_and_carry(field<5>::field5_to_field3(acc));
  }
}

static DEVICE_FORCEINLINE void partial_round_optimized(poseidon_state &state, const unsigned round) {
  const unsigned partial_round = round - HALF_NUM_FULL_ROUNDS;
  apply_non_linearity(state[0]);
  state[0] = field<3>::add_limbs(state[0], base_field::into<3>(ROUND_CONSTANTS_FOR_FUSED_S_BOXES[partial_round]));
  base_field values[STATE_WIDTH];
#pragma unroll
  for (unsigned i = 0; i < STATE_WIDTH; i++)
    values[i] = field<3>::field3_to_field2(state[i]);
  const auto vs = VS_FOR_PARTIAL_ROUNDS[partial_round];
  field<5> acc = base_field::into<5>(values[0]);
#pragma unroll
  for (unsigned i = 0; i < STATE_WIDTH - 1; i++)
    acc = field<5>::add_limbs(acc, field<4>::into<5>(base_field::mul_wide(values[i + 1], vs[i])));
  state[0] = field<3>::field3_to_field2_and_carry(field<5>::field5_to_field3(acc));
  const auto w_hats = W_HATS_FOR_PARTIAL_ROUNDS[partial_round];
#pragma unroll
  for (unsigned i = 1; i < STATE_WIDTH; i++)
    state[i] = field<3>::add_limbs(field<3>::field3_to_field2_and_carry(field<4>::field4_to_field3(base_field::mul_wide(values[0], w_hats[i - 1]))),
                                   base_field::into<3>(values[i]));
}

} // namespace poseidon
