#include "goldilocks.cuh"
#include "memory.cuh"
#include "poseidon_constants.cuh"

namespace poseidon {

using namespace goldilocks;

template <unsigned STRIDE> static DEVICE_FORCEINLINE void apply_round_constants(field<3> *state, const unsigned round, const unsigned wid) {
  constexpr unsigned COUNT = STATE_WIDTH / STRIDE;
  const auto rc = ALL_ROUND_CONSTANTS[round] + wid;
#pragma unroll
  for (unsigned i = 0; i < COUNT; i++)
    state[i] = field<3>::add_limbs(state[i], base_field::into<3>(rc[i * STRIDE]));
}

template <bool IS_FULL_ROUND, unsigned STRIDE> static DEVICE_FORCEINLINE void apply_non_linearity(field<3> *state, const unsigned wid) {
  constexpr unsigned COUNT = STATE_WIDTH / STRIDE;
#pragma unroll
  for (unsigned i = 0; i < COUNT; i++) {
    if (IS_FULL_ROUND || (i == 0 && wid == 0)) {
      const base_field f1 = base_field::field3_to_field2(state[i]);
      const base_field f2 = base_field::sqr(f1);
      const base_field f3 = base_field::mul(f1, f2);
      const base_field f4 = base_field::sqr(f2);
      state[i] = field<4>::field4_to_field3(base_field::mul_wide(f3, f4));
    }
    state[i] = field<3>::field3_to_field2_and_carry(state[i]);
  }
}

template <unsigned STRIDE> static DEVICE_FORCEINLINE void apply_mds_matrix(field<3> *state, const unsigned wid, const unsigned tid) {
  constexpr unsigned COUNT = STATE_WIDTH / STRIDE;
  __shared__ limb shared[STATE_WIDTH][3][32];
  __syncthreads();
  auto s = &shared[wid];
#pragma unroll
  for (unsigned i = 0; i < COUNT; i++) {
    field<3> value = state[i];
    auto ss = *s;
#pragma unroll
    for (unsigned j = 0; j < 3; j++)
      ss[j][tid] = value[j];
    s += STRIDE;
  }
  __syncthreads();
  field<3> values[STATE_WIDTH];
#pragma unroll
  for (unsigned i = 0; i < COUNT; i++) {
    values[i * STRIDE] = state[i];
#pragma unroll
    for (unsigned j = 1; j < STRIDE; j++) {
      const unsigned index = i * STRIDE + j;
      const unsigned shared_index = (index + wid) % STATE_WIDTH;
      auto ss = shared[shared_index];
      field<3> value;
#pragma unroll
      for (unsigned k = 0; k < 3; k++)
        value[k] = ss[k][tid];
      values[index] = value;
    }
  }

  field<3> result[COUNT];

#pragma unroll
  for (unsigned row = 0; row < COUNT; row++) {
    field<3> acc;
#pragma unroll
    for (unsigned i = 0; i < STATE_WIDTH; i++) {
      const unsigned index = MDS_MATRIX_EXPS_ORDER[i];
      const unsigned col = (index + row * STRIDE) % STATE_WIDTH;
      field<3> value = values[col];
      acc = i ? field<3>::add_limbs(acc, value) : value;
      const unsigned shift = MDS_MATRIX_SHIFTS[i];
      if (shift)
        acc = field<3>::shl(acc, shift);
    }
    result[row] = acc;
  }
#pragma unroll
  for (unsigned i = 0; i < COUNT; i++)
    state[i] = result[i];
}

template <unsigned STRIDE> static DEVICE_FORCEINLINE void permutation(field<3> *state, const unsigned wid, const unsigned tid) {
#pragma unroll
  for (unsigned round = 0; round < TOTAL_NUM_ROUNDS; round++) {
    const bool is_full_round = round < HALF_NUM_FULL_ROUNDS || round >= HALF_NUM_FULL_ROUNDS + NUM_PARTIAL_ROUNDS;
    apply_round_constants<STRIDE>(state, round, wid);
    if (is_full_round)
      apply_non_linearity<true, STRIDE>(state, wid);
    else
      apply_non_linearity<false, STRIDE>(state, wid);
    apply_mds_matrix<STRIDE>(state, wid, tid);
  }
}

template <unsigned STRIDE>
static DEVICE_FORCEINLINE void poseidon_cooperative_nodes_kernel_impl(const field<4> *values, base_field *results, const unsigned count) {
  static_assert(RATE == 2 * CAPACITY);
  static_assert(STATE_WIDTH % STRIDE == 0);
  static_assert(CAPACITY % STRIDE == 0);
  constexpr unsigned COUNT = STATE_WIDTH / STRIDE;
  unsigned tid = threadIdx.x;
  unsigned wid = threadIdx.y;
  const unsigned gid = tid + blockIdx.x * blockDim.x;
  field<3> state[COUNT];
  values += count * wid + gid;
#pragma unroll
  for (unsigned i = 0; i < CAPACITY / STRIDE; i++, values += count * STRIDE) {
    const auto value = memory::load_cs(values);
    auto v2 = reinterpret_cast<const base_field *>(&value);
#pragma unroll
    for (unsigned j = 0; j < 2; j++)
      state[j * CAPACITY / STRIDE + i] = base_field::into<3>(v2[j]);
  }
  permutation<STRIDE>(state, wid, tid);
  results += count * wid + gid;
#pragma unroll
  for (unsigned i = 0; i < CAPACITY / STRIDE; i++, results += count * STRIDE)
    memory::store_cs(results, base_field::field3_to_field2(state[i]));
}

EXTERN __launch_bounds__(4 * 32, 1) __global__ void poseidon_cooperative_nodes_kernel(const field<4> *values, base_field *results, const unsigned count) {
  poseidon_cooperative_nodes_kernel_impl<4>(values, results, count);
}

} // namespace poseidon
