#include "goldilocks.cuh"
#include "memory.cuh"
#include "poseidon_constants.cuh"
#include "poseidon_utils.cuh"

namespace poseidon2 {

using namespace goldilocks;

typedef limb block_states[STATE_WIDTH][3][32];

template <bool IS_FULL_ROUND> static DEVICE_FORCEINLINE void apply_round_constants(field<3> *state, const unsigned round, const unsigned wid) {
  const auto rc = ALL_ROUND_CONSTANTS[round] + TILE * wid;
  if (IS_FULL_ROUND) {
#pragma unroll
    for (unsigned i = 0; i < TILE; i++)
      state[i] = field<3>::add_limbs(state[i], base_field::into<3>(rc[i]));
  } else if (wid == 0) {
    state[0] = field<3>::add_limbs(state[0], base_field::into<3>(rc[0]));
  }
}

template <bool IS_FULL_ROUND> static DEVICE_FORCEINLINE void apply_non_linearity(field<3> *state, const unsigned wid) {
#pragma unroll
  for (unsigned i = 0; i < TILE; i++) {
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

static DEVICE_FORCEINLINE field<3> load_shared(const block_states &shared_states, const unsigned index) {
  field<3> result;
#pragma unroll
  for (unsigned i = 0; i < 3; i++)
    result[i] = shared_states[index][i][threadIdx.x];
  return result;
}

static DEVICE_FORCEINLINE void store_shared(const field<3> value, block_states &shared_states, const unsigned index) {
#pragma unroll
  for (unsigned i = 0; i < 3; i++)
    shared_states[index][i][threadIdx.x] = value[i];
}

static DEVICE_FORCEINLINE void apply_M_eps_matrix(field<3> *state, const unsigned wid, block_states &shared_states) {
  m4_times_tile(state);
  // Publish this thread's m4_times_tile
  __syncthreads();
  for (unsigned i = 0; i < TILE; i++)
    store_shared(state[i], shared_states, i + TILE * wid);
  __syncthreads();
  field<3> acc_tile[TILE] = {state[0], state[1], state[2], state[3]};
#pragma unroll
  for (unsigned i = 0; i < STATE_WIDTH; i += TILE) {
    if (i != TILE * wid) {
#pragma unroll
      for (unsigned j = 0; j < TILE; j++)
        acc_tile[j] = field<3>::add_limbs(acc_tile[j], load_shared(shared_states, i + j));
    }
  }
#pragma unroll
  for (unsigned i = 0; i < TILE; i++)
    state[i] = field<3>::add_limbs(acc_tile[i], state[i]);
}

static DEVICE_FORCEINLINE void apply_M_I_matrix(field<3> *state, const unsigned wid, block_states &shared_states) {
  field<3> sum{state[0]};
#pragma unroll
  for (unsigned i = 1; i < TILE; i++)
    sum = field<3>::add_limbs(sum, state[i]);
  // Publish this thread's tile sum
  __syncthreads();
  store_shared(sum, shared_states, wid);
  __syncthreads();
#pragma unroll
  for (unsigned i = 0; i < TILES_PER_STATE; i++) {
    if (i != wid)
      sum = field<3>::add_limbs(sum, load_shared(shared_states, i));
  }
#pragma unroll
  for (unsigned i = 0; i < TILE; i++)
    state[i] = field<3>::add_limbs(sum, field<3>::shl(state[i], LOG_MU_MINUS_ONE[i + TILE * wid]));
}

// https://eprint.iacr.org/2023/323.pdf Fig. 1
static DEVICE_FORCEINLINE void permutation(field<3> *state, const unsigned wid, block_states &shared_states) {
  apply_M_eps_matrix(state, wid, shared_states);
#pragma unroll
  for (unsigned round = 0; round < TOTAL_NUM_ROUNDS; round++) {
    const bool is_full_round = round < HALF_NUM_FULL_ROUNDS || round >= HALF_NUM_FULL_ROUNDS + NUM_PARTIAL_ROUNDS;
    if (is_full_round) {
      apply_round_constants<true>(state, round, wid);
      apply_non_linearity<true>(state, wid);
      apply_M_eps_matrix(state, wid, shared_states);
    } else {
      apply_round_constants<false>(state, round, wid);
      apply_non_linearity<false>(state, wid);
      apply_M_I_matrix(state, wid, shared_states);
    }
  }
}

EXTERN __global__ void poseidon2_cooperative_leaves_kernel(const base_field *values, base_field *results, const unsigned rows_count, const unsigned cols_count,
                                                           const unsigned count, bool load_intermediate, bool store_intermediate) {
  static_assert(RATE == 8);
  static_assert(CAPACITY == 4);
  __shared__ block_states shared_states;
  const unsigned gid = threadIdx.x + blockIdx.x * blockDim.x;
  if (gid >= count)
    return;
  unsigned wid = threadIdx.y;
  field<3> state[TILE] = {0};
  if (load_intermediate && wid == 2) {
    auto intermediate_results = results + gid;
#pragma unroll
    for (unsigned i = 0; i < TILE; i++, intermediate_results += count)
      state[i] = base_field::into<3>(memory::load_cs(intermediate_results));
  }
  values += gid * rows_count;
  for (unsigned offset = 0; offset < rows_count * cols_count;) {
    if (wid < 2) {
      offset += TILE * wid;
#pragma unroll
      for (unsigned i = 0; i < TILE; i++, offset++) {
        const unsigned row = offset % rows_count;
        const unsigned col = offset / rows_count;
        state[i] = col < cols_count ? base_field::into<3>(memory::load_cs(values + row + col * rows_count * count)) : field<3>{};
      }
      offset += TILE * (1 - wid);
    } else
      offset += RATE;
    permutation(state, wid, shared_states);
  }

  //  if (wid < 2)
  //    values += TILE * count * wid + gid;
  //  for (unsigned offset = 0; offset < values_count; offset += RATE) {
  //    if (wid < 2) {
  // #pragma unroll
  //      for (unsigned i = 0; i < TILE; i++, values += count)
  //        state[i] = base_field::into<3>(memory::load_cs(values));
  //      values += TILE * count;
  //    }
  //    permutation(state, wid, shared_states);
  //  }

  results += gid;
  if (wid == (store_intermediate ? 2 : 0)) {
#pragma unroll
    for (unsigned i = 0; i < TILE; i++, results += count)
      memory::store_cs(results, base_field::field3_to_field2(state[i]));
  }
}

static DEVICE_FORCEINLINE void load_nodes_to_shared(const field<4> *values, block_states &shared_states, const unsigned wid) {
  field<3> state_transposed[TILES_PER_STATE] = {0};
  const auto value = memory::load_cs(values);
  auto v2 = reinterpret_cast<const base_field *>(&value);
#pragma unroll
  for (unsigned i = 0; i < 2; i++)
    state_transposed[i] = base_field::into<3>(v2[i]);
    // un-transpose input
#pragma unroll
  for (unsigned i = 0; i < TILES_PER_STATE; i++)
    store_shared(state_transposed[i], shared_states, wid + TILE * i);
}

EXTERN __global__ void poseidon2_cooperative_nodes_kernel(const field<4> *values, base_field *results, const unsigned count) {
  static_assert(RATE == 8);
  static_assert(CAPACITY == 4);
  __shared__ block_states shared_states;
  const unsigned gid = threadIdx.x + blockIdx.x * blockDim.x;
  if (gid >= count)
    return;
  unsigned wid = threadIdx.y;
  load_nodes_to_shared(values + count * wid + gid, shared_states, wid);
  // 3 warps in the block, so warp 0 reads the fourth column
  if (wid == 0)
    load_nodes_to_shared(values + count * 3 + gid, shared_states, 3);
  __syncthreads();
  field<3> state[TILE] = {0};
  for (unsigned i = 0; i < TILE; i++)
    state[i] = load_shared(shared_states, i + TILE * wid);
  // apply_M_eps_matrix calls __syncthreads() before smem writes,
  // so we don't need another __syncthreads() here
  permutation(state, wid, shared_states);
  // We could sync, communicate warp 0's state tile, then store cooperatively,
  // but that's not worth the effort. warp 0 can fire off all stores quickly.
  if (wid == 0) {
    results += gid;
#pragma unroll
    for (unsigned i = 0; i < TILE; i++, results += count)
      memory::store_cs(results, base_field::field3_to_field2(state[i]));
  }
}

} // namespace poseidon2
