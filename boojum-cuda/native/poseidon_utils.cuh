#pragma once

namespace poseidon_common {

typedef field<3> poseidon_state[STATE_WIDTH];

template <void (*permutation)(poseidon_state &)>
static DEVICE_FORCEINLINE void single_thread_leaves_impl(const base_field *values, base_field *results, const unsigned rows_count, const unsigned cols_count,
                                                         const unsigned count, bool load_intermediate, bool store_intermediate) {
  const unsigned gid = threadIdx.x + blockIdx.x * blockDim.x;
  if (gid >= count)
    return;
  poseidon_state state{};
  if (load_intermediate) {
    auto intermediate_results = results + gid;
#pragma unroll
    for (unsigned i = STATE_WIDTH - CAPACITY; i < STATE_WIDTH; i++, intermediate_results += count)
      state[i] = base_field::into<3>(memory::load_cs(intermediate_results));
  }
  values += gid * rows_count;
  for (unsigned offset = 0; offset < rows_count * cols_count;) {
#pragma unroll
    for (unsigned i = 0; i < RATE; i++, offset++) {
      const unsigned row = offset % rows_count;
      const unsigned col = offset / rows_count;
      state[i] = col < cols_count ? base_field::into<3>(memory::load_cs(values + row + col * rows_count * count)) : field<3>{};
    }
    permutation(state);
  }
  results += gid;
  if (store_intermediate) {
#pragma unroll
    for (unsigned i = STATE_WIDTH - CAPACITY; i < STATE_WIDTH; i++, results += count)
      memory::store_cs(results, base_field::field3_to_field2(state[i]));
  } else {
#pragma unroll
    for (unsigned i = 0; i < CAPACITY; i++, results += count)
      memory::store_cs(results, base_field::field3_to_field2(state[i]));
  }
}

template <void (*permutation)(poseidon_state &)>
static DEVICE_FORCEINLINE void single_thread_nodes_impl(const field<4> *values, base_field *results, const unsigned count) {
  const unsigned gid = threadIdx.x + blockIdx.x * blockDim.x;
  if (gid >= count)
    return;
  poseidon_state state{};
  values += gid;
#pragma unroll
  for (unsigned i = 0; i < CAPACITY; i++, values += count) {
    const auto value = memory::load_cs(values);
    auto v2 = reinterpret_cast<const base_field *>(&value);
#pragma unroll
    for (unsigned j = 0; j < 2; j++)
      state[j * CAPACITY + i] = base_field::into<3>(v2[j]);
  }
  permutation(state);
  results += gid;
#pragma unroll
  for (unsigned i = 0; i < CAPACITY; i++, results += count)
    memory::store_cs(results, base_field::field3_to_field2(state[i]));
}

} // namespace poseidon_common

namespace poseidon2 {

// https://eprint.iacr.org/2023/323.pdf Appendix B
static DEVICE_FORCEINLINE void m4_times_tile(field<3> *tile) {
  field<3> t0 = field<3>::add_limbs(tile[0], tile[1]);              //  t0 = x[0] + x[1]
  field<3> t1 = field<3>::add_limbs(tile[2], tile[3]);              //  t1 = x[2] + x[3]
  field<3> t2 = field<3>::add_limbs(field<3>::shl(tile[1], 1), t1); //  t2 = 2 * x[1] + t1
  field<3> t3 = field<3>::add_limbs(field<3>::shl(tile[3], 1), t0); //  t3 = 2 * x[3] + t0
  field<3> t4 = field<3>::add_limbs(field<3>::shl(t1, 2), t3);      //  t4 = 4 * t1 + t3
  field<3> t5 = field<3>::add_limbs(field<3>::shl(t0, 2), t2);      //  t5 = 4 * t0 + t2
  field<3> t6 = field<3>::add_limbs(t3, t5);                        //  t6 = t3 + t5
  field<3> t7 = field<3>::add_limbs(t2, t4);                        //  t7 = t2 + t4
  tile[0] = t6;
  tile[1] = t5;
  tile[2] = t7;
  tile[3] = t4;
}

} // namespace poseidon2
