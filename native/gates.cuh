#pragma once

#include "goldilocks.cuh"
#include "goldilocks_extension.cuh"
#include "memory.cuh"

namespace gates {

using namespace goldilocks;

struct params {
  unsigned id;
  unsigned selector_mask;
  unsigned selector_count;
  unsigned repetitions_count;
  unsigned initial_variables_offset;
  unsigned initial_witnesses_offset;
  unsigned initial_constants_offset;
  unsigned repetition_variables_offset;
  unsigned repetition_witnesses_offset;
  unsigned repetition_constants_offset;
};

typedef void (*evaluate_fn)(const params &params, const base_field *variables, const base_field *witnesses, const base_field *constants,
                            const extension_field *challenge_bases, extension_field *challenge_powers, extension_field *quotient_sums,
                            const unsigned challenges_count, const unsigned inputs_stride);

typedef ef_double_vector_getter<memory::ld_modifier::ca> challenges_getter;

typedef ef_double_matrix_getter_setter<memory::ld_modifier::cs, memory::st_modifier::cs> quotient_getter_setter;

DEVICE_FORCEINLINE void evaluate(evaluate_fn fn, const params &params, const base_field *variables, const base_field *witnesses, const base_field *constants,
                                 challenges_getter challenges, quotient_getter_setter quotients, const unsigned challenges_count,
                                 const unsigned challenges_power_offset, const unsigned rows_count, const unsigned inputs_stride) {
  const unsigned tid = threadIdx.x;
  const unsigned row = tid + blockIdx.x * blockDim.x;
  if (row >= rows_count)
    return;
  variables += row;
  witnesses += row;
  constants += row;
  quotients += row;
  base_field selector = base_field::one();
  for (unsigned i = 0, mask = params.selector_mask; i < params.selector_count; i++, mask >>= 1) {
    base_field value = *constants;
    constants += inputs_stride;
    if (!(mask & 1))
      value = base_field::sub(base_field::one(), value);
    selector = base_field::mul(selector, value);
  }
  if (base_field::is_zero(selector))
    return;
  extern __shared__ extension_field shared[];
  extension_field *challenge_bases = shared;
  extension_field *challenge_powers = shared + challenges_count;
  extension_field *quotient_sums = shared + challenges_count * 2 + tid;
  if (tid < challenges_count) {
    extension_field challenge = challenges.get(tid);
    challenge_bases[tid] = challenge;
    challenge_powers[tid] = extension_field::pow(challenge, challenges_power_offset);
  }
  for (unsigned i = 0; i < challenges_count; i++)
    quotient_sums[i * blockDim.x] = {};
  fn(params, variables, witnesses, constants, challenge_bases, challenge_powers, quotient_sums, challenges_count, inputs_stride);
  for (unsigned i = 0; i < challenges_count; i++, quotients.inc_col()) {
    extension_field quotient = quotients.get();
    const extension_field sum = extension_field::mul(quotient_sums[i * blockDim.x], selector);
    quotient = extension_field::add(quotient, sum);
    quotients.set(quotient);
  }
}

DEVICE_FORCEINLINE base_field load(const base_field *src, const unsigned offset, const unsigned stride) { return src[offset * stride]; }

template <unsigned COUNT> DEVICE_FORCEINLINE void load(const base_field *src, base_field *dst, const unsigned stride) {
#pragma unroll
  for (unsigned i = 0; i < COUNT; i++)
    dst[i] = load(src, i, stride);
}

DEVICE_FORCEINLINE void push(const base_field &value, const extension_field *challenge_bases, extension_field *challenge_powers, extension_field *quotient_sums,
                             const unsigned challenges_count) {
  __syncthreads();
  for (unsigned i = 0, offset = 0; i < challenges_count; i++, offset += blockDim.x) {
    const extension_field quotient_addition = extension_field::mul(challenge_powers[i], value);
    quotient_sums[offset] = extension_field::add(quotient_sums[offset], quotient_addition);
  }
  __syncthreads();
  const unsigned tid = threadIdx.x;
  if (tid < challenges_count)
    challenge_powers[tid] = extension_field::mul(challenge_powers[tid], challenge_bases[tid]);
}

#define GATE_FUNCTION(name)                                                                                                                                    \
  DEVICE_FORCEINLINE void name(const params &params, const base_field *variables, const base_field *witnesses, const base_field *constants,                    \
                               const extension_field *challenge_bases, extension_field *challenge_powers, extension_field *quotient_sums,                      \
                               const unsigned challenges_count, const unsigned inputs_stride)

#define GATE_KERNEL(name)                                                                                                                                      \
  extern "C" __global__ void evaluate_##name##_kernel(const params params, const base_field *__restrict__ variables, const base_field *__restrict__ witnesses, \
                                                      const base_field *__restrict__ constants, challenges_getter challenges,                                  \
                                                      quotient_getter_setter quotients, const unsigned challenges_count,                                       \
                                                      const unsigned challenges_power_offset, const unsigned rows_count, const unsigned inputs_stride) {       \
    evaluate(name, params, variables, witnesses, constants, challenges, quotients, challenges_count, challenges_power_offset, rows_count, inputs_stride);      \
  }

#define GATE_INPUTS(src, dst, count)                                                                                                                           \
  base_field dst[count];                                                                                                                                       \
  load<count>(src, dst, inputs_stride);                                                                                                                        \
  (src) += params.repetition_##src##_offset * inputs_stride;

#define GATE_VARS(count) GATE_INPUTS(variables, v, count);

#define GATE_WITS(count) GATE_INPUTS(witnesses, w, count);

#define GATE_CONS(count) GATE_INPUTS(constants, c, count);

#define GATE_TEMPS(count) base_field t[count];

#define GATE_VAL(value) base_field::from_u64(value)

#define GATE_INIT                                                                                                                                              \
  variables += params.initial_variables_offset * inputs_stride;                                                                                                \
  witnesses += params.initial_witnesses_offset * inputs_stride;                                                                                                \
  constants += params.initial_constants_offset * inputs_stride;

#define GATE_REP for (unsigned i = 0; i < params.repetitions_count; i++)

#define GATE_ADD(x, y, z) z = base_field::add(x, y);

#define GATE_DBL(x, y) y = base_field::dbl(x);

#define GATE_SUB(x, y, z) z = base_field::sub(x, y);

#define GATE_NEG(x, y) y = base_field::neg(x);

#define GATE_MUL(x, y, z) z = base_field::mul(x, y);

#define GATE_SQR(x, y) y = base_field::sqr(x);

#define GATE_INV(x, y) y = base_field::inv(x);

#define GATE_PUSH(x) push(x, challenge_bases, challenge_powers, quotient_sums, challenges_count);

} // namespace gates
