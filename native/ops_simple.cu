#include "goldilocks_extension.cuh"
#include "memory.cuh"

namespace goldilocks {

using namespace memory;

template <typename T> struct value_getter {
  using value_type = T;
  T value;

  DEVICE_FORCEINLINE T get(const unsigned, const unsigned) const { return value; }
};

using bf = base_field;
using bf_value_getter = value_getter<bf>;
using bf_getter = wrapping_matrix_getter<matrix_getter<bf, ld_modifier::cs>>;
using bf_setter = wrapping_matrix_setter<matrix_setter<bf, st_modifier::cs>>;

using ef = extension_field;
using ef_value_getter = value_getter<ef>;
using ef_getter = wrapping_matrix_getter<ef_double_matrix_getter<ld_modifier::cs>>;
using ef_setter = wrapping_matrix_setter<ef_double_matrix_setter<st_modifier::cs>>;

using u32 = uint32_t;
using u32_value_getter = value_getter<u32>;
using u32_getter = wrapping_matrix_getter<matrix_getter<u32, ld_modifier::cs>>;
using u32_setter = wrapping_matrix_setter<matrix_setter<u32, st_modifier::cs>>;

using u64 = uint64_t;
using u64_value_getter = value_getter<u64>;
using u64_getter = wrapping_matrix_getter<matrix_getter<u64, ld_modifier::cs>>;
using u64_setter = wrapping_matrix_setter<matrix_setter<u64, st_modifier::cs>>;

template <class T, class U> using unary_fn = U (*)(const T &);

template <class T0, class T1, class U> using binary_fn = U (*)(const T0 &, const T1 &);

template <class T0, class T1, class T2, class U> using ternary_fn = U (*)(const T0 &, const T1 &, const T2 &);

template <class T, class U> DEVICE_FORCEINLINE void unary_op(unary_fn<typename T::value_type, typename U::value_type> func, const T arg, U result) {
  const unsigned row = threadIdx.x + blockIdx.x * blockDim.x;
  if (row >= result.rows)
    return;
  const unsigned col = blockIdx.y;
  const typename T::value_type arg_value = arg.get(row, col);
  const typename U::value_type result_value = func(arg_value);
  result.set(row, col, result_value);
}

template <class T0, class T1, class U>
DEVICE_FORCEINLINE void binary_op(binary_fn<typename T0::value_type, typename T1::value_type, typename U::value_type> func, const T0 arg0, const T1 arg1,
                                  U result) {
  const unsigned row = threadIdx.x + blockIdx.x * blockDim.x;
  if (row >= result.rows)
    return;
  const unsigned col = blockIdx.y;
  const typename T0::value_type arg0_value = arg0.get(row, col);
  const typename T1::value_type arg1_value = arg1.get(row, col);
  const typename U::value_type result_value = func(arg0_value, arg1_value);
  result.set(row, col, result_value);
}

template <class T0, class T1, class T2, class U>
DEVICE_FORCEINLINE void ternary_op(ternary_fn<typename T0::value_type, typename T1::value_type, typename T2::value_type, typename U::value_type> func,
                                   const T0 arg0, const T1 arg1, const T2 arg2, U result) {
  const unsigned row = threadIdx.x + blockIdx.x * blockDim.x;
  if (row >= result.rows)
    return;
  const unsigned col = blockIdx.y;
  const typename T0::value_type arg0_value = arg0.get(row, col);
  const typename T1::value_type arg1_value = arg1.get(row, col);
  const typename T2::value_type arg2_value = arg2.get(row, col);
  const typename U::value_type result_value = func(arg0_value, arg1_value, arg2_value);
  result.set(row, col, result_value);
}

template <class T> DEVICE_FORCEINLINE T return_value(const T &x) { return x; }

DEVICE_FORCEINLINE bf add(const bf &x, const bf &y) { return bf::add(x, y); }

DEVICE_FORCEINLINE ef add(const bf &x, const ef &y) { return ef::add(x, y); }

DEVICE_FORCEINLINE ef add(const ef &x, const bf &y) { return ef::add(x, y); }

DEVICE_FORCEINLINE ef add(const ef &x, const ef &y) { return ef::add(x, y); }

DEVICE_FORCEINLINE bf mul(const bf &x, const bf &y) { return bf::mul(x, y); }

DEVICE_FORCEINLINE ef mul(const bf &x, const ef &y) { return ef::mul(x, y); }

DEVICE_FORCEINLINE ef mul(const ef &x, const bf &y) { return ef::mul(x, y); }

DEVICE_FORCEINLINE ef mul(const ef &x, const ef &y) { return ef::mul(x, y); }

DEVICE_FORCEINLINE bf sub(const bf &x, const bf &y) { return bf::sub(x, y); }

DEVICE_FORCEINLINE ef sub(const bf &x, const ef &y) { return ef::sub(x, y); }

DEVICE_FORCEINLINE ef sub(const ef &x, const bf &y) { return ef::sub(x, y); }

DEVICE_FORCEINLINE ef sub(const ef &x, const ef &y) { return ef::sub(x, y); }

template <class T0, class T1, class T2, class U> DEVICE_FORCEINLINE U mul_add(const T0 &x, const T1 &y, const T2 &z) { return add(mul(x, y), z); }

template <class T0, class T1, class T2, class U> DEVICE_FORCEINLINE U mul_sub(const T0 &x, const T1 &y, const T2 &z) { return sub(mul(x, y), z); }

#define SET_BY_VAL_KERNEL(arg_t)                                                                                                                               \
  EXTERN __global__ void set_by_val_##arg_t##_kernel(const arg_t##_value_getter arg, arg_t##_setter result) { unary_op(return_value, arg, result); }

SET_BY_VAL_KERNEL(u32)
SET_BY_VAL_KERNEL(u64)
SET_BY_VAL_KERNEL(bf)
SET_BY_VAL_KERNEL(ef)

#define SET_BY_REF_KERNEL(arg_t)                                                                                                                               \
  EXTERN __global__ void set_by_ref_##arg_t##_kernel(const arg_t##_getter arg, arg_t##_setter result) { unary_op(return_value, arg, result); }

SET_BY_REF_KERNEL(u32)
SET_BY_REF_KERNEL(u64)
SET_BY_REF_KERNEL(bf)
SET_BY_REF_KERNEL(ef)

#define UNARY_KERNEL(op, arg_t)                                                                                                                                \
  EXTERN __global__ void op##_##arg_t##_kernel(const arg_t##_getter arg, arg_t##_setter result) { unary_op(arg_t::op, arg, result); }

UNARY_KERNEL(dbl, bf)
UNARY_KERNEL(dbl, ef)
UNARY_KERNEL(inv, bf)
UNARY_KERNEL(inv, ef)
UNARY_KERNEL(neg, bf)
UNARY_KERNEL(neg, ef)
UNARY_KERNEL(sqr, bf)
UNARY_KERNEL(sqr, ef)

#define PARAMETRIZED_KERNEL(op, arg_t)                                                                                                                         \
  EXTERN __global__ void op##_##arg_t##_kernel(const arg_t##_getter arg, const u32_value_getter parameter, arg_t##_setter result) {                            \
    binary_op(arg_t::op, arg, parameter, result);                                                                                                              \
  }

PARAMETRIZED_KERNEL(pow, bf)
PARAMETRIZED_KERNEL(pow, ef)
PARAMETRIZED_KERNEL(shl, bf)
PARAMETRIZED_KERNEL(shl, ef)
PARAMETRIZED_KERNEL(shr, bf)
PARAMETRIZED_KERNEL(shr, ef)

#define BINARY_KERNEL(op, arg0_t, arg1_t, result_t)                                                                                                            \
  EXTERN __global__ void op##_##arg0_t##_##arg1_t##_kernel(const arg0_t##_getter arg0, const arg1_t##_getter arg1, result_t##_setter result) {                 \
    binary_op(result_t::op, arg0, arg1, result);                                                                                                               \
  }

BINARY_KERNEL(add, bf, bf, bf)
BINARY_KERNEL(add, bf, ef, ef)
BINARY_KERNEL(add, ef, bf, ef)
BINARY_KERNEL(add, ef, ef, ef)
BINARY_KERNEL(mul, bf, bf, bf)
BINARY_KERNEL(mul, bf, ef, ef)
BINARY_KERNEL(mul, ef, bf, ef)
BINARY_KERNEL(mul, ef, ef, ef)
BINARY_KERNEL(sub, bf, bf, bf)
BINARY_KERNEL(sub, bf, ef, ef)
BINARY_KERNEL(sub, ef, bf, ef)
BINARY_KERNEL(sub, ef, ef, ef)

#define TERNARY_KERNEL(op, arg0_t, arg1_t, arg2_t, result_t)                                                                                                   \
  EXTERN __global__ void op##_##arg0_t##_##arg1_t##_##arg2_t##_kernel(const arg0_t##_getter arg0, const arg1_t##_getter arg1, const arg2_t##_getter arg2,      \
                                                                      result_t##_setter result) {                                                              \
    ternary_op(op, arg0, arg1, arg2, result);                                                                                                                  \
  }

TERNARY_KERNEL(mul_add, bf, bf, bf, bf)
TERNARY_KERNEL(mul_add, bf, bf, ef, ef)
TERNARY_KERNEL(mul_add, bf, ef, bf, ef)
TERNARY_KERNEL(mul_add, bf, ef, ef, ef)
TERNARY_KERNEL(mul_add, ef, bf, bf, ef)
TERNARY_KERNEL(mul_add, ef, bf, ef, ef)
TERNARY_KERNEL(mul_add, ef, ef, bf, ef)
TERNARY_KERNEL(mul_add, ef, ef, ef, ef)
TERNARY_KERNEL(mul_sub, bf, bf, bf, bf)
TERNARY_KERNEL(mul_sub, bf, bf, ef, ef)
TERNARY_KERNEL(mul_sub, bf, ef, bf, ef)
TERNARY_KERNEL(mul_sub, bf, ef, ef, ef)
TERNARY_KERNEL(mul_sub, ef, bf, bf, ef)
TERNARY_KERNEL(mul_sub, ef, bf, ef, ef)
TERNARY_KERNEL(mul_sub, ef, ef, bf, ef)
TERNARY_KERNEL(mul_sub, ef, ef, ef, ef)

} // namespace goldilocks
