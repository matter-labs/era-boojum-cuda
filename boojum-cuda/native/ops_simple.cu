#include "goldilocks_extension.cuh"
#include "memory.cuh"

namespace goldilocks {

using namespace memory;

template <typename T> struct value_getter {
  using value_type = T;
  T value;
  DEVICE_FORCEINLINE T get(const unsigned, const unsigned) const { return value; }
};

using bf_value_getter = value_getter<base_field>;
using bf_getter = wrapping_matrix_getter<matrix_getter<base_field, ld_modifier::cs>>;
using bf_setter = wrapping_matrix_setter<matrix_setter<base_field, st_modifier::cs>>;

using ef_value_getter = value_getter<extension_field>;
using ef_getter = wrapping_matrix_getter<ef_double_matrix_getter<ld_modifier::cs>>;
using ef_setter = wrapping_matrix_setter<ef_double_matrix_setter<st_modifier::cs>>;

using u32_value_getter = value_getter<uint32_t>;
using u32_getter = wrapping_matrix_getter<matrix_getter<uint32_t, ld_modifier::cs>>;
using u32_setter = wrapping_matrix_setter<matrix_setter<uint32_t, st_modifier::cs>>;

using u64_value_getter = value_getter<uint64_t>;
using u64_getter = wrapping_matrix_getter<matrix_getter<uint64_t, ld_modifier::cs>>;
using u64_setter = wrapping_matrix_setter<matrix_setter<uint64_t, st_modifier::cs>>;

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

#define SET_BY_VAL_KERNEL(arg_t)                                                                                                                               \
  EXTERN __global__ void set_by_val_##arg_t##_kernel(const arg_t##_value_getter arg, arg_t##_setter result) { unary_op(return_value, arg, result); }

#define SET_BY_REF_KERNEL(arg_t)                                                                                                                               \
  EXTERN __global__ void set_by_ref_##arg_t##_kernel(const arg_t##_getter arg, arg_t##_setter result) { unary_op(return_value, arg, result); }

#define UNARY_KERNEL(name, arg_t, result_t, op)                                                                                                                \
  EXTERN __global__ void name##_kernel(const arg_t##_getter arg, result_t##_setter result) { unary_op(op, arg, result); }

#define PARAMETRIZED_KERNEL(name, arg_t, result_t, op)                                                                                                         \
  EXTERN __global__ void name##_kernel(const arg_t##_getter arg, const u32_value_getter parameter, result_t##_setter result) {                                 \
    binary_op(op, arg, parameter, result);                                                                                                                     \
  }

#define BINARY_KERNEL(name, arg0_t, arg1_t, result_t, op)                                                                                                      \
  EXTERN __global__ void name##_kernel(const arg0_t##_getter arg0, const arg1_t##_getter arg1, result_t##_setter result) { binary_op(op, arg0, arg1, result); }

#define TERNARY_KERNEL(name, arg0_t, arg1_t, arg2_t, result_t, op)                                                                                             \
  EXTERN __global__ void name##_kernel(const arg0_t##_getter arg0, const arg1_t##_getter arg1, const arg2_t##_getter arg2, result_t##_setter result) {         \
    ternary_op(op, arg0, arg1, arg2, result);                                                                                                                  \
  }

template <class T> DEVICE_FORCEINLINE T return_value(const T &x) { return x; }

DEVICE_FORCEINLINE base_field add(const base_field &x, const base_field &y) { return base_field::add(x, y); }
DEVICE_FORCEINLINE extension_field add(const base_field &x, const extension_field &y) { return extension_field::add(x, y); }
DEVICE_FORCEINLINE extension_field add(const extension_field &x, const base_field &y) { return extension_field::add(x, y); }
DEVICE_FORCEINLINE extension_field add(const extension_field &x, const extension_field &y) { return extension_field::add(x, y); }
DEVICE_FORCEINLINE base_field mul(const base_field &x, const base_field &y) { return base_field::mul(x, y); }
DEVICE_FORCEINLINE extension_field mul(const base_field &x, const extension_field &y) { return extension_field::mul(x, y); }
DEVICE_FORCEINLINE extension_field mul(const extension_field &x, const base_field &y) { return extension_field::mul(x, y); }
DEVICE_FORCEINLINE extension_field mul(const extension_field &x, const extension_field &y) { return extension_field::mul(x, y); }
DEVICE_FORCEINLINE base_field sub(const base_field &x, const base_field &y) { return base_field::sub(x, y); }
DEVICE_FORCEINLINE extension_field sub(const base_field &x, const extension_field &y) { return extension_field::sub(x, y); }
DEVICE_FORCEINLINE extension_field sub(const extension_field &x, const base_field &y) { return extension_field::sub(x, y); }
DEVICE_FORCEINLINE extension_field sub(const extension_field &x, const extension_field &y) { return extension_field::sub(x, y); }

template <class T0, class T1, class T2, class U> DEVICE_FORCEINLINE U mul_add(const T0 &x, const T1 &y, const T2 &z) { return add(mul(x, y), z); }
template <class T0, class T1, class T2, class U> DEVICE_FORCEINLINE U mul_sub(const T0 &x, const T1 &y, const T2 &z) { return sub(mul(x, y), z); }

SET_BY_VAL_KERNEL(bf)
SET_BY_REF_KERNEL(bf)
SET_BY_VAL_KERNEL(ef)
SET_BY_REF_KERNEL(ef)
SET_BY_VAL_KERNEL(u32)
SET_BY_REF_KERNEL(u32)
SET_BY_VAL_KERNEL(u64)
SET_BY_REF_KERNEL(u64)

UNARY_KERNEL(dbl_bf, bf, bf, base_field::dbl)
UNARY_KERNEL(dbl_ef, ef, ef, extension_field::dbl)
UNARY_KERNEL(inv_bf, bf, bf, base_field::inv)
UNARY_KERNEL(inv_ef, ef, ef, extension_field::inv)
UNARY_KERNEL(neg_bf, bf, bf, base_field::neg)
UNARY_KERNEL(neg_ef, ef, ef, extension_field::neg)
UNARY_KERNEL(sqr_bf, bf, bf, base_field::sqr)
UNARY_KERNEL(sqr_ef, ef, ef, extension_field::sqr)

PARAMETRIZED_KERNEL(pow_bf, bf, bf, base_field::pow)
PARAMETRIZED_KERNEL(pow_ef, ef, ef, extension_field::pow)
PARAMETRIZED_KERNEL(shl, bf, bf, base_field::shl)
PARAMETRIZED_KERNEL(shr, bf, bf, base_field::shr)

BINARY_KERNEL(add_bf_bf, bf, bf, bf, add)
BINARY_KERNEL(add_bf_ef, bf, ef, ef, add)
BINARY_KERNEL(add_ef_bf, ef, bf, ef, add)
BINARY_KERNEL(add_ef_ef, ef, ef, ef, add)
BINARY_KERNEL(mul_bf_bf, bf, bf, bf, mul)
BINARY_KERNEL(mul_bf_ef, bf, ef, ef, mul)
BINARY_KERNEL(mul_ef_bf, ef, bf, ef, mul)
BINARY_KERNEL(mul_ef_ef, ef, ef, ef, mul)
BINARY_KERNEL(sub_bf_bf, bf, bf, bf, sub)
BINARY_KERNEL(sub_bf_ef, bf, ef, ef, sub)
BINARY_KERNEL(sub_ef_bf, ef, bf, ef, sub)
BINARY_KERNEL(sub_ef_ef, ef, ef, ef, sub)

TERNARY_KERNEL(mul_add_bf_bf_bf, bf, bf, bf, bf, mul_add)
TERNARY_KERNEL(mul_add_bf_bf_ef, bf, bf, ef, ef, mul_add)
TERNARY_KERNEL(mul_add_bf_ef_bf, bf, ef, bf, ef, mul_add)
TERNARY_KERNEL(mul_add_bf_ef_ef, bf, ef, ef, ef, mul_add)
TERNARY_KERNEL(mul_add_ef_bf_bf, ef, bf, bf, ef, mul_add)
TERNARY_KERNEL(mul_add_ef_bf_ef, ef, bf, ef, ef, mul_add)
TERNARY_KERNEL(mul_add_ef_ef_bf, ef, ef, bf, ef, mul_add)
TERNARY_KERNEL(mul_add_ef_ef_ef, ef, ef, ef, ef, mul_add)
TERNARY_KERNEL(mul_sub_bf_bf_bf, bf, bf, bf, bf, mul_sub)
TERNARY_KERNEL(mul_sub_bf_bf_ef, bf, bf, ef, ef, mul_sub)
TERNARY_KERNEL(mul_sub_bf_ef_bf, bf, ef, bf, ef, mul_sub)
TERNARY_KERNEL(mul_sub_bf_ef_ef, bf, ef, ef, ef, mul_sub)
TERNARY_KERNEL(mul_sub_ef_bf_bf, ef, bf, bf, ef, mul_sub)
TERNARY_KERNEL(mul_sub_ef_bf_ef, ef, bf, ef, ef, mul_sub)
TERNARY_KERNEL(mul_sub_ef_ef_bf, ef, ef, bf, ef, mul_sub)
TERNARY_KERNEL(mul_sub_ef_ef_ef, ef, ef, ef, ef, mul_sub)

} // namespace goldilocks
