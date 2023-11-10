#pragma once

#include "goldilocks.cuh"
#include "memory.cuh"

namespace goldilocks {

struct __align__(16) extension_field {
  base_field coefficients[2];

  static constexpr HOST_DEVICE_FORCEINLINE base_field non_residue() { return {7}; }
  static constexpr HOST_DEVICE_FORCEINLINE extension_field zero() { return {}; }
  static constexpr HOST_DEVICE_FORCEINLINE extension_field one() { return {1, 0}; }

  DEVICE_FORCEINLINE base_field &operator[](const unsigned idx) { return coefficients[idx]; }
  DEVICE_FORCEINLINE const base_field &operator[](const unsigned idx) const { return coefficients[idx]; }

  static DEVICE_FORCEINLINE base_field mul_by_non_residue(const base_field &x) { return base_field::mul(x, non_residue()); }

  static DEVICE_FORCEINLINE extension_field add(const extension_field &x, const base_field &y) { return {base_field::add(x[0], y), x[1]}; }

  static DEVICE_FORCEINLINE extension_field add(const base_field &x, const extension_field &y) { return {base_field::add(x, y[0]), y[1]}; }

  static DEVICE_FORCEINLINE extension_field add(const extension_field &x, const extension_field &y) {
    return {base_field::add(x[0], y[0]), base_field::add(x[1], y[1])};
  }

  static DEVICE_FORCEINLINE extension_field sub(const extension_field &x, const base_field &y) { return {base_field::sub(x[0], y), x[1]}; }

  static DEVICE_FORCEINLINE extension_field sub(const base_field &x, const extension_field &y) { return {base_field::sub(x, y[0]), base_field::neg(y[1])}; }

  static DEVICE_FORCEINLINE extension_field sub(const extension_field &x, const extension_field &y) {
    return {base_field::sub(x[0], y[0]), base_field::sub(x[1], y[1])};
  }

  static DEVICE_FORCEINLINE extension_field dbl(const extension_field &x) {
    auto a = base_field::dbl(x[0]);
    auto b = base_field::dbl(x[1]);
    return {a, b};
  }

  static DEVICE_FORCEINLINE extension_field neg(const extension_field &x) {
    auto a = base_field::neg(x[0]);
    auto b = base_field::neg(x[1]);
    return {a, b};
  }

  static DEVICE_FORCEINLINE extension_field mul(const extension_field &x, const base_field &y) { return {base_field::mul(x[0], y), base_field::mul(x[1], y)}; }

  static DEVICE_FORCEINLINE extension_field mul(const base_field &x, const extension_field &y) { return {base_field::mul(x, y[0]), base_field::mul(x, y[1])}; }

  static DEVICE_FORCEINLINE extension_field mul(const extension_field &x, const extension_field &y) {
    auto a = base_field::mul(x[0], y[0]);
    auto b = base_field::mul(x[1], y[0]);
    auto c = base_field::mul(x[0], y[1]);
    auto d = base_field::mul(x[1], y[1]);
    a = base_field::add(a, mul_by_non_residue(d));
    b = base_field::add(b, c);
    return {a, b};
  }

  static DEVICE_FORCEINLINE extension_field sqr(const extension_field &x) {
    auto a = base_field::sqr(x[0]);
    auto b = base_field::mul(x[0], x[1]);
    auto c = base_field::sqr(x[1]);
    a = base_field::add(a, mul_by_non_residue(c));
    b = base_field::dbl(b);
    return {a, b};
  }

  static DEVICE_FORCEINLINE extension_field inv(const extension_field &x) {
    auto a = x[0];
    auto b = x[1];
    auto c = base_field::sub(base_field::sqr(a), mul_by_non_residue(base_field::sqr(b)));
    c = base_field::inv(c);
    a = base_field::mul(a, c);
    b = base_field::mul(b, c);
    b = base_field::neg(b);
    return {a, b};
  }

  static DEVICE_FORCEINLINE extension_field pow(const extension_field &x, const unsigned &power) {
    extension_field result = {1};
    extension_field value = x;
    for (unsigned i = power; i != 0;) {
      if (i & 1)
        result = mul(result, value);
      i >>= 1;
      if (i)
        value = sqr(value);
    }
    return result;
  }
};

template <memory::ld_modifier LD_MODIFIER = memory::ld_modifier::none>
struct ef_double_vector_getter : memory::double_vector_getter<extension_field, base_field, LD_MODIFIER> {};

template <memory::st_modifier ST_MODIFIER = memory::st_modifier::none>
struct ef_double_vector_setter : memory::double_vector_setter<extension_field, base_field, ST_MODIFIER> {};

template <memory::ld_modifier LD_MODIFIER = memory::ld_modifier::none, memory::st_modifier ST_MODIFIER = memory::st_modifier::none>
struct ef_double_vector_getter_setter : memory::double_vector_getter_setter<extension_field, base_field, LD_MODIFIER, ST_MODIFIER> {};

template <memory::ld_modifier LD_MODIFIER = memory::ld_modifier::none>
struct ef_double_matrix_getter : memory::double_matrix_getter<extension_field, base_field, LD_MODIFIER> {};

template <memory::st_modifier ST_MODIFIER = memory::st_modifier::none>
struct ef_double_matrix_setter : memory::double_matrix_setter<extension_field, base_field, ST_MODIFIER> {};

template <memory::ld_modifier LD_MODIFIER = memory::ld_modifier::none, memory::st_modifier ST_MODIFIER = memory::st_modifier::none>
struct ef_double_matrix_getter_setter : memory::double_matrix_getter_setter<extension_field, base_field, LD_MODIFIER, ST_MODIFIER> {};

} // namespace goldilocks