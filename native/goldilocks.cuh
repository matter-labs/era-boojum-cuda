#pragma once

#include "carry_chain.cuh"
#include "common.cuh"
#include "memory.cuh"
#include <sm_32_intrinsics.h>

namespace goldilocks {

#define LIMBS_ALIGNMENT(x) ((x) % 4 == 0 ? 16 : ((x) % 2 == 0 ? 8 : 4))

typedef uint32_t limb;

template <unsigned LIMBS_COUNT> struct __align__(LIMBS_ALIGNMENT(LIMBS_COUNT)) field {
  static_assert(LIMBS_COUNT >= 2);
  __align__(LIMBS_ALIGNMENT(LIMBS_COUNT)) limb limbs[LIMBS_COUNT] = {};
  static constexpr HOST_DEVICE_FORCEINLINE field zero() { return {}; }
  static constexpr HOST_DEVICE_FORCEINLINE field one() { return {1}; }
  static constexpr HOST_DEVICE_FORCEINLINE field minus_one() { return {0, ~0u}; }
  static constexpr HOST_DEVICE_FORCEINLINE field order() { return {1, ~0u}; }
  static constexpr HOST_DEVICE_FORCEINLINE field epsilon() { return {~0u}; }
  static constexpr unsigned TWO_ADICITY = 32;
  static constexpr uint64_t ORDER_U64 = 0xffffffff00000001;
  static constexpr uint64_t MINUS_ONE_U64 = 0xffffffff00000000;
  static constexpr HOST_DEVICE_FORCEINLINE field inverse_2_pow_adicity() { return {2, ~1u}; }

  DEVICE_FORCEINLINE limb &operator[](const unsigned idx) { return limbs[idx]; }
  DEVICE_FORCEINLINE const limb &operator[](const unsigned idx) const { return limbs[idx]; }

  template <bool SUBTRACT, bool CARRY_OUT = false> static DEVICE_FORCEINLINE field add_sub_limbs(const field &x, const field &y) {
    carry_chain<LIMBS_COUNT, false, CARRY_OUT> chain;
    field result{};
#pragma unroll
    for (unsigned i = 0; i < LIMBS_COUNT; i++)
      result[i] = SUBTRACT ? chain.sub(x[i], y[i]) : chain.add(x[i], y[i]);
    return result;
  }

  template <bool CARRY_OUT = false> static DEVICE_FORCEINLINE field add_limbs(const field &x, const field &y) { return add_sub_limbs<false, CARRY_OUT>(x, y); }

  template <bool CARRY_OUT = false> static DEVICE_FORCEINLINE field sub_limbs(const field &x, const field &y) { return add_sub_limbs<true, CARRY_OUT>(x, y); }

  static DEVICE_FORCEINLINE bool is_zero(const field &x) {
    uint32_t limbs_or = x[0];
#pragma unroll
    for (unsigned i = 1; i < LIMBS_COUNT; i++)
      limbs_or |= x[i];
    return limbs_or == 0;
  }

  template <unsigned RESULT_LIMBS_COUNT> static DEVICE_FORCEINLINE field<RESULT_LIMBS_COUNT> into(const field &x) {
    field<RESULT_LIMBS_COUNT> result{};
#pragma unroll
    for (unsigned i = 0; i < LIMBS_COUNT && i < RESULT_LIMBS_COUNT; i++)
      result[i] = x[i];
    return result;
  }

  static DEVICE_FORCEINLINE field get_correction() {
    switch (LIMBS_COUNT) {
    case 2:
      return epsilon();
    case 3:
      return minus_one();
    case 4:
      return {1, ~1u};
    default:
      static_assert(LIMBS_COUNT >= 2 && LIMBS_COUNT <= 4);
    }
  }

  template <bool SUBTRACT> static DEVICE_FORCEINLINE field get_correction_from_carry_flag() {
    limb flag = ptx::subc(0, 0);
    if (!SUBTRACT)
      flag = ~flag;
    switch (LIMBS_COUNT) {
    case 2:
      return {flag};
    case 3:
      return {0, flag};
    case 4:
      return {flag & 1, flag & ~1};
    default:
      static_assert(LIMBS_COUNT >= 2 && LIMBS_COUNT <= 4);
    }
  }

  template <bool SUBTRACT, bool DOUBLE_OVERFLOW_POSSIBLE, bool OVERFLOW_LIKELY> static DEVICE_FORCEINLINE field add_sub(const field &x, const field &y) {
    static_assert(!DOUBLE_OVERFLOW_POSSIBLE || OVERFLOW_LIKELY);
    field z = add_sub_limbs<SUBTRACT, true>(x, y);
    if (DOUBLE_OVERFLOW_POSSIBLE) {
      return add_sub<SUBTRACT, false, false>(z, get_correction_from_carry_flag<SUBTRACT>());
    } else {
      if (OVERFLOW_LIKELY)
        return add_sub_limbs<SUBTRACT>(z, get_correction_from_carry_flag<SUBTRACT>());
      else
        return unlikely(SUBTRACT ? ptx::subc(0, 0) : ptx::addc(0, 0)) ? add_sub_limbs<SUBTRACT>(z, get_correction()) : z;
    }
  }

  template <bool ONE_INPUT_IS_CANONICAL = false, bool OVERFLOW_LIKELY = true> static DEVICE_FORCEINLINE field add(const field &x, const field &y) {
    return add_sub<false, !ONE_INPUT_IS_CANONICAL, OVERFLOW_LIKELY>(x, y);
  }

  template <bool RHS_IS_CANONICAL = false, bool OVERFLOW_LIKELY = true> static DEVICE_FORCEINLINE field sub(const field &x, const field &y) {
    return add_sub<true, !RHS_IS_CANONICAL, OVERFLOW_LIKELY>(x, y);
  }

  template <bool INPUT_IS_CANONICAL = false> static DEVICE_FORCEINLINE field dbl(const field &x) { return add<INPUT_IS_CANONICAL, true>(x, x); }

  template <bool INPUT_IS_CANONICAL = false> static DEVICE_FORCEINLINE field neg(const field &x) {
    return INPUT_IS_CANONICAL ? is_zero(x) ? zero() : sub_limbs<false>(order(), x) : sub(order(), x);
  }

  static DEVICE_FORCEINLINE field<2> reduce(const field<2> &x) {
    auto r = sub_limbs<true>(x, order());
    auto c = ptx::subc(0, 0);
    return likely(c) ? x : r;
  }

  static DEVICE_FORCEINLINE field<4> mul_wide(const field<2> &x, const field<2> &y) {
    const uint64_t x64 = reinterpret_cast<const uint64_t &>(x);
    const uint64_t y64 = reinterpret_cast<const uint64_t &>(y);
    field<4> result;
    auto *r64 = reinterpret_cast<uint64_t *>(&result);
    r64[0] = ptx::u64::mul_lo(x64, y64);
    r64[1] = ptx::u64::mul_hi(x64, y64);
    return result;
  }

  static DEVICE_FORCEINLINE field<4> sqr_wide(const field<2> &x) {
    const uint64_t x64 = reinterpret_cast<const uint64_t &>(x);
    field<4> result;
    auto *r64 = reinterpret_cast<uint64_t *>(&result);
    r64[0] = ptx::u64::mul_lo(x64, x64);
    r64[1] = ptx::u64::mul_hi(x64, x64);
    return result;
  }

  static DEVICE_FORCEINLINE field<2> field3_to_field2(const field<3> &x) {
    field<2> result = field<3>::into<2>(x);
    limb l = x[2];
    result = field<2>::add<true>(result, field<2>::sub_limbs({0, l}, {l}));
    return result;
  }

  static DEVICE_FORCEINLINE field<3> field3_to_field2_and_carry(const field<3> &x) {
    field<3> result = x;
    result[2] = 0;
    limb l = x[2];
    result = field<3>::add_limbs(result, field<2>::into<3>(field<2>::sub_limbs({0, l}, {l})));
    return result;
  }

  static DEVICE_FORCEINLINE field<3> field4_to_field3(const field<4> &x) {
    field<2> x2 = field<4>::into<2>(x);
    limb lo = x[2];
    limb hi = x[3];
    x2 = field<2>::sub<true, false>(x2, {hi});
    field<3> result = field<2>::into<3>(x2);
    result[2] = lo;
    return result;
  }

  static DEVICE_FORCEINLINE field<4> field5_to_field4(const field<5> &x) {
    field<2> x2 = field<5>::into<2>(x);
    limb lo = x[2];
    limb mi = x[3];
    limb hi = x[4];
    x2 = field<2>::sub<true, true>(x2, {0, hi});
    field<4> result = field<2>::into<4>(x2);
    result[2] = lo;
    result[3] = mi;
    return result;
  }

  static DEVICE_FORCEINLINE field<2> field4_to_field2(const field<4> &x) { return field3_to_field2(field4_to_field3(x)); }

  static DEVICE_FORCEINLINE field<3> field5_to_field3(const field<5> &x) { return field4_to_field3(field5_to_field4(x)); }

  static DEVICE_FORCEINLINE field<2> mul(const field<2> &x, const field<2> &y) { return field4_to_field2(mul_wide(x, y)); }

  static DEVICE_FORCEINLINE field<2> sqr(const field<2> &x) { return field4_to_field2(sqr_wide(x)); }

  static DEVICE_FORCEINLINE field shr(const field &x, const unsigned &shift) {
    field y;
#pragma unroll
    for (unsigned i = 0; i < LIMBS_COUNT - 1; i++)
      y[i] = __funnelshift_rc(x[i], x[i + 1], shift);
    y[LIMBS_COUNT - 1] = x[LIMBS_COUNT - 1] >> shift;
    return y;
  }

  static DEVICE_FORCEINLINE field shl(const field &x, const unsigned &shift) {
    field y;
    y[0] = x[0] << shift;
#pragma unroll
    for (unsigned i = 1; i < LIMBS_COUNT; i++)
      y[i] = __funnelshift_lc(x[i - 1], x[i], shift);
    return y;
  }

  template <class T> static DEVICE_FORCEINLINE void swap(T & x, T & y) {
    T temp = x;
    x = y;
    y = temp;
  }

  static DEVICE_FORCEINLINE field shl_extended(const field &x, const unsigned shift) {
    unsigned s = shift;
    field y = x;
    while (s >= 32) {
#pragma unroll
      for (unsigned i = 1; i < LIMBS_COUNT; i++)
        y[LIMBS_COUNT - i] = y[LIMBS_COUNT - i - 1];
      y[0] = 0;
      s -= 32;
    }
    return shl(y, s);
  }

  static DEVICE_FORCEINLINE void inv_safe_iteration(uint64_t & f, uint64_t & g, field<4> & c, field<4> & d, unsigned &k) {
    if (f < g) {
      swap(f, g);
      swap(c, d);
    }
    if ((f & 3) == (g & 3)) {
      f -= g;
      c = field<4>::sub_limbs(c, d);
      unsigned kk = __ffsll(f) - 1;
      f >>= kk;
      d = field<4>::shl_extended(d, kk);
      k += kk;
    } else {
      f = (f >> 2) + (g >> 2) + 1;
      c = field<4>::add_limbs(c, d);
      unsigned kk = __ffsll(f) - 1;
      f >>= kk;
      d = field<4>::shl_extended(d, kk + 2);
      k += kk + 2;
    }
  }

  static DEVICE_FORCEINLINE void inv_unsafe_iteration(uint64_t & f, uint64_t & g, field<4> & c, field<4> & d, unsigned &k) {
    if (f < g) {
      swap(f, g);
      swap(c, d);
    }
    if ((f & 3) == (g & 3)) {
      f -= g;
      c = field<4>::sub_limbs(c, d);
    } else {
      f += g;
      c = field<4>::add_limbs(c, d);
    }
    unsigned kk = __ffsll(f) - 1;
    f >>= kk;
    d = field<4>::shl_extended(d, kk);
    k += kk;
  }

  static DEVICE_FORCEINLINE field<2> inv_2exp_unchecked(const unsigned exp) { return from_u64(ORDER_U64 - (MINUS_ONE_U64 >> exp)); }

  static DEVICE_FORCEINLINE field<2> inv_2exp(const unsigned exp) {
    if (exp > TWO_ADICITY) {
      field<2> res = inverse_2_pow_adicity();
      unsigned e = exp - TWO_ADICITY;
      while (e > TWO_ADICITY) {
        res = mul(res, inverse_2_pow_adicity());
        e -= TWO_ADICITY;
      }
      return mul(res, inv_2exp_unchecked(e));
    } else {
      return inv_2exp_unchecked(exp);
    }
  }

  template <bool INPUT_IS_CANONICAL = false> static DEVICE_FORCEINLINE field<2> inv(const field<2> &x) {
    uint64_t f = to_u64(INPUT_IS_CANONICAL ? x : reduce(x));
    uint64_t g = ORDER_U64;
    field<4> c = field<4>::one();
    field<4> d = field<4>::zero();
    if (f == 0)
      return zero();
    unsigned k = __ffsll(f) - 1;
    f >>= k;
    if (f == 1)
      return inv_2exp(k);
    inv_safe_iteration(f, g, c, d, k);
    if (f == 1) {
      field<2> t = inv_2exp(k);
      if (c[3])
        t = neg(t);
      return t;
    }
    inv_safe_iteration(f, g, c, d, k);
    while (f != 1)
      inv_unsafe_iteration(f, g, c, d, k);
    while (c[3] & 0x80000000)
      c = field<4>::add_limbs(c, field<4>::order());
    return mul(field4_to_field2(c), inv_2exp(k));
  }

  static DEVICE_FORCEINLINE field<2> pow(const field<2> &x, const unsigned &power) {
    field<2> result = field<2>::one();
    field<2> value = x;
    for (unsigned i = power; i != 0;) {
      if (i & 1)
        result = field<2>::mul(result, value);
      i >>= 1;
      if (i)
        value = field<2>::sqr(value);
    }
    return result;
  }

  static DEVICE_FORCEINLINE field<2> from_u64(const uint64_t x) { return *reinterpret_cast<const field<2> *>(&x); }

  static DEVICE_FORCEINLINE uint64_t to_u64(const field<2> x) { return *reinterpret_cast<const uint64_t *>(&x); }
};

typedef field<2> base_field;

} // namespace goldilocks