#pragma once

#include <sm_32_intrinsics.h>
#include <type_traits>

namespace memory {

using namespace std;

enum class ld_modifier { none, g, cg, ca, cs, lu, cv };

template <typename T, ld_modifier MODIFIER> static constexpr DEVICE_FORCEINLINE T ld_single(const T *ptr) {
#if __CUDA_ARCH__ >= 320
  switch (MODIFIER) {
  case ld_modifier::none:
    return *ptr;
  case ld_modifier::g:
    return __ldg(ptr);
  case ld_modifier::cg:
    return __ldcg(ptr);
  case ld_modifier::ca:
    return __ldca(ptr);
  case ld_modifier::cs:
    return __ldcs(ptr);
  case ld_modifier::lu:
    return __ldlu(ptr);
  case ld_modifier::cv:
    return __ldcv(ptr);
  }
#else
  return *ptr;
#endif
}

enum class st_modifier { none, wb, cg, cs, wt };

template <typename T, st_modifier MODIFIER> static constexpr DEVICE_FORCEINLINE void st_single(T *ptr, T value) {
#if __CUDA_ARCH__ >= 320
  switch (MODIFIER) {
  case st_modifier::none:
    *ptr = value;
    break;
  case st_modifier::wb:
    __stwb(ptr, value);
    break;
  case st_modifier::cg:
    __stcg(ptr, value);
    break;
  case st_modifier::cs:
    __stcs(ptr, value);
    break;
  case st_modifier::wt:
    __stwt(ptr, value);
    break;
  }
#else
  *ptr = value;
#endif
}

template <typename T> DEVICE_FORCEINLINE void swap(T &a, T &b) {
  T temp = a;
  a = b;
  b = temp;
}

template <unsigned STRIDE> DEVICE_FORCEINLINE unsigned swap_index(unsigned index) {
  const unsigned i1 = index % STRIDE;
  const unsigned i2 = index / STRIDE;
  const unsigned i3 = i2 * STRIDE * 2;
  return i3 + i1;
}

template <typename T> DEVICE_FORCEINLINE T shfl_xor(unsigned mask, T var, int laneMask, int width = warpSize) {
  return __shfl_xor_sync(mask, var, laneMask, width);
}

template <> DEVICE_FORCEINLINE uint2 shfl_xor(unsigned mask, uint2 var, int laneMask, int width) {
  uint2 result{};
  result.x = __shfl_xor_sync(mask, var.x, laneMask, width);
  result.y = __shfl_xor_sync(mask, var.y, laneMask, width);
  return result;
}

template <> DEVICE_FORCEINLINE uint4 shfl_xor(unsigned mask, uint4 var, int laneMask, int width) {
  uint4 result{};
  result.x = __shfl_xor_sync(mask, var.x, laneMask, width);
  result.y = __shfl_xor_sync(mask, var.y, laneMask, width);
  result.z = __shfl_xor_sync(mask, var.z, laneMask, width);
  result.w = __shfl_xor_sync(mask, var.w, laneMask, width);
  return result;
}

template <typename T, unsigned STRIDE_ROW, unsigned COUNT_ROW, unsigned STRIDE_COL = STRIDE_ROW, unsigned COUNT_COL = COUNT_ROW>
DEVICE_FORCEINLINE void transpose_tile(unsigned mask, T *u, const unsigned lane_id) {
  const bool swap_rows = !(lane_id & STRIDE_ROW);
  if (swap_rows) {
#pragma unroll
    for (unsigned i = 0; i < COUNT_ROW; i++) {
      const unsigned index = swap_index<STRIDE_ROW>(i);
      swap(u[index], u[index + STRIDE_ROW]);
    }
  }
#pragma unroll
  for (unsigned i = 0; i < COUNT_COL; i++) {
    const unsigned index = swap_index<STRIDE_COL>(i);
    u[index] = shfl_xor(mask, u[index], STRIDE_COL);
  }
  if (swap_rows) {
#pragma unroll
    for (unsigned i = 0; i < COUNT_ROW; i++) {
      const unsigned index = swap_index<STRIDE_ROW>(i);
      swap(u[index], u[index + STRIDE_ROW]);
    }
  }
}

template <class T, typename U, ld_modifier MODIFIER, unsigned STRIDE> static constexpr DEVICE_FORCEINLINE T ld(const T *address, const unsigned offset) {
  static_assert(alignof(T) % alignof(U) == 0);
  static_assert(sizeof(T) % sizeof(U) == 0);
  constexpr size_t count = sizeof(T) / sizeof(U);
  T result = {};
  auto pa = reinterpret_cast<const U *>(address) + offset;
  auto pr = reinterpret_cast<U *>(&result);
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
  for (unsigned i = 0; i < count; i++) {
    const auto pai = pa + i * STRIDE;
    const auto pri = pr + i;
    *pri = ld_single<U, MODIFIER>(pai);
  }
  return result;
}

template <class T, typename U, st_modifier MODIFIER, unsigned STRIDE>
static constexpr DEVICE_FORCEINLINE void st(T *address, const T &value, const unsigned offset) {
  static_assert(alignof(T) % alignof(U) == 0);
  static_assert(sizeof(T) % sizeof(U) == 0);
  constexpr size_t count = sizeof(T) / sizeof(U);
  auto pa = reinterpret_cast<U *>(address) + offset;
  auto pv = reinterpret_cast<const U *>(&value);
#ifdef __CUDA_ARCH__
#pragma unroll
#endif
  for (unsigned i = 0; i < count; i++) {
    auto pai = pa + i * STRIDE;
    auto pvi = pv + i;
    st_single<U, MODIFIER>(pai, *pvi);
  }
}

template <typename U, unsigned STRIDE_COL> DEVICE_FORCEINLINE void transpose_tile(const unsigned stride_row, U *tile, const unsigned lane_id) {
  switch (stride_row) {
  case 0:
    transpose_tile<U, 1, STRIDE_COL>(UINT32_MAX, tile, lane_id);
    break;
  case 1:
    transpose_tile<U, 2, STRIDE_COL>(UINT32_MAX, tile, lane_id);
    break;
  case 2:
    transpose_tile<U, 4, STRIDE_COL>(UINT32_MAX, tile, lane_id);
    break;
  case 3:
    transpose_tile<U, 8, STRIDE_COL>(UINT32_MAX, tile, lane_id);
    break;
  case 4:
    transpose_tile<U, 16, STRIDE_COL>(UINT32_MAX, tile, lane_id);
    break;
  default:
    break;
  }
}

template <class T, typename U, unsigned LOG_WARP_SIZE, ld_modifier MODIFIER, bool CHECK_INACTIVE>
DEVICE_FORCEINLINE T ld_warp(const T *address, const unsigned offset, const unsigned lane_id) {
  static_assert(alignof(T) % alignof(U) == 0);
  static_assert(sizeof(T) % (sizeof(U) << LOG_WARP_SIZE) == 0);
  constexpr size_t count = sizeof(T) / (sizeof(U) << LOG_WARP_SIZE);
  constexpr unsigned threads_count = 1 << LOG_WARP_SIZE;
  const unsigned l = lane_id & (threads_count - 1);
  T result = {};
  auto pr = reinterpret_cast<U *>(&result);
#pragma unroll
  for (int i = 0; i < threads_count; i++) {
    const unsigned o = __shfl_sync(UINT32_MAX, offset, i, threads_count);
    if (CHECK_INACTIVE && o == UINT32_MAX)
      continue;
    const U *ap = reinterpret_cast<const U *>(address + o) + l;
#pragma unroll
    for (unsigned j = 0; j < count; j++) {
      const unsigned shift = j << LOG_WARP_SIZE;
      pr[i + shift] = ld_single<U, MODIFIER>(ap + shift);
    }
  }
#pragma unroll
  for (unsigned i = 0; i < count; i++) {
    const unsigned shift = i << LOG_WARP_SIZE;
    U *tile = pr + shift;
    constexpr unsigned stride = threads_count >> 1;
#pragma unroll
    for (unsigned j = 0; j < LOG_WARP_SIZE; j++)
      transpose_tile<U, stride>(j, tile, l);
  }
  return result;
}

template <class T, typename U, unsigned LOG_WARP_SIZE, st_modifier MODIFIER, bool CHECK_INACTIVE>
DEVICE_FORCEINLINE void st_warp(T *address, const unsigned offset, const T &value, const unsigned lane_id) {
  static_assert(alignof(T) % alignof(U) == 0);
  static_assert(sizeof(T) % (sizeof(U) << LOG_WARP_SIZE) == 0);
  constexpr size_t count = sizeof(T) / (sizeof(U) << LOG_WARP_SIZE);
  constexpr unsigned threads_count = 1 << LOG_WARP_SIZE;
  const unsigned l = lane_id & (threads_count - 1);
  T value_copy = value;
  auto pv = reinterpret_cast<U *>(&value_copy);
#pragma unroll
  for (unsigned i = 0; i < count; i++) {
    const unsigned shift = i << LOG_WARP_SIZE;
    U *tile = pv + shift;
    constexpr unsigned stride = threads_count >> 1;
#pragma unroll
    for (int j = LOG_WARP_SIZE - 1; j >= 0; j--)
      transpose_tile<U, stride>(j, tile, l);
  }
#pragma unroll
  for (int i = 0; i < threads_count; i++) {
    const unsigned o = __shfl_sync(UINT32_MAX, offset, i, threads_count);
    if (CHECK_INACTIVE && o == UINT32_MAX)
      continue;
    U *ap = reinterpret_cast<U *>(address + o) + l;
#pragma unroll
    for (unsigned j = 0; j < count; j++) {
      const unsigned shift = j << LOG_WARP_SIZE;
      st_single<U, MODIFIER>(ap + shift, pv[i + shift]);
    }
  }
}

template <class T, ld_modifier MODIFIER = ld_modifier::none, unsigned STRIDE = 1, typename U = enable_if_t<sizeof(T) % sizeof(uint4) == 0, uint4>>
static constexpr DEVICE_FORCEINLINE T load(const T *address, const unsigned offset = 0, [[maybe_unused]] uint4 _dummy = {}) {
  return ld<T, U, MODIFIER, STRIDE>(address, offset);
}

template <class T, unsigned LOG_WARP_SIZE, ld_modifier MODIFIER = ld_modifier::none, bool CHECK_INACTIVE = true,
          typename U = enable_if_t<sizeof(T) % (sizeof(uint4) << LOG_WARP_SIZE) == 0, uint4>>
static constexpr DEVICE_FORCEINLINE T load_warp(const T *address, const unsigned offset, const unsigned lane_id, [[maybe_unused]] uint4 _dummy = {}) {
  return ld_warp<T, U, LOG_WARP_SIZE, MODIFIER, CHECK_INACTIVE>(address, offset, lane_id);
}

template <class T, ld_modifier MODIFIER = ld_modifier::none, unsigned STRIDE = 1,
          typename U = enable_if_t<(sizeof(T) % sizeof(uint4) != 0) && (sizeof(T) % sizeof(uint2) == 0), uint2>>
static constexpr DEVICE_FORCEINLINE T load(const T *address, const unsigned offset = 0, [[maybe_unused]] uint2 _dummy = {}) {
  return ld<T, U, MODIFIER, STRIDE>(address, offset);
}

template <class T, unsigned LOG_WARP_SIZE, ld_modifier MODIFIER = ld_modifier::none, bool CHECK_INACTIVE = true,
          typename U = enable_if_t<sizeof(T) % (sizeof(uint4) << LOG_WARP_SIZE) != 0 && sizeof(T) % (sizeof(uint2) << LOG_WARP_SIZE) == 0, uint2>>
static constexpr DEVICE_FORCEINLINE T load_warp(const T *address, const unsigned offset, const unsigned lane_id, [[maybe_unused]] uint2 _dummy = {}) {
  return ld_warp<T, U, LOG_WARP_SIZE, MODIFIER, CHECK_INACTIVE>(address, offset, lane_id);
}

template <class T, ld_modifier MODIFIER = ld_modifier::none, unsigned STRIDE = 1, typename U = enable_if_t<sizeof(T) % sizeof(uint2) != 0, unsigned>>
static constexpr DEVICE_FORCEINLINE T load(const T *address, const unsigned offset = 0, [[maybe_unused]] unsigned _dummy = {}) {
  return ld<T, U, MODIFIER, STRIDE>(address, offset);
}

template <class T, unsigned LOG_WARP_SIZE, ld_modifier MODIFIER = ld_modifier::none, bool CHECK_INACTIVE = true,
          typename U = enable_if_t<sizeof(T) % (sizeof(uint2) << LOG_WARP_SIZE) != 0, unsigned>>
static constexpr DEVICE_FORCEINLINE T load_warp(const T *address, const unsigned offset, const unsigned lane_id, [[maybe_unused]] unsigned _dummy = {}) {
  return ld_warp<T, U, LOG_WARP_SIZE, MODIFIER, CHECK_INACTIVE>(address, offset, lane_id);
}

template <class T, st_modifier MODIFIER = st_modifier::none, unsigned STRIDE = 1, typename U = enable_if_t<sizeof(T) % sizeof(uint4) == 0, uint4>>
static constexpr DEVICE_FORCEINLINE void store(T *address, const T &value, const unsigned offset = 0, [[maybe_unused]] uint4 _dummy = {}) {
  st<T, U, MODIFIER, STRIDE>(address, value, offset);
}

template <class T, unsigned LOG_WARP_SIZE, st_modifier MODIFIER = st_modifier::none, bool CHECK_INACTIVE = true,
          typename U = enable_if_t<sizeof(T) % (sizeof(uint4) << LOG_WARP_SIZE) == 0, uint4>>
static constexpr DEVICE_FORCEINLINE void store_warp(T *address, const unsigned offset, const T &value, const unsigned lane_id,
                                                    [[maybe_unused]] uint4 _dummy = {}) {
  st_warp<T, U, LOG_WARP_SIZE, MODIFIER, CHECK_INACTIVE>(address, offset, value, lane_id);
}

template <class T, st_modifier MODIFIER = st_modifier::none, unsigned STRIDE = 1,
          typename U = enable_if_t<(sizeof(T) % sizeof(uint4) != 0) && (sizeof(T) % sizeof(uint2) == 0), uint2>>
static constexpr DEVICE_FORCEINLINE void store(T *address, const T &value, const unsigned offset = 0, [[maybe_unused]] uint2 _dummy = {}) {
  st<T, U, MODIFIER, STRIDE>(address, value, offset);
}

template <class T, unsigned LOG_WARP_SIZE, st_modifier MODIFIER = st_modifier::none, bool CHECK_INACTIVE = true,
          typename U = enable_if_t<sizeof(T) % (sizeof(uint4) << LOG_WARP_SIZE) != 0 && sizeof(T) % (sizeof(uint2) << LOG_WARP_SIZE) == 0, uint2>>
static constexpr DEVICE_FORCEINLINE void store_warp(T *address, const unsigned offset, const T &value, const unsigned lane_id,
                                                    [[maybe_unused]] uint2 _dummy = {}) {
  st_warp<T, U, LOG_WARP_SIZE, MODIFIER, CHECK_INACTIVE>(address, offset, value, lane_id);
}

template <class T, st_modifier MODIFIER = st_modifier::none, unsigned STRIDE = 1, typename U = enable_if_t<sizeof(T) % sizeof(uint2) != 0, unsigned>>
static constexpr DEVICE_FORCEINLINE void store(T *address, const T &value, const unsigned offset = 0, [[maybe_unused]] unsigned _dummy = {}) {
  st<T, U, MODIFIER, STRIDE>(address, value, offset);
}

template <class T, unsigned LOG_WARP_SIZE, st_modifier MODIFIER = st_modifier::none, bool CHECK_INACTIVE = true,
          typename U = enable_if_t<sizeof(T) % (sizeof(uint2) << LOG_WARP_SIZE) != 0, unsigned>>
static constexpr DEVICE_FORCEINLINE void store_warp(T *address, const unsigned offset, const T &value, const unsigned lane_id,
                                                    [[maybe_unused]] unsigned _dummy = {}) {
  st_warp<T, U, LOG_WARP_SIZE, MODIFIER, CHECK_INACTIVE>(address, offset, value, lane_id);
}

template <class T> static constexpr DEVICE_FORCEINLINE T load_g(const T *address) { return load<T, ld_modifier::g>(address); }
template <class T> static constexpr DEVICE_FORCEINLINE T load_cg(const T *address) { return load<T, ld_modifier::cg>(address); }
template <class T> static constexpr DEVICE_FORCEINLINE T load_ca(const T *address) { return load<T, ld_modifier::ca>(address); }
template <class T> static constexpr DEVICE_FORCEINLINE T load_cs(const T *address) { return load<T, ld_modifier::cs>(address); }
template <class T> static constexpr DEVICE_FORCEINLINE T load_lu(const T *address) { return load<T, ld_modifier::lu>(address); }
template <class T> static constexpr DEVICE_FORCEINLINE T load_cv(const T *address) { return load<T, ld_modifier::cv>(address); }

template <class T> static constexpr DEVICE_FORCEINLINE void store_wb(T *address, const T &value) { store<T, st_modifier::wb>(address, value); }
template <class T> static constexpr DEVICE_FORCEINLINE void store_cg(T *address, const T &value) { store<T, st_modifier::cg>(address, value); }
template <class T> static constexpr DEVICE_FORCEINLINE void store_cs(T *address, const T &value) { store<T, st_modifier::cs>(address, value); }
template <class T> static constexpr DEVICE_FORCEINLINE void store_wt(T *address, const T &value) { store<T, st_modifier::wt>(address, value); }

template <typename T, typename U = T> struct vector_accessor {
  using value_type = T;

  T *ptr;

  DEVICE_FORCEINLINE U *self() { return reinterpret_cast<U *>(this); }

  DEVICE_FORCEINLINE const U *self() const { return reinterpret_cast<const U *>(this); }

  DEVICE_FORCEINLINE U copy() const { return *self(); }

  DEVICE_FORCEINLINE U operator+(const unsigned offset) const {
    U result = *self();
    result.ptr += offset;
    return result;
  }

  DEVICE_FORCEINLINE U operator-(const unsigned offset) const {
    U result = *self();
    result.ptr -= offset;
    return result;
  }

  DEVICE_FORCEINLINE U operator+=(const unsigned offset) {
    U *s = self();
    s->ptr += offset;
    return *s;
  }
  DEVICE_FORCEINLINE U operator-=(const unsigned offset) {
    U *s = self();
    s->ptr -= offset;
    return *s;
  }

  DEVICE_FORCEINLINE U operator++() {
    U *s = self();
    s->ptr++;
    return *s;
  }

  DEVICE_FORCEINLINE U operator--() {
    U *s = self();
    s->ptr--;
    return *s;
  }

  DEVICE_FORCEINLINE const U operator++(int) {
    U copy = copy();
    self()->ptr++;
    return copy;
  }

  DEVICE_FORCEINLINE const U operator--(int) {
    U copy = copy();
    self()->ptr--;
    return copy;
  }
};

template <typename T, ld_modifier LD_MODIFIER = ld_modifier::none> struct vector_getter : vector_accessor<T, vector_getter<T, LD_MODIFIER>> {
  DEVICE_FORCEINLINE T get() const { return load<T, LD_MODIFIER>(this->ptr); }
  DEVICE_FORCEINLINE T get(const unsigned i) const { return this->operator+(i).get(); }
};

template <typename T, st_modifier ST_MODIFIER = st_modifier::none> struct vector_setter : vector_accessor<T, vector_setter<T, ST_MODIFIER>> {
  DEVICE_FORCEINLINE void set(const T &value) const { store<T, ST_MODIFIER>(this->ptr, value); }
  DEVICE_FORCEINLINE void set(const unsigned i, const T &value) const { this->operator+(i).set(value); }
};

template <typename T, ld_modifier LD_MODIFIER = ld_modifier::none, st_modifier ST_MODIFIER = st_modifier::none>
struct vector_getter_setter : vector_accessor<T, vector_getter_setter<T, LD_MODIFIER, ST_MODIFIER>> {
  DEVICE_FORCEINLINE T get() const { return load<T, LD_MODIFIER>(this->ptr); }
  DEVICE_FORCEINLINE T get(const unsigned i) const { return this->operator+(i).get(); }
  DEVICE_FORCEINLINE void set(const T &value) const { store<T, ST_MODIFIER>(this->ptr, value); }
  DEVICE_FORCEINLINE void set(const unsigned i, const T &value) const { this->operator+(i).set(value); }
};

template <typename T, typename U = T> struct matrix_accessor : vector_accessor<T, U> {
  const size_t stride;
  DEVICE_FORCEINLINE U inc_row() { return this->operator++(); }
  DEVICE_FORCEINLINE U inc_col() { return this->operator+=(this->stride); }
  DEVICE_FORCEINLINE U dec_row() { return this->operator--(); }
  DEVICE_FORCEINLINE U dec_col() { return this->operator-=(this->stride); }
  DEVICE_FORCEINLINE U add_row(const unsigned offset) { return this->operator+=(offset); }
  DEVICE_FORCEINLINE U add_col(const unsigned offset) { return this->operator+=(offset * this->stride); }
  DEVICE_FORCEINLINE U sub_row(const unsigned offset) { return this->operator-=(offset); }
  DEVICE_FORCEINLINE U sub_col(const unsigned offset) { return this->operator-=(offset * this->stride); }
};

template <typename T, ld_modifier LD_MODIFIER = ld_modifier::none> struct matrix_getter : matrix_accessor<T, matrix_getter<T, LD_MODIFIER>> {
  DEVICE_FORCEINLINE T get() const { return load<T, LD_MODIFIER>(this->ptr); }
  DEVICE_FORCEINLINE T get_at_row(const unsigned row) const { return this->copy().add_row(row).get(); }
  DEVICE_FORCEINLINE T get_at_col(const unsigned col) const { return this->copy().add_col(col).get(); }
  DEVICE_FORCEINLINE T get(const unsigned row, const unsigned col) const { return this->copy().add_row(row).add_col(col).get(); }
};

template <typename T, st_modifier ST_MODIFIER = st_modifier::none> struct matrix_setter : matrix_accessor<T, matrix_setter<T, ST_MODIFIER>> {
  DEVICE_FORCEINLINE void set(const T &value) const { store<T, ST_MODIFIER>(this->ptr, value); }
  DEVICE_FORCEINLINE void set_at_row(const unsigned row, const T &value) const { this->copy().add_row(row).set(value); }
  DEVICE_FORCEINLINE void set_at_col(const unsigned col, const T &value) const { this->copy().add_col(col).set(value); }
  DEVICE_FORCEINLINE void set(const unsigned row, const unsigned col, const T &value) const { this->copy().add_row(row).add_col(col).set(value); }
};

template <typename T, ld_modifier LD_MODIFIER = ld_modifier::none, st_modifier ST_MODIFIER = st_modifier::none>
struct matrix_getter_setter : matrix_accessor<T, matrix_getter_setter<T, LD_MODIFIER, ST_MODIFIER>> {
  DEVICE_FORCEINLINE T get() const { return load<T, LD_MODIFIER>(this->ptr); }
  DEVICE_FORCEINLINE T get_at_row(const unsigned row) const { return this->copy().add_row(row).get(); }
  DEVICE_FORCEINLINE T get_at_col(const unsigned col) const { return this->copy().add_col(col).get(); }
  DEVICE_FORCEINLINE T get(const unsigned row, const unsigned col) const { return this->copy().add_row(row).add_col(col).get(); }
  DEVICE_FORCEINLINE void set(const T &value) const { store<T, ST_MODIFIER>(this->ptr, value); }
  DEVICE_FORCEINLINE void set_at_row(const unsigned row, const T &value) const { this->copy().add_row(row).set(value); }
  DEVICE_FORCEINLINE void set_at_col(const unsigned col, const T &value) const { this->copy().add_col(col).set(value); }
  DEVICE_FORCEINLINE void set(const unsigned row, const unsigned col, const T &value) const { this->copy().add_row(row).add_col(col).set(value); }
};

template <typename T, typename U> struct double_vector_accessor {
  T internal;

  DEVICE_FORCEINLINE U *self() { return reinterpret_cast<U *>(this); }

  DEVICE_FORCEINLINE const U *self() const { return reinterpret_cast<const U *>(this); }

  DEVICE_FORCEINLINE U copy() const { return *self(); }

  DEVICE_FORCEINLINE U operator+(const unsigned offset) const {
    U result = copy();
    result.internal += offset;
    return result;
  }

  DEVICE_FORCEINLINE U operator-(const unsigned offset) const {
    U result = copy();
    result.internal -= offset;
    return result;
  }

  DEVICE_FORCEINLINE U operator+=(const unsigned offset) {
    U *s = self();
    s->internal += offset;
    return *s;
  }

  DEVICE_FORCEINLINE U operator-=(const unsigned offset) {
    U *s = self();
    s->internal -= offset;
    return *s;
  }

  DEVICE_FORCEINLINE U operator++() {
    U *s = self();
    s->internal++;
    return *s;
  }

  DEVICE_FORCEINLINE U operator--() {
    U *s = self();
    s->internal--;
    return *s;
  }

  DEVICE_FORCEINLINE const U operator++(int) {
    const U copy = copy();
    self()->internal++;
    return copy;
  }

  DEVICE_FORCEINLINE const U operator--(int) {
    const U copy = copy();
    self()->internal--;
    return copy;
  }
};

template <typename T, typename U, ld_modifier LD_MODIFIER = ld_modifier::none>
struct double_vector_getter : double_vector_accessor<matrix_getter<U, LD_MODIFIER>, double_vector_getter<T, U, LD_MODIFIER>> {
  using value_type = T;

  DEVICE_FORCEINLINE T get() const {
    auto getter = this->internal;
    const U c0 = getter.get();
    const U c1 = getter.get_at_row(getter.stride);
    return {c0, c1};
  }

  DEVICE_FORCEINLINE T get(const unsigned i) const { return this->operator+(i).get(); }
};

template <typename T, typename U, st_modifier ST_MODIFIER = st_modifier::none>
struct double_vector_setter : double_vector_accessor<matrix_setter<U, ST_MODIFIER>, double_vector_setter<T, U, ST_MODIFIER>> {
  using value_type = T;

  DEVICE_FORCEINLINE void set(const T &value) const {
    auto setter = this->internal;
    setter.set(value[0]);
    setter.set_at_row(setter.stride, value[1]);
  }

  DEVICE_FORCEINLINE void set(const unsigned i, const T &value) const { this->operator+(i).set(value); }
};

template <typename T, typename U, ld_modifier LD_MODIFIER = ld_modifier::none, st_modifier ST_MODIFIER = st_modifier::none>
struct double_vector_getter_setter
    : double_vector_accessor<matrix_getter_setter<U, LD_MODIFIER, ST_MODIFIER>, double_vector_getter_setter<T, U, LD_MODIFIER, ST_MODIFIER>> {
  using value_type = T;

  DEVICE_FORCEINLINE T get() const {
    auto setter = this->internal;
    const U c0 = setter.get();
    const U c1 = setter.get_at_row(setter.stride);
    return {c0, c1};
  }

  DEVICE_FORCEINLINE T get(const unsigned i) const { return this->operator+(i).get(); }

  DEVICE_FORCEINLINE void set(const T &value) const {
    auto getter = this->internal;
    getter.set(value[0]);
    getter.set_at_row(getter.stride, value[1]);
  }

  DEVICE_FORCEINLINE void set(const unsigned i, const T &value) const { this->operator+(i).set(value); }
};

template <typename T, typename U> struct double_matrix_accessor : double_vector_accessor<T, U> {
  DEVICE_FORCEINLINE U inc_row() { return this->operator++(); }
  DEVICE_FORCEINLINE U inc_col() { return this->operator+=(this->internal.stride << 1); }
  DEVICE_FORCEINLINE U dec_row() { return this->operator--(); }
  DEVICE_FORCEINLINE U dec_col() { return this->operator-=(this->internal.stride << 1); }
  DEVICE_FORCEINLINE U add_row(const unsigned offset) { return this->operator+=(offset); }
  DEVICE_FORCEINLINE U add_col(const unsigned offset) { return this->operator+=(offset * this->internal.stride << 1); }
  DEVICE_FORCEINLINE U sub_row(const unsigned offset) { return this->operator-=(offset); }
  DEVICE_FORCEINLINE U sub_col(const unsigned offset) { return this->operator-=(offset * this->internal.stride << 1); }
};

template <typename T, typename U, ld_modifier LD_MODIFIER = ld_modifier::none>
struct double_matrix_getter : double_matrix_accessor<matrix_getter<U, LD_MODIFIER>, double_matrix_getter<T, U, LD_MODIFIER>> {
  using value_type = T;

  DEVICE_FORCEINLINE T get() const {
    auto getter = this->internal;
    const U c0 = getter.get();
    const U c1 = getter.get_at_row(getter.stride);
    return {c0, c1};
  }

  DEVICE_FORCEINLINE T get_at_row(const unsigned row) const { return this->copy().add_row(row).get(); }

  DEVICE_FORCEINLINE T get_at_col(const unsigned col) const { return this->copy().add_col(col).get(); }

  DEVICE_FORCEINLINE T get(const unsigned row, const unsigned col) const { return this->copy().add_row(row).add_col(col).get(); }
};

template <typename T, typename U, st_modifier ST_MODIFIER = st_modifier::none>
struct double_matrix_setter : double_matrix_accessor<matrix_setter<U, ST_MODIFIER>, double_matrix_setter<T, U, ST_MODIFIER>> {
  using value_type = T;

  DEVICE_FORCEINLINE void set(const T &value) const {
    auto setter = this->internal;
    setter.set(value[0]);
    setter.set_at_row(setter.stride, value[1]);
  }

  DEVICE_FORCEINLINE void set_at_row(const unsigned row, const T &value) const { this->copy().add_row(row).set(value); }

  DEVICE_FORCEINLINE void set_at_col(const unsigned col, const T &value) const { this->copy().add_col(col).set(value); }

  DEVICE_FORCEINLINE void set(const unsigned row, const unsigned col, const T &value) const { this->copy().add_row(row).add_col(col).set(value); }
};

template <typename T, typename U, ld_modifier LD_MODIFIER = ld_modifier::none, st_modifier ST_MODIFIER = st_modifier::none>
struct double_matrix_getter_setter
    : double_matrix_accessor<matrix_getter_setter<U, LD_MODIFIER, ST_MODIFIER>, double_matrix_getter_setter<T, U, LD_MODIFIER, ST_MODIFIER>> {
  using value_type = T;

  DEVICE_FORCEINLINE T get() const {
    auto getter = this->internal;
    const U c0 = getter.get();
    const U c1 = getter.get_at_row(getter.stride);
    return {c0, c1};
  }

  DEVICE_FORCEINLINE T get_at_row(const unsigned row) const { return this->copy().add_row(row).get(); }

  DEVICE_FORCEINLINE T get_at_col(const unsigned col) const { return this->copy().add_col(col).get(); }

  DEVICE_FORCEINLINE T get(const unsigned row, const unsigned col) const { return this->copy().add_row(row).add_col(col).get(); }

  DEVICE_FORCEINLINE void set(const T &value) const {
    auto setter = this->internal;
    setter.set(value[0]);
    setter.set_at_row(setter.stride, value[1]);
  }

  DEVICE_FORCEINLINE void set_at_row(const unsigned row, const T &value) const { this->copy().add_row(row).set(value); }

  DEVICE_FORCEINLINE void set_at_col(const unsigned col, const T &value) const { this->copy().add_col(col).set(value); }

  DEVICE_FORCEINLINE void set(const unsigned row, const unsigned col, const T &value) const { this->copy().add_row(row).add_col(col).set(value); }
};

template <typename T> struct wrapping_vector_accessor {
  using value_type = typename T::value_type;

  T internal;
  unsigned count;
};

template <typename T> struct wrapping_vector_getter : wrapping_vector_accessor<T> {
  DEVICE_FORCEINLINE typename T::value_type get() const { return this->internal.get(); }
  DEVICE_FORCEINLINE typename T::value_type get(const unsigned i) const { return this->interal.get(i % this->count); }
};

template <typename T> struct wrapping_vector_setter {
  DEVICE_FORCEINLINE void set(const typename T::value_type &value) const { this->internal.set(value); }
  DEVICE_FORCEINLINE void set(const unsigned i, const typename T::value_type &value) const { this->internal.set(i % this->count, value); }
};

template <typename T> struct wrapping_vector_getter_setter {
  DEVICE_FORCEINLINE typename T::value_type get() const { return this->internal.get(); }
  DEVICE_FORCEINLINE typename T::value_type get(const unsigned i) const { return this->interal.get(i % this->count); }
  DEVICE_FORCEINLINE void set(const typename T::value_type &value) const { this->internal.set(value); }
  DEVICE_FORCEINLINE void set(const unsigned i, const typename T::value_type &value) const { this->internal.set(i % this->count, value); }
};

template <typename T> struct wrapping_matrix_accessor {
  using value_type = typename T::value_type;

  T internal;
  unsigned rows;
  unsigned cols;
};

template <typename T> struct wrapping_matrix_getter : wrapping_matrix_accessor<T> {
  DEVICE_FORCEINLINE typename T::value_type get() const { return this->internal.get(); }
  DEVICE_FORCEINLINE typename T::value_type get_at_row(const unsigned row) const { return this->internal.get_at_row(row % this->rows); }
  DEVICE_FORCEINLINE typename T::value_type get_at_col(const unsigned col) const { return this->internal.get_at_col(col % this->cols); }
  DEVICE_FORCEINLINE typename T::value_type get(const unsigned row, const unsigned col) const { return this->internal.get(row % this->rows, col % this->cols); }
};

template <typename T> struct wrapping_matrix_setter : wrapping_matrix_accessor<T> {
  DEVICE_FORCEINLINE void set(const typename T::value_type &value) const { this->internal.set(value); }
  DEVICE_FORCEINLINE void set_at_row(const unsigned row, const typename T::value_type &value) const { this->internal.set_at_row(row % this->rows, value); }
  DEVICE_FORCEINLINE void set_at_col(const unsigned col, const typename T::value_type &value) const { this->internal.set_at_col(col % this->cols, value); }
  DEVICE_FORCEINLINE void set(const unsigned row, const unsigned col, const typename T::value_type &value) const {
    this->internal.set(row % this->rows, col % this->cols, value);
  }
};

template <typename T> struct wrapping_matrix_getter_setter : wrapping_matrix_accessor<T> {
  DEVICE_FORCEINLINE typename T::value_type get() const { return this->internal.get(); }
  DEVICE_FORCEINLINE typename T::value_type get_at_row(const unsigned row) const { return this->internal.get_at_row(row % this->rows); }
  DEVICE_FORCEINLINE typename T::value_type get_at_col(const unsigned col) const { return this->internal.get_at_col(col % this->cols); }
  DEVICE_FORCEINLINE typename T::value_type get(const unsigned row, const unsigned col) const { return this->internal.get(row % this->rows, col % this->cols); }
  DEVICE_FORCEINLINE void set(const typename T::value_type &value) const { this->internal.set(value); }
  DEVICE_FORCEINLINE void set_at_row(const unsigned row, const typename T::value_type &value) const { this->internal.set_at_row(row % this->rows, value); }
  DEVICE_FORCEINLINE void set_at_col(const unsigned col, const typename T::value_type &value) const { this->internal.set_at_col(col % this->cols, value); }
  DEVICE_FORCEINLINE void set(const unsigned row, const unsigned col, const typename T::value_type &value) const {
    this->internal.set(row % this->rows, col % this->cols, value);
  }
};
} // namespace memory