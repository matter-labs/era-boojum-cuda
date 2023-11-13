#pragma once

#include "../goldilocks.cuh"
#include "../goldilocks_extension.cuh"
#include <cub/cub.cuh>

using namespace cub;

namespace goldilocks {

struct base_field_add {
  DEVICE_FORCEINLINE base_field operator()(const base_field &a, const base_field &b) const { return base_field::add(a, b); }
};

struct base_field_mul {
  DEVICE_FORCEINLINE base_field operator()(const base_field &a, const base_field &b) const { return base_field::mul(a, b); }
};

struct extension_field_add {
  DEVICE_FORCEINLINE extension_field operator()(const extension_field &a, const extension_field &b) const { return extension_field::add(a, b); }
};

struct extension_field_mul {
  DEVICE_FORCEINLINE extension_field operator()(const extension_field &a, const extension_field &b) const { return extension_field::mul(a, b); }
};

} // namespace goldilocks
