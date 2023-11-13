#include "common.cuh"
#include "context.cuh"
#include "goldilocks.cuh"
#include "goldilocks_extension.cuh"
#include "memory.cuh"

namespace goldilocks {

using namespace memory;

template <typename T, int INV_BATCH, bool batch_is_full>
DEVICE_FORCEINLINE void batch_inv_registers(const T *inputs, T *fwd_scan_and_outputs, int runtime_batch_size) {
  // If count < grid size, the kernel is inefficient no matter what (because each thread processes just one element)
  // but we should still bail out if a thread has no assigned elems at all.
  T running_prod = T::one();
  int i = 0;
#pragma unroll
  for (; i < INV_BATCH; i++)
    if (batch_is_full || i < runtime_batch_size) {
      fwd_scan_and_outputs[i] = running_prod;
      running_prod = T::mul(running_prod, inputs[i]);
    }

  T inv = T::inv(running_prod);

  i--;
#pragma unroll
  for (; i >= 0; i--) {
    if (batch_is_full || i < runtime_batch_size) {
      const auto input = inputs[i];
      // Isolates and stores this input's inv
      fwd_scan_and_outputs[i] = T::mul(fwd_scan_and_outputs[i], inv);
      // Removes this input's inv contribution
      if (i > 0)
        inv = T::mul(inv, input);
    }
  }
}

} // namespace goldilocks