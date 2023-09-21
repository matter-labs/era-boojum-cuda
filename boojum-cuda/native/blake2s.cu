#include "common.cuh"

#define ROTR32(x, y) (((x) >> (y)) ^ ((x) << (32 - (y))))

#define B2S_G(a, b, c, d, x, y)                                                                                                                                \
  v[a] = v[a] + v[b] + (x);                                                                                                                                    \
  v[d] = ROTR32(v[d] ^ v[a], 16);                                                                                                                              \
  v[c] = v[c] + v[d];                                                                                                                                          \
  v[b] = ROTR32(v[b] ^ v[c], 12);                                                                                                                              \
  v[a] = v[a] + v[b] + (y);                                                                                                                                    \
  v[d] = ROTR32(v[d] ^ v[a], 8);                                                                                                                               \
  v[c] = v[c] + v[d];                                                                                                                                          \
  v[b] = ROTR32(v[b] ^ v[c], 7);

DEVICE_FORCEINLINE uint64_t get_digest(const uint32_t input[10]) {
  const uint32_t blake2s_iv[8] = {0x6A09E667, 0xBB67AE85, 0x3C6EF372, 0xA54FF53A, 0x510E527F, 0x9B05688C, 0x1F83D9AB, 0x5BE0CD19};
  const uint8_t sigma[10][16] = {{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, {14, 10, 4, 8, 9, 15, 13, 6, 1, 12, 0, 2, 11, 7, 5, 3},
                                 {11, 8, 12, 0, 5, 2, 15, 13, 10, 14, 3, 6, 7, 1, 9, 4}, {7, 9, 3, 1, 13, 12, 11, 14, 2, 6, 5, 10, 4, 0, 15, 8},
                                 {9, 0, 5, 7, 2, 4, 10, 15, 14, 1, 11, 12, 6, 8, 3, 13}, {2, 12, 6, 10, 0, 11, 8, 3, 4, 13, 7, 5, 15, 14, 1, 9},
                                 {12, 5, 1, 15, 14, 13, 4, 10, 0, 7, 6, 3, 9, 2, 8, 11}, {13, 11, 7, 14, 12, 1, 3, 9, 5, 0, 15, 4, 8, 6, 2, 10},
                                 {6, 15, 14, 9, 11, 3, 0, 8, 12, 2, 13, 7, 1, 4, 10, 5}, {10, 2, 8, 4, 7, 6, 1, 5, 15, 11, 9, 14, 3, 12, 13, 0}};
  __align__(8) uint32_t h[8];
#pragma unroll
  for (unsigned i = 0; i < 8; i++)
    h[i] = blake2s_iv[i];
  h[0] ^= 0x01010000 ^ 32;
  uint32_t v[16];
  uint32_t m[16] = {};
#pragma unroll
  for (unsigned i = 0; i < 8; i++) {
    v[i] = h[i];
    v[i + 8] = blake2s_iv[i];
  }
  v[12] ^= 40;
  v[14] = ~v[14];
#pragma unroll
  for (unsigned i = 0; i < 10; i++)
    m[i] = input[i];
#pragma unroll
  for (auto s : sigma) {
    B2S_G(0, 4, 8, 12, m[s[0]], m[s[1]])
    B2S_G(1, 5, 9, 13, m[s[2]], m[s[3]])
    B2S_G(2, 6, 10, 14, m[s[4]], m[s[5]])
    B2S_G(3, 7, 11, 15, m[s[6]], m[s[7]])
    B2S_G(0, 5, 10, 15, m[s[8]], m[s[9]])
    B2S_G(1, 6, 11, 12, m[s[10]], m[s[11]])
    B2S_G(2, 7, 8, 13, m[s[12]], m[s[13]])
    B2S_G(3, 4, 9, 14, m[s[14]], m[s[15]])
  }
#pragma unroll
  for (unsigned i = 0; i < 2; ++i)
    h[i] ^= v[i] ^ v[i + 8];
  return *reinterpret_cast<uint64_t *>(h);
}

EXTERN __global__ void blake2s_pow_kernel(const uint64_t *seed, const uint32_t bits_count, const uint64_t max_nonce, volatile uint64_t *result) {
  __align__(8) uint32_t input_u32[10];
  auto input_u64 = reinterpret_cast<uint64_t *>(input_u32);
#pragma unroll
  for (unsigned i = 0; i < 4; i++)
    input_u64[i] = seed[i];
  for (uint64_t nonce = threadIdx.x + blockIdx.x * blockDim.x; nonce < max_nonce && *result == UINT64_MAX; nonce += blockDim.x * gridDim.x) {
    input_u64[4] = nonce;
    uint64_t digest = get_digest(input_u32);
    if (__clzll((long long)__brevll(digest)) >= bits_count)
      atomicCAS(reinterpret_cast<unsigned long long *>(const_cast<uint64_t *>(result)), UINT64_MAX, nonce);
  }
}
