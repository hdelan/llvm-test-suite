// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out -Xsycl-target-backend --cuda-gpu-arch=sm_80 
// RUN: %GPU_RUN_PLACEHOLDER %t.out
//
// Only cuda backend implements bf16
// REQUIRES: cuda

#include <CL/sycl.hpp>

constexpr int N = 32; // All vector sizes divide this

using namespace sycl;
using sycl::ext::oneapi::experimental::bfloat16;

float make_fp32(bfloat16 x) {
  uint32_t y = reinterpret_cast<bfloat16 &>(x);
  y = y << 16;
  float res = reinterpret_cast<float &>(y);
  return res;
}

bfloat16 make_bf16(float x) {
  uint32_t res = reinterpret_cast<uint32_t &>(x);
  res = res >> 16;
  return reinterpret_cast<bfloat16 &>(res);
}

bool compare_fma_relu_bf16(bfloat16 a, bfloat16 b, bfloat16 c, bfloat16 d) {
  uint32_t a_tmp = reinterpret_cast<uint16_t &>(a),
           b_tmp = reinterpret_cast<uint16_t &>(b),
           c_tmp = reinterpret_cast<uint16_t &>(c),
           d_tmp = reinterpret_cast<uint16_t &>(d);
  a_tmp <<= 16;
  b_tmp <<= 16;
  c_tmp <<= 16;
  d_tmp <<= 16;
  float a_float = reinterpret_cast<float &>(a_tmp),
        b_float = reinterpret_cast<float &>(b_tmp),
        c_float = reinterpret_cast<float &>(c_tmp),
        d_float = reinterpret_cast<float &>(d_tmp);
  float d_cmp = std::fma(a_float, b_float, c_float);
  d_cmp = d_cmp > 0 ? d_cmp : 0;

  return fabs(d_float - d_cmp) <=
         8 * fabs(d_cmp) * std::numeric_limits<cl::sycl::half>::epsilon();
}

bool compare_fma_relu_bf16x2(uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
  bfloat16 *a_beg = reinterpret_cast<bfloat16 *>(&a),
           *b_beg = reinterpret_cast<bfloat16 *>(&b),
           *c_beg = reinterpret_cast<bfloat16 *>(&c),
           *d_beg = reinterpret_cast<bfloat16 *>(&d);
  return compare_fma_relu_bf16(*a_beg, *b_beg, *c_beg, *d_beg) &&
         compare_fma_relu_bf16(*(a_beg + 1), *(b_beg + 1), *(c_beg + 1),
                               *(d_beg + 1));
}

#define TEST_BUILTIN_HALF_SCAL_IMPL(NAME)                                      \
  {                                                                            \
    buffer<half> a_buf(&a[0], N);                                              \
    buffer<half> b_buf(&b[0], N);                                              \
    buffer<half> c_buf(&c[0], N);                                              \
    buffer<half> d_buf(&d[0], N);                                              \
    q.submit([&](handler &cgh) {                                               \
      auto A = a_buf.get_access<access::mode::read>(cgh);                      \
      auto B = b_buf.get_access<access::mode::read>(cgh);                      \
      auto C = c_buf.get_access<access::mode::read>(cgh);                      \
      auto D = d_buf.get_access<access::mode::write>(cgh);                     \
      cgh.parallel_for(N, [=](id<1> index) {                                   \
        D[index] = NAME(A[index], B[index], C[index]);                         \
      });                                                                      \
    });                                                                        \
  }                                                                            \
  for (int i = 0; i < N; i++) {                                                \
    assert(fabs(d[i] - NAME(a[i], b[i], c[i])) <                               \
           std::numeric_limits<cl::sycl::half>::epsilon());                    \
  }

#define TEST_BUILTIN_HALF_VEC_IMPL(NAME, SZ)                                   \
  {                                                                            \
    buffer<half##SZ> a_buf((half##SZ *)&a[0], N / SZ);                         \
    buffer<half##SZ> b_buf((half##SZ *)&b[0], N / SZ);                         \
    buffer<half##SZ> c_buf((half##SZ *)&c[0], N / SZ);                         \
    buffer<half##SZ> d_buf((half##SZ *)&d[0], N / SZ);                         \
    q.submit([&](handler &cgh) {                                               \
      auto A = a_buf.get_access<access::mode::read>(cgh);                      \
      auto B = b_buf.get_access<access::mode::read>(cgh);                      \
      auto C = c_buf.get_access<access::mode::read>(cgh);                      \
      auto D = d_buf.get_access<access::mode::write>(cgh);                     \
      cgh.parallel_for(N / SZ, [=](id<1> index) {                              \
        D[index] = NAME(A[index], B[index], C[index]);                         \
      });                                                                      \
    });                                                                        \
  }                                                                            \
  for (int i = 0; i < N; i++) {                                                \
    assert(fabs(d[i] - NAME(a[i], b[i], c[i])) <                               \
           std::numeric_limits<cl::sycl::half>::epsilon());                    \
  }

#define TEST_BUILTIN_HALF_VEC3_IMPL(NAME)                                      \
  {                                                                            \
    buffer<half3> a_buf((half3 *)&a[0], N / 4);                                \
    buffer<half3> b_buf((half3 *)&b[0], N / 4);                                \
    buffer<half3> c_buf((half3 *)&c[0], N / 4);                                \
    buffer<half3> d_buf((half3 *)&d[0], N / 4);                                \
    q.submit([&](handler &cgh) {                                               \
      auto A = a_buf.get_access<access::mode::read>(cgh);                      \
      auto B = b_buf.get_access<access::mode::read>(cgh);                      \
      auto C = c_buf.get_access<access::mode::read>(cgh);                      \
      auto D = d_buf.get_access<access::mode::write>(cgh);                     \
      cgh.parallel_for(N / 4, [=](id<1> index) {                               \
        D[index] = NAME(A[index], B[index], C[index]);                         \
      });                                                                      \
    });                                                                        \
  }                                                                            \
  for (int i = 0; i < N; i++) {                                                \
    if (i % 4 != 3) {                                                          \
      assert(fabs(d[i] - NAME(a[i], b[i], c[i])) <                             \
             std::numeric_limits<cl::sycl::half>::epsilon());                  \
    }                                                                          \
  }

// There are currently no vec types implemented for the bfloat16 class
// TODO: test vec types once implemented
#define TEST_BUILTIN_BF16_SCAL_IMPL(NAME)                                      \
  {                                                                            \
    buffer<bfloat16> a_buf(&a[0], N);                                          \
    buffer<bfloat16> b_buf(&b[0], N);                                          \
    buffer<bfloat16> c_buf(&c[0], N);                                          \
    buffer<bfloat16> d_buf(&d[0], N);                                          \
    q.submit([&](handler &cgh) {                                               \
      auto A = a_buf.get_access<access::mode::read>(cgh);                      \
      auto B = b_buf.get_access<access::mode::read>(cgh);                      \
      auto C = c_buf.get_access<access::mode::read>(cgh);                      \
      auto D = d_buf.get_access<access::mode::write>(cgh);                     \
      cgh.parallel_for(N, [=](id<1> index) {                                   \
        D[index] = NAME(A[index], B[index], C[index]);                         \
      });                                                                      \
    });                                                                        \
  }                                                                            \
  for (int i = 0; i < N; i++) {                                                \
    assert(compare_fma_relu_bf16(a[i], b[i], c[i], d[i]));                     \
  }

#define TEST_BUILTIN_BF16X2_SCAL_IMPL(NAME)                                    \
  {                                                                            \
    buffer<uint32_t> a_buf((uint32_t *)&a[0], N / 2);                          \
    buffer<uint32_t> b_buf((uint32_t *)&b[0], N / 2);                          \
    buffer<uint32_t> c_buf((uint32_t *)&c[0], N / 2);                          \
    buffer<uint32_t> d_buf((uint32_t *)&d[0], N / 2);                          \
    q.submit([&](handler &cgh) {                                               \
      auto A = a_buf.get_access<access::mode::read>(cgh);                      \
      auto B = b_buf.get_access<access::mode::read>(cgh);                      \
      auto C = c_buf.get_access<access::mode::read>(cgh);                      \
      auto D = d_buf.get_access<access::mode::write>(cgh);                     \
      cgh.parallel_for(N / 2, [=](id<1> index) {                               \
        D[index] = NAME(A[index], B[index], C[index]);                         \
      });                                                                      \
    });                                                                        \
  }                                                                            \
  for (int i = 0; i < N; i++) {                                                \
    assert(compare_fma_relu_bf16x2(a[i], b[i], c[i], d[i]));                   \
  }

#define TEST_BUILTIN_BF16X2_VEC_IMPL(NAME, SZ)                                 \
  {                                                                            \
    buffer<uint##SZ> a_buf((uint##SZ *)&a[0], N / (2 * SZ));                   \
    buffer<uint##SZ> b_buf((uint##SZ *)&b[0], N / (2 * SZ));                   \
    buffer<uint##SZ> c_buf((uint##SZ *)&c[0], N / (2 * SZ));                   \
    buffer<uint##SZ> d_buf((uint##SZ *)&d[0], N / (2 * SZ));                   \
    q.submit([&](handler &cgh) {                                               \
      auto A = a_buf.get_access<access::mode::read>(cgh);                      \
      auto B = b_buf.get_access<access::mode::read>(cgh);                      \
      auto C = c_buf.get_access<access::mode::read>(cgh);                      \
      auto D = d_buf.get_access<access::mode::write>(cgh);                     \
      cgh.parallel_for(N / (2 * SZ), [=](id<1> index) {                        \
        D[index] = NAME(A[index], B[index], C[index]);                         \
      });                                                                      \
    });                                                                        \
  }                                                                            \
  for (int i = 0; i < N; i++) {                                                \
    assert(compare_fma_relu_bf16x2(a[i], b[i], c[i], d[i]));                   \
  }

#define TEST_BUILTIN_BF16X2_VEC3_IMPL(NAME)                                    \
  {                                                                            \
    buffer<uint3> a_buf((uint3 *)&a[0], N / (2 * 4));                          \
    buffer<uint3> b_buf((uint3 *)&b[0], N / (2 * 4));                          \
    buffer<uint3> c_buf((uint3 *)&c[0], N / (2 * 4));                          \
    buffer<uint3> d_buf((uint3 *)&d[0], N / (2 * 4));                          \
    q.submit([&](handler &cgh) {                                               \
      auto A = a_buf.get_access<access::mode::read>(cgh);                      \
      auto B = b_buf.get_access<access::mode::read>(cgh);                      \
      auto C = c_buf.get_access<access::mode::read>(cgh);                      \
      auto D = d_buf.get_access<access::mode::write>(cgh);                     \
      cgh.parallel_for(N / (2 * 4), [=](id<1> index) {                         \
        D[index] = NAME(A[index], B[index], C[index]);                         \
      });                                                                      \
    });                                                                        \
  }                                                                            \
  for (int i = 0; i < N; i++) {                                                \
    if (i % 8 > 5) {                                                           \
      assert(compare_fma_relu_bf16x2(a[i], b[i], c[i], d[i]));                 \
    }                                                                          \
  }

#define TEST_BUILTIN(NAME, type)                                               \
  TEST_BUILTIN_##type##_SCAL_IMPL(NAME);                                       \
  TEST_BUILTIN_##type##_VEC_IMPL(NAME, 2);                                     \
  TEST_BUILTIN_##type##_VEC3_IMPL(NAME);                                       \
  TEST_BUILTIN_##type##_VEC_IMPL(NAME, 4);                                     \
  TEST_BUILTIN_##type##_VEC_IMPL(NAME, 8);                                     \
  TEST_BUILTIN_##type##_VEC_IMPL(NAME, 16);

int main() {
  queue q;

  // HALF tests
  {
    std::vector<half> a(N), b(N), c(N), d(N);
    for (int i = 0; i < N; i++) {
      a[i] = i / (half)N;
      b[i] = (N - i) / (half)N;
      c[i] = -i / 4 / (half)N;
    }
    TEST_BUILTIN(sycl::ext::oneapi::experimental::fma_relu, HALF);
  }

  // BF16
  {
    std::vector<bfloat16> a(N), b(N), c(N), d(N);
    for (int i = 0; i < N; i++) {
      a[i] = make_bf16(i / (float)N);
      b[i] = make_bf16((N - i) / (float)N);
      c[i] = make_bf16(-i / 4 / (float)N);
    }
    TEST_BUILTIN_BF16_SCAL_IMPL(fma_relu);
  }

  // BF16X2
  {
    std::vector<uint16_t> a(N), b(N), c(N), d(N);
    for (int i = 0; i < N; i++) {
      auto tmp = make_bf16(i / (float)N);
      a[i] = reinterpret_cast<uint16_t &>(tmp);
      tmp = make_bf16((N - i) / (float)N);
      b[i] = reinterpret_cast<uint16_t &>(tmp);
      tmp = make_bf16(-i / 4 / (float)N);
      c[i] = reinterpret_cast<uint16_t &>(tmp);
    }
    TEST_BUILTIN(sycl::ext::oneapi::experimental::fma_relu, BF16X2);
  }
}
