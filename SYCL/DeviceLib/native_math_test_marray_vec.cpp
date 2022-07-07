// RUN: %clangxx -fsycl -fsycl-targets=%sycl_triple %s -o %t.out
// RUN: %GPU_RUN_PLACEHOLDER %t.out
// RUN: %HOST_RUN_PLACEHOLDER %t.out
// RUN: %CPU_RUN_PLACEHOLDER %t.out
// RUN: %ACC_RUN_PLACEHOLDER %t.out

#include <sycl/sycl.hpp>

using namespace sycl;

template <typename T1, typename T2> class TypeHelper;

template <typename T> bool checkEqual(vec<T, 3> A, size_t B) {
  T TB = B;
  return A.x() == TB && A.y() == TB && A.z() == TB;
}

template <typename T> bool checkEqual(vec<T, 4> A, size_t B) {
  T TB = B;
  return A.x() == TB && A.y() == TB && A.z() == TB && A.w() == TB;
}

template <typename T, size_t N> bool checkEqual(marray<T, N> A, size_t B) {
  for (int i = 0; i < N; i++) {
    if (A[i] != B) {
      return false;
    }
  }
  return true;
}

#define NATIVE_OPERATOR(NAME)                                                  \
  template <typename T>                                                        \
  void native_math_test_##NAME(queue &deviceQueue, T result, T input,          \
                               size_t ref) {                                   \
    {                                                                          \
      buffer<T, 1> buffer1(&result, 1);                                        \
      buffer<T, 1> buffer2(&input, 1);                                         \
      deviceQueue.submit([&](handler &cgh) {                                   \
        accessor<T, 1, access::mode::write, target::device> res_access(        \
            buffer1, cgh);                                                     \
        accessor<T, 1, access::mode::read, target::device> input_access(       \
            buffer2, cgh);                                                     \
        cgh.single_task<TypeHelper<class native##NAME, T>>(                    \
            [=]() { res_access[0] = sycl::native::NAME(input_access[0]); });   \
      });                                                                      \
    }                                                                          \
    assert(checkEqual(result, ref));                                           \
  }

NATIVE_OPERATOR(sin)
NATIVE_OPERATOR(tan)
NATIVE_OPERATOR(cos)
NATIVE_OPERATOR(exp)
NATIVE_OPERATOR(exp2)
NATIVE_OPERATOR(exp10)
NATIVE_OPERATOR(log)
NATIVE_OPERATOR(log2)
NATIVE_OPERATOR(log10)
NATIVE_OPERATOR(sqrt)
NATIVE_OPERATOR(rsqrt)
NATIVE_OPERATOR(recip)

#undef NATIVE_OPERATOR

#define NATIVE_OPERATOR_2(NAME)                                                \
  template <typename T>                                                        \
  void native_math_test_2_##NAME(queue &deviceQueue, T result, T input1,       \
                                 T input2, size_t ref) {                       \
    {                                                                          \
      buffer<T, 1> buffer1(&result, 1);                                        \
      buffer<T, 1> buffer2(&input1, 1);                                        \
      buffer<T, 1> buffer3(&input2, 1);                                        \
      deviceQueue.submit([&](handler &cgh) {                                   \
        accessor<T, 1, access::mode::write, target::device> res_access(        \
            buffer1, cgh);                                                     \
        accessor<T, 1, access::mode::read, target::device> input1_access(      \
            buffer2, cgh);                                                     \
        accessor<T, 1, access::mode::read, target::device> input2_access(      \
            buffer3, cgh);                                                     \
        cgh.single_task<TypeHelper<class native2##NAME, T>>([=]() {            \
          res_access[0] =                                                      \
              sycl::native::NAME(input1_access[0], input2_access[0]);          \
        });                                                                    \
      });                                                                      \
    }                                                                          \
    assert(checkEqual(result, ref));                                           \
  }

NATIVE_OPERATOR_2(divide)
NATIVE_OPERATOR_2(powr)

#undef NATIVE_OPERATOR_2

template <typename T> void native_tests_3(queue &deviceQueue) {
  native_math_test_sin(deviceQueue, T{-1, -1, -1}, T{0, 0, 0}, 0);
  native_math_test_tan(deviceQueue, T{-1, -1, -1}, T{0, 0, 0}, 0);
  native_math_test_cos(deviceQueue, T{-1, -1, -1}, T{0, 0, 0}, 1);
  native_math_test_exp(deviceQueue, T{-1, -1, -1}, T{0, 0, 0}, 1);
  native_math_test_log(deviceQueue, T{-1, -1, -1}, T{1, 1, 1}, 0);
  native_math_test_log2(deviceQueue, T{-1, -1, -1}, T{4, 4, 4}, 2);
  native_math_test_log10(deviceQueue, T{-1, -1, -1}, T{100, 100, 100}, 2);
  native_math_test_sqrt(deviceQueue, T{-1, -1, -1}, T{4, 4, 4}, 2);
  native_math_test_rsqrt(deviceQueue, T{-1, -1, -1}, T{0.25, 0.25, 0.25}, 2);
  native_math_test_recip(deviceQueue, T{-1, -1, -1}, T{0.25, 0.25, 0.25}, 4);
  native_math_test_2_divide(deviceQueue, T{-1, -1, -1}, T{4, 4, 4}, T{2, 2, 2},
                            2);
}

template <typename T> void native_tests_4(queue &deviceQueue) {
  native_math_test_sin(deviceQueue, T{-1, -1, -1, -1}, T{0, 0, 0, 0}, 0);
  native_math_test_tan(deviceQueue, T{-1, -1, -1, -1}, T{0, 0, 0, 0}, 0);
  native_math_test_cos(deviceQueue, T{-1, -1, -1, -1}, T{0, 0, 0, 0}, 1);
  native_math_test_exp(deviceQueue, T{-1, -1, -1, -1}, T{0, 0, 0, 0}, 1);
  native_math_test_exp2(deviceQueue, T{-1, -1, -1, -1}, T{2, 2, 2, 2}, 4);
  native_math_test_log(deviceQueue, T{-1, -1, -1, -1}, T{1, 1, 1, 1}, 0);
  native_math_test_log2(deviceQueue, T{-1, -1, -1, -1}, T{4, 4, 4, 4}, 2);
  native_math_test_log10(deviceQueue, T{-1, -1, -1, -1}, T{100, 100, 100, 100},
                         2);
  native_math_test_sqrt(deviceQueue, T{-1, -1, -1, -1}, T{4, 4, 4, 4}, 2);
  native_math_test_rsqrt(deviceQueue, T{-1, -1, -1, -1},
                         T{0.25, 0.25, 0.25, 0.25}, 2);
  native_math_test_recip(deviceQueue, T{-1, -1, -1, -1},
                         T{0.25, 0.25, 0.25, 0.25}, 4);
  native_math_test_2_powr(deviceQueue, T{-1, -1, -1, -1}, T{2, 2, 2, 2},
                          T{2, 2, 2, 2}, 4);
  native_math_test_2_divide(deviceQueue, T{-1, -1, -1, -1}, T{4, 4, 4, 4},
                            T{2, 2, 2, 2}, 2);
}

int main() {
  queue deviceQueue;

  native_tests_3<float3>(deviceQueue);
  native_tests_3<marray<float, 3>>(deviceQueue);

  native_tests_4<float4>(deviceQueue);
  native_tests_4<marray<float, 4>>(deviceQueue);

  std::cout << "Pass" << std::endl;
  return 0;
}
