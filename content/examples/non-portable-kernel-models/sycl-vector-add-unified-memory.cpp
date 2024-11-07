#include <iostream>
#include <sycl/sycl.hpp>

int main() {
  const int N = 10000;
  // The queue will be executed on the best device in the system
  // We use in-order queue for simplicity
  sycl::queue q{{sycl::property::queue::in_order()}};

  std::vector<float> Cref(N);

  // Allocate the shared arrays
  float *A = sycl::malloc_shared<float>(N, q);
  float *B = sycl::malloc_shared<float>(N, q);
  float *C = sycl::malloc_shared<float>(N, q);

  // Initialize data and calculate reference values on CPU
  for (int i = 0; i < N; i++) {
    A[i] = std::sin(i) * 2.3f;
    B[i] = std::cos(i) * 1.1f;
    Cref[i] = A[i] + B[i];
  }

  // Define grid dimensions
  // We can specify the block size explicitly, but we don't have to
  sycl::range<1> global_size(N);
  q.submit([&](sycl::handler &h) {
     h.parallel_for<class VectorAdd>(global_size, [=](sycl::id<1> threadId) {
       int tid = threadId.get(0);
       C[tid] = A[tid] + B[tid];
     });
   }).wait(); // Wait for the kernel to finish

  // Print reference and result values
  std::cout << "Reference: " << Cref[0] << " " << Cref[1] << " " << Cref[2]
            << " " << Cref[3] << " ... " << Cref[N - 2] << " " << Cref[N - 1]
            << std::endl;
  std::cout << "Result   : " << C[0] << " " << C[1] << " " << C[2] << " "
            << C[3] << " ... " << C[N - 2] << " " << C[N - 1] << std::endl;

  // Compare results and calculate the total error
  float error = 0.0f;
  float tolerance = 1e-6f;
  for (int i = 0; i < N; i++) {
    float diff = std::abs(Cref[i] - C[i]);
    if (diff > tolerance) {
      error += diff;
    }
  }

  std::cout << "Total error: " << error << std::endl;
  std::cout << "Reference:   " << Cref[42] << " at (42)" << std::endl;
  std::cout << "Result   :   " << C[42] << " at (42)" << std::endl;

  // Free the shared memory
  sycl::free(A, q);
  sycl::free(B, q);
  sycl::free(C, q);

  return 0;
}
