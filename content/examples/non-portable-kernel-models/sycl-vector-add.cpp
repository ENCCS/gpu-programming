#include <iostream>
#include <sycl/sycl.hpp>

int main() {
  const int N = 10000;
  // The queue will be executed on the best device in the system
  // We use in-order queue for simplicity
  sycl::queue q{{sycl::property::queue::in_order()}};

  std::vector<float> Ah(N);
  std::vector<float> Bh(N);
  std::vector<float> Ch(N);
  std::vector<float> Cref(N);

  // Initialize data and calculate reference values on CPU
  for (int i = 0; i < N; i++) {
    Ah[i] = std::sin(i) * 2.3f;
    Bh[i] = std::cos(i) * 1.1f;
    Cref[i] = Ah[i] + Bh[i];
  }

  // Allocate the arrays on GPU
  float *Ad = sycl::malloc_device<float>(N, q);
  float *Bd = sycl::malloc_device<float>(N, q);
  float *Cd = sycl::malloc_device<float>(N, q);

  q.copy<float>(Ah.data(), Ad, N);
  q.copy<float>(Bh.data(), Bd, N);

  // Define grid dimensions
  // We can specify the block size explicitly, but we don't have to
  sycl::range<1> global_size(N);
  q.submit([&](sycl::handler &h) {
    h.parallel_for<class VectorAdd>(global_size, [=](sycl::id<1> threadId) {
      int tid = threadId.get(0);
      Cd[tid] = Ad[tid] + Bd[tid];
    });
  });

  // Copy results back to CPU
  sycl::event eventCCopy = q.copy<float>(Cd, Ch.data(), N);
  // Wait for the copy to finish
  eventCCopy.wait();

  // Print reference and result values
  std::cout << "Reference: " << Cref[0] << " " << Cref[1] << " " << Cref[2]
            << " " << Cref[3] << " ... " << Cref[N - 2] << " " << Cref[N - 1]
            << std::endl;
  std::cout << "Result   : " << Ch[0] << " " << Ch[1] << " " << Ch[2] << " "
            << Ch[3] << " ... " << Ch[N - 2] << " " << Ch[N - 1] << std::endl;

  // Compare results and calculate the total error
  float error = 0.0f;
  float tolerance = 1e-6f;
  for (int i = 0; i < N; i++) {
    float diff = std::abs(Cref[i] - Ch[i]);
    if (diff > tolerance) {
      error += diff;
    }
  }

  std::cout << "Total error: " << error << std::endl;
  std::cout << "Reference:   " << Cref[42] << " at (42)" << std::endl;
  std::cout << "Result   :   " << Ch[42] << " at (42)" << std::endl;

  // Free the GPU memory
  sycl::free(Ad, q);
  sycl::free(Bd, q);
  sycl::free(Cd, q);

  return 0;
}
