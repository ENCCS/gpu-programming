#include <iostream>
#include <sycl/sycl.hpp>
#include <vector>

int main() {
  // Create an in-order queue
  sycl::queue q{sycl::property::queue::in_order()};
  // Print the device name, just for fun
  std::cout << "Running on "
            << q.get_device().get_info<sycl::info::device::name>() << std::endl;
  const int n = 1024; // Vector size

  // Allocate device and host memory for the first input vector
  float *d_x = sycl::malloc_device<float>(n, q);
  float *h_x = sycl::malloc_host<float>(n, q);
  // Bonus question: Can we use `std::vector` here instead of `malloc_host`?
  // TODO: Allocate second input vector on device and host, d_y and h_y
  // Allocate device and host memory for the output vector
  float *d_z = sycl::malloc_device<float>(n, q);
  float *h_z = sycl::malloc_host<float>(n, q);

  // Initialize values on host
  for (int i = 0; i < n; i++) {
    h_x[i] = i;
    // TODO: Initialize h_y somehow
  }
  const float alpha = 0.42f;

  q.copy<float>(h_x, d_x, n);
  // TODO: Copy h_y to d_y
  // Bonus question: Why don't we need to wait before using the data?

  // Run the kernel
  q.parallel_for(sycl::range<1>{n}, [=](sycl::id<1> i) {
    // TODO: Modify the code to compute z[i] = alpha * x[i] + y[i]
    d_z[i] = alpha * d_x[i];
  });

  // TODO: Copy d_z to h_z
  // TODO: Wait for the copy to complete

  // Check the results
  bool ok = true;
  for (int i = 0; i < n; i++) {
    float ref = alpha * h_x[i] + h_y[i]; // Reference value
    float tol = 1e-5;                    // Relative tolerance
    if (std::abs((h_z[i] - ref)) > tol * std::abs(ref)) {
      std::cout << i << " " << h_z[i] << " " << h_x[i] << " " << h_y[i]
                << std::endl;
      ok = false;
      break;
    }
  }
  if (ok)
    std::cout << "Results are correct!" << std::endl;
  else
    std::cout << "Results are NOT correct!" << std::endl;

  // Free allocated memory
  sycl::free(d_x, q);
  sycl::free(h_x, q);
  // TODO: Free d_y, h_y.
  sycl::free(d_y, q);
  sycl::free(h_y, q);

  return 0;
}
