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
  // Answer: Yes, it will work correctly. However, copies to/from
  //         device can be inefficient when using plain host memory
  //         (the same is true in CUDA/HIP), so using `sycl::malloc_host`
  //         (of cudaMallocHost / hipMallocHost) is preferrable.
  //         If you really want, you can use std::vector with
  //         a custom allocator to have the best of both worlds.
  float *d_y = sycl::malloc_device<float>(n, q);
  float *h_y = sycl::malloc_host<float>(n, q);
  // Allocate device and host memory for the output vector
  float *d_z = sycl::malloc_device<float>(n, q);
  float *h_z = sycl::malloc_host<float>(n, q);

  // Initialize values on host
  for (int i = 0; i < n; i++) {
    h_x[i] = i;
    h_y[i] = 2 * i;
  }
  const float alpha = 0.42f;

  q.copy<float>(h_x, d_x, n);
  q.copy<float>(h_y, d_y, n);
  // Bonus question: Why don't we need to wait before using the data?
  // Answer: Because the queue is created with in_order parameter.
  //         By default, SYCL queues are out-of-order, so without
  //         the "in_order" property we would have to specify
  //         data dependencies manually (or use buffers/accessors).

  // Run the kernel
  q.parallel_for(sycl::range<1>{n},
                 [=](sycl::id<1> i) { d_z[i] = alpha * d_x[i] + d_y[i]; });

  q.copy<float>(d_z, h_z, n);
  q.wait();

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
  sycl::free(d_y, q);
  sycl::free(h_y, q);
  sycl::free(d_z, q);
  sycl::free(h_z, q);

  return 0;
}
