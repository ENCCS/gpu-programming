#include <sycl/sycl.hpp>

int main() {

  sycl::queue q;
  unsigned n = 5;

  // Allocate space for 5 ints
  auto a_buf = sycl::buffer<int>(sycl::range<1>(n));
  auto b_buf = sycl::buffer<int>(sycl::range<1>(n));
  auto c_buf = sycl::buffer<int>(sycl::range<1>(n));

  // Initialize values
  // We should use curly braces to limit host accessors' lifetime
  //    and indicate when we're done working with them:
  {
    auto a_host_acc = a_buf.get_host_access();
    auto b_host_acc = b_buf.get_host_access();
    for (unsigned i = 0; i < n; i++) {
      a_host_acc[i] = i;
      b_host_acc[i] = 1;
    }
  }

  // Submit a SYCL kernel into a queue
  q.submit([&](sycl::handler &cgh) {
    // Create read accessors over a_buf and b_buf
    auto a_acc = a_buf.get_access<sycl::access_mode::read>(cgh);
    auto b_acc = b_buf.get_access<sycl::access_mode::read>(cgh);
    // Create write accesor over c_buf
    auto c_acc = c_buf.get_access<sycl::access_mode::write>(cgh);
    // Run element-wise multiplication on device
    cgh.parallel_for<class vec_add>(sycl::range<1>{n}, [=](sycl::id<1> i) {
      c_acc[i] = a_acc[i] * b_acc[i];
    });
  });

  // No need to synchronize, creating the accessor for c_buf will do it
  // automatically
  {
    const auto c_host_acc = c_buf.get_host_access();
    // Print results
    for (unsigned i = 0; i < n; i++)
      printf("c[%d] = %d\n", i, c_host_acc[i]);
  }

  return 0;
}
