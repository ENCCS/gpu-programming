#include <sycl/sycl.hpp>

int main() {

  sycl::queue q;
  unsigned n = 5;

  // Allocate shared memory (Unified Shared Memory)
  int *a = sycl::malloc_shared<int>(n, q);
  int *b = sycl::malloc_shared<int>(n, q);
  int *c = sycl::malloc_shared<int>(n, q);

  // Initialize values on host
  for (unsigned i = 0; i < n; i++) {
    a[i] = i;
    b[i] = 1;
  }

  // Run element-wise multiplication on device
  q.parallel_for(sycl::range<1>{n}, [=](sycl::id<1> i) {
     c[i] = a[i] * b[i];
   }).wait();

  // Print results
  for (unsigned i = 0; i < n; i++) {
    printf("c[%d] = %d\n", i, c[i]);
  }

  // Free shared memory allocation (Unified Memory)
  sycl::free(a, q);
  sycl::free(b, q);
  sycl::free(c, q);

  return 0;
}
