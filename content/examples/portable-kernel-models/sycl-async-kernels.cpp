#include <sycl/sycl.hpp>

int main() {

  sycl::queue q;
  unsigned n = 5;
  unsigned nx = 20;

  // Allocate shared memory (Unified Shared Memory)
  int *a = sycl::malloc_shared<int>(nx, q);

  // Launch multiple potentially asynchronous kernels on different parts of the
  // array
  for (unsigned region = 0; region < n; region++) {
    q.parallel_for(sycl::range<1>{n}, [=](sycl::id<1> i) {
      const int iShifted = i + nx / n * region;
      a[iShifted] = region + iShifted;
    });
  }

  // Synchronize
  q.wait();

  // Print results
  for (unsigned i = 0; i < nx; i++)
    printf("a[%d] = %d\n", i, a[i]);

  // Free shared memory allocation (Unified Memory)
  sycl::free(a, q);

  return 0;
}
