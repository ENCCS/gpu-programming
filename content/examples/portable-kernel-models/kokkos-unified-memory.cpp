#include <Kokkos_Core.hpp>

int main(int argc, char *argv[]) {

  // Initialize Kokkos
  Kokkos::initialize(argc, argv);

  {
    unsigned n = 5;

    // Allocate on Kokkos default memory space (Unified Memory)
    int *a = (int *)Kokkos::kokkos_malloc(n * sizeof(int));
    int *b = (int *)Kokkos::kokkos_malloc(n * sizeof(int));
    int *c = (int *)Kokkos::kokkos_malloc(n * sizeof(int));

    // Initialize values on host
    for (unsigned i = 0; i < n; i++) {
      a[i] = i;
      b[i] = 1;
    }

    // Run element-wise multiplication on device
    Kokkos::parallel_for(n, KOKKOS_LAMBDA(const int i) { c[i] = a[i] * b[i]; });

    // Kokkos synchronization
    Kokkos::fence();

    // Print results
    for (unsigned i = 0; i < n; i++)
      printf("c[%d] = %d\n", i, c[i]);

    // Free Kokkos allocation (Unified Memory)
    Kokkos::kokkos_free(a);
    Kokkos::kokkos_free(b);
    Kokkos::kokkos_free(c);
  }

  // Finalize Kokkos
  Kokkos::finalize();
  return 0;
}
