#include <Kokkos_Core.hpp>

int main(int argc, char *argv[]) {

  // Initialize Kokkos
  Kokkos::initialize(argc, argv);

  {
    unsigned n = 5;
    unsigned nx = 20;

    // Allocate on Kokkos default memory space (Unified Memory)
    int *a = (int *)Kokkos::kokkos_malloc(nx * sizeof(int));

    // Create 'n' execution space instances (maps to streams in CUDA/HIP)
    auto ex = Kokkos::Experimental::partition_space(
        Kokkos::DefaultExecutionSpace(), 1, 1, 1, 1, 1);

    // Launch 'n' potentially asynchronous kernels
    // Each kernel has their own execution space instances
    for (unsigned region = 0; region < n; region++) {
      Kokkos::parallel_for(
          Kokkos::RangePolicy<Kokkos::DefaultExecutionSpace>(
              ex[region], nx / n * region, nx / n * (region + 1)),
          KOKKOS_LAMBDA(const int i) { a[i] = region + i; });
    }

    // Sync execution space instances (maps to streams in CUDA/HIP)
    for (unsigned region = 0; region < n; region++)
      ex[region].fence();

    // Print results
    for (unsigned i = 0; i < nx; i++)
      printf("a[%d] = %d\n", i, a[i]);

    // Free Kokkos allocation (Unified Memory)
    Kokkos::kokkos_free(a);
  }

  // Finalize Kokkos
  Kokkos::finalize();
  return 0;
}
