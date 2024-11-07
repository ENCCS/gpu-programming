#include <Kokkos_Core.hpp>

int main(int argc, char *argv[]) {

  // Initialize Kokkos
  Kokkos::initialize(argc, argv);

  {
    unsigned n = 5;

    // Allocate space for 5 ints on Kokkos host memory space
    Kokkos::View<int *, Kokkos::HostSpace> h_a("h_a", n);
    Kokkos::View<int *, Kokkos::HostSpace> h_b("h_b", n);
    Kokkos::View<int *, Kokkos::HostSpace> h_c("h_c", n);

    // Allocate space for 5 ints on Kokkos default memory space (eg, GPU memory)
    Kokkos::View<int *> a("a", n);
    Kokkos::View<int *> b("b", n);
    Kokkos::View<int *> c("c", n);

    // Initialize values on host
    for (unsigned i = 0; i < n; i++) {
      h_a[i] = i;
      h_b[i] = 1;
    }

    // Copy from host to device
    Kokkos::deep_copy(a, h_a);
    Kokkos::deep_copy(b, h_b);

    // Run element-wise multiplication on device
    Kokkos::parallel_for(n, KOKKOS_LAMBDA(const int i) { c[i] = a[i] * b[i]; });

    // Copy from device to host
    Kokkos::deep_copy(h_c, c);

    // Print results
    for (unsigned i = 0; i < n; i++)
      printf("c[%d] = %d\n", i, h_c[i]);
  }

  // Finalize Kokkos
  Kokkos::finalize();
  return 0;
}
