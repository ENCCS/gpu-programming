#include <Kokkos_Core.hpp>

int main(int argc, char *argv[]) {

  // Initialize Kokkos
  Kokkos::initialize(argc, argv);

  {
    unsigned n = 10;

    // Initialize sum variable
    int sum = 0;

    // Run sum reduction kernel
    Kokkos::parallel_reduce(
        n, KOKKOS_LAMBDA(const int i, int &lsum) { lsum += i; }, sum);

    // Kokkos synchronization
    Kokkos::fence();

    // Print results
    printf("sum = %d\n", sum);
  }

  // Finalize Kokkos
  Kokkos::finalize();
  return 0;
}
