#include <Kokkos_Core.hpp>
#include <iostream>

int main() {
  Kokkos::initialize();

  int count = Kokkos::Cuda().concurrency();
  int device =
      Kokkos::Cuda().impl_internal_space_instance()->impl_internal_space_id();

  std::cout << "Hello! I'm GPU " << device << " out of " << count
            << " GPUs in total." << std::endl;

  Kokkos::finalize();

  return 0;
}
