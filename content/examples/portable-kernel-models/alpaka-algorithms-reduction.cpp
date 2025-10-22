#include <alpaka/alpaka.hpp>

namespace ap = alpaka;

auto main() -> int {
  unsigned n = 10;

  /* Select a device, possible combinations:
   * host+cpu, cuda+nvidiaGpu, hip+amdGpu, oneApi+intelGpu, oneApi+cpu,
   * oneApi+amdGpu, oneApi+nvidiaGpu
   */
  auto devSelector =
      ap::onHost::makeDeviceSelector(ap::api::hip, ap::deviceKind::amdGpu);
  ap::onHost::Device devAcc = devSelector.makeDevice(0);
  printf("Using alpaka device: %s\n", devAcc.getName().c_str());

  // Blocking device queue (requires synchronization)
  ap::onHost::Queue queue = devAcc.makeQueue(ap::queueKind::blocking);

  // Allocate unified memory that is accessible on host and device
  auto sum = ap::onHost::allocUnified<int>(devAcc, 1);

  // Run element-wise multiplication on device
  ap::onHost::reduce(queue, 0, sum, std::plus{}, ap::LinearizedIdxGenerator{n});

  // Print results
  printf("sum = %d\n", sum[0]);

  return 0;
}
