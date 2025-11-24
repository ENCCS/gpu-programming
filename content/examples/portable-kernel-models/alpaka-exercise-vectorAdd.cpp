#include <alpaka/alpaka.hpp>

namespace ap = alpaka;

auto main() -> int {
  unsigned n = 5;

  /* Select a device, possible combinations:
   * host+cpu, cuda+nvidiaGpu, hip+amdGpu, oneApi+intelGpu, oneApi+cpu,
   * oneApi+amdGpu, oneApi+nvidiaGpu
   */

  // auto devSelector = ap::onHost::makeDeviceSelector(ap::api::host,
  // ap::deviceKind::cpu);
  auto devSelector =
      ap::onHost::makeDeviceSelector(ap::api::hip, ap::deviceKind::amdGpu);
  ap::onHost::Device devAcc = devSelector.makeDevice(0);
  printf("Using alpaka device: %s\n", devAcc.getName().c_str());

  // Blocking device queue (requires synchronization)
  ap::onHost::Queue queue = devAcc.makeQueue(ap::queueKind::blocking);

  // Allocate unified memory that is accessible on host and device
  auto a = ap::onHost::allocUnified<int>(devAcc, n);
  auto b = ap::onHost::allocUnified<int>(devAcc, n);
  auto c = ap::onHost::allocUnified<int>(devAcc, n);

  // Initialize values on host
  for (unsigned i = 0; i < n; i++) {
    a[i] = i;
    b[i] = 1;
  }

  // Run element-wise vector addition on device
  ap::onHost::transform(queue, c, std::plus{}, a, b);

  for (unsigned i = 0; i < n; i++) {
    printf("c[%d] = %d\n", i, c[i]);
  }

  return 0;
}
