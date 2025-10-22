#include <alpaka/alpaka.hpp>

namespace ap = alpaka;

auto main() -> int {
  unsigned n = 5;
  unsigned nx = 20;

  /* Select a device, possible combinations:
   * host+cpu, cuda+nvidiaGpu, hip+amdGpu, oneApi+intelGpu, oneApi+cpu,
   * oneApi+amdGpu, oneApi+nvidiaGpu
   */
  auto devSelector =
      ap::onHost::makeDeviceSelector(ap::api::hip, ap::deviceKind::amdGpu);
  ap::onHost::Device devAcc = devSelector.makeDevice(0);
  printf("Using alpaka device: %s\n", devAcc.getName().c_str());

  // Non-blocking device queue (requires synchronization)
  using QueueType =
      ap::onHost::Queue<ALPAKA_TYPEOF(devAcc), ap::queueKind::NonBlocking>;
  std::vector<QueueType> queues;
  for (unsigned region = 0; region < n; region++) {
    queues.emplace_back(devAcc.makeQueue(ap::queueKind::nonBlocking));
  }

  // Allocate unified memory that is accessible on host and device
  auto a = ap::onHost::allocUnified<int>(devAcc, nx);

  // Run element-wise multiplication on device
  for (unsigned region = 0; region < n; region++) {
    unsigned nPerRegion = nx / n;
    unsigned regionOffset = nPerRegion * region;
    ap::onHost::iota<int>(queues[region], regionOffset,
        a.getSubView(regionOffset, nx - regionOffset));
  }
  // Wait for the device, includes all queues
  ap::onHost::wait(devAcc);

  for (unsigned i = 0; i < nx; i++) printf("a[%d] = %d\n", i, a[i]);

  return 0;
}
