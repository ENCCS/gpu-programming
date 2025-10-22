#include <alpaka/alpaka.hpp>

namespace ap = alpaka;

struct IdxAssignKernel {
  constexpr void operator()(ap::onAcc::concepts::Acc auto const& acc,
      ap::concepts::IMdSpan auto a,
      unsigned region,
      unsigned n) const {
    unsigned nPerRegion = a.getExtents().x() / n;
    unsigned regionOffset = nPerRegion * region;
    for (auto [idx] :
        ap::onAcc::makeIdxMap(acc, ap::onAcc::worker::threadsInGrid,
            ap::IdxRange{regionOffset, regionOffset + nPerRegion})) {
      a[idx] = idx;
    }
  }
};

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

  unsigned frameExtent = 32u;
  auto frameSpec =
      ap::onHost::FrameSpec{ap::divExZero(nx, frameExtent), frameExtent};

  // Run element-wise multiplication on device
  for (unsigned region = 0; region < n; region++) {
    queues[region].enqueue(
        frameSpec, ap::KernelBundle{IdxAssignKernel{}, a, region, n});
  }
  // Wait for the device, includes all queues
  ap::onHost::wait(devAcc);

  for (unsigned i = 0; i < nx; i++) printf("a[%d] = %d\n", i, a[i]);

  return 0;
}
