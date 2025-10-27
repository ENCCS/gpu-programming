#include <alpaka/alpaka.hpp>

namespace ap = alpaka;

struct MulKernel {
  constexpr void operator()(ap::onAcc::concepts::Acc auto const& acc,
      ap::concepts::IMdSpan auto c,
      ap::concepts::IMdSpan auto const a,
      ap::concepts::IMdSpan auto const b) const {
    for (auto idx : ap::onAcc::makeIdxMap(acc, ap::onAcc::worker::threadsInGrid,
             ap::IdxRange{c.getExtents()})) {
      c[idx] = a[idx] * b[idx];
    }
  }
};

auto main() -> int {
  unsigned n = 5;

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
  auto a = ap::onHost::allocUnified<int>(devAcc, n);
  auto b = ap::onHost::allocUnified<int>(devAcc, n);
  auto c = ap::onHost::allocUnified<int>(devAcc, n);

  // Initialize values on host
  for (unsigned i = 0; i < n; i++) {
    a[i] = i;
    b[i] = 1;
  }

  unsigned frameExtent = 32u;
  auto frameSpec =
      ap::onHost::FrameSpec{ap::divExZero(n, frameExtent), frameExtent};

  // Run element-wise multiplication on device
  queue.enqueue(frameSpec, ap::KernelBundle{MulKernel{}, c, a, b});

  for (unsigned i = 0; i < n; i++) {
    printf("c[%d] = %d\n", i, c[i]);
  }

  return 0;
}
