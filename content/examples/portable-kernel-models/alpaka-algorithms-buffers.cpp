#include <alpaka/alpaka.hpp>

namespace ap = alpaka;

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

  // Allocate memory that is accessible on host
  auto h_a = ap::onHost::allocHost<int>(n);
  auto h_b = ap::onHost::allocHostLike(h_a);
  auto h_c = ap::onHost::allocHostLike(h_a);

  // Allocate memory on the device and inherit the extents from h_a
  auto a = ap::onHost::allocLike(devAcc, h_a);
  auto b = ap::onHost::allocLike(devAcc, h_a);
  auto c = ap::onHost::allocLike(devAcc, h_a);

  // Initialize values on host
  for (unsigned i = 0; i < n; i++) {
    h_a[i] = i;
    h_b[i] = 1;
  }

  // Copy host memory element wise to the device memory
  ap::onHost::memcpy(queue, a, h_a);
  ap::onHost::memcpy(queue, b, h_b);

  // Run element-wise multiplication on device
  ap::onHost::transform(queue, c, std::multiplies{}, a, b);

  // Copy the device result back to host memory
  ap::onHost::memcpy(queue, h_c, c);

  for (unsigned i = 0; i < n; i++) {
    printf("c[%d] = %d\n", i, h_c[i]);
  }

  return 0;
}
