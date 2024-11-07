#include <iostream>
#include <sycl/sycl.hpp>

int main() {
  auto gpu_devices = sycl::device::get_devices(sycl::info::device_type::gpu);
  auto count = gpu_devices.size();
  std::cout << "Hello! I'm using a SYCL device by "
            << gpu_devices[0].get_info<sycl::info::device::vendor>()
            << ">, the first of " << count << " devices." << std::endl;
  return 0;
}
