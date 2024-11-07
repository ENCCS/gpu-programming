
#include <sycl/sycl.hpp>
#include <vector>

const static int width = 4096;
const static int height = 4096;
const static int tile_dim = 16;

// Instead of defining kernel lambda at the place of submission,
// we can define it here:
auto transposeKernelNaive(const float *in, float *out, int width, int height) {
  return [=](sycl::nd_item<2> item) {
    int x_index = item.get_global_id(1);
    int y_index = item.get_global_id(0);
    int in_index = y_index * width + x_index;
    int out_index = x_index * height + y_index;
    out[out_index] = in[in_index];
  };
}

int main() {
  std::vector<float> matrix_in(width * height);
  std::vector<float> matrix_out(width * height);

  for (int i = 0; i < width * height; i++) {
    matrix_in[i] = (float)rand() / (float)RAND_MAX;
  }

  // Create queue on the default device with profiling enabled
  sycl::queue queue{{sycl::property::queue::in_order(),
                     sycl::property::queue::enable_profiling()}};

  float *d_in = sycl::malloc_device<float>(width * height, queue);
  float *d_out = sycl::malloc_device<float>(width * height, queue);

  queue.copy<float>(matrix_in.data(), d_in, width * height);
  queue.wait();

  printf("Setup complete. Launching kernel\n");
  sycl::range<2> global_size{height, width}, local_size{tile_dim, tile_dim};
  sycl::nd_range<2> kernel_range{global_size, local_size};

  // Create events
  printf("Warm up the GPU!\n");
  for (int i = 0; i < 10; i++) {
    queue.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(kernel_range,
                       transposeKernelNaive(d_in, d_out, width, height));
    });
  }

  // Unlike in CUDA or HIP, for SYCL we have to store all events
  std::vector<sycl::event> kernel_events;
  for (int i = 0; i < 10; i++) {
    sycl::event kernel_event = queue.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(kernel_range,
                       transposeKernelNaive(d_in, d_out, width, height));
    });
    kernel_events.push_back(kernel_event);
  }

  queue.wait();

  auto first_kernel_started =
      kernel_events.front()
          .get_profiling_info<sycl::info::event_profiling::command_start>();
  auto last_kernel_ended =
      kernel_events.back()
          .get_profiling_info<sycl::info::event_profiling::command_end>();
  double total_kernel_time_ns =
      static_cast<double>(last_kernel_ended - first_kernel_started);
  double time_kernels = total_kernel_time_ns / 1e6; // convert ns to ms
  double bandwidth = 2.0 * 10000 *
                     (((double)(width) * (double)height) * sizeof(float)) /
                     (time_kernels * 1024 * 1024 * 1024);

  printf("Kernel execution complete\n");
  printf("Event timings:\n");
  printf("  %.6lf ms - transpose (naive)\n  Bandwidth %.6lf GB/s\n",
         time_kernels / 10, bandwidth);

  sycl::free(d_in, queue);
  sycl::free(d_out, queue);
  return 0;
}
