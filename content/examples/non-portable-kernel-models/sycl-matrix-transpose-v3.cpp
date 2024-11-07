
#include <sycl/sycl.hpp>
#include <vector>

const static int width = 4096;
const static int height = 4096;
const static int tile_dim = 16;

// Instead of defining kernel lambda at the place of submission,
// we can define it here:
auto transposeKernelSMNoBC(sycl::handler &cgh, const float *in, float *out,
                           int width, int height) {
  sycl::local_accessor<float, 1> tile{{tile_dim * (tile_dim + 1)}, cgh};
  return [=](sycl::nd_item<2> item) {
    int x_tile_index = item.get_group(1) * tile_dim;
    int y_tile_index = item.get_group(0) * tile_dim;
    int x_local_index = item.get_local_id(1);
    int y_local_index = item.get_local_id(0);
    int in_index =
        (y_tile_index + y_local_index) * width + (x_tile_index + x_local_index);
    int out_index =
        (x_tile_index + y_local_index) * width + (y_tile_index + x_local_index);

    tile[y_local_index * (tile_dim + 1) + x_local_index] = in[in_index];
    item.barrier();
    out[out_index] = tile[x_local_index * (tile_dim + 1) + y_local_index];
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
                       transposeKernelSMNoBC(cgh, d_in, d_out, width, height));
    });
  }

  // Unlike in CUDA or HIP, for SYCL we have to store all events
  std::vector<sycl::event> kernel_events;
  for (int i = 0; i < 10; i++) {
    sycl::event kernel_event = queue.submit([&](sycl::handler &cgh) {
      cgh.parallel_for(kernel_range,
                       transposeKernelSMNoBC(cgh, d_in, d_out, width, height));
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
  printf("  %.6lf ms - transpose (SM, no BC)\n  Bandwidth %.6lf GB/s\n",
         time_kernels / 10, bandwidth);

  sycl::free(d_in, queue);
  sycl::free(d_out, queue);
  return 0;
}
