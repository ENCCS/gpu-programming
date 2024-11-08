#include <iostream>
#include <sycl/sycl.hpp>
#include <vector>

// do final reduction on host instead of using device atomics?
constexpr bool useHostReduction = false;
constexpr unsigned int tpb = 128; // threadsPerBlock

// SYCL has built-in sycl::reduction primitive, the use of which is demonstrated
// in the "Portable kernel models" chapter. Here is how the reduction can be
// implemented manually:
auto redutionKernel(sycl::handler &cgh, double *x, double *sum, int N) {
  sycl::local_accessor<double, 1> shtmp{{2 * tpb}, cgh};
  return [=](sycl::nd_item<1> item) {
    int ibl = item.get_group(0);
    int ind = item.get_global_id(0);
    int tid = item.get_local_id(0);
    shtmp[item.get_local_id(0)] = 0;
    if (ind < N / 2) {
      shtmp[tid] = x[ind];
    } else {
      shtmp[tid] = 0.0;
    }
    if (ind + N / 2 < N) {
      shtmp[tid + tpb] = x[ind + N / 2];
    } else {
      shtmp[tid + tpb] = 0.0;
    }

    for (int s = tpb; s > 0; s >>= 1) {
      if (tid < s) {
        shtmp[tid] += shtmp[tid + s];
      }
      item.barrier();
    }
    if (tid == 0) {
      if constexpr (useHostReduction) {
        sum[ibl] = shtmp[0]; // each block saves its partial result to an array
      } else {
        // Alternatively, we could agregate everything together at index 0.
        // Only useful when there not many partial sums left and when the device
        // supports atomic operations on FP64/double operands.
        sycl::atomic_ref<double, sycl::memory_order::relaxed,
                         sycl::memory_scope::device,
                         sycl::access::address_space::global_space>
            ref(sum[0]);
        ref.fetch_add(shtmp[0]);
      }
    }
  };
}

int main() {
  unsigned int nBlocks = 100;
  int n = nBlocks * tpb;
  std::vector<double> h_in(n);

  for (int i = 0; i < n; i++) {
    h_in[i] = (double)rand() / (double)RAND_MAX;
  }

  // Create queue on the default device with profiling enabled
  sycl::queue queue{{
      sycl::property::queue::in_order(),
  }};

  double *d_in = sycl::malloc_device<double>(n, queue);
  double *d_out = sycl::malloc_device<double>(nBlocks, queue);

  printf("Copying data ...\n");
  queue.copy<double>(h_in.data(), d_in, n);

  sycl::range<1> global_size{tpb * nBlocks}, local_size{tpb};
  sycl::nd_range<1> kernel_range{global_size, local_size};

  printf("Launching the kernel ...\n");
  queue.submit([&](sycl::handler &cgh) {
    cgh.parallel_for(kernel_range, redutionKernel(cgh, d_in, d_out, n));
  });

  int n_out_elements = useHostReduction ? nBlocks : 1;
  std::vector<double> h_out(n_out_elements);
  // Copying the data back
  queue.copy<double>(d_out, h_out.data(), n_out_elements).wait();

  // Calculate the last
  double totalSum = std::accumulate(h_out.begin(), h_out.end(), 0.0);
  double totalSumRef = std::accumulate(h_in.begin(), h_in.end(), 0.0);

  std::cout << "Total sum: " << totalSum << std::endl;
  std::cout << "Reference: " << totalSumRef << std::endl;

  sycl::free(d_in, queue);
  sycl::free(d_out, queue);
  return 0;
}
