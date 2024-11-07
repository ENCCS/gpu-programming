// We're using OpenCL C++ API here; there is also C API in <CL/cl.h>
#define CL_TARGET_OPENCL_VERSION 110
#define CL_HPP_TARGET_OPENCL_VERSION 110
#include <CL/cl.hpp>

// For larger kernels, we can store source in a separate file
static const std::string kernel_source = R"(
            __kernel void dot(__global const int *a, __global const int *b, __global int *c) {
              int i = get_global_id(0);
              c[i] = a[i] * b[i];
            }
          )";

int main(int argc, char *argv[]) {

  // Initialize OpenCL
  cl::Device device = cl::Device::getDefault();
  cl::Context context(device);
  cl::CommandQueue queue(context, device);

  // Compile OpenCL program for found device.
  cl::Program program(context, kernel_source);
  program.build({device});
  cl::Kernel kernel_dot(program, "dot");

  {
    // Set problem dimensions
    unsigned n = 5;

    std::vector<int> a(n), b(n), c(n);

    // Initialize values on host
    for (unsigned i = 0; i < n; i++) {
      a[i] = i;
      b[i] = 1;
    }

    // Create buffers and copy input data to device.
    cl::Buffer dev_a(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     n * sizeof(int), a.data());
    cl::Buffer dev_b(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
                     n * sizeof(int), b.data());
    cl::Buffer dev_c(context, CL_MEM_WRITE_ONLY, n * sizeof(int));

    // Pass arguments to device kernel
    kernel_dot.setArg(0, dev_a);
    kernel_dot.setArg(1, dev_b);
    kernel_dot.setArg(2, dev_c);

    // We don't need to apply any offset to thread IDs
    queue.enqueueNDRangeKernel(kernel_dot, cl::NullRange, cl::NDRange(n),
                               cl::NullRange);

    // Read result
    queue.enqueueReadBuffer(dev_c, CL_TRUE, 0, n * sizeof(int), c.data());

    // Print results
    for (unsigned i = 0; i < n; i++)
      printf("c[%d] = %d\n", i, c[i]);
  }

  return 0;
}
