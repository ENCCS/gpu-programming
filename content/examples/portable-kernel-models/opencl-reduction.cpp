// We're using OpenCL C++ API here; there is also C API in <CL/cl.h>
#define CL_TARGET_OPENCL_VERSION 110
#define CL_HPP_TARGET_OPENCL_VERSION 110
#include <CL/cl.hpp>

// For larger kernels, we can store source in a separate file
static const std::string kernel_source = R"(
           __kernel void reduce(__global int* sum, __local int* local_mem) {
             
             // Get work group and work item information
             int gsize = get_global_size(0); // global work size
             int gid = get_global_id(0); // global work item index
             int lsize = get_local_size(0); // local work size
             int lid = get_local_id(0); // local work item index
             
             // Store reduced item into local memory
             local_mem[lid] = gid; // initialize local memory
             barrier(CLK_LOCAL_MEM_FENCE); // synchronize local memory
             
             // Perform reduction across the local work group
             for (int s = 1; s < lsize; s *= 2) { // loop over local memory with stride doubling each iteration
               if (lid % (2 * s) == 0 && (lid + s) < lsize) {
                 local_mem[lid] += local_mem[lid + s];
               }
               barrier(CLK_LOCAL_MEM_FENCE); // synchronize local memory
             }
             
             if (lid == 0) { // only one work item per work group
               atomic_add(sum, local_mem[0]); // add partial sum to global sum atomically
             }
           }
         )";

int main(int argc, char *argv[]) {

  // Initialize OpenCL
  cl::Device device = cl::Device::getDefault();
  cl::Context context(device);
  cl::CommandQueue queue(context, device);

  // Compile OpenCL program for found device
  cl::Program program(context, kernel_source);
  program.build({device});
  cl::Kernel kernel_reduce(program, "reduce");

  {
    // Set problem dimensions
    unsigned n = 10;

    // Initialize sum variable
    int sum = 0;

    // Create buffer for sum
    cl::Buffer buffer(context, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR,
                      sizeof(int), &sum);

    // Pass arguments to device kernel
    kernel_reduce.setArg(0, buffer);            // pass buffer to device
    kernel_reduce.setArg(1, sizeof(int), NULL); // allocate local memory

    // Enqueue kernel
    queue.enqueueNDRangeKernel(kernel_reduce, cl::NullRange, cl::NDRange(n),
                               cl::NullRange);

    // Read result
    queue.enqueueReadBuffer(buffer, CL_TRUE, 0, sizeof(int), &sum);

    // Print result
    printf("sum = %d\n", sum);
  }

  return 0;
}
