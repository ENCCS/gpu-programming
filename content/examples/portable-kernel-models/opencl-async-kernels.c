// We're using OpenCL C API here, since SVM support in C++ API is unstable on
// ROCm
#define CL_TARGET_OPENCL_VERSION 200
#include <CL/cl.h>
#include <stdio.h>

// For larger kernels, we can store source in a separate file
static const char *kernel_source = "              \
                    __kernel void async(__global int *a) { \
                      int i = get_global_id(0);            \
                      int region = i / get_global_size(0); \
                      a[i] = region + i;                   \
                    }                                      \
         ";

int main(int argc, char *argv[]) {
  // Initialize OpenCL
  cl_platform_id platform;
  clGetPlatformIDs(1, &platform, NULL);
  cl_device_id device;
  clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
  cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
  cl_command_queue queue = clCreateCommandQueue(context, device, 0, NULL);

  // Compile OpenCL program for found device.
  cl_program program =
      clCreateProgramWithSource(context, 1, &kernel_source, NULL, NULL);
  clBuildProgram(program, 1, &device, NULL, NULL, NULL);
  cl_kernel kernel = clCreateKernel(program, "async", NULL);

  // Set problem dimensions
  unsigned n = 5;
  unsigned nx = 20;

  // Create SVM buffer objects on host side
  int *a = clSVMAlloc(context, CL_MEM_WRITE_ONLY, nx * sizeof(int), 0);

  // Pass arguments to device kernel
  clSetKernelArgSVMPointer(kernel, 0, a);

  // Launch multiple potentially asynchronous kernels on different parts of the
  // array
  for (unsigned region = 0; region < n; region++) {
    size_t offset = (nx / n) * region;
    size_t size = nx / n;
    clEnqueueNDRangeKernel(queue, kernel, 1, &offset, &size, NULL, 0, NULL,
                           NULL);
  }

  // Create mapping for host and print results
  clEnqueueSVMMap(queue, CL_TRUE, CL_MAP_READ, a, nx * sizeof(int), 0, NULL,
                  NULL);
  for (unsigned i = 0; i < nx; i++)
    printf("a[%d] = %d\n", i, a[i]);
  clEnqueueSVMUnmap(queue, a, 0, NULL, NULL);

  // Free SVM buffers
  clSVMFree(context, a);

  return 0;
}
