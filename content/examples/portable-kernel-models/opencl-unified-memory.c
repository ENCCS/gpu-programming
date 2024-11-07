// We're using OpenCL C API here, since SVM support in C++ API is unstable on
// ROCm
#define CL_TARGET_OPENCL_VERSION 220
#include <CL/cl.h>
#include <stdio.h>

// For larger kernels, we can store source in a separate file
static const char *kernel_source =
    "                                                 \
           __kernel void dot(__global const int *a, __global const int *b, __global int *c) { \
             int i = get_global_id(0);                                                        \
             c[i] = a[i] * b[i];                                                              \
           }                                                                                  \
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
  cl_kernel kernel = clCreateKernel(program, "dot", NULL);

  // Set problem dimensions
  unsigned n = 5;

  // Create SVM buffer objects on host side
  int *a = clSVMAlloc(context, CL_MEM_READ_ONLY, n * sizeof(int), 0);
  int *b = clSVMAlloc(context, CL_MEM_READ_ONLY, n * sizeof(int), 0);
  int *c = clSVMAlloc(context, CL_MEM_WRITE_ONLY, n * sizeof(int), 0);

  // Pass arguments to device kernel
  clSetKernelArgSVMPointer(kernel, 0, a);
  clSetKernelArgSVMPointer(kernel, 1, b);
  clSetKernelArgSVMPointer(kernel, 2, c);

  // Create mappings for host and initialize values
  clEnqueueSVMMap(queue, CL_TRUE, CL_MAP_WRITE, a, n * sizeof(int), 0, NULL,
                  NULL);
  clEnqueueSVMMap(queue, CL_TRUE, CL_MAP_WRITE, b, n * sizeof(int), 0, NULL,
                  NULL);
  for (unsigned i = 0; i < n; i++) {
    a[i] = i;
    b[i] = 1;
  }
  clEnqueueSVMUnmap(queue, a, 0, NULL, NULL);
  clEnqueueSVMUnmap(queue, b, 0, NULL, NULL);

  size_t globalSize = n;
  clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, NULL, 0, NULL,
                         NULL);

  // Create mapping for host and print results
  clEnqueueSVMMap(queue, CL_TRUE, CL_MAP_READ, c, n * sizeof(int), 0, NULL,
                  NULL);
  for (unsigned i = 0; i < n; i++)
    printf("c[%d] = %d\n", i, c[i]);
  clEnqueueSVMUnmap(queue, c, 0, NULL, NULL);

  // Free SVM buffers
  clSVMFree(context, a);
  clSVMFree(context, b);
  clSVMFree(context, c);

  return 0;
}
