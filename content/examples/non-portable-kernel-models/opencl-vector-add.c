// We're using C API here; examples with C++ API can be found in the "Portable
// kernel models" chapter
#include <CL/cl.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#define N 10000

static const char *programSource =
    "__kernel void vector_add(__global const float* A, __global const float* "
    "B, __global float* C, int N) {\n"
    "    int tid = get_global_id(0);\n"
    "    if (tid < N) {\n"
    "        C[tid] = A[tid] + B[tid];\n"
    "    }\n"
    "}\n";

int main() {
  // Initialize data and calculate reference values on CPU
  float Ah[N], Bh[N], Ch[N], Cref[N];
  for (int i = 0; i < N; i++) {
    Ah[i] = sin(i) * 2.3f;
    Bh[i] = cos(i) * 1.1f;
    Ch[i] = 12.f;
    Cref[i] = Ah[i] + Bh[i];
  }

  // Use the default device
  cl_platform_id platform;
  clGetPlatformIDs(1, &platform, NULL);
  cl_device_id device;
  clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
  cl_context context = clCreateContext(NULL, 1, &device, NULL, NULL, NULL);
  cl_command_queue queue = clCreateCommandQueue(context, device, 0, NULL);

  // Build the kernel from string
  cl_program program =
      clCreateProgramWithSource(context, 1, &programSource, NULL, NULL);
  clBuildProgram(program, 1, &device, NULL, NULL, NULL);
  cl_kernel kernel = clCreateKernel(program, "vector_add", NULL);

  // Allocate the arrays on GPU
  cl_mem d_A =
      clCreateBuffer(context, CL_MEM_READ_ONLY, N * sizeof(float), NULL, NULL);
  cl_mem d_B =
      clCreateBuffer(context, CL_MEM_READ_ONLY, N * sizeof(float), NULL, NULL);
  cl_mem d_C =
      clCreateBuffer(context, CL_MEM_WRITE_ONLY, N * sizeof(float), NULL, NULL);

  clEnqueueWriteBuffer(queue, d_A, CL_TRUE, 0, N * sizeof(float), Ah, 0, NULL,
                       NULL);
  clEnqueueWriteBuffer(queue, d_B, CL_TRUE, 0, N * sizeof(float), Bh, 0, NULL,
                       NULL);

  // Set arguments and launch the kernel
  clSetKernelArg(kernel, 0, sizeof(cl_mem), &d_A);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), &d_B);
  clSetKernelArg(kernel, 2, sizeof(cl_mem), &d_C);
  cl_int N_as_cl_int = N;
  clSetKernelArg(kernel, 3, sizeof(cl_int), &N_as_cl_int);
  size_t globalSize = N;
  clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, NULL, 0, NULL,
                         NULL);

  // Copy the results back
  clEnqueueReadBuffer(queue, d_C, CL_TRUE, 0, N * sizeof(float), Ch, 0, NULL,
                      NULL);

  // Print reference and result values
  printf("Reference: %f %f %f %f ... %f %f\n", Cref[0], Cref[1], Cref[2],
         Cref[3], Cref[N - 2], Cref[N - 1]);
  printf("Result   : %f %f %f %f ... %f %f\n", Ch[0], Ch[1], Ch[2], Ch[3],
         Ch[N - 2], Ch[N - 1]);

  // Compare results and calculate the total error
  float error = 0.0f;
  float tolerance = 1e-6f;
  for (int i = 0; i < N; i++) {
    float diff = fabs(Cref[i] - Ch[i]);
    if (diff > tolerance) {
      error += diff;
    }
  }

  printf("Total error: %f\n", error);
  printf("Reference:   %f at (42)\n", Cref[42]);
  printf("Result   :   %f at (42)\n", Ch[42]);

  clReleaseMemObject(d_A);
  clReleaseMemObject(d_B);
  clReleaseMemObject(d_C);
  clReleaseKernel(kernel);
  clReleaseProgram(program);
  clReleaseCommandQueue(queue);
  clReleaseContext(context);

  return 0;
}
