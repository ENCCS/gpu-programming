#include <hip/hip_runtime.h>
#include <math.h>
#include <stdio.h>

__global__ void vector_add(float *A, float *B, float *C, int n) {
  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < n) {
    C[tid] = A[tid] + B[tid];
  }
}

int main(void) {
  const int N = 10000;
  float *Ah, *Bh, *Ch, *Cref;
  // Allocate the arrays using Unified Memory
  hipMallocManaged((void **)&Ah, N * sizeof(float));
  hipMallocManaged((void **)&Bh, N * sizeof(float));
  hipMallocManaged((void **)&Ch, N * sizeof(float));
  hipMallocManaged((void **)&Cref, N * sizeof(float));

  // Initialize data and calculate reference values on CPU
  for (int i = 0; i < N; i++) {
    Ah[i] = sin(i) * 2.3;
    Bh[i] = cos(i) * 1.1;
    Cref[i] = Ah[i] + Bh[i];
  }
  // All data at this point is on CPU

  // Define grid dimensions + launch the device kernel
  dim3 blocks, threads;
  threads = dim3(256, 1, 1);
  blocks = dim3((N + 256 - 1) / 256, 1, 1);

  // Launch Kernel
  //  use
  // hipLaunchKernelGGL(vector_add, blocks, threads, 0, 0, Ah, Bh, Ch, N); // or
  vector_add<<<blocks, threads>>>(Ah, Bh, Ch, N);
  hipDeviceSynchronize(); // Wait for the kernel to complete

  // At this point we want to access the data on the CPU
  printf("reference: %f %f %f %f ... %f %f\n", Cref[0], Cref[1], Cref[2],
         Cref[3], Cref[N - 2], Cref[N - 1]);
  printf("   result: %f %f %f %f ... %f %f\n", Ch[0], Ch[1], Ch[2], Ch[3],
         Ch[N - 2], Ch[N - 1]);

  // Confirm that results are correct
  float error = 0.0;
  float tolerance = 1e-6;
  float diff;
  for (int i = 0; i < N; i++) {
    diff = fabs(Cref[i] - Ch[i]);
    if (diff > tolerance) {
      error += diff;
    }
  }
  printf("total error: %f\n", error);
  printf("  reference: %f at (42)\n", Cref[42]);
  printf("     result: %f at (42)\n", Ch[42]);

  // Free the Unified Memory arrays
  hipFree(Ah);
  hipFree(Bh);
  hipFree(Ch);
  hipFree(Cref);

  return 0;
}
