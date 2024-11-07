#include <cuda.h>
#include <cuda_runtime.h>
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
  int i;

  // Allocate the arrays using Unified Memory
  cudaMallocManaged(&Ah, N * sizeof(float));
  cudaMallocManaged(&Bh, N * sizeof(float));
  cudaMallocManaged(&Ch, N * sizeof(float));
  cudaMallocManaged(&Cref, N * sizeof(float));

  // initialise data and calculate reference values on CPU
  for (i = 0; i < N; i++) {
    Ah[i] = sin(i) * 2.3;
    Bh[i] = cos(i) * 1.1;
    Cref[i] = Ah[i] + Bh[i];
  }

  // define grid dimensions
  dim3 blocks, threads;
  threads = dim3(256, 1, 1);
  blocks = dim3((N + 256 - 1) / 256, 1, 1);

  // Launch Kernel
  vector_add<<<blocks, threads>>>(Ah, Bh, Ch, N);
  cudaDeviceSynchronize(); // Wait for the kernel to complete

  // At this point we want to access the data on CPU
  printf("reference: %f %f %f %f ... %f %f\n", Cref[0], Cref[1], Cref[2],
         Cref[3], Cref[N - 2], Cref[N - 1]);
  printf("   result: %f %f %f %f ... %f %f\n", Ch[0], Ch[1], Ch[2], Ch[3],
         Ch[N - 2], Ch[N - 1]);

  // confirm that results are correct
  float error = 0.0;
  float tolerance = 1e-6;
  float diff;
  for (i = 0; i < N; i++) {
    diff = fabs(Cref[i] - Ch[i]);
    if (diff > tolerance) {
      error += diff;
    }
  }
  printf("total error: %f\n", error);
  printf("  reference: %f at (42)\n", Cref[42]);
  printf("     result: %f at (42)\n", Ch[42]);

  // Free the GPU arrays
  cudaFree(Ah);
  cudaFree(Bh);
  cudaFree(Ch);
  cudaFree(Cref);

  return 0;
}
