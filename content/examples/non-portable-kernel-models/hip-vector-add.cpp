
#include <hip/hip_runtime.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

__global__ void vector_add(float *A, float *B, float *C, int n) {

  int tid = threadIdx.x + blockIdx.x * blockDim.x;
  if (tid < n) {
    C[tid] = A[tid] + B[tid];
  }
}

int main(void) {
  const int N = 10000;
  float *Ah, *Bh, *Ch, *Cref;
  float *Ad, *Bd, *Cd;

  // Allocate the arrays on CPU
  Ah = (float *)malloc(N * sizeof(float));
  Bh = (float *)malloc(N * sizeof(float));
  Ch = (float *)malloc(N * sizeof(float));
  Cref = (float *)malloc(N * sizeof(float));

  // initialise data and calculate reference values on CPU
  for (int i = 0; i < N; i++) {
    Ah[i] = sin(i) * 2.3;
    Bh[i] = cos(i) * 1.1;
    Cref[i] = Ah[i] + Bh[i];
  }

  // Allocate the arrays on GPU
  hipMalloc((void **)&Ad, N * sizeof(float));
  hipMalloc((void **)&Bd, N * sizeof(float));
  hipMalloc((void **)&Cd, N * sizeof(float));

  // Transfer the data from CPU to GPU
  hipMemcpy(Ad, Ah, sizeof(float) * N, hipMemcpyHostToDevice);
  hipMemcpy(Bd, Bh, sizeof(float) * N, hipMemcpyHostToDevice);

  // define grid dimensions + launch the device kernel
  dim3 blocks, threads;
  threads = dim3(256, 1, 1);
  blocks = dim3((N + 256 - 1) / 256, 1, 1);

  // Launch Kernel
  //  use
  // hipLaunchKernelGGL(vector_add, blocks, threads, 0, 0, Ad, Bd, Cd, N); // or
  vector_add<<<blocks, threads, 0, 0>>>(Ad, Bd, Cd, N);

  // copy results back to CPU
  hipMemcpy(Ch, Cd, sizeof(float) * N, hipMemcpyDeviceToHost);

  printf("reference: %f %f %f %f ... %f %f\n", Cref[0], Cref[1], Cref[2],
         Cref[3], Cref[N - 2], Cref[N - 1]);
  printf("   result: %f %f %f %f ... %f %f\n", Ch[0], Ch[1], Ch[2], Ch[3],
         Ch[N - 2], Ch[N - 1]);

  // confirm that results are correct
  float error = 0.0;
  float tolerance = 1e-6;
  float diff;
  for (int i = 0; i < N; i++) {
    diff = abs(Cref[i] - Ch[i]);
    if (diff > tolerance) {
      error += diff;
    }
  }
  printf("total error: %f\n", error);
  printf("  reference: %f at (42)\n", Cref[42]);
  printf("     result: %f at (42)\n", Ch[42]);

  // Free the GPU arrays
  hipFree(Ad);
  hipFree(Bd);
  hipFree(Cd);

  // Free the CPU arrays
  free(Ah);
  free(Bh);
  free(Ch);
  free(Cref);

  return 0;
}
