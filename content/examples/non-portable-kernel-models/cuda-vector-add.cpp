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
  float *Ad, *Bd, *Cd;
  int i;

  // Allocate the arrays on CPU
  Ah = (float *)malloc(N * sizeof(float));
  Bh = (float *)malloc(N * sizeof(float));
  Ch = (float *)malloc(N * sizeof(float));
  Cref = (float *)malloc(N * sizeof(float));

  // initialise data and calculate reference values on CPU
  for (i = 0; i < N; i++) {
    Ah[i] = sin(i) * 2.3;
    Bh[i] = cos(i) * 1.1;
    Cref[i] = Ah[i] + Bh[i];
  }

  // Allocate the arrays on GPU
  cudaMalloc((void **)&Ad, N * sizeof(float));
  cudaMalloc((void **)&Bd, N * sizeof(float));
  cudaMalloc((void **)&Cd, N * sizeof(float));

  // Transfer the data from CPU to GPU
  cudaMemcpy(Ad, Ah, sizeof(float) * N, cudaMemcpyHostToDevice);
  cudaMemcpy(Bd, Bh, sizeof(float) * N, cudaMemcpyHostToDevice);

  // define grid dimensions + launch the device kernel
  dim3 blocks, threads;
  threads = dim3(256, 1, 1);
  blocks = dim3((N + 256 - 1) / 256, 1, 1);

  // Launch Kernel
  vector_add<<<blocks, threads>>>(Ad, Bd, Cd, N);

  // copy results back to CPU
  cudaMemcpy(Ch, Cd, sizeof(float) * N, cudaMemcpyDeviceToHost);

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
  cudaFree(Ad);
  cudaFree(Bd);
  cudaFree(Cd);

  // Free the CPU arrays
  free(Ah);
  free(Bh);
  free(Ch);
  free(Cref);

  return 0;
}
