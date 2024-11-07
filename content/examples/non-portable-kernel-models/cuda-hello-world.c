#include <cuda.h>
#include <cuda_runtime.h>
#include <stdio.h>

int main(void) {
  int count, device;

  cudaGetDeviceCount(&count);
  cudaGetDevice(&device);

  printf("Hello! I'm GPU %d out of %d GPUs in total.\n", device, count);
  return 0;
}
