#include <hip/hip_runtime.h>
#include <stdio.h>

int main(void) {
  int count, device;

  hipGetDeviceCount(&count);
  //hipSetDevice(nd); // change nd to the GPU id 
  hipGetDevice(&device);

  printf("Hello! I'm GPU %d out of %d GPUs in total.\n", device, count);
  return 0;
}
