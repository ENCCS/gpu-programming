#include <CL/opencl.h>
#include <stdio.h>
int main(void) {
  cl_uint count;
  cl_platform_id platform;
  clGetPlatformIDs(1, &platform, NULL);

  cl_device_id device;
  clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, &count);

  char deviceName[1024];
  clGetDeviceInfo(device, CL_DEVICE_NAME, sizeof(deviceName), deviceName, NULL);

  printf("Hello! I'm GPU %s out of %d GPUs in total.\n", deviceName, count);

  return 0;
}
