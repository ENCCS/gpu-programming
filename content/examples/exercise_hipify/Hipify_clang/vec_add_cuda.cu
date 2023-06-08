#include <cuda.h>
#include <cuda_runtime_api.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

// CUDA kernel, callable from host due to `__global__`
__global__ void add(const float* a, const float* b, float* c, const size_t n) {
    // Calculate the array index of this thread
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
        c[id] = a[id] + b[id];
    }
}

int main(int argc, char* argv[]) {
    printf("ENTER MAIN\n");
    // Number of elements to compute over
    const size_t num_elements = 1000000;

    // Allocate memory that can be accessed both on host and device
    float* a;
    float* b;
    float* c;
    // Should ideally catch errors here, but skip for brevity
    cudaMallocManaged(&a, num_elements * sizeof(float));
    cudaMallocManaged(&b, num_elements * sizeof(float));
    cudaMallocManaged(&c, num_elements * sizeof(float));

    // Fill our input arrays, on host, with some data to calculate
    for (int i = 0; i < num_elements; i++) {
        a[i] = sinf(i) * sinf(i);
        b[i] = cosf(i) * cosf(i);
    }

    // Define how many threads to launch on CUDA device
    const int block_size = 1024; // Number of threads in each thread block
    // Number of thread blocks in a grid
    const int grid_size = (int) ceil((float) num_elements / block_size);
	
    // Call CUDA kernel to run on device
    add<<<grid_size, block_size>>>(a, b, c, num_elements);
    // Wait for computation before doing anything with data on host
    cudaDeviceSynchronize();

    // Should print 1.0 at all entries
    printf("c[0]  : %f\n", c[0]);
    printf("c[1]  : %f\n", c[1]);
    printf("c[42] : %f\n", c[42]);
	
    // Free memory
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

    printf("EXIT SUCCESS\n");
    return EXIT_SUCCESS;
}
