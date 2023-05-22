//
// nvcc 02_matrix_thread_index_info.cu
//
#include <cstdio>
#include <cuda_runtime.h>
#include "error_checker.cuh"

void initialInt(int *matrix, int nxy);
void printMatrix(int *h_matrixA, const int nx, const int ny);
void __global__ printGPUIdx(int *d_matrixA, const int nx, const int ny);


int main(int argc, const char * argv[])
{

	printf("\n--Beginning of the main function.\n");

    int dev = 0;
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    printf("\nUsing Device %d: %s\n", dev, deviceProp.name); // Using Device 0: NVIDIA GeForce GT 1030

    int nx = 8;
    int ny = 6;
    int nxy = nx * ny;
	int size_matrix = nxy*(sizeof(int));

    // malloc host mem
    int *h_matrixA;
    h_matrixA = (int *)malloc(size_matrix);
    initialInt(h_matrixA, nxy);
    printMatrix(h_matrixA, nx, ny);

    //malloc device mem
    int *d_matrixA;
    cudaMalloc((void **)&d_matrixA, size_matrix); // 
    cudaMemcpy(d_matrixA, h_matrixA, size_matrix, cudaMemcpyHostToDevice); // copy data from CPU to GPU

    // setup excution configuration
    dim3 block(4, 2);
    dim3 grid((nx + block.x-1)/block.x, (ny + block.y-1)/block.y);
	printf("\ngrid info  >>> grid.x=%d   grid.y=%d   grid.z=%d.\n", grid.x, grid.y, grid.z);
	printf("block info >>> block.x=%d  block.y=%d  block.z=%d.\n\n", block.x, block.y, block.z);

    //invoke kernel
    printGPUIdx<<<grid, block>>>(d_matrixA, nx, ny);
    cudaDeviceSynchronize();
	printf("\n");

    //free host and device
    free(h_matrixA);
    cudaFree(d_matrixA);

    //reset device
    CHECK(cudaDeviceReset());

	printf("\n--Ending of the main function.\n\n");

    return 0;
}


void initialInt(int *matrix, int nxy){
    for(int i = 0; i < nxy; i++)
        matrix[i] = i;
}


void printMatrix(int *h_matrixA, const int nx, const int ny){
    int *ic = h_matrixA;
    printf("\nMatrix:%d, %d\n", nx, ny);
    for(int iy = 0; iy < ny; iy++){
        for(int ix=0; ix < nx; ix++)
            printf("%3d ",ic[ix]);
        ic += nx;
        printf("\n");
    }
}


void __global__ printGPUIdx(int *d_matrixA, const int nx, const int ny){
    int ix = threadIdx.x + blockIdx.x*blockDim.x;
    int iy = threadIdx.y + blockIdx.y*blockDim.y;
    unsigned int idx = iy*nx + ix;
    printf("block_id (%d %d) thread_id (%d,%d) coordinate (%d %d) global_index (%d) value (%d)\n",
		blockIdx.x, blockIdx.y, threadIdx.x, threadIdx.y, ix, iy, idx, d_matrixA[idx]);
}
