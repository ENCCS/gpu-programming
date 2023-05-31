#include "hip/hip_runtime.h"
// 
// nvcc 03_matrix_summation_GPU_2D2D_2D1D_1D1D.cu
// 
#include <hip/hip_runtime.h>
#include <stdio.h>
#include "error_checker.h"

const double EPSILON = 1.0E-8;

void matrix_initialization(float *ip, const int size);
void matrix_summation_on_CPU(float *A, float *B, float *C, const int, const int);
void check_results_from_CPU_GPU(float *fromCPU, float *fromGPU, const int);
void __global__ matrix_summation_on_GPU_1D1D(float *A, float *B, float *C, int, int);
void __global__ matrix_summation_on_GPU_2D1D(float *A, float *B, float *C, int, int);
void __global__ matrix_summation_on_GPU_2D2D(float *A, float *B, float *C, int, int);


int main(int argc, const char * argv[])
{

	printf("\n--Beginning of the main function.\n");

    // set up device
    int dev = 0;
    hipDeviceProp_t deviceProp;
    CHECK(hipGetDeviceProperties(&deviceProp, dev));
    printf("\n\tUsing Device %d: %s\n", dev, deviceProp.name);
    CHECK(hipSetDevice(dev));

    // set up data size of matrix
    int nx = 1 << 10;
    int ny = 1 << 10;

    int nxy = nx * ny;
    int size_matrix = nxy * sizeof(float);
    printf("\n\tMatrix size: nx=%d ny=%d\n", nx, ny);

    // malloc host memory
    float *h_matrixA, *h_matrixB, *h_matrixSumFromCPU, *h_matrixSumFromGPU;
    h_matrixA = (float *)malloc(size_matrix);
    h_matrixB = (float *)malloc(size_matrix);
    h_matrixSumFromCPU = (float *)malloc(size_matrix);
    h_matrixSumFromGPU = (float *)malloc(size_matrix);

    // initialize data at host side and define a timer
    hipEvent_t start, stop;
    hipEventCreate(&start);
    hipEventCreate(&stop);
    hipEventRecord(start);
    hipEventQuery(start);
    matrix_initialization(h_matrixA, nxy);
    matrix_initialization(h_matrixB, nxy);
    hipEventRecord(stop);
    hipEventSynchronize(stop);
    float elapsed_time;
    hipEventElapsedTime(&elapsed_time, start, stop); //CHECK();
    printf("\tMatrix initialization on host(CPU) elapsed %f sec\n", elapsed_time);

    memset(h_matrixSumFromCPU, 0, size_matrix);
    memset(h_matrixSumFromGPU, 0, size_matrix);

    // summation of matrix elements at host(CPU) side
    hipEventCreate(&start);
    hipEventCreate(&stop);
    hipEventRecord(start);
    hipEventQuery(start);
    matrix_summation_on_CPU(h_matrixA, h_matrixB, h_matrixSumFromCPU, nx, ny);
    hipEventRecord(stop);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&elapsed_time, start, stop);
    printf("\tMatrix summation on host(CPU) elapsed %f sec\n", elapsed_time);


    // malloc device global memory
    float *d_matrixA, *d_matrixB, *d_matrixC;
    CHECK(hipMalloc((void **)&d_matrixA, size_matrix));
    CHECK(hipMalloc((void **)&d_matrixB, size_matrix));
    CHECK(hipMalloc((void **)&d_matrixC, size_matrix));

    // transfer data from host to device
    CHECK(hipMemcpy(d_matrixA, h_matrixA, size_matrix, hipMemcpyHostToDevice));
    CHECK(hipMemcpy(d_matrixB, h_matrixB, size_matrix, hipMemcpyHostToDevice));

//---------------
    // invoke kernel at host side for summation on GPU using 2D_grid and 2D_block
    int dimx = 32;
    int dimy = 32;
    dim3 block(dimx, dimy); // (32, 32, 1)
    dim3 grid((nx + block.x - 1) / block.x, (ny + block.y - 1) / block.y); //(32, 32, 1)

    hipEventCreate(&start);
    hipEventCreate(&stop);
    hipEventRecord(start);
    hipEventQuery(start);
    hipLaunchKernelGGL(matrix_summation_on_GPU_2D2D, grid, block, 0, 0, d_matrixA, d_matrixB, d_matrixC, nx, ny);
    hipDeviceSynchronize();
    hipEventRecord(stop);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&elapsed_time, start, stop);
    printf("\n\tMatrix summation on GPU (2D_grid 2D_block) <<<(%d,%d), (%d,%d) >>> elapsed %f sec\n", 
           grid.x, grid.y, block.x, block.y, elapsed_time);
    hipGetLastError(); // check kernel error

    // copy kernel result back to host side
    hipMemcpy(h_matrixSumFromGPU, d_matrixC, size_matrix, hipMemcpyDeviceToHost);

    // comparison of computation results
    check_results_from_CPU_GPU(h_matrixSumFromCPU, h_matrixSumFromGPU, nxy);
//---------------

    // invoke kernel at host side for summation on GPU using 2D_grid and 1D_block
    dimy = 1;
	block.y = dimy; // block (32, 1, 1)
	grid.x = (nx + block.x - 1) / block.x;
	grid.y = ny; // grid (32, 1024, 1)

    hipEventCreate(&start);
    hipEventCreate(&stop);
    hipEventRecord(start);
    hipEventQuery(start);
    hipLaunchKernelGGL(matrix_summation_on_GPU_2D1D, grid, block, 0, 0, d_matrixA, d_matrixB, d_matrixC, nx, ny);
    hipDeviceSynchronize();
    hipEventRecord(stop);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&elapsed_time, start, stop);
    printf("\n\tMatrix summation on GPU (2D_grid 1D_block) <<<(%d,%d), (%d,%d) >>> elapsed %f sec\n", 
           grid.x, grid.y, block.x, block.y, elapsed_time);
    hipGetLastError(); // check kernel error

    // copy kernel result back to host side
    hipMemcpy(h_matrixSumFromGPU, d_matrixC, size_matrix, hipMemcpyDeviceToHost);

    // comparison of computation results
    check_results_from_CPU_GPU(h_matrixSumFromCPU, h_matrixSumFromGPU, nxy);
//---------------

    // invoke kernel at host side for summation on GPU using 1D_grid and 1D_block
    dimy = 1;
	block.y = dimy; // block (32, 1, 1)
	grid.x = (nx + block.x - 1) / block.x;
	grid.y = 1; // grid (32, 1, 1)

    hipEventCreate(&start);
    hipEventCreate(&stop);
    hipEventRecord(start);
    hipEventQuery(start);
    hipLaunchKernelGGL(matrix_summation_on_GPU_1D1D, grid, block, 0, 0, d_matrixA, d_matrixB, d_matrixC, nx, ny);
    hipDeviceSynchronize();
    hipEventRecord(stop);
    hipEventSynchronize(stop);
    hipEventElapsedTime(&elapsed_time, start, stop);
    printf("\n\tMatrix summation on GPU (1D_grid 1D_block) <<<(%d,%d), (%d,%d) >>> elapsed %f sec\n", 
           grid.x, grid.y, block.x, block.y, elapsed_time);
    hipGetLastError(); // check kernel error

    // copy kernel result back to host side
    hipMemcpy(h_matrixSumFromGPU, d_matrixC, size_matrix, hipMemcpyDeviceToHost);

    // comparison of computation results
    check_results_from_CPU_GPU(h_matrixSumFromCPU, h_matrixSumFromGPU, nxy);
//---------------

	// destroy start and stop events
    CHECK(hipEventDestroy(start));
    CHECK(hipEventDestroy(stop));	

    // free host memory and device global memory
    free(h_matrixA);
    free(h_matrixB);
    free(h_matrixSumFromCPU);
    free(h_matrixSumFromGPU);
    hipFree(d_matrixA);
    hipFree(d_matrixB);
    hipFree(d_matrixC);

    CHECK(hipDeviceReset());

	printf("\n--Ending of the main function.\n\n");
    return 0;
}


void matrix_initialization(float *ip, const int size)
{
    for(int i = 0; i < size; i++)
        ip[i] = (float)(rand() & 0xFF) / 10.0f;
}


void matrix_summation_on_CPU(float *matrixA, float *matrixB, float *matrixC,
	const int nx, const int ny)
{
    float *ia = matrixA;
    float *ib = matrixB;
    float *ic = matrixC;

    for (int iy = 0; iy < ny; iy++)
    {
        for (int ix = 0; ix < nx; ix++)
            ic[ix] = ia[ix] + ib[ix];
        ia += nx;
        ib += nx;
        ic += nx;
    }
}


void __global__ matrix_summation_on_GPU_2D2D(float *matrixA, float *matrixB,
	float *matrixC, int nx, int ny)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = threadIdx.y + blockIdx.y * blockDim.y;
    unsigned int idx = iy * nx + ix;
    if (ix < nx && iy < ny)
        matrixC[idx] = matrixA[idx] + matrixB[idx];
}
void __global__ matrix_summation_on_GPU_2D1D(float *matrixA, float *matrixB,
	float *matrixC, int nx, int ny)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    unsigned int iy = blockIdx.y;
    unsigned int idx = iy * nx + ix;
    if (ix < nx && iy < ny)
        matrixC[idx] = matrixA[idx] + matrixB[idx];
}
void __global__ matrix_summation_on_GPU_1D1D(float *matrixA, float *matrixB,
	float *matrixC, int nx, int ny)
{
    unsigned int ix = threadIdx.x + blockIdx.x * blockDim.x;
    if (ix < nx)
        for (int iy = 0; iy < ny; iy++)
        {
            int idx = iy * nx + ix;
            matrixC[idx] = matrixA[idx] + matrixB[idx];
        }      
}


void check_results_from_CPU_GPU(float *h_matrixSumFromCPU,
	float *h_matrixSumFromGPU, const int N)
{
    bool has_error = false;
    for (int i = 0; i < N; i++)
    {
        if (abs(h_matrixSumFromCPU[i] - h_matrixSumFromGPU[i]) > EPSILON)
        {
            has_error = true;
            printf("host %f gpu %f\n", h_matrixSumFromCPU[i], h_matrixSumFromGPU[i]);
            break;
        }
    }
    printf("\tChecking matrix summation results >>> %s\n", has_error? "|| ERROR ||":"|| NO ERROR ||");
}
