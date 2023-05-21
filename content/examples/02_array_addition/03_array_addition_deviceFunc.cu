//
// nvcc 03_array_addition_deviceFunc.cu
//
#include <math.h>
#include <stdio.h>

const double EPSILON = 1.0e-15;
const double a = 1.23;
const double b = 2.34;
const double c = 3.57;

void __global__ array_addition1(const double *vecA, const double *vecB, double *vecC, const int NX);
void __global__ array_addition2(const double *vecA, const double *vecB, double *vecC, const int NX);
void __global__ array_addition3(const double *vecA, const double *vecB, double *vecC, const int NX);
void array_check(const double *vecC, int NX);


int main(int argc, const char * argv[])
{

	printf("\n--Beginning of the main function.\n");

    const int NX = 25600004;
	int size_array = sizeof(double) * NX;

    double *h_vecA = (double *)malloc(size_array);
    double *h_vecB = (double *)malloc(size_array);
    double *h_vecC = (double *)malloc(size_array);

    for (int i = 0; i < NX; i++)
    {
        h_vecA[i] = a;
        h_vecB[i] = b;
    }

    double *d_vecA, *d_vecB, *d_vecC;
    cudaMalloc((void **)&d_vecA, size_array);
    cudaMalloc((void **)&d_vecB, size_array);
    cudaMalloc((void **)&d_vecC, size_array);
    cudaMemcpy(d_vecA, h_vecA, size_array, cudaMemcpyHostToDevice);
    cudaMemcpy(d_vecB, h_vecB, size_array, cudaMemcpyHostToDevice);

    const int block_size = 128;
    int grid_size = (NX + block_size - 1) / block_size;

	//	defining three kernel functions for array addition in GPU
    array_addition1<<<grid_size, block_size>>>(d_vecA, d_vecB, d_vecC, NX);
    cudaMemcpy(h_vecC, d_vecC, size_array, cudaMemcpyDeviceToHost);
    array_check(h_vecC, NX);

    array_addition2<<<grid_size, block_size>>>(d_vecA, d_vecB, d_vecC, NX);
    cudaMemcpy(h_vecC, d_vecC, size_array, cudaMemcpyDeviceToHost);
    array_check(h_vecC, NX);

    array_addition3<<<grid_size, block_size>>>(d_vecA, d_vecB, d_vecC, NX);
    cudaMemcpy(h_vecC, d_vecC, size_array, cudaMemcpyDeviceToHost);
    array_check(h_vecC, NX);

    free(h_vecA);
    free(h_vecB);
    free(h_vecC);
    cudaFree(d_vecA);
    cudaFree(d_vecB);
    cudaFree(d_vecC);

	printf("\n--Ending of the main function.\n\n");

    return 0;
}


double __device__ array_addition1_device(const double aa, const double bb)
{
    return (aa + bb);
}
void __global__ array_addition1(const double *vecA, const double *vecB, double *vecC, const int NX)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < NX)
        vecC[i] = array_addition1_device(vecA[i], vecB[i]); // vecC[i] = vecA[i] + vecB[i];
}


void __device__ array_addition2_device(const double vecA, const double vecB, double *vecC)
{
    *vecC = vecA + vecB;
}
void __global__ array_addition2(const double *vecA, const double *vecB, double *vecC, const int NX)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < NX)
        array_addition2_device(vecA[i], vecB[i], &vecC[i]); // vecC[i] = vecA[i] + vecB[i];
}


void __device__ array_addition3_device(const double vecA, const double vecB, double &vecC)
{
    vecC = vecA + vecB;
}
void __global__ array_addition3(const double *vecA, const double *vecB, double *vecC, const int NX)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i < NX)
        array_addition3_device(vecA[i], vecB[i], vecC[i]); // vecC[i] = vecA[i] + vecB[i];
}


void array_check(const double *vecC, const int NX)
{
    bool has_error = false;
    for (int i = 0; i < NX; i++)
    {
        if (fabs(vecC[i] - c) > EPSILON)
		{
            has_error = true;
			break;
		}
    }
    printf("\n\tChecking array addition results >>> %s\n", has_error? "|| ERROR ||":"|| NO ERROR ||");
}

