//
// nvcc 02_array_addition_gpu.cu
//
#include <math.h>
#include <stdio.h>

const double EPSILON = 1.0e-15;
const double a = 1.23;
const double b = 2.34;
const double c = 3.57;

void __global__ array_addition(const double *vecA, const double *vecB, double *vecC);
void array_check(const double *vecC, const int NX);


int main(int argc, const char * argv[])
{

	printf("\n--Beginning of the main function.\n");

    const int NX = 25600004;
	int size_array = sizeof(double) * NX;

    double *h_vecA = (double *)malloc(size_array); // 'h' for host (CPU)
    double *h_vecB = (double *)malloc(size_array);
    double *h_vecC = (double *)malloc(size_array);

    for (int i = 0; i < NX; i++)
    {
        h_vecA[i] = a;
        h_vecB[i] = b;
    }

    double *d_vecA, *d_vecB, *d_vecC; // 'd' for device (GPU) 设备中双精度类型变量指针，不看后面代码不知道这3个指针指向哪些内存区域
    cudaMalloc((void **)&d_vecA, size_array); // 分配内存（显存），确定它们将指向GPU内存，而不是CPU内存
    cudaMalloc((void **)&d_vecB, size_array); // 该函数是个CUDA运行时API函数。所有CUDA运行时API函数都以cuda开头
    cudaMalloc((void **)&d_vecC, size_array); // 完整的API函数列表 https://docs.nvidia.com/cuda/cuda-runtime-api
    cudaMemcpy(d_vecA, h_vecA, size_array, cudaMemcpyHostToDevice); // 将主机h_vecA中数据复制到设备相应变量d_vecA所指向的缓冲区
    cudaMemcpy(d_vecB, h_vecB, size_array, cudaMemcpyHostToDevice);

    const int block_size = 128;
 // int grid_size = NX / block_size; // 网格大小
    int grid_size = (NX + block_size - 1) / block_size; // 修改之后用N=25600001没问题
		// 如使用CUDA8.0，而nvcc编译时忘记指定计算能力，程序会根据默认2.0计算能力编译程序。
		// 对于2.0计算能力，网格大小在vecA方向上限为65535，小于本例中所使用值，将导致程序无法正确执行。

	printf("\n\tArray_size = %d, grid_size = %d and block_size = %d.\n", NX, grid_size, block_size); // 输出位7812，实际为7812.5
    array_addition<<<grid_size, block_size>>>(d_vecA, d_vecB, d_vecC); // 调用核函数在设备中进行计算

    cudaMemcpy(h_vecC, d_vecC, size_array, cudaMemcpyDeviceToHost); // copy data from device (GPU) to host (CPU)
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


void __global__ array_addition(const double *vecA, const double *vecB, double *vecC)
{
    const int i = blockDim.x * blockIdx.x + threadIdx.x;
    vecC[i] = vecA[i] + vecB[i];
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




















/* 编译用nvcc 02_arravecB_addition_gpu.cu可以得到正确结果
 *
 * 如果用nvcc -arch=sm_75 02_arravecB_addition_gpu.cu，结果不对
 * 用nvcc -gencode arch=compute_75,code=compute_75 02_arravecB_addition_gpu.cu，结果也不对
 *
 * 如果用nvcc -arch=sm_60 02_arravecB_addition_gpu.cu，结果对，是GPU版本问题
 * 用nvcc -gencode arch=compute_60,code=compute_60 02_arravecB_addition_gpu.cu，结果也对
 *
 * 可以肯定是GPU结构问题，架构比较古老，不能用新的算力来描述
 */