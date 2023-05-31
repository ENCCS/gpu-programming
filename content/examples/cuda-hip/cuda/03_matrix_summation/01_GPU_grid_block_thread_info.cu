//
// nvcc 01_GPU_grid_block_thread_info.cu
//
#include <cuda_runtime.h>
#include <stdio.h>
#include "error_checker.cuh"

void __global__ check_grid_block_thread_info_GPU(void);


int main(int argc, const char * argv[])
{

	printf("\n--Beginning of the main function.\n\n");

	printf("\t***************************************************\n");
	printf("\t********** Output for num_element = 1024 **********\n");
	printf("\t***************************************************\n\n");

    int num_elements = 1024;
	printf("\t\tThere are %d data, which can be distributed:\n", num_elements);

    // define grid and block structure
    dim3 block(1024); // == (block.x = 1024; block.y = 1; block.z = 1;)
    dim3 grid((num_elements + block.x - 1) / block.x);
    printf("\t\t- grid.x=%d, block.x=%d\n", grid.x, block.x);

    // reset block
    block.x = 512;
    grid.x  = (num_elements + block.x - 1) / block.x;
    printf("\t\t- grid.x=%d, block.x=%d\n", grid.x, block.x);

    // reset block
    block.x = 256;
    grid.x  = (num_elements + block.x - 1) / block.x;
    printf("\t\t- grid.x=%d, block.x=%d\n", grid.x, block.x);

    // reset block
    block.x = 128;
    grid.x  = (num_elements + block.x - 1) / block.x;
    printf("\t\t- grid.x=%d, block.x=%d\n\n", grid.x, block.x);

    CHECK(cudaDeviceSynchronize());


	printf("\t***************************************************\n");
	printf("\t*********** Output for num_element = 16 ***********\n");
	printf("\t***************************************************\n\n");

    // reset the total number of data element
    num_elements = 16;

    // reset grid and block structure
    block.x = 2;
	grid.x = (num_elements + block.x - 1) / block.x;

    // check grid and block info from host side
    printf("\t\t- CPU output -- grid.x=%d,  grid.y=%d,  grid.z=%d\n",  grid.x,  grid.y,  grid.z);
    printf("\t\t- CPU output -- block.x=%d, block.y=%d, block.z=%d\n", block.x, block.y, block.z);
	putchar('\n');

    check_grid_block_thread_info_GPU<<<grid, block>>>();

    CHECK(cudaDeviceReset());

	printf("\n--Ending of the main function.\n\n");
    return 0;
}


void __global__ check_grid_block_thread_info_GPU(void)
{
    printf("\t\t- GPU output -- gridDim=(%d, %d, %d)   blockDim=(%d, %d, %d)  blockIdx=(%d, %d, %d)  threadIdx=(%d, %d, %d)\n", 
		gridDim.x,   gridDim.y,   gridDim.z,  blockDim.x,  blockDim.y,  blockDim.z,	
		blockIdx.x,  blockIdx.y,  blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z);
}