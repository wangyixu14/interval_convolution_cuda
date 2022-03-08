#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cstdlib>
#include <time.h>
#include <cutil.h>
#include <iostream>
#include <assert.h>

// the input matrix size
#define WA 512 
#define HA 512    

#define POOL_SIZE 3
#define TILE_SIZE 14
#define BLOCK_SIZE (TILE_SIZE + POOL_SIZE - 1)

/*cuda version of max pooling */
__global__ void Max_Pooling(float* In_lower, float* In_upper, float* Out_lower, float* Out_upper)
{
    __shared__ float tile_lower[BLOCK_SIZE][BLOCK_SIZE];
    __shared__ float tile_upper[BLOCK_SIZE][BLOCK_SIZE];

    // get thread indices
    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // get the output indices
    int row_o = ty + blockIdx.y * TILE_SIZE;
    int col_o = tx + blockIdx.x * TILE_SIZE;

    // shift to obtain input indices
    int row_i = row_o - POOL_SIZE / 2;
    int col_i = col_o - POOL_SIZE / 2;

    // Load tile elements
    if(row_i >= 0 && row_i < HA && col_i >= 0 && col_i < WA){
        tile_lower[ty][tx] = In_lower[row_i * WA + col_i];
        tile_upper[ty][tx] = In_upper[row_i * WA + col_i];
    }
    else{
        tile_lower[ty][tx] = -(float)RAND_MAX;
        tile_upper[ty][tx] = -(float)RAND_MAX;
    }

    __syncthreads();

    if(tx < TILE_SIZE && ty < TILE_SIZE){
        float Value_lower = -(float)RAND_MAX;
        float Value_upper = -(float)RAND_MAX;
        
        for(int y = 0; y < POOL_SIZE; y++)
            for(int x = 0; x < POOL_SIZE; x++){
            	if (tile_upper[y + ty][x + tx] >= Value_upper){
            		Value_upper = tile_upper[y + ty][x + tx];
            		Value_lower = tile_lower[y + ty][x + tx];
            	}
            }
        if(row_o < HA && col_o < WA){
            Out_lower[row_o * WA + col_o] = Value_lower;
            Out_upper[row_o * WA + col_o] = Value_upper;
        }
    }
}


/* cpu version of max pooling */
void reference(float* in_lower, float* in_upper, float* out_lower, float* out_upper, int size_X, int size_Y, int pool_size)
{
    int p_cent = (pool_size - 1) / 2;
	for(int y = 0; y < size_Y; y++){ 
		for(int x = 0; x < size_X; x++){
			float value_upper = -(float)RAND_MAX;
			float value_lower = -(float)RAND_MAX;

			for(int j = -p_cent; j <= p_cent; j++){ 
				for(int i = -p_cent; i <= p_cent; i++){ 
					if (x + i > -1 && x + i < size_X && y + j > -1 && y + j < size_Y){
						if (in_upper[(x + i) + (y + j) * size_X] >= value_upper){
							value_upper = in_upper[(x + i) + (y + j) * size_X];
							value_lower = in_lower[(x + i) + (y + j) * size_X];
						}
					}
				}
			}

			out_lower[x + y * size_X] = value_lower;
			out_upper[x + y * size_X] = value_upper;
		}  		
	}
}

void InitInterval(float* data_lower, float* data_upper, int size)
{
	for (int i = 0; i < size; ++i){
		int v_1 = rand() / (float)RAND_MAX;
		int v_2 = rand() / (float)RAND_MAX;
		if (v_1 >= v_2){
			data_lower[i] = v_2;
			data_upper[i] = v_1;
		}
		else{
			data_lower[i] = v_1;
			data_upper[i] = v_2;
		}
	}
}

int main(int argc, char** argv)
{
	srand(2006);

	cudaEvent_t start_G, stop_G;

	cudaEventCreate(&start_G);
	cudaEventCreate(&stop_G);

	unsigned int size_A = WA * HA;
	unsigned int mem_size_A = sizeof(float) * size_A;
	float* h_A_1 = (float*)malloc(mem_size_A);
	float* h_A_2 = (float*)malloc(mem_size_A);
	float* h_B_1 = (float*)malloc(mem_size_A);
	float* h_B_2 = (float*)malloc(mem_size_A);
	float* h_cpu_1 = (float*)malloc(mem_size_A);
	float* h_cpu_2 = (float*)malloc(mem_size_A);

	InitInterval(h_A_1, h_A_2, size_A);

    unsigned int timer;
    CUT_SAFE_CALL(cutCreateTimer(&timer));
	cutStartTimer(timer);
    reference(h_A_1, h_A_2, h_cpu_1, h_cpu_2, WA, HA, POOL_SIZE);
	cutStopTimer(timer);
    printf("Host CPU Processing time: %f (ms)\n", cutGetTimerValue(timer));
	
	float* d_A_1;
	float* d_A_2;
	float* d_B_1;
	float* d_B_2;

	cudaMalloc((void**)&d_A_1, mem_size_A);
	cudaMalloc((void**)&d_A_2, mem_size_A);
	cudaMalloc((void**)&d_B_1, mem_size_A);
	cudaMalloc((void**)&d_B_2, mem_size_A);

	cudaMemcpy(d_A_1, h_A_1, mem_size_A, cudaMemcpyHostToDevice);
	cudaMemcpy(d_A_2, h_A_2, mem_size_A, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B_1, h_B_1, mem_size_A, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B_1, h_B_2, mem_size_A, cudaMemcpyHostToDevice);

	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid((WA - 1) / (BLOCK_SIZE - POOL_SIZE + 1) + 1, (WA - 1) / (BLOCK_SIZE - POOL_SIZE + 1) + 1);

	cudaEventRecord(start_G);
	Max_Pooling <<< grid, threads >>>(d_A_1, d_A_2, d_B_1, d_B_2);

	cudaDeviceSynchronize();
	cudaEventRecord(stop_G);
	cudaEventSynchronize(stop_G);

	cudaMemcpy(h_B_1, d_B_1, mem_size_A, cudaMemcpyDeviceToHost);
	cudaMemcpy(h_B_2, d_B_2, mem_size_A, cudaMemcpyDeviceToHost);

	float miliseconds = 0;
	cudaEventElapsedTime(&miliseconds, start_G, stop_G);

	printf("Time took to compute matrix A of dimensions %d x %d  on GPU is %f ms \n \n \n", WA, HA, miliseconds);

	unsigned int result_regtest_1 = cutComparefe(h_cpu_1, h_B_1, WA*HA, 0.001f);
	unsigned int result_regtest_2 = cutComparefe(h_cpu_2, h_B_2, WA*HA, 0.001f);
    printf( "Test %s\n", (result_regtest_1 == 1 && result_regtest_2 == 1) ? "PASSED" : "FAILED");

	free(h_A_1);
	free(h_A_2);
	free(h_B_1);
	free(h_B_2);

	cudaFree(d_A_1);
	cudaFree(d_A_2);
	cudaFree(d_B_1);
	cudaFree(d_B_2);

	return EXIT_SUCCESS;
}