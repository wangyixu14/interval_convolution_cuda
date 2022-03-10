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

#define KERNEL_SIZE 3
#define POOL_SIZE 3
#define TILE_SIZE 14
#define BLOCK_SIZE (TILE_SIZE + KERNEL_SIZE - 1)

float min(float* inter_res)
{
	float value_min = inter_res[0];
	for (int i = 1; i < 4; i++){
		if (inter_res[i] <= value_min) value_min = inter_res[i];
	}
	return value_min;
}

float max(float* inter_res)
{
	float value_max = inter_res[0];
	for (int i = 1; i < 4; i++){
		if (inter_res[i] >= value_max) value_max = inter_res[i];
	}
	return value_max;
}

/*cuda version of convolution */
__global__ void Convolution(float* In_lower, float* In_upper, float* Out_lower, float* Out_upper, float* Ker_lower, float* Ker_upper)
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
    int row_i = row_o - KERNEL_SIZE / 2;
    int col_i = col_o - KERNEL_SIZE / 2;

    // Load tile elements
    if(row_i >= 0 && row_i < HA && col_i >= 0 && col_i < WA){
        tile_lower[ty][tx] = In_lower[row_i * WA + col_i];
        tile_upper[ty][tx] = In_upper[row_i * WA + col_i];
    }
    else{
        tile_lower[ty][tx] = 0.0f;
        tile_upper[ty][tx] = 0.0f;
    }

    __syncthreads();

    if(tx < TILE_SIZE && ty < TILE_SIZE){
        float Value_lower = 0.0f;
        float Value_upper = 0.0f;
        float inter_res[4];
        
        for(int y = 0; y < KERNEL_SIZE; y++)
            for(int x = 0; x < KERNEL_SIZE; x++){
            	inter_res[0] = Ker_lower[y * KERNEL_SIZE + x] * tile_lower[y + ty][x + tx];
            	inter_res[1] = Ker_lower[y * KERNEL_SIZE + x] * tile_upper[y + ty][x + tx];
            	inter_res[2] = Ker_upper[y * KERNEL_SIZE + x] * tile_lower[y + ty][x + tx];
            	inter_res[3] = Ker_upper[y * KERNEL_SIZE + x] * tile_upper[y + ty][x + tx];

            	float value_min = inter_res[0];
            	float value_max = inter_res[0];
				for (int i = 1; i < 4; i++){
					if (inter_res[i] <= value_min) value_min = inter_res[i];
					if (inter_res[i] >= value_min) value_min = inter_res[i];
				}

            	Value_lower += value_min;
            	Value_upper += value_max;
            }
        if(row_o < HA && col_o < WA){
            Out_lower[row_o * WA + col_o] = Value_lower;
            Out_upper[row_o * WA + col_o] = Value_upper;
        }
    }
}


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

/*cuda version of linear operation */
__global__ void linear(float* In_lower, float* In_upper, float* Out_lower, float* Out_upper, float* node, float* bias)
{
	float inter_res[4];
    // get thread indices
    int tx = threadIdx.x;

    // get the data indices
    int index = tx + blockIdx.x * blockDim.x;
    

    inter_res[0] = In_lower[index] * node[0];
    inter_res[1] = In_lower[index] * node[1];
    inter_res[2] = In_upper[index] * node[0];
    inter_res[3] = In_upper[index] * node[1];
    __syncthreads();
    
    if (index < WA * HA){
	    float value_min = inter_res[0];
    	float value_max = inter_res[0];
		for (int i = 1; i < 4; i++){
			if (inter_res[i] <= value_min) value_min = inter_res[i];
			if (inter_res[i] >= value_min) value_min = inter_res[i];
		}
	    Out_lower[index] = value_min + bias[0];
	    Out_upper[index] = value_max + bias[1];
    }
}


/* cpu version of convolution */
void reference_convolution(float* in_lower, float* in_upper, float* out_lower, float* out_upper, int size_X, int size_Y, float* ker_lower, float* ker_upper, int kernel_size)
{
    int ker_cent = (kernel_size - 1) / 2;
    float inter_res[4];
    float value_min = 0.0f;
    float value_max = 0.0f;
	for(int y = 0; y < size_Y; y++){ 
		for(int x = 0; x < size_X; x++){
			for(int j = -ker_cent; j <= ker_cent; j++){ 
				for(int i = -ker_cent; i <= ker_cent; i++){ 
					if(x + i > -1 && x + i < size_X && y + j > -1 && y + j < size_Y){
						inter_res[0] = ker_lower[(ker_cent + i) + (ker_cent + j) * kernel_size] * in_lower[(x + i) + (y + j) * size_X];
						inter_res[1] = ker_lower[(ker_cent + i) + (ker_cent + j) * kernel_size] * in_upper[(x + i) + (y + j) * size_X];
						inter_res[2] = ker_upper[(ker_cent + i) + (ker_cent + j) * kernel_size] * in_lower[(x + i) + (y + j) * size_X];
						inter_res[3] = ker_upper[(ker_cent + i) + (ker_cent + j) * kernel_size] * in_upper[(x + i) + (y + j) * size_X];

						value_min = min(inter_res);
						value_max = max(inter_res);

						out_lower[x + y * size_X] += value_min;
						out_upper[x + y * size_X] += value_max;
					}
				}
			}
		}
	}    		
}

/* cpu version of max pooling */
void reference_max_pooling(float* in_lower, float* in_upper, float* out_lower, float* out_upper, int size_X, int size_Y, int pool_size)
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

/* cpu version of linear operation */
void reference_linear(float* in_lower, float* in_upper, float* out_lower, float* out_upper, int size_X, int size_Y, float* node, float* bias)
{
	float inter_res[4];
	float value_min = 0.0f;
    float value_max = 0.0f;
	for (int i = 0; i < size_Y; i++){
		for (int j = 0; j < size_X; j++){
			int index = i * size_X + j;
			inter_res[0] = in_lower[index] * node[0];
    		inter_res[1] = in_lower[index] * node[1];
    		inter_res[2] = in_upper[index] * node[0];
    		inter_res[3] = in_upper[index] * node[1];

    		value_max = max(inter_res);
		    value_min = min(inter_res);
		    out_lower[index] = value_min + bias[0];
		    out_upper[index] = value_max + bias[1];
		}
	}
}



void InitVec(float* Vec)
{
	int v_1 = rand() / (float)RAND_MAX;
	int v_2 = rand() / (float)RAND_MAX;
	if (v_1 >= v_2){
		Vec[0] = v_2;
		Vec[1] = v_1;
	}
	else{
		Vec[0] = v_1;
		Vec[1] = v_2;
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

	unsigned int size_C = KERNEL_SIZE * KERNEL_SIZE;
	unsigned int mem_size_C = sizeof(float) * size_C;
	float* h_C_1 = (float*)malloc(mem_size_C);
	float* h_C_2 = (float*)malloc(mem_size_C);

	InitInterval(h_A_1, h_A_2, size_A);
	InitInterval(h_C_1, h_C_2, size_C);
	unsigned int mem_size_node = sizeof(float) * 2;
	float* node = (float*)malloc(mem_size_node);
	float* bias = (float*)malloc(mem_size_node);

	InitVec(node);
	InitVec(bias);

    unsigned int timer;
    CUT_SAFE_CALL(cutCreateTimer(&timer));
	cutStartTimer(timer);
    reference_convolution(h_A_1, h_A_2, h_cpu_1, h_cpu_2, WA, HA, h_C_1, h_C_2, KERNEL_SIZE);
    // reference_max_pooling(h_cpu_1, h_cpu_2, h_A_1, h_A_2, WA, HA, POOL_SIZE);
    // reference_linear(h_A_1, h_A_2, h_cpu_1, h_cpu_2, WA, HA, node, bias);

	cutStopTimer(timer);
    printf("Host CPU Processing time: %f (ms)\n", cutGetTimerValue(timer));
	
	float* d_A_1;
	float* d_A_2;
	float* d_B_1;
	float* d_B_2;
	float* d_C_1;
	float* d_C_2;
	float* d_node;
	float* d_bias;

	cudaMalloc((void**)&d_A_1, mem_size_A);
	cudaMalloc((void**)&d_A_2, mem_size_A);
	cudaMalloc((void**)&d_B_1, mem_size_A);
	cudaMalloc((void**)&d_B_2, mem_size_A);
	cudaMalloc((void**)&d_C_1, mem_size_C);
	cudaMalloc((void**)&d_C_2, mem_size_C);
	cudaMalloc((void**)&d_node, mem_size_node);
	cudaMalloc((void**)&d_bias, mem_size_node);

	cudaMemcpy(d_A_1, h_A_1, mem_size_A, cudaMemcpyHostToDevice);
	cudaMemcpy(d_A_2, h_A_2, mem_size_A, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B_1, h_B_1, mem_size_A, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B_1, h_B_2, mem_size_A, cudaMemcpyHostToDevice);
	cudaMemcpy(d_C_1, h_C_1, mem_size_C, cudaMemcpyHostToDevice);
	cudaMemcpy(d_C_2, h_C_2, mem_size_C, cudaMemcpyHostToDevice);
	cudaMemcpy(d_node, node, mem_size_node, cudaMemcpyHostToDevice);
	cudaMemcpy(d_bias, bias, mem_size_node, cudaMemcpyHostToDevice);

	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid((WA - 1) / (BLOCK_SIZE - KERNEL_SIZE + 1) + 1, (WA - 1) / (BLOCK_SIZE - KERNEL_SIZE + 1) + 1);

	cudaEventRecord(start_G);

	Convolution <<< grid, threads >>>(d_A_1, d_A_2, d_B_1, d_B_2, d_C_1, d_C_2);
	// cudaDeviceSynchronize();
	// Max_Pooling <<< grid, threads >>>(d_B_1, d_B_2, d_A_1, d_A_2);
	// cudaDeviceSynchronize();
	// linear <<< 34, 1024 >>>(d_A_1, d_A_2, d_B_1, d_B_2, d_node, d_bias);
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
	free(h_C_1);
	free(h_C_2);
	free(h_cpu_1);
	free(h_cpu_2);
	free(node);
	free(bias);

	cudaFree(d_A_1);
	cudaFree(d_A_2);
	cudaFree(d_B_1);
	cudaFree(d_B_2);
	cudaFree(d_C_1);
	cudaFree(d_C_2);
	cudaFree(d_node);
	cudaFree(d_bias);

	return EXIT_SUCCESS;
}