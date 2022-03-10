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
    	float value_max = inter_res[0];
	    float value_min = inter_res[0];
	    for (int i = 1; i < 4; i++){
	    	if (inter_res[i] >= value_max) value_max = inter_res[i];
	    	if (inter_res[i] <= value_min) value_min = inter_res[i];
	    }
	    Out_lower[index] = value_min + bias[0];
	    Out_upper[index] = value_max + bias[1];
    }
}


/* cpu version of linear operation */
void reference(float* in_lower, float* in_upper, float* out_lower, float* out_upper, int size_X, int size_Y, float* node, float* bias)
{
	float inter_res[4];
	for (int i = 0; i < size_Y; i++){
		for (int j = 0; j < size_X; j++){
			int index = i * size_X + j;
			inter_res[0] = in_lower[index] * node[0];
    		inter_res[1] = in_lower[index] * node[1];
    		inter_res[2] = in_upper[index] * node[0];
    		inter_res[3] = in_upper[index] * node[1];

    		float value_max = inter_res[0];
		    float value_min = inter_res[0];
		    for (int p = 1; p < 4; p++){
		    	if (inter_res[p] >= value_max) value_max = inter_res[p];
		    	if (inter_res[p] <= value_min) value_min = inter_res[p];
		    }
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

	InitInterval(h_A_1, h_A_2, size_A);
	unsigned int mem_size_node = sizeof(float) * 2;
	float* node = (float*)malloc(mem_size_node);
	float* bias = (float*)malloc(mem_size_node);

	InitVec(node);
	InitVec(bias);
	

    unsigned int timer;
    CUT_SAFE_CALL(cutCreateTimer(&timer));
	cutStartTimer(timer);
    reference(h_A_1, h_A_2, h_cpu_1, h_cpu_2, WA, HA, node, bias);
	cutStopTimer(timer);
    printf("Host CPU Processing time: %f (ms)\n", cutGetTimerValue(timer));
	
	float* d_A_1;
	float* d_A_2;
	float* d_B_1;
	float* d_B_2;
	float* d_node;
	float* d_bias;

	cudaMalloc((void**)&d_A_1, mem_size_A);
	cudaMalloc((void**)&d_A_2, mem_size_A);
	cudaMalloc((void**)&d_B_1, mem_size_A);
	cudaMalloc((void**)&d_B_2, mem_size_A);
	cudaMalloc((void**)&d_node, mem_size_node);
	cudaMalloc((void**)&d_bias, mem_size_node);

	cudaMemcpy(d_A_1, h_A_1, mem_size_A, cudaMemcpyHostToDevice);
	cudaMemcpy(d_A_2, h_A_2, mem_size_A, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B_1, h_B_1, mem_size_A, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B_1, h_B_2, mem_size_A, cudaMemcpyHostToDevice);
	cudaMemcpy(d_node, node, mem_size_node, cudaMemcpyHostToDevice);
	cudaMemcpy(d_bias, bias, mem_size_node, cudaMemcpyHostToDevice);


	cudaEventRecord(start_G);
	linear <<< 34, 1024 >>>(d_A_1, d_A_2, d_B_1, d_B_2, d_node, d_bias);

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
	free(h_cpu_1);
	free(h_cpu_2);
	free(node);
	free(bias);

	cudaFree(d_A_1);
	cudaFree(d_A_2);
	cudaFree(d_B_1);
	cudaFree(d_B_2);
	cudaFree(d_node);
	cudaFree(d_bias);

	return EXIT_SUCCESS;
}