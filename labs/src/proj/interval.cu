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
#define TILE_SIZE 12
#define BLOCK_SIZE (TILE_SIZE + KERNEL_SIZE - 1)

/*cuda version of convolution */
__global__ void Convolution(float* N, float* P, float* Mc)
{
    __shared__ float tileNs[BLOCK_SIZE][BLOCK_SIZE];
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
    if(row_i >= 0 && row_i < HA && col_i >= 0 && col_i < WA)
        tileNs[ty][tx] = N[row_i*WA + col_i];
    else
        tileNs[ty][tx] = 0.0f;

    __syncthreads();

    if(tx < TILE_SIZE && ty < TILE_SIZE){
        float pValue = 0.0f;
        for(int y=0; y<KERNEL_SIZE; y++)
            for(int x=0; x<KERNEL_SIZE; x++)
                pValue += Mc[y*KERNEL_SIZE + x] * tileNs[y+ty][x+tx];
        
        if(row_o < HA && col_o < WA)
            P[row_o*WA + col_o] = pValue;
    }
}


/* cpu version of convolution */
void reference(float* in, float* out, int data_size_X, int data_size_Y, float* kernel, int kernel_size)
{
    int kern_cent = (kernel_size - 1)/2;
	for(int x = 0; x < data_size_X; x++){ 
		for(int y = 0; y < data_size_Y; y++){
			for(int i = -kern_cent; i <= kern_cent; i++){ 
				for(int j = -kern_cent; j <= kern_cent ; j++){ 
					if(x+i>-1 && x+i<data_size_X && y+j>-1 && y+j<data_size_Y){
						out[x+y*data_size_X] += 
								kernel[(kern_cent+i)+(kern_cent+j)*kernel_size] * in[(x+i) + (y+j)*data_size_X];
					}
				}
			}
		}
	}    		
}

void randomInit(float* data, int size)
{
	for (int i = 0; i < size; ++i)
		data[i] = rand() / (float)RAND_MAX;
}

int compare(float* h_cpu, float* h_B, unsigned int size){
	int res = 1;
	for(int i = 0; i < size; i++){
		if(h_cpu[i] != h_B[i]) {
			res = 0;
			printf("Location: %d, %f, %f\n", i, h_cpu[i], h_B[i] );
			// break;
		}
	}
	return res;
}

int main(int argc, char** argv)
{
	srand(2006);

	cudaEvent_t start_G, stop_G;

	cudaEventCreate(&start_G);
	cudaEventCreate(&stop_G);

	unsigned int size_A = WA * HA;
	unsigned int mem_size_A = sizeof(float) * size_A;
	float* h_A = (float*)malloc(mem_size_A);

	float* h_B = (float*)malloc(mem_size_A);

	float* h_cpu = (float*) malloc(mem_size_A);

	unsigned int size_C = KERNEL_SIZE * KERNEL_SIZE;
	unsigned int mem_size_C = sizeof(float) * size_C;
	float* h_C = (float*)malloc(mem_size_C);

	randomInit(h_A, size_A);
	randomInit(h_C, size_C);

   
    unsigned int timer;
    CUT_SAFE_CALL(cutCreateTimer(&timer));
	cutStartTimer(timer);
    reference(h_A, h_cpu, WA, HA, h_C, KERNEL_SIZE);
	cutStopTimer(timer);
    printf("Host CPU Processing time: %f (ms)\n", cutGetTimerValue(timer));
	
	float* d_A;
	float* d_B;
	float* d_C;

	cudaMalloc((void**)&d_A, mem_size_A);
	cudaMalloc((void**)&d_B, mem_size_A);
	cudaMalloc((void**)&d_C, mem_size_C);

	cudaMemcpy(d_A, h_A, mem_size_A, cudaMemcpyHostToDevice);
	cudaMemcpy(d_C, h_C, mem_size_C, cudaMemcpyHostToDevice);

	dim3 threads(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid((WA - 1) / (BLOCK_SIZE - KERNEL_SIZE + 1) + 1, (WA - 1) / (BLOCK_SIZE - KERNEL_SIZE + 1) + 1);

	cudaEventRecord(start_G);
	Convolution <<< grid, threads >>>(d_A, d_B, d_C);

	cudaDeviceSynchronize();
	cudaEventRecord(stop_G);
	cudaEventSynchronize(stop_G);

	cudaMemcpy(h_B, d_B, mem_size_A, cudaMemcpyDeviceToHost);

	float miliseconds = 0;
	cudaEventElapsedTime(&miliseconds, start_G, stop_G);

	printf("Time took to compute matrix A of dimensions %d x %d  on GPU is %f ms \n \n \n", WA, HA, miliseconds);

	// for (int i = 0;i < HA;i++)
	// {
	// 	for (int j = 0;j < WA;j++)
	// 	{
	// 		if(i == 0 or i == HA -2 )
	// 		printf("%d, %f, %f, %f\n", j, h_B[i*HA + j], h_cpu[i*HA + j], h_B[i*HA + j] - h_cpu[i*HA + j]);
	// 	}
	// }

	unsigned int result_regtest = cutComparefe(h_cpu, h_B, WA*HA, 0.001f);
    printf( "Test %s\n", (1 == result_regtest) ? "PASSED" : "FAILED");

	free(h_A);
	free(h_B);
	free(h_C);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	return EXIT_SUCCESS;
}