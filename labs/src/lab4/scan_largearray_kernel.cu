#ifndef _PRESCAN_CU_
#define _PRESCAN_CU_

// includes, kernels
#include <assert.h>
#include <iostream>


#define NUM_BANKS 32
#define LOG_NUM_BANKS 5
// Lab4: You can use any other block size you wish.
#define BLOCK_SIZE 256
// Lab4: Host Helper Functions (allocate your own data structure...)


// Lab4: Device Functions


// Lab4: Kernel Functions
__global__ void scanKernal(float* outArray, float *inArray, float* levelArr, int numElements){
	__shared__ float tileArr[BLOCK_SIZE];
	int gArrid = blockIdx.x * BLOCK_SIZE + threadIdx.x;
	if(gArrid < numElements and gArrid > 0) tileArr[threadIdx.x] = inArray[gArrid-1]; //shift one element to load the input for exclusive scan
	else tileArr[threadIdx.x] = 0; // which by defualt

	__syncthreads();

	/*Reduction step kernel code*/
	int stride = 1;
	while(stride < BLOCK_SIZE){
		int index = (threadIdx.x+1)*stride*2 - 1;
		if(index < BLOCK_SIZE){
			tileArr[index] += tileArr[index-stride];
		}
		stride *= 2;
		__syncthreads();
	}

	// __syncthreads();

	/*Post scan step*/
	stride = BLOCK_SIZE >> 1;
	while(stride > 0){
		int index = (threadIdx.x+1)*stride*2 - 1;
		if(index + stride < BLOCK_SIZE){
			tileArr[index+stride] += tileArr[index];
		}
		stride = stride >> 1;
		__syncthreads();
	}

	// __syncthreads();


    if(threadIdx.x==0) levelArr[blockIdx.x] = tileArr[BLOCK_SIZE-1];
    // if(threadIdx.x==0) printf("%f,%d, %f, %f\n", levelArr[blockIdx.x], numElements, inArray[0], inArray[1]);
	
	if(gArrid < numElements) outArray[gArrid] = tileArr[threadIdx.x];
	
	// if(threadIdx.x == 0 and blockIdx.x == 10){
	// 	for(int i = 0; i < BLOCK_SIZE; i++){
	// 		printf("%d, %f, %f, %f\n",i, tileArr[i], tileArr[BLOCK_SIZE-1], levelArr[blockIdx.x]);
	// 	}
	// }
}

__global__ void blockSumAdd(float *blockvector, float *sumvector, int numElements){
    int gArrid = blockIdx.x*BLOCK_SIZE + threadIdx.x;
    if(gArrid < numElements){
    	// printf("%d, %f, %f\n",  gArrid, blockvector[gArrid], sumvector[blockIdx.x]);
        blockvector[gArrid] += sumvector[blockIdx.x];
    }
    // if(gArrid < numElements) printf("%f\n", blockvector[gArrid]);
}

// **===-------- Lab4: Modify the body of this function -----------===**
// You may need to make multiple kernel calls, make your own kernel
// function in this file, and then call them from here.
void prescanArray(float *outArray, float *inArray, int numElements)
{
	float *L1Arr, *L2Arr, *L3Arr, *L1tem, *L2tem, *L3tem;
	int L1Len = (numElements + BLOCK_SIZE - 1) / BLOCK_SIZE;
	cudaMalloc((void **) &L1Arr, sizeof(float) * L1Len);
	cudaMalloc((void **) &L1tem, sizeof(float) * L1Len);

	int L2Len = (L1Len + BLOCK_SIZE - 1) / BLOCK_SIZE;
	cudaMalloc((void**) &L2Arr, sizeof(float) * L2Len);
	cudaMalloc((void**) &L2tem, sizeof(float) * L2Len);

	int L3Len = (L2Len + BLOCK_SIZE - 1) / BLOCK_SIZE;
	cudaMalloc((void**) &L3Arr, sizeof(float) * L3Len);
	cudaMalloc((void**) &L3tem, sizeof(float) * L3Len);	
	// printf("%d, %d\n", L1Len, L2Len);

	scanKernal<<<L1Len, BLOCK_SIZE>>>(outArray, inArray, L1Arr, numElements);
	scanKernal<<<L2Len, BLOCK_SIZE>>>(L1tem, L1Arr, L2Arr, L1Len);
	scanKernal<<<L3Len, BLOCK_SIZE>>>(L2tem, L2Arr, L3Arr, L2Len);
	blockSumAdd<<<L2Len, BLOCK_SIZE>>>(L1tem, L2tem, L1Len);
	blockSumAdd<<<L1Len, BLOCK_SIZE>>>(outArray, L1tem, numElements);
	cudaFree(L1Arr);
    cudaFree(L2Arr);
	cudaFree(L3Arr);
    cudaFree(L1tem);
    cudaFree(L2tem);
    cudaFree(L3tem);    
}
// **===-----------------------------------------------------------===**


#endif // _PRESCAN_CU_
