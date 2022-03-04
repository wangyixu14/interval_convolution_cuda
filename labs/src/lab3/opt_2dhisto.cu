#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <cutil.h>
#include "util.h"
#include "ref_2dhisto.h"

__global__ void opt_2dhisto_kernel(uint32_t* d_input, size_t inputSize, uint32_t* d_bins);

void opt_2dhisto(uint32_t* d_input, size_t inputSize, uint32_t* d_bins)
{
    /* This function should only contain grid setup 
       code and a call to the GPU histogramming kernel. 
       Any memory allocations and transfers must be done 
       outside this function */
    cudaMemset(d_bins, 0, HISTO_HEIGHT*HISTO_WIDTH*sizeof(uint32_t));
    opt_2dhisto_kernel<<<34, 1024>>> (d_input, inputSize, d_bins);
}

/* Include below the implementation of any other functions you need */

/* This kernel function is to use the atomix instruction*/
__global__ void opt_2dhisto_kernel(uint32_t* d_input, size_t inputSize, uint32_t* d_bins){
    __shared__ uint32_t block_bins[HISTO_HEIGHT*HISTO_WIDTH];
    __shared__ uint32_t shared_input[HISTO_HEIGHT*HISTO_WIDTH];
    
    block_bins[threadIdx.x] = 0;
    __syncthreads();

    int gtid = threadIdx.x + blockDim.x*blockIdx.x;
    while(gtid<inputSize){
        atomicAdd(&(block_bins[d_input[gtid]]), 1);
        gtid += gridDim.x*blockDim.x;
    }
    __syncthreads();
 
    atomicAdd(&(d_bins[threadIdx.x]), block_bins[threadIdx.x]);
    // if(d_bins[threadIdx.x] > 255) 
    //     d_bins[threadIdx.x] = 255;
    if(threadIdx.x < 8){
        for(int i = 0; i < 128; i++){
            if(d_bins[threadIdx.x*128+i] > 255) d_bins[threadIdx.x*128+i] = 255;
        }
    }

}


/*This kernel function requires each thread to scan the input and takes responsibility for the a fixed bin number(based on thread ID)*/
// __global__ void opt_2dhisto_kernel(uint32_t* d_input, size_t inputSize, uint32_t* d_bins){
//     __shared__ uint32_t block_bins[HISTO_HEIGHT*HISTO_WIDTH];
//     block_bins[threadIdx.x] = 0;
//     __syncthreads();

//     int stride = gridDim.x*blockDim.x, base = 0;
//     while(base < inputSize){
//         for(int i = 0; i < blockDim.x; i++){
//             if((base + blockDim.x*blockIdx.x + i) < inputSize && d_input[base+blockDim.x*blockIdx.x + i] == threadIdx.x) block_bins[threadIdx.x]++;
//         }
//         base += stride;
//     }
//     __syncthreads();

//     atomicAdd(&(d_bins[threadIdx.x]), block_bins[threadIdx.x]);
//     if(d_bins[threadIdx.x] > 255) 
//         d_bins[threadIdx.x] = 255;
// }

/*This kernel function makes each thread have its histogram and then merge all the threads*/
// __global__ void opt_2dhisto_kernel(uint32_t* d_input, size_t inputSize, uint32_t* d_bins){
//     uint8_t thread_bins[HISTO_HEIGHT*HISTO_WIDTH];
//     for(int i = 0; i < HISTO_HEIGHT*HISTO_WIDTH; i++){
//         thread_bins[i] = 0;
//     }
//     __syncthreads();
//     int stride = inputSize / (gridDim.x*blockDim.x);
//     for(int j = 0; j < stride; j++){
//         int base = (blockIdx.x * blockDim.x + threadIdx.x) * stride;
//         if(base + j < inputSize) thread_bins[d_input[base + j]]++;
//     }
//     __syncthreads();

//     for(int k = 0; k < HISTO_HEIGHT*HISTO_WIDTH; k++){
//         atomicAdd(&(d_bins[k]), thread_bins[k]);
//         if(d_bins[k] > 255) 
//             d_bins[k] = 255;
//     }
//     __syncthreads();
// }

