/*
 * Copyright 1993-2006 NVIDIA Corporation.  All rights reserved.
 *
 * NOTICE TO USER:   
 *
 * This source code is subject to NVIDIA ownership rights under U.S. and 
 * international Copyright laws.  
 *
 * This software and the information contained herein is PROPRIETARY and 
 * CONFIDENTIAL to NVIDIA and is being provided under the terms and 
 * conditions of a Non-Disclosure Agreement.  Any reproduction or 
 * disclosure to any third party without the express written consent of 
 * NVIDIA is prohibited.     
 *
 * NVIDIA MAKES NO REPRESENTATION ABOUT THE SUITABILITY OF THIS SOURCE 
 * CODE FOR ANY PURPOSE.  IT IS PROVIDED "AS IS" WITHOUT EXPRESS OR 
 * IMPLIED WARRANTY OF ANY KIND.  NVIDIA DISCLAIMS ALL WARRANTIES WITH 
 * REGARD TO THIS SOURCE CODE, INCLUDING ALL IMPLIED WARRANTIES OF 
 * MERCHANTABILITY, NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.   
 * IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY SPECIAL, INDIRECT, INCIDENTAL, 
 * OR CONSEQUENTIAL DAMAGES, OR ANY DAMAGES WHATSOEVER RESULTING FROM LOSS 
 * OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE 
 * OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE 
 * OR PERFORMANCE OF THIS SOURCE CODE.  
 *
 * U.S. Government End Users.  This source code is a "commercial item" as 
 * that term is defined at 48 C.F.R. 2.101 (OCT 1995), consisting  of 
 * "commercial computer software" and "commercial computer software 
 * documentation" as such terms are used in 48 C.F.R. 12.212 (SEPT 1995) 
 * and is provided to the U.S. Government only as a commercial end item.  
 * Consistent with 48 C.F.R.12.212 and 48 C.F.R. 227.7202-1 through 
 * 227.7202-4 (JUNE 1995), all U.S. Government End Users acquire the 
 * source code with only those rights set forth herein.
 */

/* Matrix multiplication: C = A * B.
 * Device code.
 */

#ifndef _MATRIXMUL_KERNEL_H_
#define _MATRIXMUL_KERNEL_H_

#include <stdio.h>
#include "matrixmul.h"
#define TILE_WIDTH 32

////////////////////////////////////////////////////////////////////////////////
//! Simple test kernel for device functionality
//! @param g_idata  input data in global memory
//! @param g_odata  output data in global memory
////////////////////////////////////////////////////////////////////////////////
// Matrix multiplication kernel thread specification
__global__ void MatrixMulKernel(Matrix M, Matrix N, Matrix P)
{
	__shared__ float tileMs[TILE_WIDTH][TILE_WIDTH + 1];
	__shared__ float tileNs[TILE_WIDTH][TILE_WIDTH + 1];

	int tx = threadIdx.x; int ty = threadIdx.y;
	int bx = blockIdx.x; int by = blockIdx.y;

	int row = by * TILE_WIDTH + ty;
	int col = bx * TILE_WIDTH + tx;

	float V = 0;

	for(int i = 0; i < ceilf(M.width / (float)TILE_WIDTH); i++){
		if(row < M.height && (i * TILE_WIDTH + tx) < M.width)
			tileMs[ty][tx] = M.elements[row * M.width + i * TILE_WIDTH + tx];
        else
            tileMs[ty][tx] = 0;
		if(col < N.width && (i * TILE_WIDTH + ty) < N.height)
			tileNs[ty][tx] = N.elements[(i * TILE_WIDTH + ty) * N.width + col];
        else
            tileNs[ty][tx] = 0;
		__syncthreads();

		for(int j = 0; j < TILE_WIDTH; j++)
			V += tileMs[ty][j] * tileNs[j][tx];
		
		__syncthreads();
	}
	if(row < P.height && col < P.width)
		P.elements[row * P.width + col] = V;

}

#endif // #ifndef _MATRIXMUL_KERNEL_H_
