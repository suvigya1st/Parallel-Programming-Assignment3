// This example demonstrates a block-wise inclusive scan (prefix sum)

#include <stdlib.h>
#include <stdio.h>
#include<iostream>
// This kernel computes, per-block, a block-sized scan
// of the input.  It assumes that the block size evenly
// divides the input size
  const int BLOCK_SIZE = 4;

                               
//This is as per text book
__global__ void inclusive_scan(const unsigned int *X,
                               unsigned int *Y, int N)
{
  extern __shared__ int XY[];
  unsigned   int i = blockIdx.x * blockDim.x + threadIdx.x;
  // load input into __shared__ memory
  if(i<N)
  {  
    XY[threadIdx.x] =X[i];  
  }  
    /*Note here stride <= threadIdx.x, means that everytime the threads with threadIdx.x less than 
    stride do not participate in loop*/
  for(unsigned int stride = 1; stride <= threadIdx.x; stride *= 2) {
  	__syncthreads();
  	XY[threadIdx.x]+= XY[threadIdx.x - stride];
  }
  /*This is executed by all threads, so that they store the final prefix sum to 
  corresponding locations in global   memory*/
   Y[i]=XY[threadIdx.x];

// wait until all threads of this block writes the output for all prefix sum within the block 
  __syncthreads();
  if (threadIdx.x < blockIdx.x) //for 1st block onwards
  {
	//update the shared memory to keep prefix sum of last elements of previous block's
	XY[threadIdx.x] = Y[threadIdx.x * blockDim.x + BLOCK_SIZE - 1];
   }
   __syncthreads();
  for (int stride = 0; stride < blockIdx.x; stride++) 
  {    //add all previous las elements to this block elements
   	Y[threadIdx.x + blockDim.x * blockIdx.x] += XY[stride];
      __syncthreads();
    
  }
}


int main(void)
{
  // use small input sizes for illustrative purposes
  const int num_blocks = 2;
  const int num_elements = num_blocks * BLOCK_SIZE;

  // generate random input in [0,5] on the host
  // taking unsigned int as precautionary measure, 
  //so that error is generated if we assign negative numbers
  unsigned int *h_input= (unsigned int *) malloc(num_elements * sizeof(unsigned int));
  for(unsigned int i = 0; i < num_elements; ++i)
  {
    h_input[i] = i;
  }

  // copy input to device memory
  unsigned int *d_input = 0;
  cudaMalloc((void**)&d_input, sizeof(unsigned int) * num_elements);
  cudaMemcpy(d_input, h_input, sizeof(unsigned int) * num_elements, cudaMemcpyHostToDevice);

  // allocate space for the result
  unsigned int *d_result = 0;
  cudaMalloc((void**)&d_result, sizeof(unsigned int) * num_elements);

  inclusive_scan<<<num_blocks, BLOCK_SIZE, BLOCK_SIZE * sizeof(unsigned int)>>>(d_input, d_result,num_elements);

  // copy result to host memory
  unsigned int *h_result=(unsigned int *) malloc(num_elements * sizeof(unsigned int));;
  cudaMemcpy(h_result, d_result, sizeof(unsigned int) * num_elements, cudaMemcpyDeviceToHost);
  // print out the results
  for(int b = 0; b < num_blocks; ++b)
  {
    std::cout << "Block " << b << std::endl << std::endl;

    std::cout << "Input: " << std::endl;
    for(int i = 0; i < BLOCK_SIZE; ++i)
    {
      printf("%2d ", h_input[b * BLOCK_SIZE + i]);
    }
    std::cout << std::endl;

    std::cout << "Result: " << std::endl;
    for(int i = 0; i < BLOCK_SIZE; ++i)
    {
      printf("%2d ", h_result[b * BLOCK_SIZE + i]);
    }
    std::cout << std::endl << std::endl << std::endl;
  }

  return 0;
}

