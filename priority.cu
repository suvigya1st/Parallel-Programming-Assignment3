
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#define THREADS 512 // 2^9
#define BLOCKS 32768 // 2^15
#define NUM THREADS*BLOCKS

# define THREADS 4

#include <iostream.h>

__global__ void better_inclusive_scan (int *X, int *Y, int n)
{
	//here X is bt, Y is tat
	extern __shared__ int XY[];
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	//load input into shared memory
	if (i<n)
	{
		XY[threadIdx.x] = X[i];
	}
	for (unsigned int stride =1; stride <= threadIdx.x; stride *=2)
	{
		__syncthreads();
		XY[threadIdx.x] += XY[threadIdx.x - stride];
	}	
	Y[i] = XY[threadIdx.x];
	
	//simplified logic
	__syncthreads();
	if (threadIdx.x < blockIdx.x)
	{
		XY[threadIdx.x] = Y[threadIdx*blockDim.x + (THREADS-1)];
	}
	__syncthreads();
	for(int stride = 0; stride < blockIdx.x; stride++)
	{
		Y[i] += XY[stride];
		__syncthreads();
	}

}
float random_int()
{
  return (int)rand()%(int)100;
}

void array_fill(float *arr, int length)
{
  srand(time(NULL));
  int i;
  for (i = 0; i < length; ++i) {
    arr[i] = random_int();
  }
}

int main()
{
	int h_pid[NUM],h_wt[NUM],h_tat[NUM],i,j,n,total=0,pos,temp,avg_wt,avg_tat;
    // printf("Enter Total Number of Process:");
    // scanf("%d",&n);
 
    //printf("\nEnter Burst Time and Priority\n");
    int *h_bt = (int*) malloc( NUM * sizeof(int));
    int *h_pr = (int*) malloc( NUM * sizeof(int));
  	array_fill(h_bt, NUM);
  	array_fill(h_pr, NUM);
  	printf("\nINITIAL ARRAY\n");
  	//print_array(values, NUM_VALS);

    //memory allocation for device copy
    int *d_pid;
    int *d_pr;
    int *d_bt;
    //int *d_wt = 0;
    int *d_tat;
    cudaMalloc((void**)&d_pid,sizeof(int)*n);
    cudaMemcpy(d_pid, h_pid, sizeof(int)*n,cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_pr,sizeof(int)*n);
    cudaMemcpy(d_pr, h_pr, sizeof(int)*n,cudaMemcpyHostToDevice);
    
    cudaMalloc((void**)&d_bt,sizeof(int)*n);
    cudaMemcpy(d_bt, h_bt, sizeof(int)*n,cudaMemcpyHostToDevice);

    //SORTING
    //kernel launch for sorting wrt d_pr and d_bt along with d_pid

    cudaMemcpy(h_bt, d_bt, sizeof(int)*n,cudaMemcpyDeviceToHost);
    cudaMemcpy(h_pr, d_pr, sizeof(int)*n,cudaMemcpyDeviceToHost);
    cudaMemcpy(h_pid, d_pid, sizeof(int)*n,cudaMemcpyDeviceToHost);
    //we get the sorted arrays


	better_inclusive_scan<<<BLOCKS,THREADS,sizeof(int)*n>>>(d_bt,d_tat,n);

	cudaMemcpy(h_tat, d_tat, sizeof(int)*n,cudaMemcpyDeviceToHost);

	cudaMemcpy(&h_wt[1], d_tat, sizeof(int)*n,cudaMemcpyDeviceToHost);	
	h_wt[0] = 0;
}	
