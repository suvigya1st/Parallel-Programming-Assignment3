# define BLOCK_SIZE 4
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
		XY[threadIdx.x] = Y[threadIdx*blockDim.x + (BLOCK_SIZE-1)];
	}
	__syncthreads();
	for(int stride = 0; stride < blockIdx.x; stride++)
	{
		Y[i] += XY[stride];
		__syncthreads();
	}

}

int main()
{
	int h_bt[20],h_pid[20],h_wt[20],h_tat[20],h_pr[20],i,j,n,total=0,pos,temp,avg_wt,avg_tat;
    printf("Enter Total Number of Process:");
    scanf("%d",&n);
 
    printf("\nEnter Burst Time and Priority\n");
    for(i=0;i<n;i++)
    {
        printf("\nP[%d]\n",i+1);
        printf("Burst Time:");
        scanf("%d",&bt[i]);
        printf("Priority:");
        scanf("%d",&pr[i]);
        p[i]=i+1;           //contains process number
    }

    //memory allocation for device copy
    int *d_pid = 0;
    int *d_pr = 0;
    int *d_bt = 0;
    //int *d_wt = 0;
    int *d_tat = 0;
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


	better_inclusive_scan<<<ceil(n/BLOCK_SIZE),BLOCK_SIZE,sizeof(int)*n>>>(d_bt,d_tat,n);

	cudaMemcpy(h_tat, d_tat, sizeof(int)*n,cudaMemcpyDeviceToHost);

	cudaMemcpy(&h_wt[1], d_tat, sizeof(int)*n,cudaMemcpyDeviceToHost);	
	h_wt[0] = 0;
}	
