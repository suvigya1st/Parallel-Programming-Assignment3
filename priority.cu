#include <stdlib.h>
#include <stdio.h>
#include <time.h>

#define THREADS 256 // 2^9
#define BLOCKS 32 // 2^15
#define NUM THREADS*BLOCKS

int seed_var =1239;

int random_int()
{
  return (int)rand()%(int)9 +1;
}

void array_fill(int *arr, int length)
{
  srand(++seed_var);
  int i;
  for (i = 0; i < length; ++i) {
    arr[i] = random_int();
  }
}

void print_array(int *arr1,int *arr2 ,int *arr3,int *arr4, int length)
{
  //srand(time(NULL));
  int i;
  for (i = 0; i < length; ++i) {
    //arr[i] = random_float();
    printf("%d\t%d\t%d\t%d\t%d\n",i+1,arr1[i],arr2[i],arr3[i],arr4[i]);
  }
}

void print_elapsed(clock_t start, clock_t stop)
{
  double elapsed = ((double) (stop - start)) / CLOCKS_PER_SEC;
  printf("Elapsed time: %.3fs\n", elapsed);
}

__device__ void swap(int *xp, int *yp)
{
    int temp = *xp;
    *xp = *yp;
    *yp = temp;
}
__global__ void bitonic_sort_step(int *d_pr, int *d_bt, int j, int k)
{
  int i, ixj; /* Sorting partners: i and ixj */
  i = threadIdx.x + blockDim.x * blockIdx.x;
  ixj = i^j;

  /* The threads with the lowest ids sort the array. */
  if ((ixj)>i) 
  {
    if ((i&k)==0) 
    {
      /* Sort ascending */
      if (d_pr[i]>d_pr[ixj]) 
      {
        /* exchange(i,ixj); */
        swap(&d_pr[i],&d_pr[ixj]);
        swap(&d_bt[i],&d_bt[ixj]);
      }
    }
    if ((i&k)!=0)
    {
      /* Sort descending */
      if (d_pr[i]<d_pr[ixj])
      {
        /* exchange(i,ixj); */
        swap(&d_pr[i], &d_pr[ixj]);
        swap(&d_bt[i], &d_bt[ixj]);
      }
    }
  }
}

void sorting_first(int *pr, int *bt)
{
  
  dim3 blocks(BLOCKS,1);
  dim3 threads(THREADS,1);

  int k;
  
  //Major step priority time basis sorting
  for(k = 2; k <= NUM; k <<= 1)
  {
    for (int j = k>>1; j > 0; j = j>>1)
    {
      bitonic_sort_step<<<blocks,threads>>>(pr,bt,j,k);
    }
  }
}


__global__ void inclusive_scan(int *X, int *Y, int N)
{
  extern __shared__ int XY[];
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  // load input into __shared__ memory
  if(i<N)
  {  
    XY[threadIdx.x] =X[i];  
  }  
    /*Note here stride <= threadIdx.x, means that everytime the threads with threadIdx.x less than 
    stride do not participate in loop*/
  for(int stride = 1; stride <= threadIdx.x; stride *= 2) {
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
  XY[threadIdx.x] = Y[threadIdx.x * blockDim.x + THREADS - 1];
   }
   __syncthreads();
  for (int stride = 0; stride < blockIdx.x; stride++) 
  {    //add all previous las elements to this block elements
    Y[threadIdx.x + blockDim.x * blockIdx.x] += XY[stride];
    __syncthreads();
    
  }
}

__global__ void work_inefficient_scan_kernel(int *X, int *Y, int InputSize)
{
  __shared__ int XY[THREADS];
  int i= blockIdx.x*blockDim.x+ threadIdx.x;
  {
    XY[threadIdx.x] = X[i];
  }
  // the code below performs iterative scan on XY
  for (unsigned int stride = 1; stride <= threadIdx.x; stride *= 2)
  {
    __syncthreads();
    XY[threadIdx.x] += XY[threadIdx.x-stride];
  }
  Y[i] = XY[threadIdx.x];
}

__global__ void work_efficient_scan_kernel(int *X, int *Y, int InputSize)
{
  extern __shared__ int XY[];
  int i= blockIdx.x*blockDim.x+ threadIdx.x;
  if (i < InputSize)
  {
    XY[threadIdx.x] = X[i];
  }
  for (unsigned int stride = 1; stride < blockDim.x; stride *= 2)
  {
    __syncthreads();
    int index = (threadIdx.x+1) * 2* stride -1;
    if (index < blockDim.x)
    {
      XY[index] += XY[index -stride];
    }
  }
  for (int stride = THREADS/4; stride > 0; stride /= 2)
  {
    __syncthreads();
    int index = (threadIdx.x+1)*stride*2 -1;
    if(index + stride < THREADS)
    {
     XY[index + stride] += XY[index];
    }
  }
  __syncthreads();
  Y[i] = XY[threadIdx.x];

  //OWN CODE
  __syncthreads();
  if(threadIdx.x < blockIdx.x)
  {
    XY[threadIdx.x] = Y[threadIdx.x*blockDim.x + (blockDim.x-1)];
  }
  __syncthreads();
  for(unsigned int stride =0; stride < blockIdx.x; stride++)
  {
    Y[i] += XY[stride];
  }
  __syncthreads();
}

void scan_next(int *bt, int *tat)
{
  dim3 blocks(BLOCKS,1);
  dim3 threads(THREADS,1);

  work_efficient_scan_kernel<<<blocks, threads, THREADS * sizeof(int)>>>(bt, tat,NUM);
}

int main()
{
  //int h_pid[NUM],h_wt[NUM],h_tat[NUM],i,j,n,total=0,pos,temp,avg_wt,avg_tat;
  // printf("Enter Total Number of Process:");
  // scanf("%d",&n);

  //printf("\nEnter Burst Time and Priority\n");
  clock_t start, stop;
  int *h_bt = (int*) malloc( NUM * sizeof(int));
  int *h_pr = (int*) malloc( NUM * sizeof(int));
  int *h_tat = (int*) malloc( NUM * sizeof(int));
  int *h_wt = (int*) malloc( NUM * sizeof(int));
  array_fill(h_bt, NUM);
  array_fill(h_pr, NUM);
  printf("INITIAL\n");
  printf("\tPR\tBT\tWT\tTAT\n");
  print_array(h_pr,h_bt,h_wt,h_tat,NUM);

  int *d_bt, *d_pr, *d_wt, *d_tat;
  size_t size = NUM * sizeof(int);

  cudaMalloc((void**) &d_bt, size);
  cudaMalloc((void**) &d_pr, size);
  cudaMalloc((void**) &d_wt, size);
  cudaMalloc((void**) &d_tat, size);

  cudaMemcpy(d_bt, h_bt, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_pr, h_pr, size, cudaMemcpyHostToDevice);


  start = clock();
  sorting_first(d_pr, d_bt);
  
  scan_next(d_bt, d_tat);
  
  //work_inefficient_scan_kernel<<<BLOCKS,THREADS,size>>>(d_bt,d_tat,NUM);


  cudaMemcpy(h_bt, d_bt, size, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_pr, d_pr, size, cudaMemcpyDeviceToHost);
  cudaMemcpy(&h_wt[1], d_tat, size, cudaMemcpyDeviceToHost);
  cudaMemcpy(h_tat, d_tat, size, cudaMemcpyDeviceToHost);

  cudaFree(d_pr);
  cudaFree(d_bt);
  cudaFree(d_wt);
  cudaFree(d_tat);
  
  stop = clock();

  printf("\nFINAL\n");
  printf("\tPR\tBT\tWT\tTAT\n");
  print_array(h_pr,h_bt,h_wt,h_tat,NUM);
  print_elapsed(start, stop);

} 