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

void print_elapsed(clock_t start, clock_t stop)
{
  double elapsed = ((double) (stop - start)) / CLOCKS_PER_SEC;
  printf("\nElapsed time: %.3fs\n", elapsed);
}

int main()
{
    int i,j,n,total=0,pos,temp,avg_wt,avg_tat;
    
    n=NUM;

    int *bt = (int*) calloc( NUM , sizeof(int));
    int *pr = (int*) calloc( NUM , sizeof(int));
    int *wt = (int*) calloc( NUM , sizeof(int));
    int *tat = (int*) calloc( NUM , sizeof(int));


    array_fill(bt, NUM);
    array_fill(pr, NUM);
    clock_t start, stop;

    printf("\nPr\tBT\tWT\tTAT");
    for(i=0;i<n;i++)
    {
        printf("\n%d\t%d\t%d\t%d",pr[i],bt[i],wt[i],tat[i]);
    }

    start = clock();

    //sorting burst time, priority and process number in ascending order using selection sort
    for(i=0;i<n;i++)
    {
        pos=i;
        for(j=i+1;j<n;j++)
        {
            if(pr[j]<pr[pos])
                pos=j;
        }
 
        temp=pr[i];
        pr[i]=pr[pos];
        pr[pos]=temp;
 
        temp=bt[i];
        bt[i]=bt[pos];
        bt[pos]=temp;
 
    }
 
    wt[0]=0;    //waiting time for first process is zero
     
    //calculate waiting time
    for(i=1;i<n;i++)
    {
        wt[i]=wt[i-1]+bt[i-1];
        //total+=wt[i];
    }
    
    for(i=0;i<n;i++)
    {
        tat[i]=bt[i]+wt[i];     //calculate turnaround time
    }
    stop = clock();

    printf("\nPr\tBT\tWT\tTAT");
    for(i=0;i<n;i++)
    {
        printf("\n%d\t%d\t%d\t%d",pr[i],bt[i],wt[i],tat[i]);
    }
 
    //avg_tat=total/n;     //average turnaround time
    //printf("\n\nAverage Waiting Time=%d",avg_wt);
    //printf("\nAverage Turnaround Time=%d\n",avg_tat);
    print_elapsed(start, stop);

    return 0;
}