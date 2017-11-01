
#include <stdlib.h>
#include <stdio.h>

int main()
{
	int seed_var = 1200;
	//srand(seed_var) ;
	for (int i=0; i<10;i++)
	{
		printf("%d\n", rand());
	}
}