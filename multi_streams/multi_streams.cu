#include <iostream>
#include <unistd.h>
#include <stdlib.h>
#include "cuda.h"
using namespace std;

__global__ void infinitekernel(float *dptr, int *dwait)
{
 	while(*dwait)	*dptr += 1;
	*dptr = 999;

}	

int main(void)
{
	cudaStream_t stream[2];
	for (int i=0; i < 2 ; i++)
		cudaStreamCreate(&stream[i]);
	float *hptr;
	float *dptr;
	int *hwait;
	int *dwait;
	hptr = (float*)malloc(sizeof(float));
	hwait = (int*)malloc(sizeof(int));
	cudaMalloc((void **)&dptr, sizeof(float));
	cudaMalloc((void **)&dwait, sizeof(int));
	*hptr = 9;
	*hwait = 1;
	cudaMemcpyAsync(dptr, hptr, sizeof(float), cudaMemcpyHostToDevice, stream[0]);
	cudaMemcpyAsync(dwait, hwait, sizeof(float), cudaMemcpyHostToDevice, stream[0]);
	infinitekernel<<<1, 1, 0, stream[1]>>>(dptr,dwait);

	for(int i=0; i<10; i++)
	{
		sleep(1);
		cudaMemcpyAsync(hptr, dptr, sizeof(float), cudaMemcpyDeviceToHost, stream[0]);
	        cout << "["<< i << " seconds]" <<"value = " << *hptr << endl;
	}

	*hwait = 0;
	cudaMemcpyAsync(dwait, hwait, sizeof(int), cudaMemcpyHostToDevice, stream[0]);
	cudaMemcpyAsync(hptr, dptr, sizeof(float), cudaMemcpyDeviceToHost, stream[0]);
	cout <<"[Finally]" << "value = "<< *hptr << endl;
}
