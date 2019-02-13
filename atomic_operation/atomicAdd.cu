#include <iostream>
#include "cuda.h"
using namespace std;

#define N 64

__global__ void atomicKernel(int* sum, int* array)
{
	const int tid = blockDim.x * blockDim.y * threadIdx.z 
		        + blockDim.x * threadIdx.y + threadIdx.x;
	array[tid] = tid;
	atomicAdd(sum, tid);

}

int main()
{
	dim3 grid(1,1,1), block(2,8,4);

	int* hSum;
	hSum = (int*)malloc(sizeof(int));
	int* dSum;
	cudaMalloc((void**)&dSum, sizeof(int));

	int* hArray;
	hArray = (int*)malloc(N*sizeof(int));
	int* dArray;
	cudaMalloc((void**)&dArray, N*sizeof(int));

	atomicKernel<<<grid,block>>>(dSum, dArray);
	cudaDeviceSynchronize();
	cudaMemcpy(hSum, dSum, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(hArray, dArray, N*sizeof(int), cudaMemcpyDeviceToHost);

	int cpuSum = 0;
	for(int i = 0; i < N; i++)
		cpuSum += hArray[i];

	cout<< "CPU Sum: " << cpuSum  << endl;
	cout<< "GPU Sum: " << *hSum  << endl;

}
