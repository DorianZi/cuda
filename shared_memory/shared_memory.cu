// input: 1 2 3  4  5  6
// ouput:  3 6 12 20 30
#include <iostream>
#include "cuda.h"
using namespace std;

#define N 64

__global__ void myKernel(int *retArray, int *Array)
{
	extern __shared__ int sharedArray[];
	const int tid = blockIdx.x * (blockDim.x * blockDim.y) 
		        + blockDim.x * threadIdx.y + threadIdx.x;
	
	sharedArray[tid] = Array[tid];
	__syncthreads();
	if ( tid == N-1 )
		retArray[tid] = sharedArray[tid];
	else
		retArray[tid] = sharedArray[tid] * sharedArray[tid+1];	
		 
}


int main()
{

	dim3 grid(2,1,1), block(4,8,1);
	int *Array_host;
	Array_host = (int*)malloc(N*sizeof(int));
	for(int i=0; i<N; i++)
		Array_host[i] = i;
	int *Array_dev;
	cudaMalloc((void**)&Array_dev, N*sizeof(int));
	cudaMemcpy(Array_dev, Array_host, N*sizeof(int), cudaMemcpyHostToDevice);

	int *retArray_host;
	retArray_host = (int*)malloc(N*sizeof(int));
	int *retArray_dev;
	cudaMalloc((void**)&retArray_dev, N*sizeof(int));
	myKernel<<<grid, block, N*sizeof(int)>>>(retArray_dev,Array_dev);
	cudaDeviceSynchronize();

	cudaMemcpy(retArray_host,retArray_dev, N*sizeof(int), cudaMemcpyDeviceToHost);

	cout << "======[Array]=====" << endl;
	for(int i=0; i<N; i++)
		cout << "Array_host[" << i <<"] = " << Array_host[i] << endl;
	cout << "======[SumArray]=====" << endl;
	for(int i=0; i<N; i++)
		cout << "retArray_host[" << i <<"] = " << retArray_host[i] << endl;

}
