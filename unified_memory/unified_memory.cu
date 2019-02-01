#include <iostream>
#include "cuda.h"
using namespace std;

#define N 64
#define ThreadPerBlock 32

__global__ void squareKernel(int* ptr)
{
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  ptr[i]  = ptr[i] * ptr[i];
}

int main(void)
{
	int *array;
	cudaMallocManaged(&array, N*sizeof(int));
	
	cout << "[Before Square]"<< endl;
	for (int i = 0; i < N; i++ ){
		array[i] = i;	
		cout << array[i];
		if(N-1 != i)
			cout << " ,";
	}
	cout << endl;

	squareKernel<<<N/ThreadPerBlock, ThreadPerBlock>>>(array);
	cudaDeviceSynchronize();
	
	cout << "[After Square]"<< endl;
	for (int i = 0; i < N; i++ ){
		cout << array[i];
		if(N-1 != i)
			cout << " ,";
	}
	cout << endl;

}
