#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <iostream>

#define N 64

	
struct User_Multiple
{
	__host__ __device__ int operator()(const int &x, const int &y){
		return x * y;
	}	
};



int main()
{
	thrust::device_vector<int> array_X(N);
	thrust::device_vector<int> array_Y(N);
	thrust::device_vector<int> array_out(N);
	thrust::fill(array_X.begin(), array_X.end(),10);
	thrust::sequence(array_Y.begin(), array_Y.end());
	thrust::transform(array_X.begin(), array_X.end(), array_Y.begin(), array_out.begin(), User_Multiple());
	std::cout << "array_out[" << N << "] = ";
	for(int i=0; i<N; i++){
		if(i < N-1)
			std::cout << array_out[i] << ", ";
		else	
			std::cout << array_out[i] << std::endl;
	}




}
