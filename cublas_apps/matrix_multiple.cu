#include <iostream>
#include "cuda.h"
#include "cublas_v2.h"
#include "cublas_api.h"
#define M 5
#define N 3
#define K 2

int main()
{
	float* h_A;
	float* h_B;
	float* h_C;
	float* d_A;
	float* d_B;
	float* d_C;
	float alpha = 1.0;
	float beta = 0.0;
	h_A = (float*)malloc(M*N*sizeof(float));
	std::cout <<"h_A[M*N] = ";
	for(int i=0; i<M*N; i++){
		h_A[i] = rand()%10;
		std::cout<< h_A[i] << " ";
	}
	std::cout << std::endl;
	cudaMalloc((void**)&d_A, M*N*sizeof(float));
	cudaMemcpy(d_A, h_A, M*N, cudaMemcpyHostToDevice);

	std::cout <<"h_B[N*K] = ";
	h_B = (float*)malloc(N*K*sizeof(float));
	for(int i=0; i<N*K; i++){
		h_B[i] = rand()%10;
		std::cout<< h_B[i] << " ";
	}
	std::cout << std::endl;
	cudaMalloc((void**)&d_B, N*K*sizeof(float));
	cudaMemcpy(d_B, h_B, N*K, cudaMemcpyHostToDevice);

	h_C = (float*)malloc(M*K*sizeof(float));
	cudaMalloc((void**)&d_C, M*K*sizeof(float));


	// start cublas handle
	cublasHandle_t handle;
	cublasCreate(&handle);
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, M, N, K, &alpha, d_A, M, d_B, K, &beta, d_C, M);
	cudaMemcpy(h_C, d_C, M*K, cudaMemcpyDeviceToHost);
	std::cout <<"h_C[M*K] = ";
	for(int i=0; i<M*K; i++)
		std::cout << h_C[i] << " ";
        std::cout << std::endl;

	free(h_A);
	free(h_B);
	free(h_C);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	cublasDestroy(handle);

}
