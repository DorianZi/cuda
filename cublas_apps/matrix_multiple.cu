// 1 2    1 2       7 10
// 3 4 *  3 4  =   15 22
// 2 6             20 28
#include <iostream>
#include "cuda.h"
#include "cublas_v2.h"
#include "cublas_api.h"
#define M 3
#define N 2
#define K 2

int main()
{
	float h_A[M*N] = {1.f,2.f,3.f,4.f,2.f,6.f};
	float h_B[N*K] = {1.f,2.f,3.f,4.f};
	float* h_C;
	float* h_C_T;
	float* d_A;
	float* d_B;
	float* d_C;
	float alpha = 1.0f;
	float beta = 0.0f;
	//h_A = (float*)malloc(M*N*sizeof(float));

	std::cout <<"A = " <<std::endl;
	for(int i=0; i<M*N; i++){
		//h_A[i] = rand()%10;
		std::cout<< h_A[i] << " ";
		if((i+1)%N == 0)
			std::cout << std::endl;
	}
	std::cout << std::endl;
	cudaMalloc((void**)&d_A, M*N*sizeof(float));
	cudaMemcpy(d_A, h_A, M*N*sizeof(float), cudaMemcpyHostToDevice);

	std::cout <<"B = " << std::endl;
	//h_B = (float*)malloc(N*K*sizeof(float));
	for(int i=0; i<N*K; i++){
		//h_B[i] = rand()%10;
		std::cout<< h_B[i] << " ";
		if((i+1)%K == 0)
			std::cout << std::endl;
	}
	std::cout << std::endl;
	cudaMalloc((void**)&d_B, N*K*sizeof(float));
	cudaMemcpy(d_B, h_B, N*K*sizeof(float), cudaMemcpyHostToDevice);

	h_C = (float*)malloc(M*K*sizeof(float));
	h_C_T = (float*)malloc(M*K*sizeof(float));
	cudaMalloc((void**)&d_C, M*K*sizeof(float));
	
	// start cublas handle
	cublasHandle_t handle;
	cublasCreate(&handle);

	// With no transposition
	cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, K, M, N, &alpha, d_B, K, d_A, N, &beta, d_C, K);
	cudaMemcpy(h_C, d_C, M*K*sizeof(float), cudaMemcpyDeviceToHost);
	std::cout <<"[No Transposition]A*B = " << std::endl;
	for(int i=0; i<M*K; i++){
		std::cout << h_C[i] << " ";
	}
        std::cout << std::endl;
	
	// With transposition
	cublasSgemm(handle, CUBLAS_OP_T, CUBLAS_OP_T, M, K, N, &alpha, d_A, N, d_B, K, &beta, d_C, M);
	cudaMemcpy(h_C_T, d_C, M*K*sizeof(float), cudaMemcpyDeviceToHost);
	std::cout <<"[Transposition]A*B =" << std::endl;
	for(int i=0; i<M*K; i++){
		std::cout << h_C_T[i] << " ";
	}
        std::cout << std::endl;

//	free(h_A);
//	free(h_B);
	free(h_C);
	free(h_C_T);
	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);
	cublasDestroy(handle);

}
