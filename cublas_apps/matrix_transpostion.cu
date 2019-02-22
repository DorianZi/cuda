#include <iostream>
#include "time.h"
#include "cuda.h"
#include "cublas_v2.h"
#define ROW 6
#define COL 3 
#define N ROW*COL

int main()
{
	float* h_A = new float[N];
	srand(time(0));
	for(int i=0; i<N; i++)
	{
		h_A[i] = rand()%10;	
		std::cout << h_A[i] << " ";
		if ((i+1)%COL == 0 )
			std::cout << std::endl;
	}
	std::cout << std::endl;

	float* d_A;
	cudaMalloc((void**)&d_A,N*sizeof(float));
	cublasSetMatrix(ROW,COL,sizeof(*h_A),h_A,ROW,d_A,ROW );
        float* d_C; 
	cudaMalloc((void**)&d_C,N*sizeof(float));

	float alpha=1.0f;
	float beta=0.0f;
	cublasHandle_t handle;
	cublasCreate(&handle);
	cublasSgeam(handle, CUBLAS_OP_T, CUBLAS_OP_N, ROW, COL, &alpha, d_A, COL, &beta, d_A, ROW, d_C,ROW);// We need d_C instead of d_A as the result store because d_A as a input arg is a const type in function
	cublasDestroy(handle);
	//cudaMemcpy(h_A,d_A,N*sizeof(float),cudaMemcpyDeviceToHost);
	cublasGetMatrix(COL, ROW, sizeof(float), d_C, COL, h_A, COL);
	for(int i=0; i<N; i++)
		{
			std::cout << h_A[i] << " ";
			if ((i+1)%ROW == 0 )
				std::cout << std::endl;
		}
	std::cout << std::endl;

	delete [] h_A;


}
