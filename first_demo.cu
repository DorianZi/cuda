#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>

#define N 9

__global__ void kernel(int *ptr)
{
	*ptr = *ptr + N;
}

int main(void)
{
    int computeMajor;
    int computeMinor;
    cudaDeviceGetAttribute(&computeMajor, cudaDevAttrComputeCapabilityMajor,0);
    cudaDeviceGetAttribute(&computeMinor, cudaDevAttrComputeCapabilityMinor,0);
    printf("Compute Capability: %d.%d\n", computeMajor, computeMinor);
    int *hptr;
    int *dptr;
    size_t size = sizeof(int);
    hptr = (int *)malloc(size);
    cudaMalloc((void **)&dptr, size);
    //memset(hptr, 1, 1);
    *hptr = 1;
    printf("%d + %d = ", *hptr, N);
    cudaMemcpy(dptr, hptr, size, cudaMemcpyHostToDevice);
    kernel<<<2,3>>>(dptr); 
    cudaMemcpy(hptr, dptr, size, cudaMemcpyDeviceToHost);
    printf("%d\n", *hptr);
    free(hptr);
    cudaFree(dptr);

}
