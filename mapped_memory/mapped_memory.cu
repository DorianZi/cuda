#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <cuda.h>
#include <iostream>
using namespace std;

__global__ void kernel(volatile int *dptr, volatile int *dpwait)
{
	*dptr = 9;
	while(!*dpwait){}
	*dptr = 999;
}

int main(void)
{
    int *hptr;
    int *dptr;
    int *hpwait;
    int *dpwait;

    cudaHostAlloc((void **)&hpwait, sizeof(bool), cudaHostAllocMapped);
    cudaHostAlloc((void **)&hptr, sizeof(int), cudaHostAllocMapped);
    *hpwait = 0;
    *hptr = 0;
    cudaHostGetDevicePointer((void **)&dptr, hptr, 0);
    cudaHostGetDevicePointer((void **)&dpwait, hpwait, 0);
    
    cout << "Before: mem is "<< *hptr << endl;    
    kernel<<<1,1>>>((volatile int *)dptr, (volatile int *)dpwait); 

    sleep(3);
    cout << "In: mem is "<< *hptr << endl;

    sleep(3);
    *hpwait = 1;
    cudaDeviceSynchronize();
    cout << "After: mem is "<< *hptr << endl;

}
