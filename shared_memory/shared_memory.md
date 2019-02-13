# Shared Memory in multiple blocks


## Goal
```
Input:  0, 1, 2, 3, ..., 31, 32, ..., 62, 63
Algorithm: 0*63, 1*62, 2*61, ..., 31*32, 32*31, ..., 62*1, 63*0
Output: 0, 62, 122, 992, 992, ..., 62, 0

```

## One Single Block
```
__global__ void myKernel(int *outArray, int *inArray)
{
    __shared__ int sharedArray[64];
    const int tid = blockDim.x * threadIdx.y + threadIdx.x;
    sharedArray[tid] = inArray[tid];
    __syncthreads();

    outArray[tid] = sharedArray[tid] * sharedArray[63-tid];
}
...
...
dim3 grid(1,1,1), block(8,8,1);
myKernel<<<grid,block>>>(outArray,inArray)   //inArray[64] = {0, 1, 2, 3, 4, ..., 63}


//outArray[64] = {0, 62, 122, 992, 992, ..., 62, 0}

```

## Multiple Blocks
```
__global__ void myKernel(int *outArray, int *inArray)
{
    __shared__ int sharedArray[64];
    const int tid = blockDim.x * threadIdx.y + threadIdx.x;
    sharedArray[tid] = inArray[tid];
    __syncthreads();

    outArray[tid] = sharedArray[tid] * sharedArray[63-tid];
}
...
...
dim3 grid(2,1,1), block(4,8,1);
myKernel<<<grid,block>>>(outArray,inArray)   //inArray[64] = {0, 1, 2, 3, 4, ..., 63}


//outArray[64] = {0, 0, ...., 0}

```
In the multiple blocks (2 blocks in this case):
1) each blocks has allocated a 64 unit sharedArray 
2) thread id is 0~31 for the first block, 32~63 for the second block
Therefore, in the first block for example, "sharedArray[tid] = inArray[tid]" only accounts for tid in [0,31], and sharedArray[63-tid] is always ZERO
