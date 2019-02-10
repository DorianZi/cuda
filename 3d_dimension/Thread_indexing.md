#CUDA Thread Indexing

![Grid of Thread Blocks](https://docs.nvidia.com/cuda/cuda-c-programming-guide/graphics/grid-of-thread-blocks.png "Grid of Thread Blocks")

## 3-Dimension Programming Model
```
dim3 grid(2,4,1);    // Define 2*4*1 blocks in a grid with: gridDim.x=2, gridDim.y=4, gridDim.z=1
dim3 block(2,4,8);   // Define 2*4*8 threads in a block with: blockdDim.x=2, blockDim.y=4, blockDim.z=8
...
myKernel<<<grid,block>>>(...)

```

## 3-Dimension Thread Indexing in Kernel
```
__global__ void myKernel(...)
{
    int blockId = blockIdx.x + blockIdx.y * gridDim.x + gridDim.x * gridDim.y * blockIdx.z;
    int threadId = blockId * (blockDim.x * blockDim.y * blockDim.z)
                   + (threadIdx.z * (blockDim.x * blockDim.y))
                   + (threadIdx.y * blockDim.x) + threadIdx.x;
}
```


## Notes
Referring to [Doc](https://cs.calvin.edu/courses/cs/374/CUDA/CUDA-Thread-Indexing-Cheatsheet.pdf) 
