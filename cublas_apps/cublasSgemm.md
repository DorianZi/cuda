# Matrix Transposition and Compute in cuBLAS 

## Matrix theory
```
   C = AB
=> C^T = (AB)^T
=> C^T = (B^T)(A^T)
=> C = ((B^T)(A^T))^T
```

## How is Matrix stored in CPU/GPU memory
```
#define N
float *h_A[N] = {1,2,3,4,2,6};
float *d_A;
cudaMalloc((void**)A, N*sizeof(float));
cudaMemcpy(d_A, h_A, N*sizeof(float), cudaMemcpyHostToDevice);
```
If regarded as 2x3 Matrix:
1) h_A in CPU stands for a matrix of 
       ```
       [1,2,3
        4,2,6]
       ```
2) d_A in GPU stands for a matrix of 
       ```
       [1,3,2
        2,4,6]
       ```
Therefore, they are completely different matrixes
To get the same matrix as CPU, we can, in GPU regard it as 3x2 Matrix:
d_A in GPU stands for a matrix of
       ```
       [1,4
        2,2
        3,6]
       ```
Then do transpostion: 
        ```
       [1,2,3
        4,2,6]
       ```


## Compute Matrix
We have arrays in GPU memory: 
```
  A = {1,2,3,4,2,6}
  B = {1,2,3,4}
```
And intend to compute the matrix :
```
[1, 2    *   [1, 2      [ 7, 10
 3, 4         3, 4]  =   15, 22    
 2, 6]                   20, 28] 
```
How to do it in cublas features?


### cublasSgemm: Compute with transposition
Consider the cublasSgemm function which is to compute 
```
C = alpha*A*B + beta*C
```
Here we set alpha=1.0 and beta=0.0, so it's actually
```
C = A*B
```

```
// A = {1,2,3,4,2,6}
// In CPU view, we define A_ROW = 3, A_COL = 2, the Matrix is:
//                         [1, 2   
//                          3, 4  
//                          2, 6]
// In GPU view, we define A_ROW_GPU = 2, A_COL_GPU = 3, the Matrix is: 
//                             [1, 3, 2   
//                              2, 4, 6]
// Same for B={1,2,3,4}


float alpha=1.0;
float beta=0.0;
// C = A*B
cublasSgemm(
  handle,
  CUBLAS_OP_T,    // Do transposition of matrix A before compute
  CUBLAS_OP_T,    // Do transposition of matrix B before compute
  A_ROW,          // row count of C is A_COL_GPU 
  B_COL,          // column count of C is B_ROW_GPU
  A_COL,          // The common count
  &alpha,         // 1.0 
  A,              // matrix A
  A_COL,          // = A_ROW_GPU
  B,              // matrix B
  B_COL,          // = B_ROW_GPU
  &beta,          // 0
  C,              // matrix C
  A_ROW           // = A_COL_GPU
);

// cublasSgemm returns C = {7 15 20 10 22 28} with size A_ROW*B_COL=3*2, which in GPU view is:
// [7  10
//  15 22
//  20 28]
```
After copy back to CPU, need to do a transposition


### cublasSgemm: Compute with no transposition
Still, we set alpha=1.0 and beta=0.0, to compute C^T = (B^T)*(A^T)

```
// A = {1,2,3,4,2,6}
// A_ROW = 2, A_COL = 3
// A_ROW_GPU = 3, A_COL_GPU = 2
// B = {1,2,3,4}
// B_ROW = 2, B_COL = 2
// B_ROW_GPU = 2, B_COL_GPU = 2

float alpha=1.0;
float beta=0.0;
// C^T = (B^T)*(A^T)
cublasSgemm(
  handle,
  CUBLAS_OP_N,    // NO transposition of matrix B 
  CUBLAS_OP_N,    // No transposition of matrix A
  B_COL,          // = B_ROW_GPU
  A_ROW,          // = A_COL_GPU
  B_ROW,          // the common count
  &alpha,         // 1.0 
  B,              // matrix B
  B_COL,          // = B_ROW_GPU
  A,              // matrix A
  A_COL,          // = A_ROW_GPU
  &beta,          // 0
  C,              // matrix C
  B_COL           // = B_ROW_GPU
);

// cublasSgemm returns C = {7 10 15 22 20 28} with size B_COL*A_ROW=2*3, which in GPU view is:
// [ 7 15 20
//  10 22 28]
```
After copy back to CPU, do NOT need to do a transposition
