# Matrix Transposition and Compute in cuBLAS 

## Matrix theory
```
   C = AB
=> C^T = (AB)^T
=> C^T = (B^T)(A^T)
=> C = ((B^T)(A^T))^T
```

## How is Matrix stored in GPU memory
```
#define N
float *h_A[N] = {1,2,3,4,5,6};
float *A;
cudaMalloc((void**)A, N*sizeof(float));
cudaMemcpy(A, h_A, N*sizeof(float), cudaMemcpyHostToDevice);

```
If regarded as 2x3 Matrix:
1) h_A stands for a matrix of [1,2,3
                               4,5,6]
2)   A stands for a matrix of [1,3,5
                               2,4,6]

## Compute C=A*B with cublasSgemm
When calling a compute function, it's able to achieve transposition of the matrix.
Consider the cublasSgemm function which is to compute C = alpha\*A\*B + beta\*C
Here we set alpha=1.0 and beta=0.0, so it's actually C = A\*B

```
// A = [1,3,5
//      2,4,6]
// A_ROW = 2, A_COL = 3
// B = [7,10,13,16
//      8,11,14,17
//      9,12,15,18]
// B_ROW = 3, B_COL = 4

float alpha=1.0;
float beta=0.0;
// C = A*B
cublasSgemm(
  handle,
  CUBLAS_OP_T,    // Do transposition of matrix A
  CUBLAS_OP_T,    // Do transposition of matrix B
  A_ROW,          // row count of C
  B_COL,          // column count of C
  A_COL,          // column count of A
  &alpha,         // 1.0 
  A,              // matrix A
  A_COL,          // with doing transpostion of matrix A, the leading dimension of A is its column count
  B,              // matrix B
  B_COL,          // with doing transpostion of matrix B, the leading dimension of B is its column count
  &beta,          // 0
  C,              // matrix C
  A_ROW           // The leading dimension of C is its column count
);
```
