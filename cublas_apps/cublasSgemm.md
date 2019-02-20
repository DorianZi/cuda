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
float *h_A[N] = {1,2,3,4,5,6};
float *A;
cudaMalloc((void**)A, N*sizeof(float));
cudaMemcpy(A, h_A, N*sizeof(float), cudaMemcpyHostToDevice);
```
If regarded as 2x3 Matrix:
1) h_A in CPU stands for a matrix of [1,2,3
                                      4,5,6]
2)   A in GPU stands for a matrix of [1,3,5
                                      2,4,6]

## Compute Matrix
We have arrays in GPU memory: 
  A = {1,2,3,4,5,6}
  B = {7,8,9,10,11,12,13,14,15,16,17,18}
And intend to compute the matrix :
[1,2,3    *   [ 7, 8, 9,10
 4,5,6]        11,12,13,14         
               15,16,17,18]
How to do it in cublas features?


### cublasSgemm: Compute with transposition
Consider the cublasSgemm function which is to compute C = alpha\*A\*B + beta\*C
Here we set alpha=1.0 and beta=0.0, so it's actually C = A\*B

```
// A = {1,2,3,4,5,6}
// A_ROW = 2, A_COL = 3
// B = {7,8,9,10,11,12,13,14,15,16,17,18}
// B_ROW = 3, B_COL = 4

float alpha=1.0;
float beta=0.0;
// C = A*B
cublasSgemm(
  handle,
  CUBLAS_OP_T,    // Do transposition of matrix A, so A = [1,2,3
                  //                                       4,5,6]
  CUBLAS_OP_T,    // Do transposition of matrix B, so B = [ 7, 8, 9,10
                  //                                       11,12,13,14
                  //                                       15,16,17,18]
  A_ROW,          // row count of C is 2
  B_COL,          // column count of C is4
  A_COL,          // column count of A is 3
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

### cublasSgemm: Compute with no transposition
with the theory of C^T = (B^T)*(A^T) , convert to compute to 
      [ 7,11,15        [ 1,4
C^T =   8,12,16    *     2,5
        9,13,17          3,6]
       10,14,18]
Still, we set alpha=1.0 and beta=0.0, to compute C^T = (B^T)*(A^T)

```
// A = {1,2,3,4,5,6}
// A_ROW = 2, A_COL = 3
// B = {7,8,9,10,11,12,13,14,15,16,17,18}
// B_ROW = 3, B_COL = 4

float alpha=1.0;
float beta=0.0;
// C^T = (B^T)*(A^T)
cublasSgemm(
  handle,
  CUBLAS_OP_N,    // NO transposition of matrix B, so B = [ 7,11,15       
                  //                                        8,12,16   
                  //                                        9,13,17 
                  //                                       10,14,18]
  CUBLAS_OP_N,    // No transposition of matrix A, so A = [ 1,4
                  //                                        2,5
                  //                                        3,6]
  B_COL,          // row count of C^T is 4
  A_ROW,          // column count of C^T is 2
  B_ROW,          // column count of B^T is 3
  &alpha,         // 1.0 
  B,              // matrix B
  B_COL,          // with no transpostion of matrix B, the leading dimension of B is its row count (original column count)
  A,              // matrix B
  A_COL,          // with no transpostion of matrix A, the leading dimension of A is its row count (original column count)
  &beta,          // 0
  C,              // matrix C
  B_COL           // The leading dimension of C is its row count (original column count)
);
```
