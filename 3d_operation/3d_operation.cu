//This program calculate the distances of the points to a constant point
#include <iostream>
#include "cuda.h"
using namespace std;

#define N 512
struct Point3D
{
	float x;
	float y;
	float z;
};


__device__ float distance3D(Point3D p1, Point3D p2)
{
	return sqrtf( (p1.x - p2.x)*(p1.x - p2.x) + 
		     (p1.y - p2.y)*(p1.y - p2.y) + 
                     (p1.z - p2.z)*(p1.z - p2.z) );
}

__global__ void distKernel(float *dDistArray, Point3D *dpfocus, Point3D *dpArray)
{
	const int block_id = (blockDim.x * blockDim.y * blockDim.z) * (gridDim.x * gridDim.y * blockIdx.z + gridDim.x * blockIdx.y + blockIdx.x);
	const int thread_id = block_id + blockDim.x * blockDim.y * threadIdx.z + blockDim.x * threadIdx.y + threadIdx.x;
	//dDistArray[thread_id] = thread_id;
	dDistArray[thread_id] = distance3D(*dpfocus , dpArray[thread_id]);
	//dDistArray[0] = 1;
}

int main(void)
{
	dim3 grid(2,2,2), block(2,4,8);
	
	//float array to store distances
	float *distArray=NULL;
	float *dDistArray=NULL;
	distArray = (float*)malloc(N * sizeof(float));
	cudaMalloc((void**)&dDistArray, N * sizeof(float));
	
	//points array to store points
	Point3D *pArray=NULL;
	Point3D *dpArray=NULL;
	pArray = (Point3D*)malloc(N * sizeof(Point3D));
	for(int i=0; i<N; i++){
		pArray[i].x = i;
		pArray[i].y = i+1;
		pArray[i].z = i+2;
	}
	cudaMalloc((void**)&dpArray, N * sizeof(Point3D));
	cudaMemcpy(dpArray, pArray, N * sizeof(Point3D), cudaMemcpyHostToDevice);
	
	//point to store the focus
	Point3D *pfocus = NULL;
	Point3D *dpfocus = NULL;
	pfocus = (Point3D*)malloc(sizeof(Point3D));
	pfocus->x = 1;
       	pfocus->y = 1;
	pfocus->z = 1;
	cudaMalloc((void**)&dpfocus, sizeof(Point3D));
	cudaMemcpy(dpfocus, pfocus, sizeof(Point3D), cudaMemcpyHostToDevice);

	//point points
	cout << "Focus point is: "
		<< "{" << pfocus->x << "," << pfocus->y << "," << pfocus->z << "}" << endl; 
	cout << "Points are:" << endl;
	for(int i=0; i<N; i++)
		cout << "{" << pArray[i].x << "," << pArray[i].y << "," << pArray[i].z << "}" << endl;

	//launch kernel
	distKernel<<<grid,block>>>(dDistArray, dpfocus, dpArray);
	cudaDeviceSynchronize();

	//copy distances back
	cudaMemcpy(distArray, dDistArray, N * sizeof(float), cudaMemcpyDeviceToHost);
	cout << "Distances are:" << endl;
	for(int i=0; i<N; i++)
		cout << "distArray[" << i << "] = " << distArray[i] << endl;


}


