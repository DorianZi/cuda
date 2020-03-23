#include <iostream>
#include "opencv2/opencv.hpp"
#include "math.h"
using namespace std;
using namespace cv;

#define CUDACHECK(call) cudaCheck(call, #call, __FILE__, __LINE__)

inline void cudaCheck(cudaError_t err, const char *call, const char* file, const unsigned line){
	if(err != cudaSuccess){
		cout<<file<<"("<<line<<"): "<<call<<" failed with "<< "'"<<cudaGetErrorString(err)<<"'"<<endl;
		abort();
		exit(1);
	}
}

__constant__ int core_dev[2][9] =  { {-1, 0, 1, -2, 0, 2, -1, 0, 1},\
		            {-1,-2,-1,  0, 0, 0,  1, 2, 1} } ;

__global__ void compute_sobel(uchar* inptr, uchar* outptr, int cols, int rows){
	int p_id = blockDim.x * blockIdx.x + threadIdx.x;
	//        <cols>
	//        a b c 
	// <rows> d p f
	//        g h i
	if(p_id >= rows*cols) return;
	int val[9];
	val[0] = p_id - cols - 1>=0?  *(inptr + p_id - cols - 1):0 ;
	val[1] = p_id - cols>=0? *(inptr + p_id - cols):0;
	val[2] = p_id - cols + 1>=0? *(inptr + p_id - cols + 1):0;
	val[3] = p_id - 1>=0? *(inptr + p_id - 1):0 ;
	val[4] = *(inptr + p_id);
	val[5] = p_id + 1>=rows*cols? 0: *(inptr + p_id + 1) ;
	val[6] = p_id + cols - 1>=rows*cols? 0:*(inptr + p_id + cols - 1) ;
	val[7] = p_id + cols>=rows*cols? 0:*(inptr + p_id + cols) ;
	val[8] = p_id + cols + 1>=rows*cols? 0:*(inptr + p_id + cols + 1) ;

	int sum_1 = 0;
	int sum_2 = 0;
	int sum;
	for(int i=0 ; i<9 ; i++ ){
		sum_1 += val[i] * core_dev[0][i];
		sum_2 += val[i] * core_dev[1][i];
	}
	sum = sqrt((double)(sum_1*sum_1 + sum_2*sum_2));
	sum = (sum>=0 && sum<=255)? sum:(sum<0? 0:255);
	*(outptr+p_id) = sum;
}

int main(int argc, char** argvs){
	//int core_h[][9] = { {-1, 0, 1, -2, 0, 2, -1, 0, 1},\
		            {-1,-2,-1,  0, 0, 0,  1, 2, 1} } ;
	Mat image = imread(argvs[1],IMREAD_GRAYSCALE);
	Mat outimage = Mat::ones(image.rows, image.cols, CV_8UC1 );
	cout<<"channels = "<< image.channels()<<endl;
	uchar* inptr;
	uchar* outptr;
	
	CUDACHECK(cudaMalloc((void**)&inptr, sizeof(uchar) * image.cols * image.rows));
	CUDACHECK(cudaMemcpy(inptr, image.data, sizeof(uchar) * image.cols * image.rows,  cudaMemcpyHostToDevice));
	
	CUDACHECK(cudaMalloc((void**)&outptr, sizeof(uchar) * image.cols * image.rows));

        //CUDACHECK(cudaMemcpyToSymbol(core_dev[0], core_h[0], sizeof(int) * 9 *2));

	dim3 grid(1024,1,1);
        dim3 block(1024,1,1);
	compute_sobel<<<grid, block>>>(inptr, outptr, image.cols, image.rows);
	CUDACHECK(cudaGetLastError());
	CUDACHECK(cudaDeviceSynchronize());
	CUDACHECK(cudaMemcpy(outimage.data,outptr, sizeof(uchar) * image.cols * image.rows, cudaMemcpyDeviceToHost));
	CUDACHECK(cudaDeviceSynchronize());
	imshow("output",outimage);
	waitKey(0);
	return 0;
}
