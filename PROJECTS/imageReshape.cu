#include <stdio.h>
#include <iostream>
#include <opencv2/opencv.hpp>
using namespace std;
using namespace cv;

//      1111 1111 1111 11
//uchar   1   2    3    4
//       1111 1111 1111 11

__global__ void launchkernel(uchar* inptr, int rows, int cols, uchar* outptr, int out_rows, int out_cols){
	int thread = blockDim.x * blockIdx.x +  threadIdx.x;
	if(thread >= out_rows*out_cols) return ;
	int out_row = thread / out_cols;
	//   a  p_tmp1  b
	//      p
	//   c  p_tmp2  d
	for(int bias=0; bias < 3 ; bias++){
		int out_col = thread % out_cols;		
		float p_row = (float)rows / (float)out_rows * (float)out_row;
		float p_col = (float)cols / (float)out_cols * (float)out_col;
		int a_row = (int)p_row, a_col = (int)p_col;
		int b_row = a_row,      b_col = a_col+1;
		int c_row = a_row+1,    c_col = a_col;
		int d_row = a_row+1,    d_col = a_col+1;
		uchar a_val = *((uchar*)((char*)inptr + a_row * cols * 3) + a_col * 3 + bias);
		uchar b_val = *((uchar*)((char*)inptr + b_row * cols * 3) + b_col * 3 + bias);
		uchar c_val = *((uchar*)((char*)inptr + c_row * cols * 3) + c_col * 3 + bias);
		uchar d_val = *((uchar*)((char*)inptr + d_row * cols * 3) + d_col * 3 + bias);
		float p_tmp1_val = a_val + (p_col-a_col)/(float)(b_col-a_col)*(b_val - a_val);
		float p_tmp2_val = c_val + (p_col-c_col)/(float)(d_col-c_col)*(d_val - c_val);
		uchar p_val = uchar(p_tmp1_val + (p_row-a_row)/(float)(c_row-a_row)*(p_tmp2_val - p_tmp1_val));
		uchar* pPtr = (uchar*)((char*)outptr + out_row * out_cols * 3) + out_col * 3 + bias;

		*pPtr = p_val;
	}

} 

int main(int argc, char** argvs){
	cudaDeviceProp prop;
	cudaGetDeviceProperties(&prop,0);
	cout<<"==========GPU Info=========="<<endl;
	cout<<"maxThreadsPerBlock: " << prop.maxThreadsPerBlock<<endl;
	cout<<"maxThreadsDim[3]: " << "[ " <<prop.maxThreadsDim[0] << ", "
		                           <<prop.maxThreadsDim[1] << ", "
					   <<prop.maxThreadsDim[2] << " ]" << endl;
	cout<<"maxGridSize[3]: " << "[ " <<prop.maxGridSize[0] << ", "
		                      <<prop.maxGridSize[1] << ", "
		                      <<prop.maxGridSize[2] << " ]" << endl;
	cout<<"============================"<<endl;
	
	cv::Mat image= cv::imread(argvs[1]);
	cout<<"Input Image Size: "<<image.rows<<" * "<<image.cols<<endl;
	
	uchar *inptr;
	uchar *outptr;
	int times = atoi(argvs[2]);
	int out_cols = image.cols / times;
	int out_rows = image.rows / times;
	int imgByteSize = sizeof(uchar) * image.cols * 3 * image.rows;
	int outByteSize = sizeof(uchar) *  out_cols * 3 * out_rows;
	cudaMalloc(&inptr, imgByteSize);
	cudaMalloc(&outptr, outByteSize);
	cudaMemcpy(inptr, image.data, imgByteSize, cudaMemcpyHostToDevice);
	dim3 block(1024,1,1);
	dim3 grid(1024,1,1);	
	launchkernel<<<grid,block>>>(inptr, image.rows, image.cols, outptr, out_rows, out_cols);
	cudaDeviceSynchronize();
	
	Mat _dst = Mat::ones(out_rows,out_cols, CV_8UC3);
	cudaMemcpy(_dst.data, outptr, outByteSize, cudaMemcpyDeviceToHost);
	
	

	//launchkernel();
	cout<<"Output Image Size: "<<_dst.rows<<" * "<<_dst.cols<<endl;
	cv::imshow("Before", image);
	cv::imshow("After",_dst);
	cv::waitKey(0);
	return 0;
}
