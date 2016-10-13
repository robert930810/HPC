#include <cmath>
#include <cuda.h>
#include <cv.h>
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <stdio.h>
#include <time.h>

using namespace cv;
using namespace std;
#define CHANNELS 3

__global__ void  convolutionSobelGPUkernel(unsigned char *M, char *d_Gx, char *d_Gy, unsigned char *resultado,int m, int n, int widthM){
  int col = blockIdx.x*blockDim.x + threadIdx.x;
  int row = blockIdx.y*blockDim.y + threadIdx.y;

  if(col < n && row < m){
    int Gx = 0, Gy = 0;
    int start_col = col - (widthM/2);
    int start_row = row - (widthM/2);

    for (int i = 0; i < widthM ; i++) {
      for (int j = 0; j < widthM; j++) {
        int curRow = start_row + i;
        int curCol = start_col + j;
        if(curRow > -1 && curRow < m && curCol > -1 && curCol < n){
          Gx += M[curRow*n + curCol]*d_Gx[i*widthM + j];
          Gy += M[curRow*n + curCol]*d_Gy[i*widthM + j];
        }
      }
    }

    if(Gx < 0)
      Gx = 0;
    else{
    if(Gx > 255)
      Gx = 255;
    }

    if(Gy < 0)
      Gy = 0;
    else{
    if(Gy > 255)
      Gy = 255;
    }

    resultado[row*n + col] = (unsigned char)sqrtf((Gx * Gx) + (Gy * Gy));
  }
}


int main(int argc, char **argv) {

  Mat img;
  img = imread("./inputs/img4.jpg", CV_LOAD_IMAGE_COLOR); // cargamos img
  Size s = img.size();

  // mascaras para el filtro de Sobel
  int maskwidth = 3;
  char h_Gx[] = {-1, 0, 1, -2, 0, 2, -1, 0, 1};
  char h_Gy[] = {-1, -2, -1, 0, 0, 0, 1, 2, 1};
  // Convirtiendo imagen en escala de grises con openCV
  Mat grayImg;// solo se trabaja con una imagen en escala de grises
  cvtColor(img, grayImg, CV_BGR2GRAY);
  Mat grad_x, abs_grad_x;
  //aplicamos el filtro de sobel con openCV
  Sobel(grayImg, grad_x, CV_8UC1, 1, 0, 3, 1, 0, BORDER_DEFAULT);
  convertScaleAbs(grad_x, abs_grad_x);

	unsigned char *h_imgSobel;
  h_imgSobel = (unsigned char *)malloc(sizeof(unsigned char) * s.width * s.height);

  char *d_Gx,*d_Gy;
  unsigned char *d_img,*d_imgSobel;

  cudaMalloc((void**)&d_Gx,(maskwidth*maskwidth)*sizeof(char));
  cudaMalloc((void**)&d_Gy,(maskwidth*maskwidth)*sizeof(char));
  cudaMalloc((void**)&d_img,(s.width*s.height)*sizeof(unsigned char));
  cudaMalloc((void**)&d_imgSobel,(s.width*s.height)*sizeof(unsigned char));

  cudaMemcpy(d_Gx,h_Gx,(maskwidth*maskwidth)*sizeof(char),cudaMemcpyHostToDevice);
  cudaMemcpy(d_Gy,h_Gy,(maskwidth*maskwidth)*sizeof(char),cudaMemcpyHostToDevice);
  cudaMemcpy(d_img,grayImg.data,(s.width*s.height)*sizeof(unsigned char),cudaMemcpyHostToDevice);

  int blockSize = 32;
  dim3 DimGrid(ceil(s.width/float(blockSize)), ceil(s.height/float(blockSize)), 1);
  dim3 DimBlock(blockSize,blockSize,1);

  convolutionSobelGPUkernel<<<DimGrid,DimBlock>>>(d_img,d_Gx,d_Gy,d_imgSobel,s.height,s.width,maskwidth);

  cudaMemcpy(h_imgSobel,d_imgSobel,(s.width*s.height)*sizeof(unsigned char),cudaMemcpyDeviceToHost);

  // Generando la imagen de salida
  Mat imgSobelCuda;
  imgSobelCuda.create(s.height, s.width, CV_8UC1);
  imgSobelCuda.data = h_imgSobel;


  // Guardando la imagen generada por CUDA
  imwrite("./outputs/1088310731.png", imgSobelCuda);

  // Guardando la imagen generada por openCV
  //imwrite("./outputs/1088318976.png", abs_grad_x);
  cout << "La imagen esta lista." << std::endl;
  return 0;
}