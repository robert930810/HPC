#include <stdio.h>
#include <cuda.h>
#include <time.h>

__global__ void  convolutionGPUkernel_2D(int *M, int *mascara,int *resultado,int m, int n, int widthM){
  int col = blockIdx.x*blockDim.x + threadIdx.x;
  int row = blockIdx.y*blockDim.y + threadIdx.y;

  if(col < n && row < m){
    int p = 0;
    int start_col = col - (widthM/2);
    int start_row = row - (widthM/2);
    for (int i = 0; i < widthM ; i++) {
      for (int j = 0; j < widthM; j++) {
        int curRow = start_row + i;
        int curCol = start_col + j;
        if(curRow > -1 && curRow < m && curCol > -1 && curCol < n){
          p += M[curRow*m + curCol]*mascara[i*widthM + j];
        }
      }
    }
    resultado[row*n + col] = p;
  }
}

void  convolutionCPU2D(int *M, int *mascara,int *resultado,int m, int n, int widthM){
  for(int row = 0;row<m;row++){
    for(int col=0;col<n;col++){
      int p = 0;
      int start_col = col - (widthM/2);
      int start_row = row - (widthM/2);
      for (int i = 0; i < widthM ; i++) {
        for (int j = 0; j < widthM; j++) {
          int curRow = start_row + i;
          int curCol = start_col + j;
          if(curRow > -1 && curRow < m && curCol > -1 && curCol < n){
            p += M[curRow*m + curCol]*mascara[i*widthM + j];
          }
        }
      }
      resultado[row*n + col] = p;
    }
  }
}

void inicializarMat(int *Ma , int m, int n){
  for(int i = 0; i <= m*n +1; i++){
    Ma[i] = 1;
  }
}

int printData(int *Mat, int m,int n, int tipo){

  if(tipo == 1)
    printf("================ Matriz ================ \n");
  if(tipo == 2)
    printf("================ Mascara ================ \n");
  if(tipo == 3)
    printf("================ Resultado ================ \n");

  for(int i = 0; i < m; ++i){
    for(int j = 0; j < n; ++j){
      printf("%d ", Mat[(i*m)+j]);
    }
    printf("\n");
  }
  printf("=============================\n\n");
  return 0;
}

int main()
{
  int *h_n,*h_mascara,*h_r,*d_n,*d_mascara,*d_r,*h_result;
  clock_t startGPU, endGPU,startCPU, endCPU;
  double gpu_time_used,cpu_time_used;
  // dimensiones
  int Mm=7, Mn= 7, mascaraWidth = 5;

  // asignacion de memoria en el host , matrices de Mm*Mn , mascaraM*mascaraN
  h_n= (int*)malloc((Mm*Mn)*sizeof(int));
  h_mascara= (int*)malloc((mascaraWidth*mascaraWidth)*sizeof(int));
  h_result= (int*)malloc((Mm*Mn)*sizeof(int));
  h_r= (int*)malloc((Mm*Mn)*sizeof(int));
  //inicializacion
  inicializarMat(h_n,Mm,Mn);
  inicializarMat(h_mascara,Mm,Mn);
  startCPU = clock();
  convolutionCPU2D(h_n,h_mascara,h_r,Mm,Mn,mascaraWidth);
  endCPU = clock();
  cpu_time_used = ((double) (endCPU - startCPU)) / CLOCKS_PER_SEC;

  cudaMalloc((void**)&d_n,(Mm*Mn)*sizeof(int));
  cudaMalloc((void**)&d_mascara,(mascaraWidth*mascaraWidth)*sizeof(int));
  cudaMalloc((void**)&d_r,(Mm*Mn)*sizeof(int));

  startGPU = clock();
  cudaMemcpy(d_n,h_n,(Mm*Mn)*sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(d_mascara,h_mascara,(mascaraWidth*mascaraWidth)*sizeof(int),cudaMemcpyHostToDevice);

  int blockSize = 4;
  dim3 dimBlock(blockSize, blockSize, 1);
  dim3 dimGrid((Mm / blockSize) + 1, (Mn / blockSize) + 1, 1);
  convolutionGPUkernel_2D<<< dimGrid, dimBlock >>>(d_n, d_mascara, d_r, Mm,Mn, mascaraWidth);

  cudaMemcpy(h_result,d_r,(Mm*Mn)*sizeof(int),cudaMemcpyDeviceToHost);
  endGPU = clock();
  printData(h_n,Mm,Mn,1);
  printData(h_mascara,mascaraWidth,mascaraWidth,2);
  printData(h_r,Mm,Mn,3);
  printData(h_result,Mm,Mn,3);

  gpu_time_used = ((double) (endGPU - startGPU)) / CLOCKS_PER_SEC;
  printf("Tiempo CPU : %.10f\n", cpu_time_used);
  printf("Tiempo GPU : %.10f\n", gpu_time_used);


  return 0;
}