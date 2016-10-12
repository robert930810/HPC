#include <stdio.h>
#include <cuda.h>
#include <time.h>

__global__ void  convolutionGPUkernel_1D(int *h_n, int *h_mascara,int *h_r,int n, int mascara){
  int mitadMascara= (mascara/2);
  int i = blockIdx.x * blockDim.x + threadIdx.x;
	if(i<n){
		int p=0;// almacena los valores temporales
    int k= i - mitadMascara;
		for (int j =0; j < mascara; j++){
      if(k < n  && k >= 0){
        p += h_n[k]*h_mascara[j];
      }
      else
        p+=0;
      k++;
		}
    h_r[i]=p;
	}
}
void convolutionCPU_1D(int *h_n,int *h_mascara,int *h_r,int n, int mascara){
	int mitadMascara= (mascara/2);
	for(int i=0;i<n;i++){
		int p=0;// almacena los valores temporales
    int k= i - mitadMascara;
		for (int j =0; j < mascara; j++){
      if(k < n  && k >= 0){
        p += h_n[k]*h_mascara[j];
      }
      else
        p+=0;
      k++;
		}
    h_r[i]=p;
	}
}

void inicializarVec(int *vec , int t){
  for(int i = 0; i < t ; i++)
    vec[i] = i + 1;
}

void imprimirVec(int *h_r,int n){
	for (int i = 0; i < n; i++)
		printf(" %d ",h_r[i]);
	printf("\n");
}

int main()
{
	int *h_n,*h_mascara,*h_r,*d_n,*d_mascara,*d_r,*h_result;
  clock_t start, end, startGPU, endGPU;
  double cpu_time_used, gpu_time_used;
	// dimensiones
	int n= 10, mascara = 5;

	// asignacion de memoria en el host
	h_n= (int*)malloc(n*sizeof(int));
	h_mascara= (int*)malloc(mascara*sizeof(int));
	h_r= (int*)malloc(n*sizeof(int));
  h_result= (int*)malloc(n*sizeof(int));
	//inicializacion
	inicializarVec(h_n,n);
	inicializarVec(h_mascara,mascara);
	imprimirVec(h_n,n);
	imprimirVec(h_mascara,mascara);
  start = clock();
	convolutionCPU_1D(h_n,h_mascara,h_r,n,mascara);
  end = clock();
  cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
  printf("Tiempo CPU: %.10f\n", cpu_time_used);

  imprimirVec(h_r,n);

  //asignacion de memoria en el device
  cudaMalloc((void**)&d_n,n*sizeof(int));
  cudaMalloc((void**)&d_mascara,mascara*sizeof(int));
  cudaMalloc((void**)&d_r,n*sizeof(int));

  // se copian los valores de los vectores al device
  startGPU = clock();
  cudaMemcpy(d_n,h_n,n*sizeof(int),cudaMemcpyHostToDevice);
  cudaMemcpy(d_mascara,h_mascara,mascara*sizeof(int),cudaMemcpyHostToDevice);

  int blockSize = 4;
  dim3 dimBlock(blockSize, 1, 1);
  dim3 dimGrid((n / blockSize) + 1, 1, 1);
  convolutionGPUkernel_1D<<< dimGrid, dimBlock >>>(d_n, d_mascara, d_r, n, mascara);

  cudaMemcpy(h_result,d_r,n*sizeof(int),cudaMemcpyDeviceToHost);
  endGPU = clock();
  gpu_time_used = ((double) (endGPU - startGPU)) / CLOCKS_PER_SEC;
  printf("Tiempo GPU : %.10f\n", gpu_time_used);

  imprimirVec(h_result,n);

	return 0;
}
