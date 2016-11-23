#include "mpi.h"
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

//cuda = 0 MPI = 1
#define CudaOrMPI 1

#define ROWA 17000
#define COLA 17000
#define COLB 17000
#define MASTER 0               /* taskid of first task */
#define FROM_MASTER 1          /* setting a message type */
#define FROM_WORKER 2          /* setting a message type */

void cuda_mult_matriz(double *h_a,double *h_b, double *h_c,int ROWS, int COL_A, int COL_B);

void mpi_mult_matriz(double *a, double *b,double *c, int ARows,int ACols, int BCols){
double multResult;
  for(int k=0;k<BCols;k++){
     for(int i=0;i<ARows;i++){
        multResult= 0;
        for(int j=0;j<ACols;j++){
           multResult+=a[ACols*i+j]*b[j*BCols+i];
        }
        c[i*BCols+k]= multResult;
     }
  }
}

bool compareWidth(double *MPI, double *CUDA, int rows, int cols) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      if (MPI[i * cols + j] != CUDA[i * cols + j])
        return false;
    }
  }
  return true;
}

bool compareTo(double *mat, int rows, int cols) {
  for (int i = 0; i < rows; i++) {
    for (int j = 0; j < cols; j++) {
      if (mat[i * cols + j] != COLA)
        return false;
    }
  }
  return true;
}


int main (int argc, char *argv[])
{
   int   numtasks,              /* number of tasks in partition */
         taskid,                /* a task identifier */
         numworkers,            /* number of worker tasks */
         source,                /* task id of message source */
         dest,                  /* task id of message destination */
         mtype,                 /* message type */
         nRows,                  /* nRows of matrix A sent to each worker */
         averow, extra, offset,nElements, /* used to determine nRows sent to each worker */
         i, j, k, rc;           /* misc */
   double *a,*b, *c, multResult;


   MPI_Status status;

   MPI_Init(&argc,&argv);
   MPI_Comm_rank(MPI_COMM_WORLD,&taskid);
   MPI_Comm_size(MPI_COMM_WORLD,&numtasks);
   if (numtasks < 1 ) {
      printf("Need at least two MPI tasks. Quitting...\n");
     MPI_Abort(MPI_COMM_WORLD, rc);
     exit(1);
   }
   numworkers = numtasks-1;

   clock_t start,end;
   double mpi_time;
   

/**************************** master task ************************************/
   if (taskid == MASTER)
   {

       a = (double*)malloc(COLA*ROWA*sizeof(double));
 	     b = (double*)malloc(COLA*COLB*sizeof(double));
	     c = (double*)malloc(ROWA*COLB*sizeof(double));
       printf("mpi_mm has started with %d tasks.\n",numtasks);
       printf("Initializing arrays...\n");

       for (i=0; i<ROWA*COLA; i++){
           a[i]= 1;
       }

       for (i=0; i<COLB*COLA; i++){
           b[i]= 1;
       }
       
      start = clock();
      if(numworkers ==0){
	printf("trabajando con un solo nodo \n");
          if(CudaOrMPI == 0){
            cuda_mult_matriz(a,b,c,ROWA,COLA,COLB);
          }else if(CudaOrMPI == 1){
            mpi_mult_matriz(a,b,c,ROWA,COLA,COLB);
          }else{
            printf("Error funcion no definida \n");
          }
      }else{

        /* Send vector data to the worker tasks */
        averow = ROWA/numworkers;
        extra = ROWA%numworkers;
        offset = 0;
        mtype = FROM_MASTER;
        for (dest=1; dest<=numworkers; dest++)
        {
           nRows = (dest <= extra) ? averow+1 : averow;
           MPI_Send(&offset, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);
           MPI_Send(&nRows, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);

  		     nElements=nRows*COLA;

  		     MPI_Send(&nElements, 1, MPI_INT, dest, mtype, MPI_COMM_WORLD);

           MPI_Send(&a[offset*COLA], nElements, MPI_DOUBLE, dest, mtype, MPI_COMM_WORLD);

           MPI_Send(b, COLA*COLB, MPI_DOUBLE, dest, mtype, MPI_COMM_WORLD);
           offset = offset + nRows;
        }

        /* Receive results from worker tasks */
        mtype = FROM_WORKER;
        for (i=1; i<=numworkers; i++)
        {
           source = i;
           MPI_Recv(&offset, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
           MPI_Recv(&nRows, 1, MPI_INT, source, mtype, MPI_COMM_WORLD, &status);
           MPI_Recv(&c[offset*COLB], nRows*COLB, MPI_DOUBLE, source, mtype, MPI_COMM_WORLD, &status);
        }
      }
      end = clock();
      mpi_time = ((double) (end - start)) / CLOCKS_PER_SEC;

     /*
      printf("******************************************************\n");
      for (i=0; i<ROWA; i++)
      {
	     printf("\n");
	     for(j=0;j<COLB;j++){
         printf(" %.2f ", c[(i*COLB)+j]);
	       }
      }
	*/

      printf("\n******************************************************\n");
      if(CudaOrMPI == 0){
        printf("Tiempo suma matrices CUDA : %.10f\n", mpi_time);
      }else if(CudaOrMPI == 1){
        printf("Tiempo suma matrices MPI : %.10f\n", mpi_time);
      }else{
        printf("Error funcion no definida \n");
      }
      if(compareTo(c,ROWA,COLB)){
        printf (" == Successful Processing == \n");
      }else{
        printf ("ERROR:  Failed Processing  .\n");
      }

  	free(a);
  	free(b);
  	free(c);
   }


/**************************** worker task ************************************/
   if (taskid > MASTER)
   {

       mtype = FROM_MASTER;
       MPI_Recv(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
       MPI_Recv(&nRows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
       MPI_Recv(&nElements, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD, &status);
       double *matBuffA= (double*)malloc(nElements*sizeof(double));
	     double *matBuffB = (double*)malloc(COLA*COLB*sizeof(double));
	     double *matBuffC = (double*)malloc(nRows*COLB*sizeof(double));


      MPI_Recv(matBuffA, nElements, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD, &status);
       MPI_Recv(matBuffB, COLA*COLB, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD, &status);
      if(CudaOrMPI == 0){
        cuda_mult_matriz(matBuffA,matBuffB,matBuffC,nRows,COLA,COLB);
      }else if(CudaOrMPI == 1){
        mpi_mult_matriz(matBuffA,matBuffB,matBuffC,nRows,COLA,COLB);
      }else{
        printf("Error funcion no definida \n");
      }
      

       mtype = FROM_WORKER;
       MPI_Send(&offset, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
       MPI_Send(&nRows, 1, MPI_INT, MASTER, mtype, MPI_COMM_WORLD);
       MPI_Send(matBuffC, nRows*COLB, MPI_DOUBLE, MASTER, mtype, MPI_COMM_WORLD);
	     free(matBuffA);
	     free(matBuffB);
	     free(matBuffC);
}

   MPI_Finalize();

}
