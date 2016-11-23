mpic++ -c mpi_mult_matriz.c -o mpi_mult_matriz.o
/usr/local/cuda/bin/nvcc -c cuda_mult_matriz.cu -o cuda_mult_matriz.o -Wno-deprecated-gpu-targets
mpic++ mpi_mult_matriz.o cuda_mult_matriz.o -o mpiWithCuda_mm -L/usr/local/cuda/lib64/ -lcudart
