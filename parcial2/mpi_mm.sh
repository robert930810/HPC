#!/bin/bash

#SBATCH --job-name=mpiWithCuda_mm
#SBATCH --output=mpiWithCuda.out
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:1

export CUDA_VISIBLE_DEVICES=0
mpirun mpiWithCuda_mm

