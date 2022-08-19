#!/bin/bash

#PBS -l select=1:ncpus=1:mem=1mb:gpu_type=P100
#PBS -l walltime=00:00:10
#PBS -q short_gpuQ

module load cuda-11.1
module load gcc91

$HOME/aca-cuda/build/app $HOME/cat.jpg
