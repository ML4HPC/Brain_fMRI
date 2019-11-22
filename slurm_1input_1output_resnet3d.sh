#!/bin/bash

if [ "$#" -lt 2 ]
then
    echo 'Provide output directory and mri_type as command line arguments!'
    exit 1
fi

#SBATCH -A v302955
#SBATCH -p volta
#SBATCH -N 1
#SBATCH --gres=gpu:4
outdir=$1
mri_type=$2
srun ./run_1_input_single_output_resnet3d.sh $1 $2