#!/bin/bash

if [ "$#" -lt 3 ]
then
    echo 'Provide all command line arguments!'
    exit 1
fi

valid_bsz=10
normalize=True
outdir=$1
mri_type=$2
saved_state=$3
target_idx=11
metric=MSE
cwd=$(pwd)
cur_filepath="${cwd}/${0}"
python3 eval_1_input_single_output_resnet3d.py --data_dir=../data/processed/ --output_dir=$outdir --saved_state=$saved_state --valid_batch_size=$valid_bsz --normalize=$normalize --mri_type=$mri_type --target_idx=$target_idx --metric=$metric
cp $cur_filepath $outdir 


