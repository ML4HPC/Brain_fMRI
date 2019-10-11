#!/bin/bash

if [ "$#" -lt 2 ]
then
    echo 'Provide output directory as command line argument!'
    exit 1
fi

valid_bsz=10
normalize=True
outdir=$1
mri_type=$2
cwd=$(pwd)
cur_filepath="${cwd}/${0}"
python3 eval_1_input_multi_output_resnet3d.py --data_dir=../data/processed_resid/ --output_dir=$outdir --valid_batch_size=$valid_bsz --normalize=$normalize --mri_type=$mri_type
cp $cur_filepath $outdir 


