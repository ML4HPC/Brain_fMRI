#!/bin/bash

if [ "$#" -lt 1 ]
then
    echo 'Provide output directory as command line argument!'
    exit 1
fi

train_bsz=4
valid_bsz=8
epoch=10
lr=0.001
momentum=0.5
weight_decay=0.001
outdir=$1
cwd=$(pwd)
cur_filepath="${cwd}/${0}"
rm -rf $outdir
python3 run_6_input_multi_output_resnet3d.py --data_dir=../data/processed/ --output_dir=$outdir --train_batch_size=$train_bsz --valid_batch_size=$valid_bsz --epoch=$epoch --optimizer=sgd --lr=$lr --momentum=$momentum --weight_decay=$weight_decay
cp $cur_filepath $outdir 


