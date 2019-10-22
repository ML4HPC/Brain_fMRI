#!/bin/bash

if [ "$#" -lt 2 ]
then
    echo 'Provide output directory as command line argument!'
    exit 1
fi

train_bsz=3
valid_bsz=10
epoch=10
normalize=True
optim=adam
lr=0.001
weight_decay=0
outdir=$1
mri_type=$2
site=[16]
cwd=$(pwd)
cur_filepath="${cwd}/${0}"
rm -rf $outdir
python3 run_1_input_multi_output_site_resnet3d.py --data_dir=../data/processed_norm/ --output_dir=$outdir --train_batch_size=$train_bsz --valid_batch_size=$valid_bsz --epoch=$epoch --normalize=$normalize --optimizer=$optim --lr=$lr --weight_decay=$weight_decay --mri_type=$mri_type --site=$site
cp $cur_filepath $outdir 


