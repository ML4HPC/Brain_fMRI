if [$# -ne 1]
then
    echo 'Provide output directory as command line argument!'
    exit 1
fi

bsz=4
outdir=$1
rm -rf $outdir
python3 run_multi_resnet3d.py --data_dir=../data_3d/ --output_dir=$outdir --train_batch_size=$bsz --valid_batch_size=4 --epoch=7 --optimizer=sgd --normalize==True --log=True
