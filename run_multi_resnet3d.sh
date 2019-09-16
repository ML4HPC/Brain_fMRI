bsz=6
outdir=./3d_multi_resnet_sgd_norm_log_batch_${bsz}_results/
rm -rf $outdir
python3 run_multi_resnet3d.py --data_dir=../data_3d/ --output_dir=$outdir --train_batch_size=$bsz --valid_batch_size=4 --epoch=7 --optimizer=sgd --normalize==True --log=True
