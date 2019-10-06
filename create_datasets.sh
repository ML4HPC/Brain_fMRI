#!/bin/bash

csv_data_dir="/home/seungwook/data/intelligence/"
struct_data_dir="/home/seungwook/data/data_T1_T2_201909/"
dti_data_dir="/home/seungwook/data/data_DTI_tensor/"
output_dir="/home/seungwook/data/processed/"

for mri_type in T1 T2
do
    echo $mri_type
    python3 readcsv.py --data_dir=$csv_data_dir --output_dir=$output_dir
    python3 readimage.py --data_dir=$struct_data_dir --output_dir=$output_dir --mri_type=$mri_type
    python3 data_loader.py --data_dir=$output_dir --output_dir=$output_dir --mri_type=$mri_type
done

for mri_type in MD FA AD RD
do
    echo $mri_type
    python3 readcsv.py --data_dir=$csv_data_dir --output_dir=$output_dir
    python3 readimage.py --data_dir=$dti_data_dir --output_dir=$output_dir --mri_type=$mri_type
    python3 data_loader.py --data_dir=$output_dir --output_dir=$output_dir --mri_type=$mri_type
done


