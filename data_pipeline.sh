#!/bin/bash

python3 readcsvtxt.py --path=../data_3d/
python3 readimage.py --data_dir=../data_3d/ --output_dir=./tmp/
mv ./tmp/* ./
rm -rf tmp/
python3 data_loader.py
mv *.npy ../data_3d/