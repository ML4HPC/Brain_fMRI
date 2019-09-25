import os
import numpy as np
import argparse


def check_count_nan(path):
    filepath = os.path.join(path, 'train_data_img.npy')
    train_images = np.load(filepath, allow_pickle=True)

    nan_count = []
    nan_ratio = []

    for img in train_images:
        x = np.array(img.dataobj).flatten()
        nan_idx = np.where(np.isnan(x))
        cur_nan_count = nan_idx.shape[0]
        nan_count.append(cur_nan_count)
        cur_ratio = cur_nan_count / x.shape[0] * 100
        nan_ratio.append(cur_ratio)

    return nan_count, nan_ratio


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Check DTI data for NaN')
    parser.add_argument('--data_dir', help='Path to dataset images')
    args = parser.parse_args()

    nan_count, nan_ratio = check_count_nan(args.data_dir)
    