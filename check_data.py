import os
import numpy as np
import argparse
import IPython


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
    
    np.save('nan_count.npy', nan_count)
    np.save('nan_ratio.npy', nan_ratio)

    return nan_count, nan_ratio

def check_dim(path, mri_type):
    correct_dim = None
    if mri_type == 'T1' or mri_type == 'T2':
        correct_dim = 256
    else:
        correct_dim = 190
    
    train_images = np.load(os.path.join(path, 'train_data_img_{}.npy'.format(mri_type)), allow_pickle=True)
    train_subjects = np.load(os.path.join(path, 'train_subjects.npy'), allow_pickle=True)

    valid_images = np.load(os.path.join(path, 'valid_data_img_{}.npy'.format(mri_type)), allow_pickle=True)
    valid_subjects = np.load(os.path.join(path, 'valid_subjects.npy'), allow_pickle=True)

    test_images = np.load(os.path.join(path, 'test_data_img_{}.npy'.format(mri_type)), allow_pickle=True)
    test_subjects = np.load(os.path.join(path, 'test_subjects.npy'), allow_pickle=True)

    train_err = []
    valid_err = []
    test_err = []

    for i in range(len(train_images)):
        x_shape = np.array(train_images[i].dataobj).astype(float, copy=False).shape()

        if x_shape[0] != correct_dim or x_shape[1] != correct_dim or x_shape[2] != correct_dim:
            train_err.append(train_subjects[i])
    
    for i in range(len(valid_images)):
        x_shape = np.array(valid_images[i].dataobj).astype(float, copy=False).shape()

        if x_shape[0] != correct_dim or x_shape[1] != correct_dim or x_shape[2] != correct_dim:
            valid_err.append(valid_subjects[i])
    
    for i in range(len(test_images)):
        x_shape = np.array(test_images[i].dataobj).astype(float, copy=False).shape()

        if x_shape[0] != correct_dim or x_shape[1] != correct_dim or x_shape[2] != correct_dim:
            test_err.append(test_subjects[i])
    

    return train_err, valid_err, test_err



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Check DTI data for NaN')
    parser.add_argument('--data_dir', help='Path to dataset images')
    parser.add_argument('--mri_type', default=None, help='MRI Type')
    args = parser.parse_args()

    # nan_count, nan_ratio = check_count_nan(args.data_dir)
    train_err, valid_err, test_err = check_dim(args.data_dir, args.mri_type)
    IPython.embed()

    
