import os
import numpy as np
import argparse

def create_dataset(path, output_dir, mri_type):
    train_target = np.load(os.path.join(path, 'csv_train_target.npy'), allow_pickle=True)
    valid_target = np.load(os.path.join(path, 'csv_valid_target.npy'), allow_pickle=True)
    test_target = np.load(os.path.join(path, 'csv_test_target.npy'), allow_pickle=True)

    all_img = np.load(os.path.join(path, 'all_img_{}.npy'.format(mri_type)), allow_pickle=True)

    train_data_img = []
    valid_data_img = []
    test_data_img  = []

    train_data_target = []
    valid_data_target = []
    test_data_target = []

    train_subjects = []
    valid_subjects = []
    test_subjects = []

    for key in train_target.item().keys():
        if key in all_img.item().keys():
            train_data_img.append(all_img.item()[key])
            train_data_target.append([np.float(target) for target in train_target.item()[key]])
            train_subjects.append(key)

    for key in valid_target.item().keys():
        if key in all_img.item().keys():
            valid_data_img.append(all_img.item()[key])
            valid_data_target.append([np.float(target) for target in valid_target.item()[key]])
            valid_subjects.append(key)

    for key in test_target.item().keys():
        if key in all_img.item().keys():
            test_data_img.append(all_img.item()[key])
            test_data_target.append([np.float(target) for target in test_target.item()[key]])
            test_subjects.append(key)

    assert(len(train_data_img) == len(train_data_target))
    assert(len(valid_data_img) == len(valid_data_target))
    assert(len(test_data_img) == len(test_data_img))

    np.save(os.path.join(output_dir, 'train_data_img_{}.npy'.format(mri_type)), train_data_img)
    np.save(os.path.join(output_dir, 'valid_data_img_{}.npy'.format(mri_type)), valid_data_img)
    np.save(os.path.join(output_dir, 'test_data_img_{}.npy'.format(mri_type)), test_data_img)
    
    np.save(os.path.join(output_dir, 'train_data_target.npy'), train_data_target)
    np.save(os.path.join(output_dir, 'valid_data_target.npy'), valid_data_target)
    np.save(os.path.join(output_dir, 'test_data_target.npy'), test_data_target)

    np.save(os.path.join(output_dir, 'train_subjects.npy'), train_subjects)
    np.save(os.path.join(output_dir, 'valid_subjects.npy'), valid_subjects)
    np.save(os.path.join(output_dir, 'test_subjects.npy'), test_subjects)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read and process images')
    parser.add_argument('--data_dir', help='Path to dataset images')
    parser.add_argument('--output_dir', help='Path to directory for saving outputs')
    parser.add_argument('--mri_type', help='T1 / T2 / FA / MD / AD / RD')
    args = parser.parse_args()

    if not os.path.isdir(args.output_dir):
        try:
            os.mkdir(args.output_dir)
        except:
            raise Exception('Could not create output directory')

    print('Creating datasets: train, valid, and test for {}!'.format(args.mri_type))
    create_dataset(args.data_dir, args.output_dir, args.mri_type)



