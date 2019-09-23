import os
import numpy as np
import argparse

def create_dataset(path, output_dir):
    train_target = np.load(os.path.join(path, 'csv_train_target.npy'), allow_pickle=True)
    valid_target = np.load(os.path.join(path, 'csv_valid_target.npy'), allow_pickle=True)
    test_target = np.load(os.path.join(path, 'csv_test_target.npy'), allow_pickle=True)

    train_img = np.load(os.path.join(path, 'train_img.npy'), allow_pickle=True)
    valid_img = np.load(os.path.join(path, 'valid_img.npy'), allow_pickle=True)
    test_img = np.load(os.path.join(path, 'test_img.npy'), allow_pickle=True)

    train_data_img = []
    valid_data_img = []
    test_data_img = []
    train_data_target = []
    valid_data_target = []
    test_data_target = []

    for key in train_target.item().keys():
        if key in train_img.item().keys():
            train_data_img.append(train_img.item()[key])
            train_data_target.append(np.float(train_target.item()[key]))

    for key in valid_target.item().keys():
        if key in valid_img.item().keys():
            valid_data_img.append(valid_img.item()[key])
            valid_data_target.append(np.float(valid_target.item()[key]))
            #valid_data[key] = [valid_img.item()[key], valid_target.item()[key]]
            #print(valid_data[key])
            #print(valid_data[key][0].shape)

    for key in test_target.item().keys():
        if key in test_img.item().keys():
            test_data_img.append(test_img.item()[key])
            test_data_target.append(np.float(test_target.item()[key]))

    assert(len(train_data_img) == len(train_data_target))
    assert(len(valid_data_img) == len(valid_data_target))
    assert(len(test_data_img) == len(test_data_target))

    np.save(os.path.join(output_dir, 'train_data_img.npy'), train_data_img)
    np.save(os.path.join(output_dir, 'valid_data_img.npy'), valid_data_img)
    np.save(os.path.join(output_dir,'test_data_img.npy'), test_data_img)
    np.save(os.path.join(output_dir, 'train_data_target.npy'), train_data_target)
    np.save(os.path.join(output_dir,'valid_data_target.npy'), valid_data_target)
    np.save(os.path.join(output_dir,'test_data_target.npy'), test_data_target)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read and process images')
    parser.add_argument('--data_dir', help='Path to dataset images')
    parser.add_argument('--output_dir', help='Path to directory for saving outputs')
    args = parser.parse_args()

    if not os.path.isdir(args.output_dir):
        try:
            os.mkdir(args.output_dir)
        except:
            raise Exception('Could not create output directory')

    print('Creating datasets: train, valid, and test!')
    create_dataset(args.data_dir, args.output_dir)
