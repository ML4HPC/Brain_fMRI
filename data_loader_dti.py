import os
import numpy as np
import argparse

def create_dataset(path, output_dir):
    train_target = np.load(os.path.join(path, 'csv_train_target_dti.npy'), allow_pickle=True)
    valid_target = np.load(os.path.join(path, 'csv_valid_target_dti.npy'), allow_pickle=True)
    test_target = np.load(os.path.join(path, 'csv_test_target_dti.npy'), allow_pickle=True)

    all_img_fa = np.load(os.path.join(path, 'all_img_FA.npy'), allow_pickle=True)
    all_img_md = np.load(os.path.join(path, 'all_img_MD.npy'), allow_pickle=True)

    train_data_img_fa = []
    valid_data_img_fa = []
    test_data_img_fa = []
    train_data_img_md = []
    valid_data_img_md = []
    test_data_img_md = []

    train_data_target = []
    valid_data_target = []
    test_data_target = []


    for key in train_target.item().keys():
        if key in all_img_fa.item().keys():
            train_data_img_fa.append(all_img_fa.item()[key])
            train_data_target.append(np.float(train_target.item()[key]))
        
        if key in all_img_md.item().keys():
            train_data_img_md.append(all_img_md.item()[key])


    for key in valid_target.item().keys():
        if key in all_img_fa.item().keys():
            valid_data_img_fa.append(all_img_fa.item()[key])
            valid_data_target.append(np.float(valid_target.item()[key]))

        if key in all_img_md.item().keys():
            valid_data_img_md.append(all_img_md.item()[key])

    for key in test_target.item().keys():
        if key in all_img_fa.item().keys():
            test_data_img_fa.append(all_img_fa.item()[key])
            test_data_target.append(np.float(test_target.item()[key]))

        if key in all_img_md.item().keys():
            test_data_img_md.append(all_img_md.item()[key])

    assert(len(train_data_img_fa) == len(train_data_img_md))
    assert(len(valid_data_img_fa) == len(valid_data_img_md))
    assert(len(test_data_img_fa) == len(test_data_img_md))

    assert(len(train_data_img_fa) == len(train_data_target))
    assert(len(valid_data_img_fa) == len(valid_data_target))
    assert(len(test_data_img_fa) == len(test_data_target))

    np.save(os.path.join(output_dir, 'train_data_img_fa.npy'), train_data_img_fa)
    np.save(os.path.join(output_dir, 'train_data_img_md.npy'), train_data_img_md)
    np.save(os.path.join(output_dir, 'valid_data_img_fa.npy'), valid_data_img_fa)
    np.save(os.path.join(output_dir, 'valid_data_img_md.npy'), valid_data_img_md)
    np.save(os.path.join(output_dir, 'test_data_img_fa.npy'), test_data_img_fa)
    np.save(os.path.join(output_dir, 'test_data_img_md.npy'), test_data_img_md)

    np.save(os.path.join(output_dir, 'train_data_target.npy'), train_data_target)
    np.save(os.path.join(output_dir, 'valid_data_target.npy'), valid_data_target)
    np.save(os.path.join(output_dir, 'test_data_target.npy'), test_data_target)

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



