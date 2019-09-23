import os
import csv
import numpy as np
import argparse
from readcsv import readcsv


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read target csv for dti')
    parser.add_argument('--data_dir', help='Path to dataset images')
    parser.add_argument('--output_dir', help='Path to directory for saving outputs')
    args = parser.parse_args()

    train = "training_fluid_intelligence_sri.csv"
    valid = "validation_fluid_intelligence_sri.csv"
    test = "test_fluid_intelligence_sri.csv"

    if not os.path.isdir(args.output_dir):
        try:
            os.mkdir(args.output_dir)
        except:
            raise Exception('Could not create output directory')
    
    csv_train = readcsv(args.data_dir, train)
    csv_valid = readcsv(args.data_dir, valid)
    csv_test = readcsv(args.data_dir, test)

    for key, value in csv_train.items():
        print(key, value)

    for key, value in csv_valid.items():
        print(key, value)
    
    for key, value in csv_test.items():
        print(key, value)


    print('saving train dict!')
    np.save(os.path.join(args.output_dir, 'csv_train_target_vol.npy'), csv_train)
    print('saving valid dict!')
    np.save(os.path.join(args.output_dir, 'csv_valid_target_vol.npy'), csv_valid)
    print('saving test dict!')
    np.save(os.path.join(args.output_dir, 'csv_test_target_vol.npy'), csv_valid)
    
