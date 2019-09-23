import os
import csv
import numpy as np
import argparse

def readcsv(path, filename):
    with open(path+filename) as csvfile:
        csv_dict = {}
        csv_data = csv.reader(csvfile, delimiter=',')
        next(csv_data)
        for row in csv_data:
            csv_dict[row[0]] = row[1]
    return csv_dict

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read target csv for dti')
    parser.add_argument('--data_dir', help='Path to dataset images')
    parser.add_argument('--output_dir', help='Path to directory for saving outputs')
    args = parser.parse_args()

    # Using residualized fluid intelligence scores using our own brain volumetric variable
    train = "training_fluid_intelligence_sri.csv"
    valid = "validation_fluid_intelligence_sri.csv"
    test = "test_fluid_intelligence_sri.csv"
    
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
    np.save(os.path.join(args.output_dir, 'csv_train_target.npy'), csv_train)
    print('saving valid dict!')
    np.save(os.path.join(args.output_dir, 'csv_valid_target.npy'), csv_valid)
    print('saving test dict!')
    np.save(os.path.join(args.output_dir, 'csv_test_target.npy'), csv_valid)
