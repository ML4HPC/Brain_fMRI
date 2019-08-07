import torch
import numpy as np
from mri_dataset import MRIDataset
from model_3d import eval
import argparse
import os
import IPython


if __name__ == "__main__":
    # Parsing arguments
    parser = argparse.ArgumentParser(description='Calculating mean & std dev for normalization')
    parser.add_argument('--data_dir')
    parser.add_argument('--train_batch_size', type=int, default=100)
    parser.add_argument('--resize', type=int, default=0)
    args = parser.parse_args()


    # Load and create datasets
    train_img = np.load(os.path.join(args.data_dir, 'train_data_img.npy'), allow_pickle=True)
    train_target = np.load(os.path.join(args.data_dir, 'train_data_target.npy'), allow_pickle=True)
    train_dataset = MRIDataset(train_img, train_target, args.resize)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size)
    
    mean = 0.0
    for batch_img, _ in train_loader:
        mean += torch.sum(batch_img)
    
    mean /= len(train_loader.dataset) * 120 * 120 * 120

    var = 0.0
    for batch_img, _ in train_loader:
        var += torch.sum(np.power(np.subtract(batch_img, mean), 2))
    
    var /= len(train_loader.dataset) * 120 * 120 * 120
    std = np.sqrt(var)

    print('Mean: {}\n Std: {}\n'.format(mean, std))