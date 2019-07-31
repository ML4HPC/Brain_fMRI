import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from mri_dataset import MRIDataset
from model_3d import CNN, train, eval
import argparse
import os
from lr_finder import LRFinder
import IPython

if __name__ == "__main__":
    torch.cuda.set_device(0)

    # Parsing arguments
    parser = argparse.ArgumentParser(description='3D CNN for regression')
    parser.add_argument('--data_dir')
    parser.add_argument('--output_dir')
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.5)
    parser.add_argument('--resize', type=float, default=0)
    args = parser.parse_args()

    # Setting device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = CNN()
    model.cuda()

    # Setting up data parallelism, if available
    if torch.cuda.device_count() > 1:
        print('Using Data Parallelism with multiple GPUs available')
        model = nn.DataParallel(model)

    # Load and create datasets
    train_img = np.load(os.path.join(args.data_dir, 'train_data_img.npy'), allow_pickle=True)
    valid_img = np.load(os.path.join(args.data_dir, 'valid_data_img.npy'), allow_pickle=True)
    train_target = np.load(os.path.join(args.data_dir, 'train_data_target.npy'), allow_pickle=True)
    valid_target = np.load(os.path.join(args.data_dir, 'valid_data_target.npy'), allow_pickle=True)

    train_dataset = MRIDataset(train_img, train_target, args.resize)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size)
    valid_dataset = MRIDataset(valid_img, valid_target, args.resize)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.valid_batch_size)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    loss = nn.L1Loss()
    lr_finder = LRFinder(model, optimizer, loss, device="cuda")
    lr_finder.range_test(train_loader, end_lr=100, num_iter=100)
    lr_finder.plot()
    IPython.embed()
