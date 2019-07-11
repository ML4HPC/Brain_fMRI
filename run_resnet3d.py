import torch
import torch.optim as optim
import resnet3d
import numpy as np
from mri_dataset import MRIDataset
from model_3d import train, eval
import argparse
import os


if __name__ == "__main__":
    # Parsing arguments
    parser = argparse.ArgumentParser(description='ResNet3D for regression')
    parser.add_argument('--data_dir')
    parser.add_argument('--output_dir')
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--train_batch_size', type=int, default=2)
    parser.add_argument('--valid_batch_size', type=int, default=4)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.5)
    args = parser.parse_args()

    #model = resnet3d.resnet3D50(num_classes=10)
    model = resnet3d.ResNet3DRegressor()
    # Setting device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Load and create datasets
    train_img = np.load(os.path.join(args.data_dir, 'train_data_img.npy'))
    valid_img = np.load(os.path.join(args.data_dir, 'valid_data_img.npy'))
    train_target = np.load(os.path.join(args.data_dir, 'train_data_target.npy'))
    valid_target = np.load(os.path.join(args.data_dir, 'valid_data_target.npy'))

    train_dataset = MRIDataset(train_img, train_target)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size)
    valid_dataset = MRIDataset(valid_img, valid_target)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.valid_batch_size)

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    train(model, args.epoch, train_loader, valid_loader, optimizer, args.output_dir)
    eval(model, valid_loader)
    
    