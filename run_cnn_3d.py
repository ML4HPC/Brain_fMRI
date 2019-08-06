import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from mri_dataset import MRIDataset
from model_3d import CNN, CNN1, train, eval
import argparse
import os
import apex

if __name__ == "__main__":
    torch.cuda.set_device(0)

    # Parsing arguments
    parser = argparse.ArgumentParser(description='3D CNN for regression')
    parser.add_argument('--data_dir')
    parser.add_argument('--output_dir')
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--train_batch_size', type=int, default=2)
    parser.add_argument('--valid_batch_size', type=int, default=4)
    parser.add_argument('--checkpoint_state', default='')
    parser.add_argument('--checkpoint_epoch', type=int, default=0)
    parser.add_argument('--checkpoint_opt', default='')
    parser.add_argument('--resize', type=int, default=0)
    parser.add_argument('--normalize', type=bool, default=False)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.5)
    parser.add_argument('--optimizer', default='sgd', help='Optimizer type: adam, sgd')
    parser.add_argument('--sync_bn', default=False, help='Use sync batch norm or not (True/False)')
    parser.add_argument('--model', type=int, default=0, help='CNN or CNN1')
    args = parser.parse_args()

    # Setting device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    if args.model == 0:
        model = CNN()
    elif args.model == 1:
        model = CNN1()

    # Setting up data parallelism, if available
    if torch.cuda.device_count() > 1:
        print('Using Data Parallelism with multiple GPUs available')
        model = nn.DataParallel(model)

    # Load from checkpoint, if available
    if args.checkpoint_state:
        saved_state = torch.load(args.checkpoint_state, map_location='cpu')
        model.load_state_dict(saved_state)
        print('Loaded model from checkpoint')

    # Convert async batch norm to sync batch norm, if applicable
    if args.sync_bn:
        model = apex.parallel.convert_syncbn_model(model)
        print('Using sync batch norm')

    model.cuda()
    # Load and create datasets
    train_img = np.load(os.path.join(args.data_dir, 'train_data_img.npy'), allow_pickle=True)
    valid_img = np.load(os.path.join(args.data_dir, 'valid_data_img.npy'), allow_pickle=True)
    train_target = np.load(os.path.join(args.data_dir, 'train_data_target.npy'), allow_pickle=True)
    valid_target = np.load(os.path.join(args.data_dir, 'valid_data_target.npy'), allow_pickle=True)

    train_dataset = MRIDataset(train_img, train_target, args.resize, args.normalize)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size)
    valid_dataset = MRIDataset(valid_img, valid_target, args.resize, args.normalize)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.valid_batch_size)

    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.001)
    
    if args.checkpoint_state:
        saved_opt_state = torch.load(args.checkpoint_opt)
        optimizer.load_state_dict(saved_opt_state)
        print('Loaded optimizer from saved state')
        
    loss = nn.L1Loss()

    if not args.checkpoint_state:
        train(model, args.epoch, train_loader, valid_loader, optimizer, loss, args.output_dir)
    else:
        train(model, args.epoch, train_loader, valid_loader, optimizer, loss, args.output_dir, checkpoint_epoch=args.checkpoint_epoch)

    
