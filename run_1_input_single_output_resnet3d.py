import torch
import torch.optim as optim
import torch.nn as nn
import multi_input_resnet3d 
import numpy as np
from mri_dataset import MRIDataset
from model_3d import train, eval
import argparse
import os
import apex
import IPython

if __name__ == "__main__":
    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = True
    
    # Parsing arguments
    parser = argparse.ArgumentParser(description='Single input, single output ResNet3D for regression')
    parser.add_argument('--data_dir', help='Directory path for datasets')
    parser.add_argument('--output_dir')
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--train_batch_size', type=int, default=3)
    parser.add_argument('--valid_batch_size', type=int, default=10)
    parser.add_argument('--checkpoint_state', default='')
    parser.add_argument('--checkpoint_epoch', type=int, default=0)
    parser.add_argument('--checkpoint_opt', default='')
    parser.add_argument('--resize', type=int, default=0)
    parser.add_argument('--normalize', type=bool, default=False)
    parser.add_argument('--log', type=bool, default=False)
    parser.add_argument('--nan', type=bool, default=True)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--momentum', type=float, default=0.5)
    parser.add_argument('--weight_decay', type=float, default=0.001)
    parser.add_argument('--optimizer', default='sgd', help='Optimizer type: adam, sgd')
    parser.add_argument('--sync_bn', default=False, help='Use sync batch norm or not (True/False)')
    parser.add_argument('--mri_type', default=None, help='MRI type: T1, T2, FA, MD, RD, AD')
    parser.add_argument('--target_idx', default=0, type=int, help='Index of target lable that we want to use from the multi-label dataset')
    args = parser.parse_args()

    if args.mri_type == 'T1' or args.mri_type == 'T2':
        model = multi_input_resnet3d.one_struct_input_single_output_resnet3D50(devices=[0,1,2,3], output_classes=1)
    else:
        model = multi_input_resnet3d.one_dti_input_single_output_resnet3D50(devices=[0,1,2,3], output_classes=1)

    # Load from checkpoint, if available
    if args.checkpoint_state:
        saved_state = torch.load(args.checkpoint_state)
        model.load_state_dict(saved_state)
        print('Loaded model from checkpoint')

    # Convert async batch norm to sync batch norm, if applicable
    if args.sync_bn:
        model = apex.parallel.convert_syncbn_model(model)
        print('Using sync batch norm')

    # Load and create datasets
    train_img       =   np.load(os.path.join(args.data_dir, 'train_data_img_{}.npy'.format(args.mri_type)), allow_pickle=True)
    valid_img       =   np.load(os.path.join(args.data_dir, 'train_data_img_{}.npy'.format(args.mri_type)), allow_pickle=True)
    test_img        =   np.load(os.path.join(args.data_dir, 'train_data_img_{}.npy'.format(args.mri_type)), allow_pickle=True)

    print('Extracting target index of {} from multi-label dataset'.format(args.target_idx))
    # Loading only target at target_idx
    train_target    =   np.load(os.path.join(args.data_dir, 'train_data_target.npy'), allow_pickle=True)
    train_target    =   [t[args.target_idx] for t in train_target]
    valid_target    =   np.load(os.path.join(args.data_dir, 'valid_data_target.npy'), allow_pickle=True)
    valid_target    =   [t[args.target_idx] for t in valid_target]
    test_target     =   np.load(os.path.join(args.data_dir, 'test_data_target.npy'), allow_pickle=True)
    test_target     =   [t[args.target_idx] for t in test_target]

    norms = None
    if args.normalize:
        type_order  =   ['T1', 'T2', 'FA', 'MD', 'RD', 'AD']
        norms_idx   =   type_order.index(args.mri_type)
        means       =   np.load(os.path.join(args.data_dir, 'means_reordered.npy'), allow_pickle=True)
        stds        =   np.load(os.path.join(args.data_dir, 'stds_reordered.npy'), allow_pickle=True)
        norms       =   list(zip(means, stds))[norms_idx]

    train_dataset   =   MRIDataset(train_img, train_target, args.resize, norms, args.log, args.nan)
    train_loader    =   torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    valid_dataset   =   MRIDataset(valid_img, valid_target, args.resize, norms, args.log, args.nan)
    valid_loader    =   torch.utils.data.DataLoader(valid_dataset, batch_size=args.valid_batch_size)
    test_dataset    =   MRIDataset(test_img,test_target, args.resize, norms, args.log, args.nan)
    test_loader     =   torch.utils.data.DataLoader(test_dataset, batch_size=args.valid_batch_size)

    if args.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.checkpoint_state:
        saved_opt_state = torch.load(args.checkpoint_opt, map_location='cpu')
        optimizer.load_state_dict(saved_opt_state)
        print('Loaded optimizer from saved state')

    # Creating list of losses of size of output
    loss = None
    reg_loss = nn.L1Loss()
    bin_loss = nn.BCEWithLogitsLoss()
    cat_loss = nn.CrossEntropyLoss()
    
    bin_idx = [1]
    cat_idx = [2, 3, 4, 5, 6]
    
    if args.target_idx in bin_idx:
        loss = bin_loss
    elif args.target_idx in cat_idx:
        loss = cat_loss
    else:
        loss = reg_loss

    if not args.checkpoint_state:
        train(model, args.epoch, train_loader, valid_loader, test_loader, optimizer, loss, args.output_dir)
    else:
        train(model, args.epoch, train_loader, valid_loader, test_loader, optimizer, loss, args.output_dir, args.checkpoint_epoch)
