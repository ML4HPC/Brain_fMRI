import torch
import torch.optim as optim
import torch.nn as nn
import multi_input_resnet3d 
import numpy as np
from mri_dataset import MultiMRIDataset
from model_3d_multi import train_multi, eval_multi
import argparse
import os
import apex
import IPython

if __name__ == "__main__":
    torch.cuda.set_device(0)
    torch.backends.cudnn.benchmark = True

    # Parsing arguments
    parser = argparse.ArgumentParser(description='Multi-channel/input ResNet3D for regression')
    parser.add_argument('--data_dir', help='Directory path for datasets')
    parser.add_argument('--output_dir')
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--train_batch_size', type=int, default=2)
    parser.add_argument('--valid_batch_size', type=int, default=4)
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
    args = parser.parse_args()

    if args.mri_type == 'T1' or args.mri_type == 'T2':
        model = multi_input_resnet3d.one_struct_input_multi_output_resnet3D50(devices=[0,1,2,3])
    else:
        model = multi_input_resnet3d.one_dti_input_multi_output_resnet3D50(devices=[0,1,2,3])

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

    train_target    =   np.load(os.path.join(args.data_dir, 'train_data_target.npy'), allow_pickle=True)
    valid_target    =   np.load(os.path.join(args.data_dir, 'valid_data_target.npy'), allow_pickle=True)
    test_target     =   np.load(os.path.join(args.data_dir, 'test_data_target.npy'), allow_pickle=True)

    norms = None
    if args.normalize:
        type_order  =   ['T1', 'T2', 'FA', 'MD', 'RD', 'AD']
        norms_idx   =   type_order.index(args.mri_type)
        means       =   np.load(os.path.join(args.data_dir, 'means_reordered.npy'), allow_pickle=True)
        stds        =   np.load(os.path.join(args.data_dir, 'stds_reordered.npy'), allow_pickle=True)
        norms       =   list(zip(means, stds))[norms_idx]

    train_dataset   =   MultiMRIDataset(train_img, train_target, args.resize, norms, args.log, args.nan)
    train_loader    =   torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    valid_dataset   =   MultiMRIDataset(valid_img, valid_target, args.resize, norms, args.log, args.nan)
    valid_loader    =   torch.utils.data.DataLoader(valid_dataset, batch_size=args.valid_batch_size)
    test_dataset    =   MultiMRIDataset(test_img,test_target, args.resize, norms, args.log, args.nan)
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
    losses = [None] * 21
    reg_loss = nn.L1Loss()
    bin_loss = nn.BCEWithLogitsLoss()
    cat_loss = nn.CrossEntropyLoss()
    
    bin_idx = [1]
    cat_idx = [2, 3, 4, 5, 6]
    
    for i in range(len(losses)):
        if i in bin_idx:
            losses[i] = bin_loss
        elif i in cat_idx:
            losses[i] = cat_loss
        else:
            losses[i] = reg_loss

    if not args.checkpoint_state:
        train_multi(model, args.epoch, train_loader, valid_loader, test_loader, optimizer, losses, args.output_dir)
    else:
        train_multi(model, args.epoch, train_loader, valid_loader, test_loader, optimizer, losses, args.output_dir, args.checkpoint_epoch)
