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

    # Parsing arguments
    parser = argparse.ArgumentParser(description='Multi-channel/input ResNet3D for regression')
    parser.add_argument('--data_dir', help='Directory path for datasets')
    parser.add_argument('--output_dir', help='Directory path for outputs')
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--valid_batch_size', type=int, default=3)
    parser.add_argument('--saved_state', default=None)
    parser.add_argument('--resize', type=int, default=0)
    parser.add_argument('--normalize', type=int, default=0)
    parser.add_argument('--log', type=bool, default=False)
    parser.add_argument('--nan', type=bool, default=True)
    parser.add_argument('--mri_type', default=None, help='MRI type: T1, T2, FA, MD, RD, AD')
    args = parser.parse_args()

    if args.mri_type == 'T1' or args.mri_type == 'T2':
        model = multi_input_resnet3d.one_struct_input_multi_output_resnet3D50(devices=[0,1,2,3])
    else:
        model = multi_input_resnet3d.one_dti_input_multi_output_resnet3D50(devices=[0,1,2,3])

    # Load from checkpoint, if available
    if args.saved_state:
        saved_state = torch.load(args.saved_state)
        model.load_state_dict(saved_state['model_state_dict'])
        print('Loaded model from saved state')

    # Load and create datasets
    valid_img       =   np.load(os.path.join(args.data_dir, 'train_data_img_{}.npy'.format(args.mri_type)), allow_pickle=True)
    test_img        =   np.load(os.path.join(args.data_dir, 'train_data_img_{}.npy'.format(args.mri_type)), allow_pickle=True)

    valid_target    =   np.load(os.path.join(args.data_dir, 'valid_data_target.npy'), allow_pickle=True)
    test_target     =   np.load(os.path.join(args.data_dir, 'test_data_target.npy'), allow_pickle=True)

    norms = None
    if args.normalize:
        type_order  =   ['T1', 'T2', 'FA', 'MD', 'RD', 'AD']
        norms_idx   =   type_order.index(args.mri_type)
        means       =   np.load(os.path.join(args.data_dir, 'means_reordered.npy'), allow_pickle=True)
        stds        =   np.load(os.path.join(args.data_dir, 'stds_reordered.npy'), allow_pickle=True)
        norms       =   list(zip(means, stds))[norms_idx]


    valid_dataset   =   MultiMRIDataset(valid_img, valid_target, args.resize, norms, args.log, args.nan)
    valid_loader    =   torch.utils.data.DataLoader(valid_dataset, batch_size=args.valid_batch_size)
    test_dataset    =   MultiMRIDataset(test_img,test_target, args.resize, norms, args.log, args.nan)
    test_loader     =   torch.utils.data.DataLoader(test_dataset, batch_size=args.valid_batch_size)

    eval_multi(model, valid_loader, save=True, output_dir=args.output_dir, valid_type='valid')
    eval_multi(model, test_loader, save=True, output_dir=args.output_dir, valid_type='test')

    
