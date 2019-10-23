import torch
import torch.optim as optim
import torch.nn as nn
import multi_input_resnet3d 
import numpy as np
from mri_dataset import MRIDatasetBySite
from sklearn.metrics import r2_score, mean_squared_error
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
    parser.add_argument('--saved_state', default='')
    parser.add_argument('--resize', type=int, default=0)
    parser.add_argument('--normalize', type=bool, default=False)
    parser.add_argument('--log', type=bool, default=False)
    parser.add_argument('--nan', type=bool, default=True)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--site', default=None)
    parser.add_argument('--site_excl', type=int, default=0)
    parser.add_argument('--mri_type', default=None, help='MRI type: T1, T2, FA, MD, RD, AD')
    parser.add_argument('--target_idx', default=0, type=int, help='Index of target lable that we want to use from the multi-label dataset')
    parser.add_argument('--metric', default='R2', help='Scoring metric: R2, MSE')
    args = parser.parse_args()

    if args.mri_type == 'T1' or args.mri_type == 'T2':
        model = multi_input_resnet3d.one_struct_input_single_output_resnet3D50(devices=[0,1,2,3], output_classes=1)
    else:
        model = multi_input_resnet3d.one_dti_input_single_output_resnet3D50(devices=[0,1,2,3], output_classes=1)

    # Load from saved state, if available
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

    if args.site:
        parsed_site =   list(map(int, args.site.strip('[]').split(',')))

    valid_dataset   =   MRIDatasetBySite(valid_img, valid_target, args.resize, norms, args.log, args.nan, parsed_site, args.target_idx, args.site_excl)
    valid_loader    =   torch.utils.data.DataLoader(valid_dataset, batch_size=args.valid_batch_size)
    test_dataset    =   MRIDatasetBySite(test_img,test_target, args.resize, norms, args.log, args.nan, parsed_site, args.target_idx, args.site_excl)
    test_loader     =   torch.utils.data.DataLoader(test_dataset, batch_size=args.valid_batch_size)

    # Metrics
    if args.metric.upper() == 'R2':
        metric = r2_score
    elif args.metric.upper() == 'MSE':
        metric = mean_squared_error

    valid_score     =   eval(model, valid_loader, metric, save=True, output_dir=args.output_dir, valid_type='valid')
    test_score      =   eval(model, test_loader, metric, save=True, output_dir=args.output_dir, valid_type='test')
    print('Validation: {} Test: {}'.format(valid_score, test_score))
