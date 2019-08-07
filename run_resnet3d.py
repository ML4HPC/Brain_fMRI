import torch
import torch.optim as optim
import torch.nn as nn
import resnet3d
import numpy as np
from mri_dataset import MRIDataset
from model_3d import train, eval
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

    #model = resnet3d.resnet3D50(num_classes=10)
    #model = resnet3d.ResNet3DRegressor()
    model = resnet3d.PipelinedResNet3dRegressor(devices=[0,1,2,3])
    
        # Load from checkpoint, if available
    if args.checkpoint_state:
        saved_state = torch.load(args.checkpoint_state, map_location='cpu')
        model.load_state_dict(saved_state)
        print('Loaded model from checkpoint')

    # Convert async batch norm to sync batch norm, if applicable
    if args.sync_bn:
        model = apex.parallel.convert_syncbn_model(model)
        print('Using sync batch norm')
    

    # Setting device
    #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #model = model.to(device)

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
        saved_opt_state = torch.load(args.checkpoint_opt, map_location='cpu')
        optimizer.load_state_dict(saved_opt_state)
        print('Loaded optimizer from saved state')
    
    loss = nn.L1Loss()

    if not args.checkpoint_state:
        train(model, args.epoch, train_loader, valid_loader, optimizer, loss, args.output_dir)
    else:
        train(model, args.epoch, train_loader, valid_loader, optimizer, loss, args.output_dir, checkpoint_epoch=args.checkpoint_epoch)
"""
def train_split():
    model.train()
    loss = nn.L1Loss()

    model.cuda()
    loss = loss.cuda()
    best_mse = float('inf')

    # Create output directory and results file
    try:
        os.mkdir(output_dir)
        results = open(os.path.join(output_dir, 'results.txt'), 'w+')
    except: 
        raise Exception('Output directory / results file cannot be created')

    for i in range(epoch):
        for batch_idx, (batch_img, batch_target) in enumerate(train_loader):
            batch_img = batch_img.unsqueeze(1)

            optimizer.zero_grad()

            batch_img = batch_img.cuda()
            batch_target = batch_target.float().cuda()

            output = model(batch_img)
            print('current output is: ', output.cpu().detach().numpy(), 'the ground truth is: ', batch_target.cpu().detach().numpy())
            res = loss(output.squeeze(), batch_target)
            res.backward() 
            optimizer.step()
            print('current residue is: ', res.cpu().detach().numpy())
        
        cur_mse = eval(model, valid_loader)
        results.write('Epoch {}: {}\n'.format(epoch, cur_mse))
            
        if cur_mse < best_mse:
            best_mse = cur_mse
            torch.save(model.state_dict(), os.path.join(output_dir, '{}_epoch_{}.pth'.format(model._get_name(), i)))
"""
