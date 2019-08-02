import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from mri_dataset import MRIDataset, SliceMRIDataset
from model_3d import CNN, train, eval
import argparse
import os
from skorch import NeuralNetRegressor
from sklearn.model_selection import GridSearchCV
from scipy.stats import randint as sp_randint
import IPython

if __name__ == "__main__":
    torch.cuda.set_device(0)

    # Parsing arguments
    parser = argparse.ArgumentParser(description='3D CNN for regression')
    parser.add_argument('--data_dir')
    parser.add_argument('--output_dir')
    parser.add_argument('--epoch', type=int, default=30)
    #parser.add_argument('--train_batch_size', type=int, default=2)
    #parser.add_argument('--valid_batch_size', type=int, default=4)
    parser.add_argument('--checkpoint_state', default='')
    parser.add_argument('--optimizer', default='sgd', help='Optimizer type')
    parser.add_argument('--resize', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.5)
    args = parser.parse_args()

    # Setting device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = CNN()
    model.cuda()

    # Setting up data parallelism, if available
    if torch.cuda.device_count() > 1:
        print('Using Data Parallelism with multiple GPUs available')
        model = nn.DataParallel(model)

    # # Load from checkpoint, if available
    # if args.checkpoint_state:
    #     saved_state = torch.load(args.checkpoint_state)
    #     model.load_state_dict(saved_state)
    #     print('Loaded model from checkpoint')

    # Load and create datasets
    train_img = np.load(os.path.join(args.data_dir, 'train_data_img.npy'), allow_pickle=True)
    #valid_img = np.load(os.path.join(args.data_dir, 'valid_data_img.npy'), allow_pickle=True)
    train_target = np.load(os.path.join(args.data_dir, 'train_data_target.npy'), allow_pickle=True)
    #valid_target = np.load(os.path.join(args.data_dir, 'valid_data_target.npy'), allow_pickle=True)

    train_dataset = MRIDataset(train_img, train_target, args.resize)
    #train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size)
    #valid_dataset = MRIDataset(valid_img, valid_target, args.resize)
    #valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.valid_batch_size)

    if args.optimizer == 'sgd':
        optimizer = torch.optim.SGD
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam
    
    net = NeuralNetRegressor(model, max_epochs=3, lr=0.001, optimizer=optimizer, verbose=1, batch_size=10)
    
    params = {
        'lr':[0.001, 0.01, 0.02, 0.04, 0.1]
    }

    slice_dataset = SliceMRIDataset(train_dataset)
    gs = GridSearchCV(net, params, refit=False, cv=3, n_jobs=-1, scoring='neg_mean_squared_error', verbose=10)
    #loss = nn.L1Loss()
    
    gs.fit(slice_dataset, train_dataset.get_y_data())
    print(gs.best_params_)
    IPython.embed()

    # train(model, args.epoch, train_loader, valid_loader, optimizer, loss, args.output_dir)
    # eval(model, valid_loader, loss)

