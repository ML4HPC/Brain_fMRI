import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
from mri_dataset import MultiMRIDataset
from model_3d_multi import MultiCNN, train_multi, eval_multi
import argparse
import os

if __name__ == "__main__":
    torch.cuda.set_device(0)

    # Parsing arguments
    parser = argparse.ArgumentParser(description='Evaluating Multi 3D CNN for regression')
    parser.add_argument('--data_dir')
    parser.add_argument('--output_dir')
    parser.add_argument('--valid_batch_size', type=int, default=4)
    parser.add_argument('--checkpoint_state', default='')
    parser.add_argument('--resize', type=int, default=0)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--momentum', type=float, default=0.5)
    args = parser.parse_args()

    # Setting device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = MultiCNN()
    model.cuda()

    # Setting up data parallelism, if available
    if torch.cuda.device_count() > 1:
        print('Using Data Parallelism with multiple GPUs available')
        model = nn.DataParallel(model)

    # Load from checkpoint, if available
    if args.checkpoint_state:
        saved_state = torch.load(args.checkpoint_state)
        model.load_state_dict(saved_state)
        print('Loaded model from checkpoint')

    # Load and create datasets
    #train_img = np.load(os.path.join(args.data_dir, 'train_data_img.npy'), allow_pickle=True)
    #train_target = np.load(os.path.join(args.data_dir, 'train_data_target.npy'), allow_pickle=True)
    # train_dataset = MRIDataset(train_img, train_target, args.resize)
    # train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size)  

    valid_img = np.load(os.path.join(args.data_dir, 'valid_data_img.npy'), allow_pickle=True)
    valid_target = np.load(os.path.join(args.data_dir, 'valid_data_target.npy'), allow_pickle=True)
    valid_dataset = MultiMRIDataset(valid_img, valid_target, args.resize)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=args.valid_batch_size)
    
    loss_fi = nn.L1Loss()
    loss_age = nn.L1Loss()
    loss_gender = nn.BCEWithLogitsLoss()
    loss_race = nn.CrossEntropyLoss()
    loss_edu = nn.CrossEntropyLoss()
    loss_married = nn.CrossEntropyLoss()
    loss_site = nn.CrossEntropyLoss()
    losses = [loss_fi, loss_age, loss_gender, loss_race, loss_edu, loss_married, loss_site]
    
    cur_mse = eval_multi(model, valid_loader, losses, save=True, output_dir=args.output_dir)
    print('MSE Score: {}'.format(cur_mse))

    