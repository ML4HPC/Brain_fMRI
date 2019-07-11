import torch
import resnet3d
import numpy as np
from mri_dataset import MRIDataset
from model_3d import train


def train_tmp(model, epoch, train_loader):
    model.train()
    loss = nn.L1Loss()

    model.cuda()
    loss = loss.cuda()

    for i in range(epoch):
        for batch_idx, (batch_img, batch_target) in enumerate(train_loader):
            print(torch.cuda.current_device)
            batch_img = batch_img.unsqueeze(1)
            batch_img, batch_target = Variable(batch_img), Variable(batch_target)

            optimizer.zero_grad()

            batch_img = batch_img.cuda()
            batch_target = batch_target.cuda()

            output = model(batch_img)

            print('allocated: {} GB'.format(torch.cuda.memory_allocated()/1024*3))

            break
            # print('current output is: ', output.cpu().detach().numpy(), 'the ground truth is: ', batch_target.cpu().detach().numpy())
            # res = loss(output.squeeze(), batch_target)
            # res.backward() 
            # optimizer.step()
            # print('current residue is: ', res.cpu().detach().numpy())

if __name__ == "__main__":
    #model = resnet3d.resnet3D50(num_classes=10)
    model = resnet3d.ResNet3DRegressor()
    # Setting device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    # Load and create datasets
    train_img = np.load('train_data_img.npy')
    valid_img = np.load('valid_data_img.npy')
    train_target = np.load('train_data_target.npy')
    valid_target = np.load('valid_data_target.npy')

    train_dataset = MRIDataset(train_img, train_target)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=4)
    valid_dataset = MRIDataset(valid_img, valid_target)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=4)

    train(model, 1, train_loader)
    
    