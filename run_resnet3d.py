import torch
import torch.optim as optim
import resnet3d
import numpy as np
from mri_dataset import MRIDataset
from model_3d import train


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
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=2)
    valid_dataset = MRIDataset(valid_img, valid_target)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=4)

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)

    train(model, 1, train_loader, optimizer)
    
    