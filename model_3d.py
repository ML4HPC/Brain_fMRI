## https://discuss.pytorch.org/t/solved-concatenate-time-distributed-cnn-with-lstm/15435/4
## https://machinelearningmastery.com/cnn-long-short-term-memory-networks/

import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from mri_dataset import MRIDataset


def flatten(t):
    t = t.reshape(1, -1)
    t = t.squeeze()
    return t 


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv3d(1,10, kernel_size=5)
        self.conv2 = nn.Conv3d(10,20, kernel_size=5)
        self.conv3 = nn.Conv3d(20,40, kernel_size=6)
        self.conv4 = nn.Conv3d(40,80, kernel_size=5)
        self.drop = nn.Dropout2d()
        self.fc1 = nn.Linear(11*11*11*80, 4840)   # 11x11x11 x80
        self.fc2 = nn.Linear(4840, 2420)
        self.fc3 = nn.Linear(2420, 1)


    def forward(self,x):
        x = F.relu(F.max_pool3d(self.conv1(x),2))
        x = self.drop(x)
        x = F.relu(F.max_pool3d(self.conv2(x),2))
        x = self.drop(x)
        x = F.relu(F.max_pool3d(self.conv3(x),2))
        x = self.drop(x)
        x = F.relu(F.max_pool3d(self.conv4(x),2))
        x = self.drop(x)
        
        x = x.view(-1, 11*11*11*80)
        x = self.fc1(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x



class CombineRNN(nn.Module):
    def __init__(self):
        super(CombineRNN, self).__init__()
        self.cnn = CNN()
        self.rnn = nn.LSTM(
            input_size = 2420,
            hidden_size = 256,
            num_layers = 1,
            batch_first = True)
        self.linear1 = nn.Linear(256,128)
        self.linear2 = nn.Linear(128,1)

    def forward(self, x):
        batch_size, time_steps, C, H, W = x.size()
        c_in = x.view(batch_size*time_steps, C, H, W)  # batch_size*time_steps becomes dummy_batch
        c_out = self.cnn(c_in)
        r_in = c_out.view(batch_size, time_steps, -1) # transform back to [batch_size, time_steps, feature]
        r_out, (h_n, h_c) = self.rnn(r_in)
        r_out2 = self.linear1(r_out[:,-1,:])
        r_output = self.linear2(r_out2)

        return r_output





class Args:
    def __init__(self):
        self.cuda = True
        self.no_cuda = False
        self.seed = 1
        self.batch_size = 50
        self.test_batch_size = 1000
        self.epoch = 10
        self.lr = 0.01
        self.momentum = 0.5
        self.log_interval = 10

args = Args()


def train(epoch, train_loader):
    model.train()
    loss = nn.L1Loss()

    model.cuda()
    loss = loss.cuda()

    for i in range(epoch):
        for batch_idx, (batch_img, batch_target) in enumerate(train_loader):
            batch_img = batch_img.unsqueeze(1)
            batch_img, batch_target = Variable(batch_img), Variable(batch_target)

            optimizer.zero_grad()

            batch_img = batch_img.cuda()
            batch_target = batch_target.cuda()

            output = model(batch_img)
            print('current output is: ', output.cpu().detach().numpy(), 'the ground truth is: ', batch_target.cpu().detach().numpy())
            res = loss(output.squeeze(), batch_target)
            res.backward() 
            optimizer.step()
            print('current residue is: ', res.cpu().detach().numpy())

def eval(valid_loader):
    model.eval()
    loss = nn.L1Loss()

    model.cuda()
    loss = loss.cuda()
    
    for batch_idx, (batch_img, batch_target) in enumerate(valid_loader):
        batch_img = batch_img.unsqueeze(1)
        batch_img, batch_target = Variable(batch_img), Variable(batch_target)
        optimizer.zero_grad()

        batch_img = batch_img.cuda()
        batch_target = batch_target.cuda()

        output = model(batch_img)
        print('current output is: ', output.cpu().detach().numpy(), 'the ground truth is: ', batch_target.cpu().detach().numpy())
        res = loss(output.squeeze(), batch_target)
        #res.backward() 
        #optimizer.step()
        print('current residue is: ', res.cpu().detach().numpy())


# Setting device
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = CNN()

# Data Parallelism if multiple GPUs available
if torch.cuda.device_count() > 1:
    print('Using Data Parallelism with multiple GPUs available')
    model = nn.DataParallel(model)

model.to(device)
lr = 0.01 * 1000
momentum = 0.5
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum).to(device)

# Load and create datasets
train_img = np.load('train_data_img.npy')
valid_img = np.load('valid_data_img.npy')
train_target = np.load('train_data_target.npy')
valid_target = np.load('valid_data_target.npy')

train_dataset = MRIDataset(train_img, train_target)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8)
valid_dataset = MRIDataset(valid_img, valid_target)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=8)

print("now in training!\n")
train(10, train_loader)
print("training finished!\n")

torch.save(model.state_dict(), './model_saved.pth')
print('Saving model')

#print("now in evaluation!\n")
#model.eval()
#output = model(input_data)
#print("evaluation finished!\n")

#print(output)

#output = output.detach().numpy()
#print(output)
#print(output.shape)











