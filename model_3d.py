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
from sklearn.metrics import mean_squared_error
import logging
import os, sys, time
import IPython

# Setting up logger
LOGGER = logging.getLogger(__name__)
out_hdlr = logging.StreamHandler(sys.stdout)
out_hdlr.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
out_hdlr.setLevel(logging.INFO)
LOGGER.addHandler(out_hdlr)
LOGGER.setLevel(logging.INFO)


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
        self.bn1 = nn.BatchNorm3d(80)
        self.drop = nn.Dropout2d()
        self.fc1 = nn.Linear(11*11*11*80, 4840)   # 11x11x11 x80
        #self.fc1 = nn.Linear(3*3*3*80, 4840)
        self.bn2 = nn.BatchNorm1d(4840)
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
        
        x = self.bn1(x)
        x = x.view(-1, 11*11*11*80)
        #x = x.view(-1, 3*3*3*80)
        x = self.fc1(x)
        #x = self.bn2(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.fc3(x)

        return x

class CNN1(nn.Module):
    def __init__(self):
        super(CNN1, self).__init__()
        self.conv1 = nn.Conv3d(1,10, kernel_size=5)
        self.conv2 = nn.Conv3d(10,20, kernel_size=5)
        self.conv3 = nn.Conv3d(20,40, kernel_size=6)
        self.conv4 = nn.Conv3d(40,80, kernel_size=5)
        self.bn1 = nn.BatchNorm3d(80)
        self.drop = nn.Dropout2d()
        self.fc1 = nn.Linear(11*11*11*80, 2420)   # 11x11x11 x80
        #self.fc1 = nn.Linear(3*3*3*80, 2420)
        #self.bn2 = nn.BatchNorm1d(2420)
        self.fc2 = nn.Linear(2420, 1210)
        self.fc3 = nn.Linear(1210, 1)


    def forward(self,x):
        x = F.relu(F.max_pool3d(self.conv1(x),2))
        x = self.drop(x)
        x = F.relu(F.max_pool3d(self.conv2(x),2))
        x = self.drop(x)
        x = F.relu(F.max_pool3d(self.conv3(x),2))
        x = self.drop(x)
        x = F.relu(F.max_pool3d(self.conv4(x),2))
        x = self.drop(x)
        
        x = self.bn1(x)
        x = x.view(-1, 11*11*11*80)
       # x = x.view(-1, 3*3*3*80)
        x = self.fc1(x)
        #x = self.bn2(x)
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


def train(model, epoch, train_loader, valid_loader, optimizer, loss, output_dir, checkpoint_epoch=0):
    model.train()
    loss = nn.L1Loss()

    loss = loss.cuda()
    best_mse = float('inf')

    if checkpoint_epoch <= 0:
        # Create output directory and results file
        try:
            os.mkdir(output_dir)
            #results = open(os.path.join(output_dir, 'results.txt'), 'w+')
        except: 
            raise Exception('Output directory / results file cannot be created')

    results = open((output_dir+'/results.txt'), 'a+')

    start_epoch = 0
    if checkpoint_epoch > 0:
        start_epoch = checkpoint_epoch

    for i in range(start_epoch, epoch):
        epoch_start = time.time()

        for batch_idx, (batch_img, batch_target) in enumerate(train_loader):
            LOGGER.info('Starting batch {}: [{}/{}]'.format(batch_idx, batch_idx * len(batch_img), len(train_loader.dataset)))
            batch_img = batch_img.unsqueeze(1)

            optimizer.zero_grad()

            batch_img = batch_img.cuda()
            batch_target = batch_target.float().cuda()

            output = model(batch_img)
            res = loss(output.squeeze(), batch_target)
            res.backward() 
            optimizer.step()
            
            LOGGER.info('End batch {}: [{}/{}]'.format(batch_idx, batch_idx * len(batch_img), len(train_loader.dataset)))

            if batch_idx % 10 == 0:
                LOGGER.info('Train Epoch {}: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(i, batch_idx * len(batch_img), len(train_loader.dataset), len(batch_img) * batch_idx / len(train_loader.dataset) * 100, res.item()))
            
            #torch.cuda.empty_cache()
            #del batch_img, batch_target
        
        epoch_end = time.time()
        epoch_train_time = epoch_end - epoch_start

        cur_mse = eval(model, valid_loader, loss)
        results.write('Epoch {}: {} ({} s)\n'.format(i, cur_mse, epoch_train_time))
        torch.save(model.state_dict(), os.path.join(output_dir, '{}_epoch_{}.pth'.format(model._get_name(), i)))
        torch.save(optimizer.state_dict(), os.path.join(output_dir, 'optimizer.pth'))

        if cur_mse < best_mse:
            best_mse = cur_mse
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_{}_epoch.pth'.format(i)))
    
    results.close()
            

def eval(model, valid_loader, loss):
    model.eval()
    loss = nn.L1Loss()
    
    model.cuda()
    loss = loss.cuda()
    
    target_true = []
    target_pred = []

    with torch.no_grad():
        for batch_idx, (batch_img, batch_target) in enumerate(valid_loader):
            LOGGER.info('Evaluating batch {}: [{}/{}]'.format(batch_idx, batch_idx * len(batch_img), len(valid_loader.dataset)))
            batch_img = batch_img.unsqueeze(1)

            batch_img = batch_img.cuda()
            batch_target = batch_target.float().cuda()

            output = model(batch_img)
            #LOGGER.info('current output is: {}\nground truth is: {}'.format(output.cpu().detach().numpy(), batch_target.cpu().detach().numpy()))
            res = loss(output.squeeze(), batch_target)

            # Adding predicted and true targets
            target_true.extend(batch_target.cpu())
            for pred in output:
                target_pred.extend(pred.cpu())

            if batch_idx % 10 == 0:
                LOGGER.info('Eval Progress: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                batch_idx * len(batch_img), len(valid_loader.dataset), 
                valid_loader.batch_size * batch_idx / len(valid_loader), res.item()))     
    
    target_true = np.subtract(np.exp(target_true), 40)
    target_pred = np.subtract(np.exp(target_pred),40)
    print('Target true:')
    print(target_true)
    print('Target pred:')
    print(target_pred)
    mse = mean_squared_error(target_true, target_pred)
    LOGGER.info('Mean squared error: {}'.format(mse))

    return mse
    


"""
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
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

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
train(model, 10, train_loader)
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
"""










