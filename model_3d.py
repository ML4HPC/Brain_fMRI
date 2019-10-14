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
from sklearn.metrics import r2_score
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


def train(model, epoch, train_loader, valid_loader, test_loader, optimizer, loss, output_dir, metric=r2_score, checkpoint_epoch=0):
    model.train()

    best_score = float('-inf')

    if checkpoint_epoch <= 0:
        # Create output directory and results file
        try:
            os.mkdir(output_dir)
            #results = open(os.path.join(output_dir, 'results.txt'), 'w+')
        except: 
            raise Exception('Output directory / results file cannot be created')

    results = open(os.path.join(output_dir, 'results.txt'), 'a+')
    loss_hist = []

    start_epoch = 0
    if checkpoint_epoch > 0:
        start_epoch = checkpoint_epoch

    for i in range(start_epoch, epoch):
        progress = 0
        epoch_start = time.time()

        for batch_idx, (batch_img, batch_target) in enumerate(train_loader):            
            LOGGER.info('Starting batch {}: [{}/{}]'.format(batch_idx, progress, len(train_loader.dataset)))
            batch_img = batch_img.unsqueeze(1).cuda()

            optimizer.zero_grad()
            cur_loss = 0 
            
            outputs = model(batch_img).squeeze()
            batch_target = batch_target.squeeze().float().cuda()
            cur_loss = loss(outputs, batch_target)
             
            loss_hist.append(cur_loss.item())
            cur_loss.backward() 
            optimizer.step()
            
            progress += len(batch_img)
            LOGGER.info('End batch {}: [{}/{}]'.format(batch_idx, batch_idx * len(batch_img), len(train_loader.dataset)))

            if batch_idx % 10 == 0:
                LOGGER.info('Train Epoch {}: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(i, progress, len(train_loader.dataset), progress / len(train_loader.dataset) * 100, cur_loss.item()))
            
            #torch.cuda.empty_cache()
            #del batch_img, batch_target
        
        epoch_end = time.time()
        epoch_train_time = epoch_end - epoch_start

        cur_score= eval(model, valid_loader, metric)
        test_score = eval(model, test_loader, metric)
        results.write('Epoch {}: Validation {} Test {} ({} s)\n'.format(i, cur_score, test_score, epoch_train_time))
        results.flush()
        torch.save({
            'epoch': i,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }, os.path.join(output_dir, '{}_epoch_{}.pth'.format(model._get_name(), i)))

        # Higher the better
        if cur_score > best_score:
            best_score = cur_score
            torch.save({
                'epoch': i,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
            }, os.path.join(output_dir, 'best_epoch_{}.pth'.format(i)))

        np.save(os.path.join(output_dir, 'loss_history_train.npy'), loss_hist)
    
    results.close()


def eval(model, valid_loader, metric, save=False, output_dir=None, valid_type=None):
    model.eval()

    target_true = []
    target_pred = []

    with torch.no_grad():
        progress = 0
        for batch_idx, (batch_img, batch_target) in enumerate(valid_loader):
            LOGGER.info('Evaluating batch {}: [{}/{}]'.format(batch_idx, progress, len(valid_loader.dataset)))
            
            devs = model.get_devices()
            batch_img = batch_img.unsqueeze(1).float().to(devs[0])

            outputs = model(batch_img).squeeze()

            # Adding predicted and true targets
            target_true.extend(batch_target.squeeze().cpu())
            target_pred.extend(outputs.cpu().numpy())

            if batch_idx % 10 == 0:
                LOGGER.info('Eval Progress: [{}/{} ({:.0f}%)]'.format(
                progress, len(valid_loader.dataset), progress / len(valid_loader.dataset) * 100))     
            
            progress += len(batch_img)
    
    if valid_loader.dataset.log:
        target_true = np.subtract(np.exp(target_true), 40)
        target_pred = np.subtract(np.exp(target_pred), 40)

    # MSE of fluid intelligence
    score = metric(target_true, target_pred)
    LOGGER.info('Score: {}'.format(score))

    if save:
        try:
            np.save(os.path.join(output_dir, '{}_target_true.npy'.format(valid_type)), target_true)
            np.save(os.path.join(output_dir, '{}_target_pred.npy'.format(valid_type)), target_pred)
        except:
            raise Exception('Could not save ground truth & predictions to file')

    return score

def train_multi_input(model, epoch, train_loader, valid_loader, test_loader, optimizer, loss, output_dir, checkpoint_epoch=0):
    model.train()

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
    loss_hist = []

    start_epoch = 0
    if checkpoint_epoch > 0:
        start_epoch = checkpoint_epoch

    for i in range(start_epoch, epoch):
        epoch_start = time.time()

        for batch_idx, (batch_img, batch_target) in enumerate(train_loader):
            LOGGER.info('Starting batch {}: [{}/{}]'.format(batch_idx, batch_idx * len(batch_img), len(train_loader.dataset)))

            optimizer.zero_grad()
            
            new_batch_img = []
            
            # Moving each type of image to respective GPUs
            for j in range(len(batch_img)):
                devs = model.get_devices()
                new_batch_img.append(batch_img[j].unsqueeze(1).to(devs[j]))
            
            batch_target = batch_target.float().cuda()
            output = model(new_batch_img)
            res = loss(output.squeeze(), batch_target)
            res.backward() 
            optimizer.step()

            loss_hist.append(res.item())
            
            LOGGER.info('End batch {}: [{}/{}]'.format(batch_idx, batch_idx * train_loader.batch_size, len(train_loader.dataset)))

            if batch_idx % 10 == 0:
                LOGGER.info('Train Epoch {}: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    i, batch_idx * train_loader.batch_size, len(train_loader.dataset), 
                    train_loader.batch_size * batch_idx / len(train_loader.dataset) * 100, res.item()))
            
            #torch.cuda.empty_cache()
            #del batch_img, batch_target
        
        epoch_end = time.time()
        epoch_train_time = epoch_end - epoch_start

        cur_mse = eval(model, valid_loader, loss)
        test_mse = eval(model, test_loader, loss)
        results.write('Epoch {}: Validation {} Test {} ({} s)\n'.format(i, cur_mse, test_mse, epoch_train_time))
        results.flush()
        torch.save({
            'epoch': i,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': loss
        }, '{}_epoch_{}.pth'.format(model._get_name(), i))

        if cur_mse < best_mse:
            best_mse = cur_mse
            torch.save({
                'epoch': i,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss
            }, 'best_epoch_{}.pth'.format(i))
        
        np.save(os.path.join(output_dir, 'loss_history_train.npy'), loss_hist)
    
    results.close()


    


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










