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

class MultiCNN(nn.Module):
    def __init__(self):
        super(MultiCNN, self).__init__()
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
        self.fc_age = nn.Linear(2420, 1)
        self.fc_gender = nn.Linear(2420, 1)
        self.fc_race = nn.Linear(2420, 5)
        self.fc_edu = nn.Linear(2420, 23)
        self.fc_married = nn.Linear(2420, 1)
        self.fc_site = nn.Linear(2420, 21)
        self.sigmoid = nn.Sigmoid()


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
        x_fi = self.fc3(x)

        x_age = self.fc_age(x)

        x_gender = self.fc_gender(x)
        x_gender = self.sigmoid(x_gender)

        x_race = self.fc_race(x)
        x_race = F.log_softmax(x, dim = 1)

        x_edu = self.fc_edu(x)
        x_edu = F.log_softmax(x_edu, dim = 1)

        x_married = self.fc_married(x)
        x_married = self.sigmoid(x_married)
        
        x_site = self.fc_site(x)
        x_site = F.log_softmax(x_site, dim = 1)

        return [x_fi, x_age, x_gender, x_race, x_edu, x_married, x_site]


def train_multi(model, epoch, train_loader, valid_loader, optimizer, losses, output_dir, checkpoint_epoch=0):
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
            batch_target = [target.cuda() for target in batch_target]

            outputs = model(batch_img)
            loss = 0
            #res = loss(output.squeeze(), batch_target)
            for j in range(len(outputs)):
                criterion = losses[j]
                output = outputs[j]
                target = [t[j] for t in batch_target]

                if output.shape[1] == 1:
                    output = output.squeeze()

                IPython.embed()
                if j in [3, 4, 6]:
                    cur_loss = criterion(output, target.long())
                else:
                    cur_loss = criterion(output, target.float())
                loss += cur_loss
            
            loss.backward() 
            optimizer.step()
            
            LOGGER.info('End batch {}: [{}/{}]'.format(batch_idx, batch_idx * len(batch_img), len(train_loader.dataset)))

            if batch_idx % 10 == 0:
                LOGGER.info('Train Epoch {}: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(i, batch_idx * len(batch_img), len(train_loader.dataset), len(batch_img) * batch_idx / len(train_loader.dataset) * 100, loss.item()))
            
            #torch.cuda.empty_cache()
            #del batch_img, batch_target
        
        epoch_end = time.time()
        epoch_train_time = epoch_end - epoch_start

        cur_mse = eval(model, valid_loader, loss)
        results.write('Epoch {}: {} ({} s)\n'.format(i, cur_mse, epoch_train_time))
        results.flush()
        torch.save(model.state_dict(), os.path.join(output_dir, '{}_epoch_{}.pth'.format(model._get_name(), i)))
        torch.save(optimizer.state_dict(), os.path.join(output_dir, 'optimizer.pth'))

        if cur_mse < best_mse:
            best_mse = cur_mse
            torch.save(model.state_dict(), os.path.join(output_dir, 'best_{}_epoch.pth'.format(i)))
    
    results.close()
            

def eval_multi(model, valid_loader, loss):
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
            batch_target = [target.float().cuda() for target in batch_target]

            outputs = model(batch_img)
            fi_output = outputs[0].squeeze()
            #LOGGER.info('current output is: {}\nground truth is: {}'.format(output.cpu().detach().numpy(), batch_target.cpu().detach().numpy()))
            #res = loss(output.squeeze(), batch_target)

            # Adding predicted and true targets
            target_true.extend(batch_target[0].cpu())
            for pred in fi_output:
                target_pred.extend(pred.cpu())

            if batch_idx % 10 == 0:
                LOGGER.info('Eval Progress: [{}/{} ({:.0f}%)]'.format(
                batch_idx * len(batch_img), len(valid_loader.dataset), 
                valid_loader.batch_size * batch_idx / len(valid_loader)))     
    
    target_true = np.subtract(np.exp(target_true), 40)
    target_pred = np.subtract(np.exp(target_pred), 40)
    print('Target true:')
    print(target_true)
    print('Target pred:')
    print(target_pred)
    mse = mean_squared_error(target_true, target_pred)
    LOGGER.info('Mean squared error: {}'.format(mse))

    return mse
