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
        self.fc_married = nn.Linear(2420, 7)
        self.fc_site = nn.Linear(2420, 22)


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

        x_race = self.fc_race(x)
        x_race = F.log_softmax(x, dim = 1)

        x_edu = self.fc_edu(x)
        x_edu = F.log_softmax(x_edu, dim = 1)

        x_married = self.fc_married(x)
        x_married = F.log_softmax(x_married, dim = 1)
        
        x_site = self.fc_site(x)
        x_site = F.log_softmax(x_site, dim = 1)

        return [x_fi, x_age, x_gender, x_race, x_edu, x_married, x_site]


def train_multi(model, epoch, train_loader, valid_loader, test_loader, optimizer, losses, output_dir, checkpoint_epoch=0):
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

            outputs = model(batch_img)
            loss = 0
            
            for j in range(len(outputs)):
                criterion = losses[j]
                output = outputs[j]
                target = torch.tensor([t[j] for t in batch_target]).squeeze().cuda()

                # For cross entropy loss, need long tensors
                if k in [2, 3, 4, 5, 6]:
                    cur_loss = criterion(output, target.long())
                else:
                    cur_loss = criterion(output, target.float())
                
                loss += cur_loss
            
            loss_hist.append(loss.item())
            loss.backward() 
            optimizer.step()
            
            progress += len(batch_img)
            LOGGER.info('End batch {}: [{}/{}]'.format(batch_idx, batch_idx * len(batch_img), len(train_loader.dataset)))

            if batch_idx % 10 == 0:
                LOGGER.info('Train Epoch {}: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(i, progress, len(train_loader.dataset), progress / len(train_loader.dataset) * 100, loss.item()))
            
            #torch.cuda.empty_cache()
            #del batch_img, batch_target
        
        epoch_end = time.time()
        epoch_train_time = epoch_end - epoch_start

        cur_mse = eval_multi(model, valid_loader, losses)
        results.write('Epoch {}: {} ({} s)\n'.format(i, cur_mse, epoch_train_time))
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
            

def eval_multi(model, valid_loader, losses, save=False, output_dir=None):
    model.eval()

    target_true = [[] for i in range(21)]
    target_pred = [[] for i in range(21)]

    with torch.no_grad():
        progress = 0
        for batch_idx, (batch_img, batch_target) in enumerate(valid_loader):
            LOGGER.info('Evaluating batch {}: [{}/{}]'.format(batch_idx, progress, len(valid_loader.dataset)))
            
            devs = model.get_devices()
            batch_img = batch_img.unsqueeze(1).float().to(devs[0])

            outputs = model(batch_img)
    
            # Adding predicted and true targets
            for j in range(len(outputs)):
                target_true[j].extend(torch.tensor([t[j] for t in batch_target]).squeeze().cpu())
                target_pred[j].extend(outputs[j].squeeze().cpu())

            if batch_idx % 10 == 0:
                LOGGER.info('Eval Progress: [{}/{} ({:.0f}%)]'.format(
                progress, len(valid_loader.dataset), progress / len(valid_loader)))     
            
            progress += len(batch_img)
    
    if valid_loader.dataset.log:
        target_true = np.subtract(np.exp(target_true), 40)
        target_pred = np.subtract(np.exp(target_pred), 40)

    # MSE of fluid intelligence
    mse = mean_squared_error(target_true[11], target_pred[11])
    LOGGER.info('Mean squared error: {}'.format(mse))

    if save:
        try:
            np.save(os.path.join(output_dir, 'target_true.npy'), target_true)
            np.save(os.path.join(output_dir, 'target_pred.npy'), target_pred)
        except:
            raise Exception('Could not save ground truth & predictions to file')

    return mse

def train_multi_input_output(model, epoch, train_loader, valid_loader, test_loader, optimizer, losses, output_dir, checkpoint_epoch=0):
    model.train()

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

        progress = 0
        for batch_idx, (batch_img, batch_target) in enumerate(train_loader):
            LOGGER.info('Starting batch {}: [{}/{}]'.format(batch_idx, progress, len(train_loader.dataset)))

            optimizer.zero_grad()
            loss = 0
            
            # Moving each type of image to respective GPUs
            for j in range(len(batch_img)):
                devs = model.get_devices()
                batch_img[j] = batch_img[j].unsqueeze(1).float().to(devs[j])
            
            outputs = model(batch_img)

            for k in range(len(outputs)):
                criterion = losses[k]
                output = outputs[k].squeeze()
                target = torch.tensor([t[k] for t in batch_target]).squeeze().cuda()

                # For cross entropy loss, need long tensors
                if k in [2, 3, 4, 5, 6]:
                    cur_loss = criterion(output, target.long())
                else:
                    cur_loss = criterion(output, target.float())
                
                loss += cur_loss
            
            loss_hist.append(loss.item())
            loss.backward()
            optimizer.step()
            
            progress += len(batch_img[0])
            LOGGER.info('End batch {}: [{}/{}]'.format(batch_idx, progress, len(train_loader.dataset)))

            if batch_idx % 10 == 0:
                LOGGER.info('Train Epoch {}: [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    i, progress, len(train_loader.dataset), progress / len(train_loader.dataset) * 100,
                    loss.item()))

        
        epoch_end = time.time()
        epoch_train_time = epoch_end - epoch_start

        cur_mse = eval_multi_input_output(model, valid_loader, loss)
        test_mse = eval_multi_input_output(model, test_loader, loss)
        results.write('Epoch {}: Validation {} Test {} ({} s)\n'.format(i, cur_mse, test_mse, epoch_train_time))
        results.flush()
        torch.save({
            'epoch': i,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'loss': losses
        }, '{}_epoch_{}.pth'.format(model._get_name(), i))

        if cur_mse < best_mse:
            best_mse = cur_mse
            torch.save({
                'epoch': i,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': losses
            }, 'best_epoch_{}.pth'.format(i))
        
        np.save(os.path.join(output_dir, 'loss_history_train.npy'), loss_hist)
    
    results.close()

def eval_multi_input_output(model, valid_loader, losses, save=False, output_dir=None):
    model.eval()
    
    target_true = [[] for i in range(21)]
    target_pred = [[] for i in range(21)]

    with torch.no_grad():
        progress = 0
        for batch_idx, (batch_img, batch_target) in enumerate(valid_loader):
            LOGGER.info('Evaluating batch {}: [{}/{}]'.format(batch_idx, progress, len(valid_loader.dataset)))
            
            for j in range(len(batch_img)):
                devs = model.get_devices()
                batch_img[j] = batch_img[j].unsqueeze(1).float().to(devs[j])

            outputs = model(batch_img)
    
            # Adding predicted and true targets
            for j in range(len(outputs)):
                target_true[j].extend(torch.tensor([t[j] for t in batch_target]).squeeze().cpu())
                target_pred[j].extend(outputs[j].squeeze().cpu())

            if batch_idx % 10 == 0:
                LOGGER.info('Eval Progress: [{}/{} ({:.0f}%)]'.format(
                progress, len(valid_loader.dataset), progress / len(valid_loader)))     
            
            progress += len(batch_img[0])
    
    if valid_loader.dataset.log:
        target_true = np.subtract(np.exp(target_true), 40)
        target_pred = np.subtract(np.exp(target_pred), 40)

    # MSE of fluid intelligence
    mse = mean_squared_error(target_true[11], target_pred[11])
    LOGGER.info('Mean squared error: {}'.format(mse))

    if save:
        try:
            np.save(os.path.join(output_dir, 'target_true.npy'), target_true)
            np.save(os.path.join(output_dir, 'target_pred.npy'), target_pred)
        except:
            raise Exception('Could not save ground truth & predictions to file')

    return mse
