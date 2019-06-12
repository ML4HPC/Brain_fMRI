## https://discuss.pytorch.org/t/solved-concatenate-time-distributed-cnn-with-lstm/15435/4
## https://machinelearningmastery.com/cnn-long-short-term-memory-networks/

import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable



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






'''
input_data = np.ndarray(shape=(20,30,1,28,28), dtype='float')  ## batch,time,H,W
print("input data dimension: ", input_data.size)

target = np.ones(shape=(20, 1), dtype='float') # batch, score
print("input target dimension: ", target)
'''

train_img = np.load('train_data_img.npy')
valid_img = np.load('valid_data_img.npy')
train_target = np.load('train_data_target.npy')
valid_target = np.load('valid_data_target.npy')

'''
for _, train_list in enumerate(train_data.item().values()):
    train_img, train_target = train_list[0], train_list[1]

for _, valid_list in enumerate(valid_data.item().values()):
    valid_img, valid_target = valid_list[0], valid_list[1]
'''


#train_img = np.array(list(train_img), dtype=np.float)
#valid_img = np.asarray(valid_img)

#print(valid_img.shape)

'''
train_img = torch.FloatTensor(train_img)
train_target = torch.FloatTensor(train_target)
valid_img = torch.FloatTensor(valid_img)
valid_target = torch.FloatTensor(valid_target)
'''



def train(epoch, input_data, target, batch_size):
    model.train()
    loss = nn.L1Loss()

    model.cuda()
    loss = loss.cuda()
    




    for i in range(epoch):


        
        batch_idx = random.sample(range(1, len(target)), batch_size)
        print(batch_idx)


        batch_img = []
        batch_target = []
        for i in batch_idx:
            batch_img.append(np.array(input_data[i].dataobj))
            batch_target.append(target[i])
        
        batch_img = np.array(batch_img)
        batch_target = np.array(batch_target)
    

        batch_img = torch.FloatTensor(batch_img)
        batch_target = torch.FloatTensor(batch_target)

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







#input_data = torch.FloatTensor(input_data)
#print(input_data.size())



#model = CombineRNN()
model = CNN()

lr = 0.01 * 1000
momentum = 0.5
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)




print("now in training!\n")
train(1000, train_img, train_target, 8)
print("training finished!\n")

#print("now in evaluation!\n")
#model.eval()
#output = model(input_data)
#print("evaluation finished!\n")

#print(output)

#output = output.detach().numpy()
#print(output)
#print(output.shape)











