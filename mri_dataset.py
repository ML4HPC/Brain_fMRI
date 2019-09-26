import torch
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
import numpy as np
from scipy.ndimage import zoom

class MRIDataset(Dataset):
    def __init__(self, input_data, target, resize, normalize=False, log=False, nan=False):
        self.X_data = input_data
        self.Y_data = target
        self.resize = resize
        self.normalize = normalize
        self.log = log
        self.nan = nan
        self.mean = 70.4099
        self.std = 190.856

        if self.normalize:
            print('Normalization applied to dataset')
        if self.log:
            print('Log applied to dataset')
        if self.nan:
            print('NaN replaced with 0s')
    
    def __len__(self):
        return len(self.Y_data)

    def get_x_data(self):
        return [np.array(x.dataobj) for x in self.X_data]
    
    def get_y_data(self):
        return [y for y in self.Y_data]
        
    def __getitem__(self, idx):
        x = np.array(self.X_data[idx].dataobj)
        y = self.Y_data[idx]
        
        if self.nan:
            x = np.nan_to_num(x)
            assert(not np.isnan(x).any())
            
        if self.resize > 0:
            np.resize(x, (self.resize, self.resize, self.resize))
        
        if self.normalize:
            x = np.divide(np.subtract(x, self.mean), self.std)
        
        if self.log:
            y = np.log(y+100)
    
        return (x, y)

class MultiMRIDataset(Dataset):
    def __init__(self, input_data, target, resize, normalize=False, log=False):
        self.X_data = input_data
        self.Y_data = target
        self.resize = resize
        self.normalize = normalize
        self.log = log
        self.mean = 70.4099
        self.std = 190.856

        if self.normalize:
            print('Normalization applied to dataset')
        if self.log:
            print('Log applied to dataset')
    
    def __len__(self):
        return len(self.Y_data)

    def get_x_data(self):
        return [np.array(x.dataobj) for x in self.X_data]
    
    def get_y_data(self):
        return [y for y in self.Y_data]
        
    def __getitem__(self, idx):
        # img_data = np.array(self.X_data[idx].dataobj)
        # if self.resize:
        #     img_data = zoom(img_data, self.resize)
        #dim = 90
        x = np.array(self.X_data[idx].dataobj)
        y = self.Y_data[idx]

        # Converting age from months to year
        y[1] = y[1] / 12.0

        if self.resize > 0:
            np.resize(x, (self.resize, self.resize, self.resize))
        
        if self.normalize:
            x = np.divide(np.subtract(x, self.mean), self.std)
        
        if self.log:
            y[0] = np.log(y[0]+100)
    
        return (x, y)

class ThreeInputMRIDataset(Dataset):
    def __init__(self, input_data1, input_data2, input_data3, target, resize, normalize=False, log=False):
        self.dataset1 = MRIDataset(input_data1, target, resize, normalize, log)
        self.dataset2 = MRIDataset(input_data2, target, resize, normalize=False, log=False, nan=True)
        self.dataset3 = MRIDataset(input_data3, target, resize, normalize=False, log=False, nan=True)
        self.resize = resize
        self.normalize = normalize
        self.log = log
    
    def __len__(self):
        return len(self.dataset1.get_y_data())

    def get_x_data(self):
        return [self.dataset1.get_x_data(), self.dataset2.get_x_data(), self.dataset3.get_x_data()]
    
    def get_y_data(self):
        return self.dataset1.get_y_data()
        
    def __getitem__(self, idx):
        return ([self.dataset1[idx][0], self.dataset2[idx][0], self.dataset3[idx][0]], self.dataset1[idx][1])
    
    
class SliceMRIDataset(Dataset):
    def __init__(self, dataset, collate_fn=default_collate):
        self.dataset = dataset
        self.collate_fn = collate_fn
        self._indices = list(range(len(self.dataset)))
    
    def __len__(self):
        return len(self.dataset)
    
    @property
    def shape(self):
        return len(self),
    
    def __getitem__(self, i):
        if isinstance(i, (int, np.integer)):
            Xb = self.dataset[i][0]
            return Xb
        
        if isinstance(i, slice):
            i = self._indices[i]
        
        Xb = self.collate_fn([self.dataset[j][0] for j in i])

        return Xb
        

            
        
