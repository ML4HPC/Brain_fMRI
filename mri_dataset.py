from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
import numpy as np
from scipy.ndimage import zoom

class MRIDataset(Dataset):
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

        if self.resize > 0:
            np.resize(x, (self.resize, self.resize, self.resize))
        
        if self.normalize:
            x = np.divide(np.subtract(x, self.mean), self.std)
        
        if self.log:
            y = np.log(y+40)
    
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

        if self.resize > 0:
            np.resize(x, (self.resize, self.resize, self.resize))
        
        if self.normalize:
            x = np.divide(np.subtract(x, self.mean), self.std)
        
        if self.log:
            y[0] = np.log(y[0]+40)
    
        return (x, y)
    

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
        

            
        
