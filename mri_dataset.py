from torch.utils.data import Dataset
import numpy as np
from scipy.ndimage import zoom

class MRIDataset(Dataset):
    def __init__(self, input_data, target, resize):
        self.X_data = input_data
        self.Y_data = target
        if resize:
            self.resize = (resize, resize, resize)
        else:
            self.resize = 0
    
    def __len__(self):
        return len(self.Y_data)
        
    def __getitem__(self, idx):
        # img_data = np.array(self.X_data[idx].dataobj)
        # if self.resize:
        #     img_data = zoom(img_data, self.resize)
        dim = 90
        x = np.array(self.X_data[idx].dataobj)
        x = np.resize(x, (dim, dim, dim))
    
        return (x, self.Y_data[idx])

        # return (img_data, self.Y_data[idx])
