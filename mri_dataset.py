from torch.utils.data import Dataset
import numpy as np
from scipy.ndimage import zoom

class MRIDataset(Dataset):
    def __init__(self, input_data, target, resize):
        self.X_data = input_data
        self.Y_data = target
        self.resize = resize
    
    def __len__(self):
        return len(self.Y_data)

    def get_x_data(self):
        return [x.dataobj for x in self.X_data]
    
    def get_y_data(self):
        return self.Y_data
        
    def __getitem__(self, idx):
        # img_data = np.array(self.X_data[idx].dataobj)
        # if self.resize:
        #     img_data = zoom(img_data, self.resize)
        #dim = 90
        x = np.array(self.X_data[idx].dataobj)
        
        if self.resize > 0:
            x = np.resize(x, (self.resize, self.resize, self.resize))
    
        return (x, self.Y_data[idx])

        # return (img_data, self.Y_data[idx])
