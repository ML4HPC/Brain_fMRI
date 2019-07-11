from torch.utils.data import Dataset
import numpy as np

class MRIDataset(Dataset):
    def __init__(self, input_data, target):
        self.X_data = input_data
        self.Y_data = target
    
    def __len__(self):
        return len(self.Y_data)
        
    def __getitem__(self, idx):
        return (np.array(self.X_data[idx].dataobj), self.Y_data[idx])