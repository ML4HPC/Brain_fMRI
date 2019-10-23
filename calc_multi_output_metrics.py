import torch
import torch.optim as optim
import torch.nn as nn
import multi_input_resnet3d 
import numpy as np
from mri_dataset import MultiMRIDataset
from model_3d_multi import train_multi, eval_multi
from metrics import outputs_multi_classification_report, outputs_bin_classification_report
from sklearn.metrics import r2_score, mean_squared_error
import argparse
import os
import apex
import IPython

if __name__ == "__main__":
    # Parsing arguments
    parser = argparse.ArgumentParser(description='Multi-channel/input ResNet3D for regression')
    parser.add_argument('--data_dir', help='Directory path for datasets')
    parser.add_argument('--output_dir', help='Directory path for outputs')
    args = parser.parse_args()

    # Load and create datasets
    valid_target_pred   =   np.load(os.path.join(args.data_dir, 'valid_target_pred.npy'), allow_pickle=True)
    valid_target_true   =   np.load(os.path.join(args.data_dir, 'valid_target_true.npy'), allow_pickle=True)

    test_target_pred    =   np.load(os.path.join(args.data_dir, 'test_target_pred.npy'), allow_pickle=True)
    test_target_true    =   np.load(os.path.join(args.data_dir, 'test_target_true.npy'), allow_pickle=True)

    # Indices for each type of variable
    bin_idx = [1]
    cat_idx = [2, 3, 4, 5, 6]
    scores = []

    for i in range(len(valid_target_pred)):
        score = None
        y_true = valid_target_true[i]
        y_pred = valid_target_pred[i]

        if i in bin_idx:
            score = outputs_bin_classification_report(y_true, y_pred)
        elif i in cat_idx:
            score = outputs_multi_classification_report(y_true, y_pred)
        else:
            cur_r2 = r2_score(y_true, y_pred)
            cur_mse = mean_squared_error(y_true, y_pred)
            score = {'R2': cur_r2, 'MSE': cur_mse}
        
        scores.append(score)


    
