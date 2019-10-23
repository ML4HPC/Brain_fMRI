import torch
import numpy as np
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
    valid_scores = []
    test_scores = []

    for i in range(len(valid_target_pred)):
        print('Processing class {}/21'.format(i+1))
        score = None
        y_true = torch.from_numpy(valid_target_true[i])
        y_pred = torch.from_numpy(valid_target_pred[i])

        if i in bin_idx:
            score = outputs_bin_classification_report(y_true, y_pred)
        elif i in cat_idx:
            score = outputs_multi_classification_report(y_true, y_pred)
        else:
            cur_r2 = r2_score(y_true, y_pred)
            cur_mse = mean_squared_error(y_true, y_pred)
            score = {'R2': cur_r2, 'MSE': cur_mse}
        
        valid_scores.append(score)
    
    for i in range(len(test_target_pred)):
        print('Processing class {}/21'.format(i+1))
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
        
        test_scores.append(score)
    
    np.save(os.path.join(args.output_dir, 'multi_output_valid_scores.npy'), valid_scores)
    np.save(os.path.join(args.output_dir, 'multi_output_test_scores.npy'), test_scores)
    print('Scores saved')
    


    
