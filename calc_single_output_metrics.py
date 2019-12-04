import os
import numpy as np
import argparse
from sklearn.metrics import r2_score, mean_squared_error

def calc_mse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred)

def calc_r2(y_true, y_pred):
    return r2_score(y_true, y_pred)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculating single output metrics (MSE + R2)')
    parser.add_argument('--data_dir', help='Directory path for datasets')
    args = parser.parse_args()

    valid_pred = np.load(os.path.join(args.data_dir, 'valid_target_pred.npy'))
    valid_true = np.load(os.path.join(args.data_dir, 'valid_target_true.npy'))
    test_pred = np.load(os.path.join(args.data_dir, 'test_target_pred.npy'))
    test_true = np.load(os.path.join(args.data_dir, 'test_target_true.npy'))

    valid_r2 = calc_r2(valid_true, valid_pred)
    test_r2 = calc_r2(test_true, test_pred)

    valid_mse = calc_mse(valid_true, valid_pred)
    test_mse = calc_mse(test_true, test_pred)

    with open(os.path.join(args.data_dir, 'metrics.txt'), 'w+') as output:
        output.write('Valid R2: {}\t Test R2: {}\nValid MSE: {}\t Test MSE: {}\n'.format(valid_r2, test_r2, valid_mse, test_mse))
    
    
        
    
    