import torch
import numpy as np
from mri_dataset import MRIDataset
from model_3d import eval
import argparse
import os
import IPython


if __name__ == "__main__":
    # Parsing arguments
    parser = argparse.ArgumentParser(description='Calculating mean & std dev for normalization')
    parser.add_argument('--data_dir')
    parser.add_argument('--train_batch_size', type=int, default=100)
    parser.add_argument('--resize', type=int, default=0)
    args = parser.parse_args()


    # Load and create datasets
    train_img_T1 = np.load(os.path.join(args.data_dir, 'train_data_img_T1.npy'), allow_pickle=True)
    train_img_T2 = np.load(os.path.join(args.data_dir, 'train_data_img_T2.npy'), allow_pickle=True)
    train_img_FA = np.load(os.path.join(args.data_dir, 'train_data_img_FA.npy'), allow_pickle=True)
    train_img_MD = np.load(os.path.join(args.data_dir, 'train_data_img_MD.npy'), allow_pickle=True)
    train_img_AD = np.load(os.path.join(args.data_dir, 'train_data_img_AD.npy'), allow_pickle=True)
    train_img_RD = np.load(os.path.join(args.data_dir, 'train_data_img_RD.npy'), allow_pickle=True)

    train_dataset_T1 = MRIDataset(train_img_T1, [], args.resize)
    train_dataset_T2 = MRIDataset(train_img_T2, [], args.resize)
    train_dataset_FA = MRIDataset(train_img_FA, []], args.resize)
    train_dataset_MD = MRIDataset(train_img_MD, [], args.resize)
    train_dataset_AD = MRIDataset(train_img_AD, [], args.resize)
    train_dataset_RD = MRIDataset(train_img_RD, [], args.resize)

    train_loader_T1 = torch.utils.data.DataLoader(train_dataset_T1, batch_size=args.train_batch_size)
    train_loader_T2 = torch.utils.data.DataLoader(train_dataset_T2, batch_size=args.train_batch_size)
    train_loader_FA = torch.utils.data.DataLoader(train_dataset_FA, batch_size=args.train_batch_size)
    train_loader_MD = torch.utils.data.DataLoader(train_dataset_MD, batch_size=args.train_batch_size)
    train_loader_AD = torch.utils.data.DataLoader(train_dataset_AD, batch_size=args.train_batch_size)
    train_loader_RD = torch.utils.data.DataLoader(train_dataset_RD, batch_size=args.train_batch_size)

    means = [0.0] * 6
    dims = [0.0] * 6
    vars = [0.0] * 6
    stds = [0.0] * 6

    train_loader_list = [train_loader_T1, train_loader_T2, train_loader_FA, train_loader_MD, train_loader_AD, train_loader_RD]

    for i in range(len(train_loader_list)):
        print('Processing {}\n'.format(i))
        is_first = True
        train_loader = train_loader_list[i]
        for batch_img, _ in train_loader:
            if is_first:
                img_shape = batch_img[0].shape
                dims[i] = img_shape[0] * img_shape[1] * img_shape[2]
                is_first = False
            means[i] += np.sum(batch_img)
        
        means[i] /= (len(train_loader.dataset) * dims[i])
        
        for batch_img, _ in train_loader:
            vars[i] += np.sum(np.power(np.subtract(batch_img, means[i]), 2))
        
        vars[i] /= (len(train_loader.dataset) * dims[i])
        stds[i] = np.sqrt(vars[i])
    
    np.save('means.npy', means)
    np.save('stds.npy', stds)
    