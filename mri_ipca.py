import torch
import os
import numpy as np
import argparse
from mri_dataset import MultiMRIDataset
from sklearn.decomposition import PCA, IncrementalPCA

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='IPCA for images')
    parser.add_argument('--data_dir', help='Directory path for datasets')
    parser.add_argument('--mri_type', help='T1')
    args = parser.parse_args()

    train_img       =   np.load(os.path.join(args.data_dir, 'train_data_img_{}.npy'.format(args.mri_type)), allow_pickle=True)
    train_target    =   np.load(os.path.join(args.data_dir, 'train_data_target.npy'), allow_pickle=True)

    ipca            =   IncrementalPCA(n_components=2, batch_size=10)
    train_dataset   =   MultiMRIDataset(train_img, train_target, resize=0)
    train_loader    =   torch.utils.data.DataLoader(train_dataset, batch_size=10, shuffle=True)

    sites           =   []
    transformed_img =   []

    print('Fitting IPCA')
    # Partially fitting batch at a time
    for batch_img, _ in train_loader:
        batch_img = [img.numpy().flatten() for img in batch_img]
        ipca.partial_fit(batch_img)

    print('Transforming images')
    # Transforming the imgs batch at a time
    for batch_img, batch_target in train_loader:
        batch_img = [img.flatten() for img in batch_img]
        cur_trans = ipca.transform(batch_img)
        transformed_img.extend(cur_trans)

        cur_sites = [t[6] for t in batch_target]
        sites.extend(cur_sites)

    
    


    

