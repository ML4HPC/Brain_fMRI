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
    """
    i = 0
    while i <  len(train_dataset):
        print('Processing {}'.format(i))

        batch_size = min(10, len(train_dataset)-i-1)
        batch_img = [None] * batch_size
        
        for j in range(batch_size):
            batch_img[j] = train_dataset[i][0].flatten()
            i += 1
        # batch_img = [img.numpy().flatten() for img in batch_img]
        ipca.partial_fit(batch_img)

    
    print('Transforming images')
    # Transforming the imgs batch at a time
    i = 0
    while i <  len(train_dataset):
        print('Processing {}'.format(i))

        batch_size = min(10, len(train_dataset)-i-1)
        batch_img = [None] * batch_size
        batch_target = []

        for j in range(batch_size):
            x, y = train_dataset[i]
            batch_img[j] = x.flatten()
            batch_target.append(y[6])
            i += 1
        #  batch_img = [img.flatten() for img in batch_img]
        cur_trans = ipca.transform(batch_img)
        transformed_img.extend(cur_trans)

        sites.extend(batch_target)
    """

    for batch_idx, (batch_img, _) in enumerate(train_loader):
        print('Processing batch idx: {}'.format(batch_idx))
        batch_img = [img.numpy().flatten() for img in batch_img]
        ipca.partial_fit(batch_img)
    
    for batch_idx, (batch_img, batch_target) in enumerate(train_loader):
        print('Processing batch idx: {}'.format(batch_idx))

        batch_img = [img.numpy().flatten() for img in batch_img]
        batch_target = [t[6] for t in batch_target]
        cur_trans = ipca.transform(batch_img)
        transformed_img.extend(cur_trans)
        sites.extend(batch_target)

    
    print('Saving pca data')
    np.save('pca_img.npy', transformed_img)
    np.save('pca_sites.npy', sites)
    
    


    

