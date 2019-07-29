import os
import nibabel as nib
import numpy as np
import argparse
from scipy.ndimage import zoom

path = '/global/cscratch1/sd/yanzhang/data_brain/image03/'
train = 'training/'
valid = 'validation/'
test = 'testing/'
lattername = '/baseline/structural/t1_brain.nii.gz'


def readimages(path, data_for, lattername, resize):
    filenames = os.listdir(path+data_for)
    images = {}
    i = 0
    for name in filenames:
        i += 1
        if i % 300 == 0:
            print(i)
        #print(name)
        full_path = path+data_for+name+lattername
        img = nib.load(full_path)

        # Resize, if necessary
        img_data = np.array(img.dataobj)
        if resize:
            img_data = zoom(img_data, resize)
            
        resized_img = nib.Nifti1Image(img_data, img.affine, img.header)

        images[name] = img
#        print(img)
    return images

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read and process images')
    parser.add_argument('--resize', type=float)
    args = parser.parse_args()

    print('processing training!')
    train_img = readimages(path, train, lattername, args.resize)
    print('processing valid!')
    valid_img = readimages(path, valid, lattername, args.resize)
    print('processing test!')
    test_img = readimages(path, test, lattername, args.resize)


    print('saving images!')
    np.save('train_img.npy', train_img)
    np.save('valid_img.npy', valid_img)
    np.save('test_img.npy', test_img)
    print('done saving!')


