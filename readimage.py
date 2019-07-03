import os
import nibabel as nib
import numpy as np

path = '/global/cscratch1/sd/yanzhang/data_brain/image03/'
train = 'training/'
valid = 'validation/'
test = 'testing/'
lattername = '/baseline/structural/t1_brain.nii.gz'


def readimages(path, data_for, lattername):
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
        images[name] = img
#        print(img)
    return images

print('processing training!')
train_img = readimages(path, train, lattername)
print('processing valid!')
valid_img = readimages(path, valid, lattername)
print('processing test!')
test_img = readimages(path, test, lattername)


print('saving images!')
np.save('train_img.npy', train_img)
np.save('valid_img.npy', valid_img)
np.save('test_img.npy', test_img)
print('done saving!')


