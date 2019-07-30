import os
import nibabel as nib
import numpy as np
import argparse
from scipy.ndimage import zoom

#path = '/global/cscratch1/sd/yanzhang/data_brain/image03/'
train = 'training'
valid = 'validation'
test = 'testing'
lattername = '/baseline/structural/t1_brain.nii.gz'


def readimages(path, data_for, lattername, resize, output_dir):
    filenames = os.listdir(path+data_for)
    images = {}
    batch_idx = 0
    for i in range(len(filenames)):
        batch_idx += 1
        name = filenames[i]
        if i % 100 == 0:
            print(i)
        #print(name)
        full_path = path+data_for+'/'+name+lattername
        img = nib.load(full_path)

        # Resize, if necessary
        img_data = np.array(img.dataobj)
        if resize:
            img_data = zoom(img_data, resize)

        resized_img = nib.Nifti1Image(img_data, img.affine, img.header)

        images[name] = resized_img

        if (i != 0)  and (i % 1000 == 0 or i == (len(filenames) - 1)):
            np.save(os.path.join(args.output_dir, '{}_img_{}.npy'.format(data_for, batch_idx)), images)
            images.clear()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read and process images')
    parser.add_argument('--resize', type=float)
    parser.add_argument('--data_dir', help='Path to dataset images')
    parser.add_argument('--output_dir', help='Path to directory for saving outputs')
    args = parser.parse_args()

    try:
        os.mkdir(args.output_dir)
    except:
        raise Exception('Could not create output directory')

    print('processing training!')
    readimages(args.data_dir, train, lattername, args.resize, args.output_dir) 
    #np.save(os.path.join(args.output_dir, 'train_img.npy'), train_img)
    #print('saved train')
    print('processing valid!')
    readimages(args.data_dir, valid, lattername, args.resize, args.output_dir) 
    # np.save(os.path.join(args.output_dir, 'valid_img.npy'), valid_img)
    print('saved train')
    print('processing test!')
    readimages(args.data_dir, test, lattername, args.resize, args.output_dir)
    #np.save(os.path.join(args.output_dir, 'test_img.npy'), test_img)
    print('done saving!')


