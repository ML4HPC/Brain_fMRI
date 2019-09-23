import os
import nibabel as nib
import numpy as np
import argparse
from scipy.ndimage import zoom
import IPython


def readimages_dti(path, resize,  dti_type, output_dir):
    filenames = os.listdir(path)
    images = {}
    batch_idx = 0
    for i in range(len(filenames)):
        batch_idx += 1
        name = filenames[i]

        sub_start = name.index('-') + 1
        sub_end = name.index('_')

        if name[sub_end+1:sub_end+3] != dti_type.upper():
            continue
        
        subject = name[sub_start:sub_end]

        # Print progress
        if i % 100 == 0:
            print(i)
        
        full_path = path+'/'+name
        img = nib.load(full_path)

        # Resize, if necessary
        if resize:
            img_data = np.array(img.dataobj)
            img_data = zoom(img_data, resize)
            img = nib.Nifti1Image(img_data, img.affine, img.header)

        images[subject] = img

        if resize:
            if (i != 0)  and (i % 1000 == 0 or i == (len(filenames) - 1)):
                np.save(os.path.join(args.output_dir, 'img_{}.npy'.format(batch_idx)), images)
                images.clear()
    
    return images


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Read and process images')
    parser.add_argument('--resize', type=float, default=None)
    parser.add_argument('--data_dir', help='Path to dataset images')
    parser.add_argument('--output_dir', help='Path to directory for saving outputs')
    parser.add_argument('--dti_type', help="Type of DTI: FA / MD")
    args = parser.parse_args()

    if not os.path.isdir(args.output_dir):
        try:
            os.mkdir(args.output_dir)
        except:
            raise Exception('Could not create output directory')

    print('Processing all images: train, valid, and test!')
    all_img = readimages_dti(args.data_dir, args.resize, args.dti_type, args.output_dir,) 
    np.save(os.path.join(args.output_dir, 'all_img_{}.npy'.format(args.dti_type)), all_img)


