import os
import numpy as np
import argparse
import logging

# Setting up logger
LOGGER = logging.getLogger(__name__)
out_hdlr = logging.StreamHandler(sys.stdout)
out_hdlr.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
out_hdlr.setLevel(logging.INFO)
LOGGER.addHandler(out_hdlr)
LOGGER.setLevel(logging.INFO)


def agg_data(data_dir, output_dir):
    LOGGER.info('Aggregating data')
    filenames = sorted(os.listdir(args.data_dir))

    agg = {}
    
    for name in filenames:
        LOGGER.info('Collating next file')
        if not agg:
            agg = np.load(os.path.join(data_dir, name))
        else:
            new_data = np.load(os.path.join(data_dir, name))
            agg.update(new_data)        

    return agg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Aggregate all images')
    parser.add_argument('--data_dir', help='Path to dataset images')
    parser.add_argument('--output_dir', help='Path to directory for saving outputs')
    parser.add_argument('--data_class', help='Class of data: train, valid, test')
    args = parser.parse_args()

    try:
        os.mkdir(args.output_dir)
    except:
        raise Exception('Cannot create output folder')

    agg = agg_data(args.data_dir, args.output_dir)
    LOGGER.info('Saving aggregated data')
    np.save(agg, os.path.join(args.output_dir, args.data_class))
    
    
    
    
    