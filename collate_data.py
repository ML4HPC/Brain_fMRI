import os
import numpy as np
import argparse
import logging
import sys

# Setting up logger
LOGGER = logging.getLogger(__name__)
out_hdlr = logging.StreamHandler(sys.stdout)
out_hdlr.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
out_hdlr.setLevel(logging.INFO)
LOGGER.addHandler(out_hdlr)
LOGGER.setLevel(logging.INFO)


def agg_data(data_dir, output_dir, data_class):
    LOGGER.info('Aggregating data')
    filenames = sorted(os.listdir(args.data_dir))

    agg = {}
    
    for name in filenames:
        LOGGER.info('Collating next file: {}'.format(name))
        if not agg:
            agg = np.load(os.path.join(data_dir, name), allow_pickle=True).item()
        else:
            new_data = np.load(os.path.join(data_dir, name), allow_pickle=True).item()
            agg.update(new_data)  

    LOGGER.info('Saving aggregated data')
    np.save(os.path.join(output_dir, data_class), agg)      

    return agg


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Aggregate all images')
    parser.add_argument('--data_dir', help='Path to dataset images')
    parser.add_argument('--output_dir', help='Path to directory for saving outputs')
    parser.add_argument('--data_class', help='Class of data: train, valid, test')
    args = parser.parse_args()

    agg_data(args.data_dir, args.output_dir, args.data_class)

    
    
    
    
    
