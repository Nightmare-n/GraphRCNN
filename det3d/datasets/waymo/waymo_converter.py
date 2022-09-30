"""Tool to convert Waymo Open Dataset to pickle files.
    Adapted from https://github.com/WangYueFt/pillar-od
    # Copyright (c) Massachusetts Institute of Technology and its affiliates.
    # Licensed under MIT License
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob, argparse, tqdm, pickle, os 

import waymo_decoder
import tensorflow as tf
from waymo_open_dataset import dataset_pb2

from multiprocessing import Pool 

try:
    tf.enable_eager_execution()
except:
    pass

from det3d.utils import file_client
from easydict import EasyDict


backend = EasyDict({
    'name': 'HardDiskBackend',
    'kwargs': {}
})
client = getattr(file_client, backend.name)(
    **backend.kwargs
)

fnames = None 
LIDAR_PATH = None
ANNO_PATH = None 

def convert(idx):
    fname = fnames[idx]
    dataset = tf.data.TFRecordDataset(client._map_path(fname), compression_type='')
    for frame_id, data in enumerate(dataset):
        frame = dataset_pb2.Frame()
        frame.ParseFromString(bytearray(data.numpy()))
        decoded_frame = waymo_decoder.decode_frame(frame, frame_id)
        decoded_annos = waymo_decoder.decode_annos(frame, frame_id)

        client.dump_pickle(decoded_frame, os.path.join(LIDAR_PATH, 'seq_{}_frame_{}.pkl'.format(idx, frame_id)))
        client.dump_pickle(decoded_annos, os.path.join(ANNO_PATH, 'seq_{}_frame_{}.pkl'.format(idx, frame_id)))


def main():
    print("Number of files {}".format(len(fnames)))

    with Pool(128) as p: # change according to your cpu
        r = list(tqdm.tqdm(p.imap(convert, range(len(fnames))), total=len(fnames)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Waymo Data Converter')
    parser.add_argument('--root_path', type=str, required=True)
    parser.add_argument('--raw_data_tag', type=str, required=True)
    parser.add_argument('--processed_data_tag', type=str, required=True)
    parser.add_argument('--split', type=str, required=True)

    args = parser.parse_args()

    split_dir = os.path.join(args.root_path, 'ImageSets', args.split + '.txt')
    with client.get_local_path(split_dir) as path:
        fnames = [os.path.join(args.root_path, args.raw_data_tag, x.strip()) for x in open(path).readlines()]
    LIDAR_PATH = os.path.join(args.root_path, args.processed_data_tag, args.split, 'lidar')
    ANNO_PATH = os.path.join(args.root_path, args.processed_data_tag, args.split, 'annos')
    
    main()
