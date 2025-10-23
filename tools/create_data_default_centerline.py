# Copyright (c) OpenMMLab. All rights reserved.
import argparse
from os import path as osp

from typing import List, Tuple, Union
from projects.SeqGrowGraph.seq_grow_graph import nuscenes_converter_default_centerline as nuscenes_converter


def nuscenes_data_prep(root_path,
                       info_prefix,
                       version,
                       dataset_name,
                       out_dir,
                       map_size,
                       max_sweeps=10):
    """Prepare data related to nuScenes dataset.

    Related data consists of '.pkl' files recording basic infos,
    2D annotations and groundtruth database.

    Args:
        root_path (str): Path of dataset root.
        info_prefix (str): The prefix of info filenames.
        version (str): Dataset version.
        dataset_name (str): The dataset class name.
        out_dir (str): Output directory of the groundtruth database info.
        max_sweeps (int, optional): Number of input consecutive frames.
            Default: 10
    """
    nuscenes_converter.create_nuscenes_infos(
        root_path, info_prefix=info_prefix, version=version, max_sweeps=max_sweeps)


parser = argparse.ArgumentParser(description='Data converter arg parser')
parser.add_argument('dataset', metavar='nuscenes', help='name of the dataset')
parser.add_argument(
    '--root-path',
    type=str,
    default='./data/nuscenes',
    help='specify the root path of dataset')

parser.add_argument(
    '--nusc_canbus_root',
    type=str,
    default='./data/nuscenes/',
    help='specify the root path of nuscene can bus dataset')

parser.add_argument(
    '--version',
    type=str,
    default='v1.0',
    required=False,
    help='specify the dataset version, no need for kitti')
parser.add_argument(
    '--max-sweeps',
    type=int,
    default=10,
    required=False,
    help='specify sweeps of lidar per example')
parser.add_argument(
    '--with-plane',
    action='store_true',
    help='Whether to use plane information for kitti.')
parser.add_argument(
    '--map_resolution',
    type=float,
    default=0.5,
    required=False,  # True
    help='specify map centerline sample resolution')
parser.add_argument(
    '--map_size',
    type=float,
    nargs='+',
    default=[-70, 70, -70, 70],
    required=False,  # True
    help='map size : -x x -y y')
parser.add_argument(
    '--out-dir',
    type=str,
    default='./data/nuscenes',
    required=False,
    help='name of info pkl')
parser.add_argument('--extra-tag', type=str, default='nuscenes_centerline')
parser.add_argument(
    '--workers', type=int, default=4, help='number of threads to be used')
args = parser.parse_args()

if __name__ == '__main__':
    map_size = [(args.map_size[0], args.map_size[1]), (args.map_size[2], args.map_size[3])]
    if args.dataset == 'nuscenes' and args.version != 'v1.0-mini':
        train_version = f'{args.version}-trainval'
        nuscenes_data_prep(
            root_path=args.root_path,
            info_prefix=args.extra_tag,
            version=train_version,
            dataset_name='NuScenesDataset',
            out_dir=args.out_dir,
            map_size=map_size,
            max_sweeps=args.max_sweeps)
    elif args.dataset == 'nuscenes' and args.version == 'v1.0-mini':
        train_version = f'{args.version}'
        nuscenes_data_prep(
            root_path=args.root_path,
            can_bus_root_path=args.nusc_canbus_root,
            info_prefix=args.extra_tag,
            version=train_version,
            dataset_name='NuScenesDataset',
            out_dir=args.out_dir,
            max_sweeps=args.max_sweeps)