# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
import mmcv
import mmengine
import numpy as np
import os
import torch
import random
import math
import cv2
from mmdet3d.registry import TRANSFORMS
from einops import rearrange
from typing import List, Optional, Union
from mmdet.datasets.transforms import LoadAnnotations
from mmengine.fileio import get
from .depth_map_utils import fill_in_multiscale
from projects.SeqGrowGraph.seq_grow_graph.bz_roadnet_reach_dist_eval import get_geom
from .centerline_utils import SceneGraph, sentance2seq, sentance2bzseq, sentance2bzseq2, nodesbetween2seq,sentance2bzseqNew
from projects.SeqGrowGraph.seq_grow_graph.core.centerline import PryCenterLine, PryOrederedCenterLine, OrderedSceneGraph, NusOrederedBzCenterLine, NusOrederedRMcontinuedBzCenterLine,OrderedBzLaneGraph, OrderedBzSceneGraph, OrderedBzSceneGraphNew,OrderedBzPlSceneGraph, PryOrederedBzPlCenterLine, get_semiAR_seq, match_keypoints, float2int, get_semiAR_seq_fromInt, PryMonoOrederedBzCenterLine, PryMonoOrederedBzPlCenterLine, AV2OrederedBzCenterLine, AV2OrderedBzSceneGraph, AV2OrderedBzLaneGraph, AV2OrederedRMcontinuedBzCenterLine,AV2OrederedBzCenterLine_new, AV2OrderedBzSceneGraph_new, NusOrederedBzCenterLine, NusClearOrederedBzCenterLine, Laneseq2Graph,NusOrederedBzCenterLineIsometry,NusOrederedBzCenterLineEqualQuantity,NusOrederedBzCenterLineRandomCam
from projects.SeqGrowGraph.seq_grow_graph.core.centerline import seq2nodelist, EvalMapGraph, seq2bznodelist, EvalMapBzGraph, EvalMapBzPlGraph, convert_coeff_coord, seq2bzplnodelist, convert_plcoeff_coord
from projects.SeqGrowGraph.seq_grow_graph.core.centerline.structures.pryordered_bz_centerline import divide_line_with_shapely, split_line_include_original_points
from math import factorial

LOCATIONS = ['boston-seaport', 'singapore-onenorth', 'singapore-queenstown',
             'singapore-hollandvillage']

@TRANSFORMS.register_module()
class CustomLoadMultiViewImageFromFiles(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self, to_float32=False, padding=True,pad_val=128, color_type='unchanged'):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.padding = padding
        self.pad_val = pad_val

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data. \
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        filename = results['img_filename']
        # img is of shape (h, w, c, num_views)
        # img = np.stack(
        #     [mmcv.imread(name, self.color_type) for name in filename], axis=-1)
        img_list = [mmcv.imread(name, self.color_type) for name in filename]
        img_shape_list = [img.shape for img in img_list]
        max_h = max([shape[0] for shape in img_shape_list])
        max_w = max([shape[1] for shape in img_shape_list])
        size = (max_h, max_w)
        # import pdb;pdb.set_trace()
        img_list = [mmcv.impad(
                    img, shape=size, pad_val=self.pad_val) for img in img_list]

        img = np.stack(img_list,axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)
        results['filename'] = filename
        # unravel to list, see `DefaultFormatBundle` in formating.py
        # which will transpose each image separately and then stack into array
        results['img'] = [img[..., i] for i in range(img.shape[-1])]
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, '
        repr_str += f"color_type='{self.color_type}')"
        return repr_str

@TRANSFORMS.register_module()
class OrgLoadMultiViewImageFromFiles(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self, to_float32=False,
                 color_type='unchanged',
                 backend_args: Optional[dict] = None,):
        self.to_float32 = to_float32
        self.color_type = color_type
        self.backend_args = backend_args

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data. \
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        filename = results['img_filename']
        # img is of shape (h, w, c, num_views)
        img_bytes = [
            get(name, backend_args=self.backend_args) for name in filename
        ]
        img = [
            mmcv.imfrombytes(img_byte, flag=self.color_type)
            for img_byte in img_bytes
        ]
        img = np.stack(img, axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)
        results['filename'] = filename
        # unravel to list, see `DefaultFormatBundle` in formating.py
        # which will transpose each image separately and then stack into array
        results['img'] = [img[..., i] for i in range(img.shape[-1])]
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, '
        repr_str += f"color_type='{self.color_type}')"
        return repr_str


@TRANSFORMS.register_module()
class LoadFrontViewImageFromFiles(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self, to_float32=False, color_type='unchanged'):
        self.to_float32 = to_float32
        self.color_type = color_type

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data. \
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        filename = results['img_filename']
        # img is of shape (h, w, c, num_views)
        img = np.stack(
            [mmcv.imread(name, self.color_type) for name in filename if 'CAM_FRONT/' in name], axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)
        results['filename'] = filename
        # unravel to list, see `DefaultFormatBundle` in formating.py
        # which will transpose each image separately and then stack into array
        results['img'] = [img[..., i] for i in range(img.shape[-1])]
        results['img_shape'] = img.shape
        results['ori_shape'] = img.shape
        # Set initial values for default meta_keys
        results['pad_shape'] = img.shape
        results['scale_factor'] = 1.0
        num_channels = 1 if len(img.shape) < 3 else img.shape[2]
        results['img_norm_cfg'] = dict(
            mean=np.zeros(num_channels, dtype=np.float32),
            std=np.ones(num_channels, dtype=np.float32),
            to_rgb=False)
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, '
        repr_str += f"color_type='{self.color_type}')"
        return repr_str


@TRANSFORMS.register_module()
class LoadMonoCenterlineSegFromPkl(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.
    """

    def __init__(self, grid_conf, thickness=2, data_root=None):
        self.line = 255
        self.data_root = data_root
        self.thickness = thickness
        self.grid_conf = grid_conf
        dx, bx, nx = self.gen_dx_bx(self.grid_conf['xbound'],
                                    self.grid_conf['ybound'],
                                    self.grid_conf['zbound'],)
        self.dx = dx
        self.bx = bx
        self.nx = nx
        self.pc_range = np.concatenate(
            (self.bx - self.dx / 2., self.bx - self.dx / 2. + self.nx * self.dx))

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data. \
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        centerline_seg = np.zeros((int(self.nx[1]), int(self.nx[0])))
        center_lines = results['center_lines']['centerlines']
        for i in range(len(center_lines)):
            center_line = center_lines[i]
            inbev_x = np.logical_and(
                center_line[:, 0] < self.pc_range[3], center_line[:, 0] >= self.pc_range[0])
            inbev_y = np.logical_and(
                center_line[:, 1] < self.pc_range[4], center_line[:, 1] >= self.pc_range[1])
            inbev_xy = np.logical_and(inbev_x, inbev_y)

            center_line_homo = np.concatenate([center_line, np.ones(
                (center_line.shape[0], 1))], axis=1).reshape(center_line.shape[0], 4, 1)
            lidar2ego = results['lidar2ego']
            ego2lidar = np.linalg.inv(lidar2ego)
            lidar2img = results['lidar2img'][0]

            coords = lidar2img @ ego2lidar @ center_line_homo
            coords = np.squeeze(coords, axis=-1)

            depth = coords[..., 2]
            on_img = (coords[..., 2] > 1e-5)
            coords[..., 2] = np.clip(coords[..., 2], 1e-5, 1e5)
            coords[..., 0:2] /= coords[..., 2:3]
            coords = coords[..., :2]
            h, w = results['pad_shape'][0][:2]

            on_img = (on_img & (coords[..., 0] < w)
                      & (coords[..., 0] >= 0)
                      & (coords[..., 1] < h)
                      & (coords[..., 1] >= 0))

            keep_bev = np.logical_and(on_img, inbev_xy)
            # keep_bev = inbev_xy

            center_line = (center_line[keep_bev, :] -
                           self.pc_range[:3]) / self.dx
            center_line = np.floor(center_line).astype(np.int)
            for pt_i in range(len(center_line)-1):
                cv2.line(centerline_seg, tuple(center_line[pt_i, :2]), tuple(
                    center_line[pt_i+1, :2]), self.line, self.thickness)
        if self.data_root:
            filename = os.path.join(self.data_root, results['token'] + '.png')
            cv2.imwrite(filename, centerline_seg)
        centerline_seg[centerline_seg == self.line] = 1
        results['middle_seg'] = centerline_seg.astype(np.int64)
        return results

    @staticmethod
    def gen_dx_bx(xbound, ybound, zbound):
        dx = np.array([row[2] for row in [xbound, ybound, zbound]])
        bx = np.array(
            [row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])
        nx = np.floor(np.array([(row[1] - row[0]) / row[2]
                      for row in [xbound, ybound, zbound]]))
        return dx, bx, nx

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(line={self.line}, '
        return repr_str


@TRANSFORMS.register_module()
class CenterlineFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, results):
        prob = random.uniform(0, 1)
        centerlines = results['center_lines']
        lidar2ego = results['lidar2ego']
        if prob > self.prob:
            results['flip'] = None
        else:
            rot_mat = np.eye(4, dtype=lidar2ego.dtype)
            h_or_v = random.uniform(0, 1)
            if h_or_v > 0.5:
                flip_type = 'horizontal'
                rot_mat[0, 0] = -1.
            else:
                rot_mat[1, 1] = -1.
                flip_type = 'vertical'
            lidar2ego = rot_mat @ lidar2ego
            centerlines.flip(flip_type)
        results['center_lines'] = centerlines
        results['lidar2ego'] = lidar2ego
        return results


@TRANSFORMS.register_module()
class MonoCenterlineFlip(object):
    def __init__(self, prob=0.5):
        self.prob = prob

    def __call__(self, results):
        prob = random.uniform(0, 1)
        centerlines = results['center_lines']
        lidar2ego = results['lidar2ego']
        if prob > self.prob:
            results['flip'] = None
        else:
            rot_mat = np.eye(4, dtype=lidar2ego.dtype)
            h_or_v = random.uniform(0, 1)
            if h_or_v > 0.5:
                flip_type = 'horizontal'
                rot_mat[0, 0] = -1.
            else:
                rot_mat[1, 1] = -1.
                flip_type = 'vertical'
            lidar2ego = rot_mat @ lidar2ego
            centerlines.flip(flip_type)
        results['center_lines'] = centerlines
        results['lidar2ego'] = lidar2ego
        return results


@TRANSFORMS.register_module()
class CenterlineRotateScale(object):
    def __init__(self, prob=0.5,
                 max_rotate_degree=22.5,
                 scaling_ratio_range=(0.95, 1.05)):
        assert scaling_ratio_range[0] <= scaling_ratio_range[1]
        assert scaling_ratio_range[0] > 0
        self.prob = prob
        self.max_rotate_degree = max_rotate_degree
        self.scaling_ratio_range = scaling_ratio_range

    def __call__(self, results):
        prob = random.uniform(0, 1)
        centerlines = results['center_lines']
        lidar2ego = results['lidar2ego']
        if prob > self.prob:
            results['aug_matrix'] = None
        else:
            rotation_degree = random.uniform(-self.max_rotate_degree,
                                             self.max_rotate_degree)
            rotation_matrix = self._get_rotation_matrix(rotation_degree)
            scaling_ratio = random.uniform(self.scaling_ratio_range[0],
                                           self.scaling_ratio_range[1])
            scaling_matrix = self._get_scaling_matrix(scaling_ratio)
            centerlines.scale(scaling_ratio)
            centerlines.rotate(rotation_matrix)
            centerlines.filter_bev()
            aug_matrix = np.eye(4, dtype=lidar2ego.dtype)
            aug_matrix[:3, :3] = rotation_matrix @ scaling_matrix
            lidar2ego = aug_matrix @ lidar2ego
            results['center_lines'] = centerlines
            results['lidar2ego'] = lidar2ego
        return results

    @staticmethod
    def _get_rotation_matrix(rotate_degrees):
        radian = math.radians(rotate_degrees)
        rotation_matrix = np.array(
            [[np.cos(radian), -np.sin(radian), 0.],
             [np.sin(radian), np.cos(radian), 0.],
             [0., 0., 1.]],
            dtype=np.float32)
        return rotation_matrix

    @staticmethod
    def _get_scaling_matrix(scale_ratio):
        scaling_matrix = np.array(
            [[scale_ratio, 0., 0.], [0., scale_ratio, 0.],
             [0., 0., 1.]],
            dtype=np.float32)
        return scaling_matrix


@TRANSFORMS.register_module()
class MonoCenterlineRotateScale(object):
    def __init__(self, prob=0.5,
                 max_rotate_degree=22.5,
                 scaling_ratio_range=(0.95, 1.05)):
        assert scaling_ratio_range[0] <= scaling_ratio_range[1]
        assert scaling_ratio_range[0] > 0
        self.prob = prob
        self.max_rotate_degree = max_rotate_degree
        self.scaling_ratio_range = scaling_ratio_range

    def __call__(self, results):
        prob = random.uniform(0, 1)
        centerlines = results['center_lines']
        lidar2ego = results['lidar2ego']
        ego2lidar = np.linalg.inv(lidar2ego)
        lidar2img = results['lidar2img'][0]
        img_size = results['pad_shape'][0][:2]
        if prob > self.prob:
            results['aug_matrix'] = None
        else:
            rotation_degree = random.uniform(-self.max_rotate_degree,
                                             self.max_rotate_degree)
            rotation_matrix = self._get_rotation_matrix(rotation_degree)
            scaling_ratio = random.uniform(self.scaling_ratio_range[0],
                                           self.scaling_ratio_range[1])
            scaling_matrix = self._get_scaling_matrix(scaling_ratio)
            centerlines.scale(scaling_ratio)
            centerlines.rotate(rotation_matrix)
            centerlines.filter_fvcam(lidar2img, ego2lidar, img_size)
            aug_matrix = np.eye(4, dtype=lidar2ego.dtype)
            aug_matrix[:3, :3] = rotation_matrix @ scaling_matrix
            lidar2ego = aug_matrix @ lidar2ego
            results['center_lines'] = centerlines
            results['lidar2ego'] = lidar2ego
        return results

    @staticmethod
    def _get_rotation_matrix(rotate_degrees):
        radian = math.radians(rotate_degrees)
        rotation_matrix = np.array(
            [[np.cos(radian), -np.sin(radian), 0.],
             [np.sin(radian), np.cos(radian), 0.],
             [0., 0., 1.]],
            dtype=np.float32)
        return rotation_matrix

    @staticmethod
    def _get_scaling_matrix(scale_ratio):
        scaling_matrix = np.array(
            [[scale_ratio, 0., 0.], [0., scale_ratio, 0.],
             [0., 0., 1.]],
            dtype=np.float32)
        return scaling_matrix


@TRANSFORMS.register_module()
class TransformLane2Graph(object):
    def __init__(self):
        pass

    def __call__(self, results):
        centerlines = results['center_lines']
        # converter = LaneLine2NodesConverter(results)
        nodes, nodes_adj = centerlines.export_node_adj()  # get nodes and adj
        centerlines.sub_graph_split()  # split sub graph
        scene_graph = SceneGraph(centerlines.subgraphs_nodes, centerlines.subgraphs_adj,
                                 centerlines.subgraphs_points_in_between_nodes)  # subgraph dfs already
        scene_sentance, scene_sentance_list = scene_graph.sequelize_new()
        centerline_sequence = sentance2seq(
            scene_sentance_list, centerlines.pc_range, centerlines.dx)
        # results['sequence'] = centerline_sequence
        if len(centerline_sequence) % 4 != 0:
            centerline_sequence = centerline_sequence[:(
                centerline_sequence//4*4)]
        centerline_coord = np.stack(
            [centerline_sequence[::4], centerline_sequence[1::4]], axis=1)
        centerline_label = centerline_sequence[2::4]
        centerline_connect = centerline_sequence[3::4]

        results['centerline_sequence'] = centerline_sequence
        results['centerline_coord'] = centerline_coord
        results['centerline_label'] = centerline_label
        results['centerline_connect'] = centerline_connect

        # # visulization
        # gt_nodelist = seq2nodelist(centerline_sequence)
        # gt_nodegraph = MapGraph(results['sample_idx'], gt_nodelist)
        # img_name = results['filename'][0].split('/')[-1].split('.jpg')[0]
        # gt_nodegraph.visulization(centerlines.nx, 'ordered_centerline', 'no_ordered', img_name, scale=5)
        # import pdb;pdb.set_trace()

        # TODO: delete it!!!
        # raw_centerlines = results['raw_center_lines']
        # nodes, nodes_adj = raw_centerlines.export_node_adj()  # get nodes and adj
        # raw_centerlines.sub_graph_split()  # split sub graph
        # scene_graph = SceneGraph(raw_centerlines.subgraphs_nodes, raw_centerlines.subgraphs_adj, raw_centerlines.subgraphs_points_in_between_nodes)  # subgraph dfs already
        # scene_sentance, scene_sentance_list = scene_graph.sequelize_new()
        # centerline_sequence = sentance2seq(scene_sentance_list)
        # # results['sequence'] = centerline_sequence
        # if len(centerline_sequence) % 4 != 0:
        #     centerline_sequence = centerline_sequence[:(centerline_sequence//4*4)]
        # gt_nodelist = seq2nodelist(centerline_sequence)
        # gt_nodegraph = MapGraph(results['sample_idx'], gt_nodelist)
        # img_name = results['filename'][0].split('/')[-1].split('.jpg')[0]
        # gt_nodegraph.visulization('aug_centerline', 'raw', img_name, scale=5)
        return results


@TRANSFORMS.register_module()
class TransformOrderedLane2Graph(object):
    def __init__(self, orderedDFS=True):
        self.order = orderedDFS

    def __call__(self, results):
        centerlines = results['center_lines']
        nodes, nodes_adj = centerlines.export_node_adj()  # get nodes and adj
        centerlines.sub_graph_split()  # split sub graph
        scene_graph = OrderedSceneGraph(centerlines.subgraphs_nodes, centerlines.subgraphs_adj,
                                        centerlines.subgraphs_points_in_between_nodes)  # subgraph dfs already
        scene_sentance, scene_sentance_list = scene_graph.sequelize_new(
            orderedDFS=self.order)
        centerline_sequence = sentance2seq(
            scene_sentance_list, centerlines.pc_range, centerlines.dx)
        # results['sequence'] = centerline_sequence
        if len(centerline_sequence) % 4 != 0:
            centerline_sequence = centerline_sequence[:(
                centerline_sequence//4*4)]
        centerline_coord = np.stack(
            [centerline_sequence[::4], centerline_sequence[1::4]], axis=1)
        centerline_label = centerline_sequence[2::4]
        centerline_connect = centerline_sequence[3::4]

        results['centerline_sequence'] = centerline_sequence
        results['centerline_coord'] = centerline_coord
        results['centerline_label'] = centerline_label
        results['centerline_connect'] = centerline_connect

        # # visulization
        # gt_nodelist = seq2nodelist(centerline_sequence)
        # gt_nodegraph = EvalMapGraph(results['sample_idx'], gt_nodelist)
        # img_name = results['filename'][0].split('/')[-1].split('.jpg')[0]
        # gt_nodegraph.visulization(centerlines.nx, 'ordered_centerline', 'aug', img_name, scale=5)
        # import pdb;pdb.set_trace()

        # TODO: delete it!!!
        # raw_centerlines = results['raw_center_lines']
        # nodes, nodes_adj = raw_centerlines.export_node_adj()  # get nodes and adj
        # raw_centerlines.sub_graph_split()  # split sub graph
        # scene_graph = SceneGraph(raw_centerlines.subgraphs_nodes, raw_centerlines.subgraphs_adj, raw_centerlines.subgraphs_points_in_between_nodes)  # subgraph dfs already
        # scene_sentance, scene_sentance_list = scene_graph.sequelize_new()
        # centerline_sequence = sentance2seq(scene_sentance_list)
        # # results['sequence'] = centerline_sequence
        # if len(centerline_sequence) % 4 != 0:
        #     centerline_sequence = centerline_sequence[:(centerline_sequence//4*4)]
        # gt_nodelist = seq2nodelist(centerline_sequence)
        # gt_nodegraph = MapGraph(results['sample_idx'], gt_nodelist)
        # img_name = results['filename'][0].split('/')[-1].split('.jpg')[0]
        # gt_nodegraph.visulization('aug_centerline', 'raw', img_name, scale=5)
        return results


@TRANSFORMS.register_module()
class NoiseOrderedLane2Graph(object):
    def __init__(self, noise_range=[-2, 2]):
        self.noise_range = noise_range

    def __call__(self, results):
        centerline_coord = results['centerline_coord']
        noise_coord = np.random.randint(
            self.noise_range[0], self.noise_range[1], centerline_coord.shape)
        noise_centerline_coord = centerline_coord + noise_coord
        noise_centerline_coord = np.clip(noise_centerline_coord, 0, 200)
        results['noise_centerline_coord'] = noise_centerline_coord
        return results


def comb(n, k):
    return factorial(n) // (factorial(k) * factorial(n - k))


def get_bezier_coeff(points, n_control):
    """points.shape: [n, 2]"""
    if len(points)<10:
        points = np.linspace(points[0], points[-1], num=10)
    n_points = len(points)
    A = np.zeros((n_points, n_control))
    t = np.arange(n_points) / (n_points - 1)

    for i in range(n_points):
        for j in range(n_control):
            A[i, j] = comb(n_control - 1, j) * np.power(1 - t[i], n_control - 1 - j) * np.power(t[i], j)
    A_BE = A[:, 1:-1]  # (L. N-2)
    points_BE = points - np.stack(
        ((A[:, 0] * points[0][0] + A[:, -1] * points[-1][0]), (A[:, 0] * points[0][1] + A[:, -1] * points[-1][1]))).T
    try:
        conts = np.linalg.lstsq(A_BE, points_BE, rcond=None)
    except:
        raise Exception("Maybe there are some lane whose point number is one!")

    res = conts[0]
    fin_res = np.r_[[points[0]], res, [points[-1]]]
    # fin_res = fin_res.astype(int)
    return fin_res


@TRANSFORMS.register_module()
class TransformGraph2SeqWOBezier(object):
    def __init__(self, n_control=3, orderedDFS=True,n_point=12):
        self.order = orderedDFS
        self.n_control = n_control
        self.split_connect=571
        self.split_node=572
        self.summary_split = 570
        self.coeff_start = 350 
        self.idx_start=250
        self.split_line=569
        self.n_point=n_point

    def __call__(self, results):
        centerlines = results['center_lines']
        nodes, nodes_adj = centerlines.export_node_adj()  # get nodes and adj
        centerlines.sub_graph_split()  # split sub graph
        scene_graph = OrderedBzSceneGraphNew(centerlines.subgraphs_nodes, centerlines.subgraphs_adj,
                                          centerlines.subgraphs_points_in_between_nodes, self.n_control)  # subgraph dfs already
        scene_sentance_list,adj_list,new_subgraphs_points_in_between_nodes_list = scene_graph.sequelize_new(
            orderedDFS=self.order)
        


        centerline_sequence_list = self.sentance2bzseqNew(
            scene_sentance_list,adj_list, new_subgraphs_points_in_between_nodes_list,centerlines.pc_range, centerlines.dx, centerlines.bz_pc_range, centerlines.bz_nx,self.n_control)
        
        centerline_sequence=self.seqlist2seq(centerline_sequence_list)

        results['centerline_sequence'] = centerline_sequence
      
        results['n_control'] = self.n_control


        # TODO: delete it!!!

        return results
    
    
    def sentance2bzseqNew(self,sentance,adj_list,new_subgraphs_points_in_between_nodes_list,pc_range, dx, bz_pc_range, bz_nx,ncontrol):
        """ for each node, seq: x, y, cls, IDX
        if type == start or continue , IDX = 0"""

        seq_list = []
        
        for idx, (sub_sent,adj,new_subgraphs_points_in_between_nodes) in enumerate(zip(sentance,adj_list,new_subgraphs_points_in_between_nodes_list)):
            for node_i,node in enumerate(sub_sent):
            
                # node.position[0], node.position[1] = (node.position[0] - pc_range[0]) / dx[0], (node.position[1] - pc_range[1]) / dx[0]
                node.position = (node.position - pc_range[:3]) / dx
                if idx != 0:
            
                    node.sque_index = node.sque_index + sentance[idx-1][-1].sque_index +1
                seq = np.round(node.position[:2]).astype(int).tolist()  # x y
                seq.append(node.sque_index)  # cls
                father_idx = np.argwhere(adj[node_i] == -1)
                father_seq=[]
                if len(father_idx) != 0 :
        
                    father_idx = father_idx[np.where(father_idx < node_i)]
                    for f_idx in father_idx:
                        centerlines=new_subgraphs_points_in_between_nodes [(f_idx, node_i)]
                        if not isinstance(centerlines,tuple):
                            centerlines=centerlines,
                        for centerline in centerlines:
                            
                            midline=np.squeeze(centerline)
                            split_points,_=divide_line_with_shapely(midline,self.n_point)
                                                                        
                
                            split_points=np.array(split_points[1:-1])

                            split_points=(split_points- pc_range[:3]) / dx
                            split_points=np.around(split_points[:,:2]).reshape(-1).astype(int)
                        
                            if idx!=0:
                                father_seq.append([f_idx + sentance[idx-1][-1].sque_index +1,split_points])
                            else:
                                father_seq.append([f_idx ,split_points])
                            
                seq.append(father_seq)
                
                child_idx=np.argwhere(adj[node_i] == 1)
                child_seq=[]
                if len(child_idx)!=0:
                    child_idx=child_idx[np.where(child_idx < node_i)]
                    for c_idx in child_idx:
                        centerlines=new_subgraphs_points_in_between_nodes [(node_i, c_idx)]
                        if not isinstance(centerlines,tuple):
                            centerlines=centerlines,
                        for centerline in centerlines:
                            
                            midline=np.squeeze(centerline)
                            split_points,_=divide_line_with_shapely(midline,self.n_point)

                            split_points=np.array(split_points[1:-1])
                
                            split_points=(split_points- pc_range[:3]) / dx
                            split_points=np.around(split_points[:,:2]).reshape(-1).astype(int)
                        
                            if idx!=0:
                                child_seq.append([c_idx + sentance[idx-1][-1].sque_index +1,split_points])
                            else:
                                child_seq.append([c_idx ,split_points])
                        
                        
             
                
                seq.append(child_seq)
                seq_list.append(seq)
       
        return  seq_list


    
    def seqlist2seq(self,gt_lines_sequence_list):
        seq=[]
        for gt_lines_sequence in gt_lines_sequence_list:
            x,y,idx,father_seqs,child_seqs=gt_lines_sequence
            
            seq+=[x,y,idx]
            for father_seq in father_seqs:
                idx,midline=father_seq
                seq.append(idx)
                seq+=midline.tolist()
                seq.append(self.split_line)
            seq.append(self.split_connect)
        
            for child_seq in child_seqs:
                idx,midline=child_seq
                seq.append(idx)
                seq+=midline.tolist()
                seq.append(self.split_line)
            seq.append(self.split_node)
    
        return seq 
    

    
@TRANSFORMS.register_module()
class TransformGraph2Seq(object):
    def __init__(self, n_control=3, orderedDFS=True,grid_conf=None):
        self.order = orderedDFS
        self.n_control = n_control
        self.split_connect=571
        self.split_node=572
        self.summary_split = 570
        self.coeff_start = 350 
        self.idx_start=250
        self.grid_conf=grid_conf

    def __call__(self, results):
        centerlines = results['center_lines']
        nodes, nodes_adj = centerlines.export_node_adj()  # get nodes and adj
        centerlines.sub_graph_split()  # split sub graph
        scene_graph = OrderedBzSceneGraphNew(centerlines.subgraphs_nodes, centerlines.subgraphs_adj,
                                          centerlines.subgraphs_points_in_between_nodes, self.n_control)  # subgraph dfs already
        scene_sentance_list,adj_list,new_subgraphs_points_in_between_nodes_list = scene_graph.sequelize_new(
            orderedDFS=self.order)
        
        if self.grid_conf is not None:
            pc_range = get_geom(self.grid_conf)[3]
        else:
            pc_range=centerlines.pc_range

        centerline_sequence_list = self.sentance2bzseqNew(
            scene_sentance_list,adj_list, new_subgraphs_points_in_between_nodes_list,pc_range, centerlines.dx, centerlines.bz_pc_range, centerlines.bz_nx,self.n_control)
        
        centerline_sequence=self.seqlist2seq_with_start(centerline_sequence_list)
        results['centerline_sequence'] = centerline_sequence
        results['n_control'] = self.n_control
        return results
    
    
    def sentance2bzseqNew(self,sentance,adj_list,new_subgraphs_points_in_between_nodes_list,pc_range, dx, bz_pc_range, bz_nx,ncontrol):
        """ for each node, seq: x, y, cls, IDX
        if type == start or continue , IDX = 0"""

        seq_list = []
        
        for idx, (sub_sent,adj,new_subgraphs_points_in_between_nodes) in enumerate(zip(sentance,adj_list,new_subgraphs_points_in_between_nodes_list)):
            for node_i,node in enumerate(sub_sent):
            
                # node.position[0], node.position[1] = (node.position[0] - pc_range[0]) / dx[0], (node.position[1] - pc_range[1]) / dx[0]
                node.position = (node.position - pc_range[:3]) / dx
                if idx != 0:
            
                    node.sque_index = node.sque_index + sentance[idx-1][-1].sque_index +1
                seq = node.position[:2].astype(int).tolist()  # x y
                seq.append(node.sque_index)  # cls
                father_idx = np.argwhere(adj[node_i] == -1)
                father_seq=[]
                if len(father_idx) != 0 :
                    father_idx = father_idx[np.where(father_idx < node_i)]
                    for f_idx in father_idx:
                        centerlines=new_subgraphs_points_in_between_nodes [(f_idx, node_i)]
                        if not isinstance(centerlines,tuple):
                            centerlines=centerlines,
                        for centerline in centerlines:
                            fin_res=get_bezier_coeff(centerline[:, :2],
                                                    n_control=ncontrol)
                            coeff=fin_res[1:-1]
                            
                            coeff=(coeff- bz_pc_range[:2]) / dx[:2]
                            coeff[:,0] = np.clip(coeff[:,0], 0, bz_nx[0]-1)
                            coeff[:,1] = np.clip(coeff[:,1], 0, bz_nx[1]-1)
                            coeff=np.squeeze(coeff.reshape(-1))
                            
                            coeff=coeff.astype(int)
                            if idx!=0:
                                father_seq.append([f_idx + sentance[idx-1][-1].sque_index +1,coeff])
                            else:
                                father_seq.append([f_idx ,coeff])
                            
                seq.append(father_seq)
                
                child_idx=np.argwhere(adj[node_i] == 1)
                child_seq=[]
                if len(child_idx)!=0:
                    child_idx=child_idx[np.where(child_idx < node_i)]
                    for c_idx in child_idx:
                        centerlines=new_subgraphs_points_in_between_nodes  [(node_i, c_idx)]
                        if not isinstance(centerlines,tuple):
                            centerlines=centerlines,
                        for centerline in centerlines:
                            fin_res=get_bezier_coeff(centerline[:, :2],
                                                n_control=ncontrol)
                            coeff=fin_res[1:-1]
                            
                            coeff=(coeff- bz_pc_range[:2]) / dx[:2]
                            coeff[:,0] = np.clip(coeff[:,0], 0, bz_nx[0]-1)
                            coeff[:,1] = np.clip(coeff[:,1], 0, bz_nx[1]-1)
                            coeff=np.squeeze(coeff.reshape(-1))
                            coeff=coeff.astype(int)
                            if idx!=0:
                                child_seq.append([c_idx + sentance[idx-1][-1].sque_index+1 ,coeff])
                            else:
                                child_seq.append([c_idx  ,coeff])
                    
                seq.append(child_seq)
                seq_list.append(seq)
       
        return  seq_list

    def seqlist2seq_with_start(self,gt_lines_sequence_list):
        seq=[]
        for gt_lines_sequence in gt_lines_sequence_list:
            x,y,idx,father_seqs,child_seqs=gt_lines_sequence
            
            seq+=[x,y,idx+self.idx_start]
            for father_seq in father_seqs:
                idx,coeff=father_seq
                idx+=self.idx_start
                seq.append(idx)
                coeff=coeff+self.coeff_start
                seq+=coeff.tolist()
            seq.append(self.split_connect)
        
            for child_seq in child_seqs:
                idx,coeff=child_seq
                idx+=self.idx_start
                seq.append(idx)
                coeff=coeff+self.coeff_start
                seq+=coeff.tolist()
            seq.append(self.split_node)
    
        return seq 
    


@TRANSFORMS.register_module()
class TransformGraph2SeqCubic(TransformGraph2Seq):
    def __init__(self, n_control=3, orderedDFS=True,grid_conf=None):
        self.order = orderedDFS
        self.n_control = n_control
        self.split_connect=571
        self.split_node=572
        self.summary_split = 570
        self.coeff_start = 350 
        self.idx_start=250
        self.grid_conf=grid_conf



@TRANSFORMS.register_module()
class TransformOrderedBzLane2Graph(object):
    def __init__(self, n_control=3, orderedDFS=True):
        self.order = orderedDFS
        self.n_control = n_control

    def __call__(self, results):
        centerlines = results['center_lines']
        nodes, nodes_adj = centerlines.export_node_adj()  # get nodes and adj
        centerlines.sub_graph_split()  # split sub graph
        scene_graph = OrderedBzSceneGraph(centerlines.subgraphs_nodes, centerlines.subgraphs_adj,
                                          centerlines.subgraphs_points_in_between_nodes, self.n_control)  # subgraph dfs already
        scene_sentance, scene_sentance_list = scene_graph.sequelize_new(
            orderedDFS=self.order)
        centerline_sequence = sentance2bzseq(
            scene_sentance_list, centerlines.pc_range, centerlines.dx, centerlines.bz_pc_range, centerlines.bz_nx)
        clause_length = 4 + 2*(self.n_control-2)
        if len(centerline_sequence) % clause_length != 0:
            centerline_sequence = centerline_sequence[:(
                centerline_sequence//clause_length*clause_length)]
        centerline_coord = np.stack(
            [centerline_sequence[::clause_length], centerline_sequence[1::clause_length]], axis=1)
        centerline_label = centerline_sequence[2::clause_length]
        centerline_connect = centerline_sequence[3::clause_length]
        centerline_coeff = np.stack(
            [centerline_sequence[k::clause_length] for k in range(4, clause_length)], axis=1)

        results['centerline_sequence'] = centerline_sequence
        results['centerline_coord'] = centerline_coord
        results['centerline_label'] = centerline_label
        results['centerline_connect'] = centerline_connect
        results['centerline_coeff'] = centerline_coeff
        results['n_control'] = self.n_control
        return results


@TRANSFORMS.register_module()
class TransformLaneGraph(object):
    def __init__(self, n_control=3, orderedDFS=True, vertex_id_start=200, connect_start=250, coeff_start=300):
        self.order = orderedDFS
        self.n_control = n_control
        self.vertex_id_start = vertex_id_start
        self.connect_start = connect_start
        self.coeff_start = coeff_start

    def __call__(self, results):
        centerlines = results['center_lines']
        nodes, nodes_adj = centerlines.export_node_adj()  # get nodes and adj
        centerlines.sub_graph_split()  # split sub graph
        scene_graph = Laneseq2Graph(centerlines.subgraphs_nodes, centerlines.subgraphs_adj,
                                    centerlines.subgraphs_points_in_between_nodes, self.n_control)  # subgraph dfs already
        graph_nodes, graph_betweens = scene_graph.sequelize_new(
            orderedDFS=self.order)
        vert_sentence, edge_sentence = nodesbetween2seq(graph_nodes,
                                                        graph_betweens,
                                                        centerlines.pc_range,
                                                        centerlines.dx,
                                                        centerlines.bz_pc_range,
                                                        centerlines.bz_nx,
                                                        self.vertex_id_start,
                                                        self.connect_start,
                                                        self.coeff_start)
        vert_sentence = np.array(vert_sentence)
        edge_sentence = np.array(edge_sentence)

        results['vert_sentence'] = vert_sentence
        results['edge_sentence'] = edge_sentence
        results['n_control'] = self.n_control
        return results


@TRANSFORMS.register_module()
class PaddingSequence3(object):
    def __init__(self, max_box_num):
        self.num_center_classes = 576
        self.box_range = 200
        self.coeff_range = 200
        self.num_classes = 4
        self.category_start = 200
        self.connect_start = 250
        self.coeff_start = 350
        # n/a and padding share the same label to be eliminated from loss calculation
        self.no_known = 575
        self.start = 574
        self.end = 573
        self.noise_connect = 572
        self.noise_label = 571
        self.noise_coeff = 570
        self.max_box_num = max_box_num

    def __call__(self, results):
        centerline_coord = np.array(results['centerline_coord'])
        centerline_label = np.array(results['centerline_label'])
        centerline_connect = np.array(results['centerline_connect'])
        centerline_coeff = np.array(results['centerline_coeff'])

        max_box = len(centerline_coord)
        num_box = max(max_box + 2, self.max_box_num)  # 100
        n_control = results['n_control']
        coeff_dim = (n_control - 2) * 2
        box = centerline_coord.astype(np.int64)
        box = box.reshape(-1, 2)
        label = centerline_label.astype(
            np.int64) + self.category_start  # [8,1]
        label = label.reshape(-1, 1)
        connect = centerline_connect.astype(
            np.int64) + self.connect_start  # [8,1]
        connect = connect.reshape(-1, 1)
        coeff = centerline_coeff.astype(np.int64) + self.coeff_start  # [8,1]
        coeff = coeff.reshape(-1, coeff_dim)
        box_label = np.concatenate(
            [box, label, connect, coeff], axis=-1)  # [8, 5]

        random_box = np.random.rand(num_box - box_label.shape[0], 2)
        random_box = (random_box * (self.box_range - 1)).astype(np.int64)
        random_label = np.random.randint(
            0, self.num_classes, (num_box - box_label.shape[0], 1)).astype(np.int64)
        random_label = random_label + self.category_start
        random_connect = np.random.randint(
            0, num_box, (num_box - box_label.shape[0], 1)).astype(np.int64)
        random_connect = random_connect + self.connect_start
        random_coeff = np.random.rand(
            num_box - box_label.shape[0], coeff_dim).astype(np.int64)
        random_coeff = (random_coeff * (self.coeff_range - 1)).astype(np.int64)
        random_coeff = random_coeff + self.coeff_start
        random_box_label = np.concatenate(
            [random_box, random_label, random_connect, random_coeff], axis=-1)  # [92, 5]

        input_seq = np.concatenate(
            [box_label, random_box_label], axis=0)  # [100, 5]

        input_seq = np.concatenate([np.ones((1)).astype(
            np.int64) * self.start, input_seq.flatten()])  # [501]
        # input_seq = np.expand_dims(input_seq, axis=0)

        output_na = np.ones(
            (num_box - box_label.shape[0], 1)).astype(np.int64) * self.no_known  # [92,3]
        output_noise = np.ones(
            (num_box - box_label.shape[0], 1)).astype(np.int64) * self.no_known  # [92,1]
        output_noise_connect = np.ones(
            (num_box - box_label.shape[0], 1)).astype(np.int64) * self.no_known  # [92,1]
        output_noise_coeff = np.ones(
            (num_box - box_label.shape[0], coeff_dim)).astype(np.int64) * self.no_known  # [92,1]
        output_end = np.ones(
            (num_box - box_label.shape[0], 1)).astype(np.int64) * self.end  # [92, 1]
        output_seq = np.concatenate(
            [output_na, output_noise, output_noise_connect, output_noise_coeff, output_end], axis=-1)  # [92,5]

        output_seq = np.concatenate([box_label.flatten(), np.ones(
            (1)).astype(np.int64) * self.end, output_seq.flatten()])
        # output_seq = np.expand_dims(output_seq, axis=0)

        results['input_seq'] = input_seq
        results['output_seq'] = output_seq
        return results


@TRANSFORMS.register_module()
class TransformUnitOrderedBzLane2Graph(object):
    def __init__(self, n_control=3, orderedDFS=True):
        self.order = orderedDFS
        self.n_control = n_control

    def __call__(self, results):
        centerlines = results['center_lines']
        nodes, nodes_adj = centerlines.export_node_adj()  # get nodes and adj
        scene_graph = AV2OrderedBzSceneGraph_new([centerlines.all_nodes], [centerlines.adj],
                                                 [centerlines.points_in_between_nodes],
                                                 self.n_control)
        scene_sentance, scene_sentance_list = scene_graph.sequelize_new(
            orderedDFS=self.order)
        centerline_sequence = sentance2bzseq2(
            scene_sentance_list, centerlines.pc_range, centerlines.dx, centerlines.bz_pc_range, centerlines.bz_nx)
        clause_length = 4 + 2*(self.n_control-2)
        if len(centerline_sequence) % clause_length != 0:
            centerline_sequence = centerline_sequence[:(
                centerline_sequence//clause_length*clause_length)]
        centerline_coord = np.stack(
            [centerline_sequence[::clause_length], centerline_sequence[1::clause_length]], axis=1)
        centerline_label = centerline_sequence[2::clause_length]
        centerline_connect = centerline_sequence[3::clause_length]
        centerline_coeff = np.stack(
            [centerline_sequence[k::clause_length] for k in range(4, clause_length)], axis=1)

        results['centerline_sequence'] = centerline_sequence
        results['centerline_coord'] = centerline_coord
        results['centerline_label'] = centerline_label
        results['centerline_connect'] = centerline_connect
        results['centerline_coeff'] = centerline_coeff
        results['n_control'] = self.n_control

        return results


@TRANSFORMS.register_module()
class TransformOrderedBzPlLane2Graph(object):
    def __init__(self, n_control=3, orderedDFS=True, max_box_len=18, sub_graph=False, semi_AR=True):
        self.order = orderedDFS
        self.n_control = n_control
        self.semi_AR = semi_AR
        self.subgraph = sub_graph
        self.max_box_len = max_box_len

    def __call__(self, results):
        centerlines = results['center_lines']
        nodes, nodes_adj = centerlines.export_node_adj()  # get nodes and adj
        centerlines.sub_graph_split()  # split sub graph
        scene_graph = OrderedBzPlSceneGraph(centerlines.subgraphs_nodes, centerlines.subgraphs_adj,
                                            centerlines.subgraphs_points_in_between_nodes, self.n_control)  # subgraph dfs already

        keypoints_all, nodelist_all = scene_graph.sequelize_semiAR(
            orderedDFS=self.order)

        keypoints_all, nodelist_all = float2int(
            keypoints_all, nodelist_all, centerlines.pc_range, centerlines.dx, centerlines.nx, centerlines.bz_pc_range, centerlines.bz_nx)

        keypoint_node_pairs, ori2sort_idx_map = match_keypoints(
            keypoints_all, nodelist_all)


        centerline_sequences = get_semiAR_seq_fromInt(
            keypoint_node_pairs, ori2sort_idx_map)

        clause_length = 4 + 2*(self.n_control-2)
        centerline_coords = []
        centerline_labels = []
        centerline_connects = []
        centerline_coeffs = []
        for i, centerline_sequence in enumerate(centerline_sequences):
            if len(centerline_sequence) % clause_length != 0:
                centerline_sequence = centerline_sequence[:(
                    centerline_sequence//clause_length*clause_length)]
            max_len = (self.max_box_len - 2) * clause_length
            if len(centerline_sequence) > max_len:
                centerline_sequence = centerline_sequence[:max_len]
            centerline_sequences[i] = centerline_sequence
            centerline_coords.append(np.stack(
                [centerline_sequence[::clause_length], centerline_sequence[1::clause_length]], axis=1))
            centerline_labels.append(centerline_sequence[2::clause_length])
            centerline_connects.append(centerline_sequence[3::clause_length])
            centerline_coeffs.append(np.stack(
                [centerline_sequence[k::clause_length] for k in range(4, clause_length)], axis=1))
        results['centerline_sequences'] = centerline_sequences
        results['centerline_coords'] = centerline_coords
        results['centerline_labels'] = centerline_labels
        results['centerline_connects'] = centerline_connects
        results['centerline_coeffs'] = centerline_coeffs
        results['n_control'] = self.n_control

        return results


@TRANSFORMS.register_module()
class TransformOrderedPlBzLane2Graph(object):
    def __init__(self, n_control=3, bins=200, orderedDFS=True):
        self.order = orderedDFS
        self.n_control = n_control
        self.bins = bins

    def seq2nodepairs(self, centerline_coord, centerline_label, centerline_connect):
        node_list = {}
        node_pairs = []
        # type_idx_map = {'start': 0, 'continue': 1, 'fork': 2, 'merge': 3}
        # idx_type_map = {0: 'start', 1: 'continue', 2: "fork", 3: 'merge'}
        idx = 0
        epsilon = 2

        if len(centerline_coord) == 0:
            return np.array([]).astype(np.int64)

        for i in range(len(centerline_coord)):
            label = centerline_label[i]
            if label > 3 or label < 0:
                label = 1
            if idx == 0:
                assert label == 0
                label = 0

            coord = centerline_coord[i]
            if label == 3:  # merge
                if centerline_connect[i] in node_list:
                    next_coord = node_list[centerline_connect[i]]
                else:
                    next_coord = np.array([self.bins, self.bins])
                node_pairs.append(np.concatenate([coord, next_coord]))
            elif label == 2:  # fork
                last_coordnp = centerline_coord[i - 1]
                coordnp = centerline_coord
                tmp = np.sum((coordnp - last_coordnp) ** 2)
                if tmp >= epsilon:  # split fork
                    idx = idx + 1
                    node_list[idx] = coord
                if centerline_connect[i] in node_list:
                    last_coord = node_list[centerline_connect[i]]
                else:
                    last_coord = np.array([self.bins, self.bins])
                node_pairs.append(np.concatenate([last_coord, coord]))
            elif label == 1:  # continue
                last_coord = node_list[idx]
                idx = idx + 1
                node_list[idx] = coord
                node_pairs.append(np.concatenate([last_coord, coord]))
            elif label == 0:  # start
                idx = idx + 1
                node_list[idx] = coord
                node_pairs.append(np.concatenate(
                    [np.array([self.bins, self.bins]), coord]))
        assert 0 not in node_list
        node_pairs = np.concatenate(node_pairs)
        return node_pairs

    def __call__(self, results):
        centerlines = results['center_lines']
        nodes, nodes_adj = centerlines.export_node_adj()  # get nodes and adj
        centerlines.sub_graph_split()  # split sub graph
        scene_graph = OrderedBzSceneGraph(centerlines.subgraphs_nodes, centerlines.subgraphs_adj,
                                          centerlines.subgraphs_points_in_between_nodes, self.n_control)  # subgraph dfs already
        scene_sentance, scene_sentance_list = scene_graph.sequelize_new(
            orderedDFS=self.order)
        centerline_sequence = sentance2bzseq(
            scene_sentance_list, centerlines.pc_range, centerlines.dx, centerlines.bz_pc_range, centerlines.bz_nx)
        clause_length = 4 + 2*(self.n_control-2)
        if len(centerline_sequence) % clause_length != 0:
            centerline_sequence = centerline_sequence[:(
                centerline_sequence//clause_length*clause_length)]
        centerline_coord = np.stack(
            [centerline_sequence[::clause_length], centerline_sequence[1::clause_length]], axis=1)
        centerline_label = centerline_sequence[2::clause_length]
        centerline_connect = centerline_sequence[3::clause_length]
        centerline_coeff = np.stack(
            [centerline_sequence[k::clause_length] for k in range(4, clause_length)], axis=1)
        centerline_pairs = self.seq2nodepairs(
            centerline_coord, centerline_label, centerline_connect)
        assert centerline_coeff.shape[0] * 4 == centerline_pairs.shape[0]

        results['centerline_sequence'] = centerline_sequence
        results['centerline_coord'] = centerline_coord
        results['centerline_label'] = centerline_label
        results['centerline_connect'] = centerline_connect
        results['centerline_coeff'] = centerline_coeff
        results['n_control'] = self.n_control
        results['centerline_pairs'] = centerline_pairs

        return results


@TRANSFORMS.register_module()
class LoadPryCenterline(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.
    """

    def __init__(self, grid_conf):
        self.grid_conf = grid_conf

    def __call__(self, results):
        """Call function to load multi-view image from files.
        """
        results['center_lines'] = PryCenterLine(
            results['center_lines'], self.grid_conf)
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        return repr_str


@TRANSFORMS.register_module()
class LoadPryOrderedCenterline(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.
    """

    def __init__(self, grid_conf):
        self.grid_conf = grid_conf

    def __call__(self, results):
        """Call function to load multi-view image from files.
        """
        results['center_lines'] = PryOrederedCenterLine(
            results['center_lines'], self.grid_conf)
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        return repr_str


@TRANSFORMS.register_module()
class LoadNusOrderedBzCenterline(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.
    """

    def __init__(self, grid_conf, bz_grid_conf,cam_intrinsic=None):
        self.grid_conf = grid_conf
        self.bz_grid_conf = bz_grid_conf
        self.cam_intrinsic=cam_intrinsic

    def __call__(self, results):
        """Call function to load multi-view image from files.
        """
        results['center_lines'] = NusOrederedBzCenterLine(
            results['center_lines'], self.grid_conf, self.bz_grid_conf,self.cam_intrinsic,results['token'])
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        return repr_str



@TRANSFORMS.register_module()
class LoadNusOrderedBzCenterlineFrontCamera(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.
    """

    def __init__(self, grid_conf, bz_grid_conf,cam_intrinsic=None):
        self.grid_conf = grid_conf
        self.bz_grid_conf = bz_grid_conf
       

    def __call__(self, results):
        """Call function to load multi-view image from files.
        """
   
        cam_intrinsic=results["cams"]["CAM_FRONT"]["cam_intrinsic"]
        if len(cam_intrinsic)==0:
            cam_intrinsic=np.array([
                        [
                            1266.417203046554,
                            0.0,
                            816.2670197447984
                        ],
                        [
                            0.0,
                            1266.417203046554,
                            491.50706579294757
                        ],
                        [
                            0.0,
                            0.0,
                            1.0
                        ]
                    ])
        else:
            cam_intrinsic=cam_intrinsic[:3,:3]
        results['center_lines'] = NusOrederedBzCenterLine(
            results['center_lines'], self.grid_conf, self.bz_grid_conf,cam_intrinsic,results['token'])
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        return repr_str
    
    

@TRANSFORMS.register_module()
class LoadNusOrderedRMcontinuedBzCenterline(object):
    """Load multi channel images from a list of separate channel files.
        去除直线上的continue点
    Expects results['img_filename'] to be a list of filenames.
    """

    def __init__(self, grid_conf, bz_grid_conf,cam_intrinsic=None):
        self.grid_conf = grid_conf
        self.bz_grid_conf = bz_grid_conf
        self.cam_intrinsic=cam_intrinsic

    def __call__(self, results):
        """Call function to load multi-view image from files.
        """
        results['center_lines'] = NusOrederedRMcontinuedBzCenterLine(
            results['center_lines'], self.grid_conf, self.bz_grid_conf,self.cam_intrinsic,results['token'])
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        return repr_str


@TRANSFORMS.register_module()
class LoadNusOrderedBzCenterlineSeg(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.
    """

    def __init__(self, grid_conf, bz_grid_conf,data_root=None,thickness=2,cam_intrinsic=None):
        self.grid_conf = grid_conf
        self.line = 255
        self.data_root = data_root
        self.bz_grid_conf = bz_grid_conf
        self.cam_intrinsic=cam_intrinsic
        self.thickness = thickness
        dx, bx, nx = self.gen_dx_bx(self.grid_conf['xbound'],
                                    self.grid_conf['ybound'],
                                    self.grid_conf['zbound'],)
        self.dx = dx
        self.bx = bx
        self.nx = nx
        self.pc_range = np.concatenate(
            (self.bx - self.dx / 2., self.bx - self.dx / 2. + self.nx * self.dx))

    def __call__(self, results):
        """Call function to load multi-view image from files.
        """
        centerline_seg = np.zeros((int(self.nx[1]), int(self.nx[0])))
        center_lines = results['center_lines']['centerlines']
        for i in range(len(center_lines)):
            center_line = center_lines[i]
            inbev_x = np.logical_and(
                center_line[:, 0] < self.pc_range[3], center_line[:, 0] >= self.pc_range[0])
            inbev_y = np.logical_and(
                center_line[:, 1] < self.pc_range[4], center_line[:, 1] >= self.pc_range[1])
            inbev_xy = np.logical_and(inbev_x, inbev_y)
            if not max(inbev_xy):
                continue
            center_line = (center_line[inbev_xy, :] -
                           self.pc_range[:3]) / self.dx
            center_line = np.floor(center_line).astype(np.int32)
            for pt_i in range(len(center_line)-1):
                cv2.line(centerline_seg, tuple(center_line[pt_i, :2]), tuple(
                    center_line[pt_i+1, :2]), self.line, self.thickness)
        if self.data_root:
            filename = os.path.join(
                self.data_root, results['sample_idx'] + '.png')
            cv2.imwrite(filename, centerline_seg)
        centerline_seg[centerline_seg == self.line] = 1
        results['center_seg'] = centerline_seg.astype(np.int64)
        
        
        results['center_lines'] = NusOrederedBzCenterLine(
            results['center_lines'], self.grid_conf, self.bz_grid_conf,self.cam_intrinsic)
        return results

    @staticmethod
    def gen_dx_bx(xbound, ybound, zbound):
        dx = np.array([row[2] for row in [xbound, ybound, zbound]])
        bx = np.array(
            [row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])
        nx = np.floor(np.array([(row[1] - row[0]) / row[2]
                      for row in [xbound, ybound, zbound]]))
        return dx, bx, nx
    
    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        return repr_str



@TRANSFORMS.register_module()
class LoadNusClearOrderedBzCenterline(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.
    """

    def __init__(self, grid_conf, bz_grid_conf, clear=True):
        self.grid_conf = grid_conf
        self.bz_grid_conf = bz_grid_conf
        self.clear = clear

    def __call__(self, results):
        """Call function to load multi-view image from files.
        """
        results['center_lines'] = NusClearOrederedBzCenterLine(
            results['center_lines'], self.grid_conf, self.bz_grid_conf, clear=self.clear)
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        return repr_str


@TRANSFORMS.register_module()
class LoadUnitOrderedBzCenterline(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.
    """

    def __init__(self, grid_conf, bz_grid_conf, epsilon):
        self.grid_conf = grid_conf
        self.bz_grid_conf = bz_grid_conf
        self.epsilon = epsilon

    def __call__(self, results):
        """Call function to load multi-view image from files.
        """
        results['center_lines'] = AV2OrederedBzCenterLine_new(
            results['center_lines'], results['sample_idx'], self.grid_conf, self.bz_grid_conf, epsilon=self.epsilon)
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        return repr_str


@TRANSFORMS.register_module()
class LoadPryOrderedBzCenterlineFromFiles(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.
    """

    def __init__(self, grid_conf, bz_grid_conf, data_root):
        self.grid_conf = grid_conf
        self.bz_grid_conf = bz_grid_conf
        self.data_root = data_root

    def __call__(self, results):
        """Call function to load multi-view image from files.
        """
        token_name = results['sample_idx']+'.pkl'
        center_lines = mmcv.load(os.path.join(self.data_root, token_name))
        results['center_lines'] = NusClearOrederedBzCenterLine(
            center_lines, self.grid_conf, self.bz_grid_conf)
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        return repr_str


@TRANSFORMS.register_module()
class LoadMonoPryOrderedBzCenterline(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.
    """

    def __init__(self, grid_conf, bz_grid_conf):
        self.grid_conf = grid_conf
        self.bz_grid_conf = bz_grid_conf

    def __call__(self, results):
        """Call function to load multi-view image from files.
        """
        center_lines = PryMonoOrederedBzCenterLine(
            results['center_lines'], self.grid_conf, self.bz_grid_conf)
        lidar2ego = results['lidar2ego']
        ego2lidar = np.linalg.inv(lidar2ego)
        lidar2img = results['lidar2img'][0]
        img_size = results['pad_shape'][0][:2]
        center_lines.filter_fvcam(lidar2img, ego2lidar, img_size)
        results['center_lines'] = center_lines
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        return repr_str


@TRANSFORMS.register_module()
class LoadPryOrderedBzPlCenterline(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.
    """

    def __init__(self, grid_conf, bz_grid_conf):
        self.grid_conf = grid_conf
        self.bz_grid_conf = bz_grid_conf

    def __call__(self, results):
        """Call function to load multi-view image from files.
        """
        results['center_lines'] = PryOrederedBzPlCenterLine(
            results['center_lines'], self.grid_conf, self.bz_grid_conf)
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        return repr_str


@TRANSFORMS.register_module()
class LoadMonoPryOrderedBzPlCenterline(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.
    """

    def __init__(self, grid_conf, bz_grid_conf):
        self.grid_conf = grid_conf
        self.bz_grid_conf = bz_grid_conf

    def __call__(self, results):
        """Call function to load multi-view image from files.
        """
        center_lines = PryMonoOrederedBzPlCenterLine(
            results['center_lines'], self.grid_conf, self.bz_grid_conf)
        lidar2ego = results['lidar2ego']
        ego2lidar = np.linalg.inv(lidar2ego)
        lidar2img = results['lidar2img'][0]
        img_size = results['pad_shape'][0][:2]
        center_lines.filter_fvcam(lidar2img, ego2lidar, img_size)
        results['center_lines'] = center_lines
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        return repr_str


@TRANSFORMS.register_module()
class LoadPryCenterlineFromFiles(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.
    """

    def __init__(self, data_root):
        self.data_root = data_root

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data. \
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        filename = os.path.join(
            self.data_root, 'centerline_pryseq', results['sample_idx'] + '.pkl')
        centerline = mmcv.load(filename)
        centerline_sequence = np.array(centerline['centerline_sequence'])
        if len(centerline_sequence) % 4 != 0:
            centerline_sequence = centerline_sequence[:(
                centerline_sequence//4*4)]
        centerline_coord = np.stack(
            [centerline_sequence[::4], centerline_sequence[1::4]], axis=1)
        centerline_label = centerline_sequence[2::4]
        centerline_connect = centerline_sequence[3::4]

        results['centerline_sequence'] = centerline_sequence
        results['centerline_coord'] = centerline_coord
        results['centerline_label'] = centerline_label
        results['centerline_connect'] = centerline_connect
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        return repr_str


@TRANSFORMS.register_module()
class LoadCenterlineFromFiles(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.
    """

    def __init__(self, data_root):
        self.data_root = data_root

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data. \
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        filename = os.path.join(
            self.data_root, 'centerline_seq', results['sample_idx'] + '.pkl')
        centerline = mmcv.load(filename)
        centerline_coord = centerline['centerline_coord']
        centerline_label = centerline['centerline_label']
        if np.sum(centerline_coord < 0) != 0 or np.sum(centerline_coord >= 200) != 0:
            centerline_coord = np.zeros((0, 2)).astype(np.int64)
            centerline_label = np.zeros((0,)).astype(np.int64)
        results['centerline_coord'] = centerline_coord
        results['centerline_label'] = centerline_label
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        return repr_str


@TRANSFORMS.register_module()
class LoadDepthFromLidar(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.
    """

    def __init__(self, downsample):
        self.downsample = downsample

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data. \
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        lidar2img = []
        for i in range(len(results['lidar2img'])):
            lidar2img.append(results['lidar2img'][i])
        lidar2img = np.asarray(lidar2img)

        points = results['points'].tensor
        lidar2img = points.new_tensor(lidar2img)

        points_coord = torch.cat((points[:, :3], torch.ones_like(
            points[..., :1])), dim=-1).unsqueeze(0).unsqueeze(-1)
        lidar2img = lidar2img.unsqueeze(1)
        coords = torch.matmul(
            lidar2img, points_coord).squeeze(-1).detach().cpu().numpy()

        depth = coords[..., 2]
        on_img = (coords[..., 2] > 1e-5)
        coords[..., 2] = np.clip(coords[..., 2], 1e-5, 1e5)
        coords[..., 0:2] /= coords[..., 2:3]
        coords = coords[..., :2]
        h, w = results['pad_shape'][0][:2]

        on_img = (on_img & (coords[..., 0] < w)
                  & (coords[..., 0] >= 0)
                  & (coords[..., 1] < h)
                  & (coords[..., 1] >= 0))

        depth_maps = []
        for ci in range(6):
            masked_coords = coords[ci][on_img[ci]].astype(np.int32).T
            depth_map = np.zeros((h, w))
            depth_map[masked_coords[1, :],
                      masked_coords[0, :]] = depth[ci][on_img[ci]]
            depth_map, _ = fill_in_multiscale(depth_map)
            # depth_map[depth_map==0.0] = 100.0
            # down sample
            h_d = h // self.downsample
            w_d = w // self.downsample
            depth_map = depth_map.reshape(
                h_d, self.downsample, w_d, self.downsample)
            depth_map = depth_map.transpose(0, 2, 1, 3).reshape(h_d, w_d, -1)
            depth_map = np.min(depth_map, axis=-1)
            depth_maps.append(depth_map)

            # import cv2
            # import os
            # save_dir = f"vis/depth/"
            # if not os.path.exists(save_dir):
            #     os.mkdir(save_dir)
            # name = results['filename'][ci].split('/')[-1].split('.jpg')[0]
            # depth_map_draw = np.clip(depth_map / 60, 0, 1)
            # depth_map_draw = (depth_map_draw * 255).astype(np.uint8)
            # depth_map_draw = cv2.applyColorMap(depth_map_draw, cv2.COLORMAP_JET)
            # cv2.imwrite(os.path.join(save_dir, f"{name}_dpt.jpg"), depth_map_draw)
            # img = cv2.imread(results['filename'][ci])
            # cv2.imwrite(os.path.join(save_dir, f"{name}.jpg"), img)
        depth_maps = np.stack(depth_maps)
        results['lidar_depth'] = depth_maps
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(line={self.downsample}, '
        return repr_str


@TRANSFORMS.register_module()
class LoadDepthSupFromLidar(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.
    """

    def __init__(self, downsample):
        self.downsample = downsample

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data. \
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        lidar2img = []
        for i in range(len(results['lidar2img'])):
            lidar2img.append(results['lidar2img'][i])
        lidar2img = np.asarray(lidar2img)

        points = results['points'].tensor
        lidar2img = points.new_tensor(lidar2img)

        points_coord = torch.cat((points[:, :3], torch.ones_like(
            points[..., :1])), dim=-1).unsqueeze(0).unsqueeze(-1)
        lidar2img = lidar2img.unsqueeze(1)
        coords = torch.matmul(
            lidar2img, points_coord).squeeze(-1).detach().cpu().numpy()

        depth = coords[..., 2]
        on_img = (coords[..., 2] > 1e-5)
        coords[..., 2] = np.clip(coords[..., 2], 1e-5, 1e5)
        coords[..., 0:2] /= coords[..., 2:3]
        coords = coords[..., :2]
        h, w = results['pad_shape'][0][:2]

        on_img = (on_img & (coords[..., 0] < w)
                  & (coords[..., 0] >= 0)
                  & (coords[..., 1] < h)
                  & (coords[..., 1] >= 0))

        depth_maps = []
        for ci in range(6):
            masked_coords = coords[ci][on_img[ci]].astype(np.int32).T
            depth_map = np.ones((h, w)) * 200
            depth_map[masked_coords[1, :],
                      masked_coords[0, :]] = depth[ci][on_img[ci]]
            # depth_map, _ = fill_in_multiscale(depth_map)
            # depth_map[depth_map==0.0] = 100.0
            # down sample
            h_d = h // self.downsample
            w_d = w // self.downsample
            depth_map = depth_map.reshape(
                h_d, self.downsample, w_d, self.downsample)
            depth_map = depth_map.transpose(0, 2, 1, 3).reshape(h_d, w_d, -1)
            depth_map = np.min(depth_map, axis=-1)
            # ignore_mask = depth_map > 150
            # depth_map[ignore_mask] = 0
            depth_maps.append(depth_map)

            # import cv2
            # import os
            # save_dir = f"vis/depth_sup/"
            # if not os.path.exists(save_dir):
            #     os.mkdir(save_dir)
            # name = results['filename'][ci].split('/')[-1].split('.jpg')[0]
            # depth_map_draw = np.clip(depth_map / 60, 0, 1)
            # depth_map_draw = (depth_map_draw * 255).astype(np.uint8)
            # depth_map_draw = cv2.applyColorMap(depth_map_draw, cv2.COLORMAP_JET)
            # cv2.imwrite(os.path.join(save_dir, f"{name}_dpt.jpg"), depth_map_draw)
            # img = cv2.imread(results['filename'][ci])
            # cv2.imwrite(os.path.join(save_dir, f"{name}.jpg"), img)
        depth_maps = np.stack(depth_maps)
        results['lidar_depth'] = depth_maps
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(line={self.downsample}, '
        return repr_str


@TRANSFORMS.register_module()
class LoadCenterlineSegFromPkl(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.
    """

    def __init__(self, grid_conf, thickness=2, data_root=None):
        self.line = 255
        self.data_root = data_root
        self.thickness = thickness
        self.grid_conf = grid_conf
        dx, bx, nx = self.gen_dx_bx(self.grid_conf['xbound'],
                                    self.grid_conf['ybound'],
                                    self.grid_conf['zbound'],)
        self.dx = dx
        self.bx = bx
        self.nx = nx
        self.pc_range = np.concatenate(
            (self.bx - self.dx / 2., self.bx - self.dx / 2. + self.nx * self.dx))

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data. \
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        centerline_seg = np.zeros((int(self.nx[1]), int(self.nx[0])))
        center_lines = results['center_lines']['centerlines']
        for i in range(len(center_lines)):
            center_line = center_lines[i]
            inbev_x = np.logical_and(
                center_line[:, 0] < self.pc_range[3], center_line[:, 0] >= self.pc_range[0])
            inbev_y = np.logical_and(
                center_line[:, 1] < self.pc_range[4], center_line[:, 1] >= self.pc_range[1])
            inbev_xy = np.logical_and(inbev_x, inbev_y)
            if not max(inbev_xy):
                continue
            center_line = (center_line[inbev_xy, :] -
                           self.pc_range[:3]) / self.dx
            center_line = np.floor(center_line).astype(np.int32)
            for pt_i in range(len(center_line)-1):
                cv2.line(centerline_seg, tuple(center_line[pt_i, :2]), tuple(
                    center_line[pt_i+1, :2]), self.line, self.thickness)
        if self.data_root:
            filename = os.path.join(
                self.data_root, results['sample_idx'] + '.png')
            cv2.imwrite(filename, centerline_seg)
        centerline_seg[centerline_seg == self.line] = 1
        results['center_seg'] = centerline_seg.astype(np.int64)
        return results

    @staticmethod
    def gen_dx_bx(xbound, ybound, zbound):
        dx = np.array([row[2] for row in [xbound, ybound, zbound]])
        bx = np.array(
            [row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])
        nx = np.floor(np.array([(row[1] - row[0]) / row[2]
                      for row in [xbound, ybound, zbound]]))
        return dx, bx, nx

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(line={self.line}, '
        return repr_str


@TRANSFORMS.register_module()
class LoadRoadSegmentation(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.
    """

    def __init__(self, grid_conf, data_root=None, layer_names=None, save_path=None):
        self.data_root = data_root
        self.grid_conf = grid_conf
        self.layer_names = layer_names
        self.save_path = save_path
        dx, bx, nx, pc_range, ego_points = self.get_geom(grid_conf)
        self.dx = dx
        self.bx = bx
        self.nx = nx
        self.pc_range = pc_range
        self.ego_points = ego_points

        layer_dict = dict()
        for location in LOCATIONS:
            layer_dict[location] = dict()
        for location in layer_dict:
            for layer_name in layer_names:
                mask = mmengine.load(os.path.join(
                    data_root, layer_name, location+'.pkl'))
                layer_dict[location][layer_name] = mask
        self.layer_dict = layer_dict

        self.label_map = {'others': 0}
        label = 1
        for layer_name in layer_names:
            self.label_map[layer_name] = label
            label += 1

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data. \
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        token = results['sample_idx']
        location = results['location']
        ego2global = results['ego2global']

        points = np.dot(self.ego_points, ego2global.T)
        points = points[:, :, 0, :2]
        points = np.round(points * 10).astype(np.int)

        road_seg = np.zeros(
            (int(self.nx[1])*5, int(self.nx[0])*5)).astype(np.int64)
        for layer_name in self.layer_names:
            mask = self.layer_dict[location][layer_name]
            x_grid = points[..., 0]
            y_grid = points[..., 1]
            cropped_mask = mask[y_grid, x_grid].astype(bool)
            road_seg[cropped_mask] = self.label_map[layer_name]
        if self.save_path:
            filename = os.path.join(
                self.save_path, results['sample_idx'] + '.png')
            cv2.imwrite(filename, road_seg)
        results['road_seg'] = road_seg.astype(np.int64)
        return results

    @staticmethod
    def gen_dx_bx(xbound, ybound, zbound):
        dx = np.array([row[2] for row in [xbound, ybound, zbound]])
        bx = np.array(
            [row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])
        nx = np.floor(np.array([(row[1] - row[0]) / row[2]
                      for row in [xbound, ybound, zbound]]))
        return dx, bx, nx

    @staticmethod
    def get_geom(grid_conf):
        xbound, ybound, zbound = grid_conf['xbound'], grid_conf['ybound'], grid_conf['zbound']
        dx = np.array([row[2] for row in [xbound, ybound, zbound]])
        bx = np.array(
            [row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])
        nx = np.floor(np.array([(row[1] - row[0]) / row[2]
                      for row in [xbound, ybound, zbound]]))
        pc_range = np.concatenate((bx - dx / 2., bx - dx / 2. + nx * dx))

        x = np.arange(pc_range[0], pc_range[3], 0.1)
        y = np.arange(pc_range[1], pc_range[4], 0.1)
        z = np.array([0.])
        xx, yy, zz = np.meshgrid(x, y, z)
        points = np.stack([xx, yy, zz], axis=-1)
        ego_points = np.concatenate((points, np.ones(
            (points.shape[0], points.shape[1], points.shape[2], 1))), axis=-1)
        return dx, bx, nx, pc_range, ego_points

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        return repr_str


@TRANSFORMS.register_module()
class LoadMiddleSegFromFiles(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.
    """

    def __init__(self, data_root):
        self.data_root = data_root
        self.line = 255

    def __call__(self, results):
        """Call function to load multi-view image from files.

        Args:
            results (dict): Result dict containing multi-view image filenames.

        Returns:
            dict: The result dict containing the multi-view image data. \
                Added keys and values are described below.

                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        filename = os.path.join(
            self.data_root, 'centerline_seg', results['sample_idx'] + '.png')
        middle_seg = mmcv.imread(filename, flag='grayscale').astype(np.int64)
        # middle_seg = np.flip(middle_seg, axis=0)
        middle_seg = middle_seg.T
        # middle_seg = np.flip(middle_seg, axis=1)
        middle_seg[middle_seg == self.line] = 1
        results['middle_seg'] = middle_seg
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(line={self.line}, '
        return repr_str


@TRANSFORMS.register_module()
class LoadMapsFromFiles(object):
    def __init__(self, k=None):
        self.k = k

    def __call__(self, results):
        map_filename = results['map_filename']
        maps = np.load(map_filename)
        map_mask = maps['arr_0'].astype(np.float32)

        maps = map_mask.transpose((2, 0, 1))
        results['gt_map'] = maps
        maps = rearrange(
            maps, 'c (h h1) (w w2) -> (h w) c h1 w2 ', h1=16, w2=16)
        maps = maps.reshape(256, 3*256)
        results['map_shape'] = maps.shape
        results['maps'] = maps
        return results


@TRANSFORMS.register_module()
class LoadMultiViewImageFromMultiSweepsFiles(object):
    """Load multi channel images from a list of separate channel files.
    Expects results['img_filename'] to be a list of filenames.
    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self,
                 sweeps_num=5,
                 to_float32=False,
                 file_client_args=dict(backend='disk'),
                 pad_empty_sweeps=False,
                 sweep_range=[3, 27],
                 sweeps_id=None,
                 color_type='unchanged',
                 sensors=['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT',
                          'CAM_BACK', 'CAM_BACK_LEFT', 'CAM_BACK_RIGHT'],
                 test_mode=True,
                 prob=1.0,
                 ):

        self.sweeps_num = sweeps_num
        self.to_float32 = to_float32
        self.color_type = color_type
        self.file_client_args = file_client_args.copy()
        self.file_client = None
        self.pad_empty_sweeps = pad_empty_sweeps
        self.sensors = sensors
        self.test_mode = test_mode
        self.sweeps_id = sweeps_id
        self.sweep_range = sweep_range
        self.prob = prob
        if self.sweeps_id:
            assert len(self.sweeps_id) == self.sweeps_num

    def __call__(self, results):
        """Call function to load multi-view image from files.
        Args:
            results (dict): Result dict containing multi-view image filenames.
        Returns:
            dict: The result dict containing the multi-view image data. \
                Added keys and values are described below.
                - filename (str): Multi-view image filenames.
                - img (np.ndarray): Multi-view image arrays.
                - img_shape (tuple[int]): Shape of multi-view image arrays.
                - ori_shape (tuple[int]): Shape of original image arrays.
                - pad_shape (tuple[int]): Shape of padded image arrays.
                - scale_factor (float): Scale factor.
                - img_norm_cfg (dict): Normalization configuration of images.
        """
        sweep_imgs_list = []
        timestamp_imgs_list = []
        imgs = results['img']
        img_timestamp = results['img_timestamp']
        lidar_timestamp = results['timestamp']
        img_timestamp = [lidar_timestamp -
                         timestamp for timestamp in img_timestamp]
        sweep_imgs_list.extend(imgs)
        timestamp_imgs_list.extend(img_timestamp)
        nums = len(imgs)
        if self.pad_empty_sweeps and len(results['sweeps']) == 0:
            for i in range(self.sweeps_num):
                sweep_imgs_list.extend(imgs)
                mean_time = (self.sweep_range[0] +
                             self.sweep_range[1]) / 2.0 * 0.083
                timestamp_imgs_list.extend(
                    [time + mean_time for time in img_timestamp])
                for j in range(nums):
                    results['filename'].append(results['filename'][j])
                    results['lidar2img'].append(
                        np.copy(results['lidar2img'][j]))
                    results['intrinsics'].append(
                        np.copy(results['intrinsics'][j]))
                    results['extrinsics'].append(
                        np.copy(results['extrinsics'][j]))
        else:
            if self.sweeps_id:
                choices = self.sweeps_id
            elif len(results['sweeps']) <= self.sweeps_num:
                choices = np.arange(len(results['sweeps']))
            elif self.test_mode:
                choices = [
                    int((self.sweep_range[0] + self.sweep_range[1])/2) - 1]
            else:
                if np.random.random() < self.prob:
                    if self.sweep_range[0] < len(results['sweeps']):
                        sweep_range = list(range(self.sweep_range[0], min(
                            self.sweep_range[1], len(results['sweeps']))))
                    else:
                        sweep_range = list(
                            range(self.sweep_range[0], self.sweep_range[1]))
                    choices = np.random.choice(
                        sweep_range, self.sweeps_num, replace=False)
                else:
                    choices = [
                        int((self.sweep_range[0] + self.sweep_range[1])/2) - 1]

            for idx in choices:
                sweep_idx = min(idx, len(results['sweeps']) - 1)
                sweep = results['sweeps'][sweep_idx]
                if len(sweep.keys()) < len(self.sensors):
                    sweep = results['sweeps'][sweep_idx - 1]
                results['filename'].extend(
                    [sweep[sensor]['data_path'] for sensor in self.sensors])

                img = np.stack([mmcv.imread(sweep[sensor]['data_path'],
                               self.color_type) for sensor in self.sensors], axis=-1)

                if self.to_float32:
                    img = img.astype(np.float32)
                img = [img[..., i] for i in range(img.shape[-1])]
                sweep_imgs_list.extend(img)
                sweep_ts = [lidar_timestamp - sweep[sensor]
                            ['timestamp'] / 1e6 for sensor in self.sensors]
                timestamp_imgs_list.extend(sweep_ts)
                for sensor in self.sensors:
                    results['lidar2img'].append(sweep[sensor]['lidar2img'])
                    results['intrinsics'].append(sweep[sensor]['intrinsics'])
                    results['extrinsics'].append(sweep[sensor]['extrinsics'])
        results['img'] = sweep_imgs_list
        results['timestamp'] = timestamp_imgs_list

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        repr_str += f'(to_float32={self.to_float32}, '
        repr_str += f"color_type='{self.color_type}')"
        return repr_str


@TRANSFORMS.register_module()
class LoadAnnotationsLine3D(LoadAnnotations):
    """Load Annotations3D.

    Load instance mask and semantic mask of points and
    encapsulate the items into related fields.

    Args:
        with_bbox_3d (bool, optional): Whether to load 3D boxes.
            Defaults to True.
        with_label_3d (bool, optional): Whether to load 3D labels.
            Defaults to True.
        with_attr_label (bool, optional): Whether to load attribute label.
            Defaults to False.
        with_mask_3d (bool, optional): Whether to load 3D instance masks.
            for points. Defaults to False.
        with_seg_3d (bool, optional): Whether to load 3D semantic masks.
            for points. Defaults to False.
        with_bbox (bool, optional): Whether to load 2D boxes.
            Defaults to False.
        with_label (bool, optional): Whether to load 2D labels.
            Defaults to False.
        with_mask (bool, optional): Whether to load 2D instance masks.
            Defaults to False.
        with_seg (bool, optional): Whether to load 2D semantic masks.
            Defaults to False.
        with_bbox_depth (bool, optional): Whether to load 2.5D boxes.
            Defaults to False.
        poly2mask (bool, optional): Whether to convert polygon annotations
            to bitmasks. Defaults to True.
        seg_3d_dtype (dtype, optional): Dtype of 3D semantic masks.
            Defaults to int64
        file_client_args (dict): Config dict of file clients, refer to
            https://github.com/open-mmlab/mmcv/blob/master/mmcv/fileio/file_client.py
            for more details.
    """

    def __init__(self,
                 with_line=True,
                 with_bbox_3d=True,
                 with_label_3d=True,
                 with_attr_label=False,
                 with_mask_3d=False,
                 with_seg_3d=False,
                 with_bbox=False,
                 with_label=False,
                 with_mask=False,
                 with_seg=False,
                 with_bbox_depth=False,
                 poly2mask=True,
                 seg_3d_dtype='int',
                 file_client_args=dict(backend='disk')):
        super().__init__(
            with_bbox,
            with_label,
            with_mask,
            with_seg,
            poly2mask,
            file_client_args=file_client_args)
        self.with_line = with_line
        self.with_bbox_3d = with_bbox_3d
        self.with_bbox_depth = with_bbox_depth
        self.with_label_3d = with_label_3d
        self.with_attr_label = with_attr_label
        self.with_mask_3d = with_mask_3d
        self.with_seg_3d = with_seg_3d
        self.seg_3d_dtype = seg_3d_dtype

    def _load_bboxes_3d(self, results):
        """Private function to load 3D bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box annotations.
        """
        results['gt_bboxes_3d'] = results['ann_info']['gt_bboxes_3d']
        results['bbox3d_fields'].append('gt_bboxes_3d')
        return results

    def _load_lines(self, results):
        """Private function to load 3D bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box annotations.
        """
        # results['gt_lines'] = results['ann_info']['gt_lines']
        results['gt_lines'] = results['ann_info']['gt_lines']
        return results

    def _load_bboxes_depth(self, results):
        """Private function to load 2.5D bounding box annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 2.5D bounding box annotations.
        """
        results['centers2d'] = results['ann_info']['centers2d']
        results['depths'] = results['ann_info']['depths']
        return results

    def _load_labels_3d(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded label annotations.
        """
        results['gt_labels_3d'] = results['ann_info']['gt_labels_3d']
        return results

    def _load_attr_labels(self, results):
        """Private function to load label annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded label annotations.
        """
        results['attr_labels'] = results['ann_info']['attr_labels']
        return results

    def _load_masks_3d(self, results):
        """Private function to load 3D mask annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D mask annotations.
        """
        pts_instance_mask_path = results['ann_info']['pts_instance_mask_path']

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            mask_bytes = self.file_client.get(pts_instance_mask_path)
            pts_instance_mask = np.frombuffer(mask_bytes, dtype=np.int)
        except ConnectionError:
            mmcv.check_file_exist(pts_instance_mask_path)
            pts_instance_mask = np.fromfile(
                pts_instance_mask_path, dtype=np.int32)

        results['pts_instance_mask'] = pts_instance_mask
        results['pts_mask_fields'].append('pts_instance_mask')
        return results

    def _load_semantic_seg_3d(self, results):
        """Private function to load 3D semantic segmentation annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing the semantic segmentation annotations.
        """
        pts_semantic_mask_path = results['ann_info']['pts_semantic_mask_path']

        if self.file_client is None:
            self.file_client = mmcv.FileClient(**self.file_client_args)
        try:
            mask_bytes = self.file_client.get(pts_semantic_mask_path)
            # add .copy() to fix read-only bug
            pts_semantic_mask = np.frombuffer(
                mask_bytes, dtype=self.seg_3d_dtype).copy()
        except ConnectionError:
            mmcv.check_file_exist(pts_semantic_mask_path)
            pts_semantic_mask = np.fromfile(
                pts_semantic_mask_path, dtype=np.int32)

        results['pts_semantic_mask'] = pts_semantic_mask
        results['pts_seg_fields'].append('pts_semantic_mask')
        return results

    def __call__(self, results):
        """Call function to load multiple types annotations.

        Args:
            results (dict): Result dict from :obj:`mmdet3d.CustomDataset`.

        Returns:
            dict: The dict containing loaded 3D bounding box, label, mask and
                semantic segmentation annotations.
        """
        results = super().__call__(results)
        # if self.with_line:
        #     results = self._load_lines(results)
        #     if results is None:
        #         return None
        if self.with_bbox_3d:
            results = self._load_bboxes_3d(results)
            if results is None:
                return None
        if self.with_bbox_depth:
            results = self._load_bboxes_depth(results)
            if results is None:
                return None
        if self.with_label_3d:
            results = self._load_labels_3d(results)
        if self.with_attr_label:
            results = self._load_attr_labels(results)
        if self.with_mask_3d:
            results = self._load_masks_3d(results)
        if self.with_seg_3d:
            results = self._load_semantic_seg_3d(results)

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        indent_str = '    '
        repr_str = self.__class__.__name__ + '(\n'
        repr_str += f'{indent_str}with_bbox_3d={self.with_bbox_3d}, '
        repr_str += f'{indent_str}with_label_3d={self.with_label_3d}, '
        repr_str += f'{indent_str}with_attr_label={self.with_attr_label}, '
        repr_str += f'{indent_str}with_mask_3d={self.with_mask_3d}, '
        repr_str += f'{indent_str}with_seg_3d={self.with_seg_3d}, '
        repr_str += f'{indent_str}with_bbox={self.with_bbox}, '
        repr_str += f'{indent_str}with_label={self.with_label}, '
        repr_str += f'{indent_str}with_mask={self.with_mask}, '
        repr_str += f'{indent_str}with_seg={self.with_seg}, '
        repr_str += f'{indent_str}with_bbox_depth={self.with_bbox_depth}, '
        repr_str += f'{indent_str}poly2mask={self.poly2mask})'
        return repr_str


@TRANSFORMS.register_module()
class LoadAV2OrderedBzCenterline(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.
    """

    def __init__(self, grid_conf, bz_grid_conf, epsilon=1e-5):
        self.grid_conf = grid_conf
        self.bz_grid_conf = bz_grid_conf
        self.epsilon = epsilon

    def __call__(self, results):
        """Call function to load multi-view image from files.
        """
        # import pdb
        # pdb.set_trace()
        # results['center_lines'] = AV2OrederedBzCenterLine(results['center_lines'], results['token'], self.grid_conf, self.bz_grid_conf, self.epsilon)
        results['center_lines'] = AV2OrederedBzCenterLine(results['center_lines'], results['sample_idx'], self.grid_conf,
                                                          self.bz_grid_conf, self.epsilon)
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        return repr_str


@TRANSFORMS.register_module()
class LoadAV2OrderedRMcontinuedBzCenterline(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.
    """

    def __init__(self, grid_conf, bz_grid_conf, epsilon=1):
        self.grid_conf = grid_conf
        self.bz_grid_conf = bz_grid_conf
        self.epsilon = epsilon

    def __call__(self, results):
        """Call function to load multi-view image from files.
        """
        # import pdb
        # pdb.set_trace()
        # results['center_lines'] = AV2OrederedBzCenterLine(results['center_lines'], results['token'], self.grid_conf, self.bz_grid_conf, self.epsilon)
        results['center_lines'] = AV2OrederedRMcontinuedBzCenterLine(results['center_lines'], results['sample_idx'], self.grid_conf,
                                                          self.bz_grid_conf, self.epsilon)
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        return repr_str




@TRANSFORMS.register_module()
class LoadAV2OrderedBzCenterline_new(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.
    """

    def __init__(self, grid_conf, bz_grid_conf, epsilon=1e-5, centerline_path=None):
        self.grid_conf = grid_conf
        self.bz_grid_conf = bz_grid_conf
        self.epsilon = epsilon
        self.centerline_path = centerline_path

    def __call__(self, results):
        """Call function to load multi-view image from files.
        """
        file_name = os.path.join(self.centerline_path,
                                 results['token'] + '.pkl')
        center_lines = mmcv.load(file_name)
        results['center_lines'] = AV2OrederedBzCenterLine_new(
            center_lines, results['token'], self.grid_conf, self.bz_grid_conf, self.epsilon)
        # results['center_lines'] = AV2OrederedBzCenterLine_new(results['center_lines'], results['sample_idx'], self.grid_conf,
        #                                                       self.bz_grid_conf, self.epsilon)
        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        return repr_str


@TRANSFORMS.register_module()
class LoadAV2OrderedBzCenterline_test(object):
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.
    """

    def __init__(self, grid_conf, bz_grid_conf, epsilon=1e-5, centerline_path=None):
        self.grid_conf = grid_conf
        self.bz_grid_conf = bz_grid_conf
        self.epsilon = epsilon
        self.centerline_path = centerline_path

    def __call__(self, results):
        """Call function to load multi-view image from files.
        """
        # file_name = os.path.join(self.centerline_path, results['token'] + '.pkl')
        # center_lines = mmcv.load(file_name)
        results['center_lines'] = AV2OrederedBzCenterLine_new(
            results['center_lines'], results['token'], self.grid_conf, self.bz_grid_conf, self.epsilon)

        return results

    def __repr__(self):
        """str: Return a string that describes the module."""
        repr_str = self.__class__.__name__
        return repr_str


@TRANSFORMS.register_module()
class TransformAV2OrderedBzLane2Graph(object):
    def __init__(self, n_control=3, orderedDFS=True):
        self.order = orderedDFS
        self.n_control = n_control

    def __call__(self, results):
        centerlines = results['center_lines']
        nodes, nodes_adj = centerlines.export_node_adj()  # get nodes and adj

        # for i in range(len(nodes_adj)):
        #     for j in range(len(nodes_adj)):
        #         if nodes_adj[i][j] == 1 and nodes_adj[i][j] == nodes_adj[j][i]:
        #             print(i, j)

        centerlines.sub_graph_split()  # split sub graph

        scene_graph = AV2OrderedBzSceneGraph(centerlines.subgraphs_nodes, centerlines.subgraphs_adj,
                                             centerlines.subgraphs_points_in_between_nodes, self.n_control)  # subgraph dfs already
        scene_sentance, scene_sentance_list = scene_graph.sequelize_new(
            orderedDFS=self.order)
        centerline_sequence = sentance2bzseq2(
            scene_sentance_list, centerlines.pc_range, centerlines.dx, centerlines.bz_pc_range, centerlines.bz_nx)
        clause_length = 4 + 2*(self.n_control-2)
        if len(centerline_sequence) % clause_length != 0:
            centerline_sequence = centerline_sequence[:(
                centerline_sequence//clause_length*clause_length)]
        centerline_coord = np.stack(
            [centerline_sequence[::clause_length], centerline_sequence[1::clause_length]], axis=1)
        centerline_label = centerline_sequence[2::clause_length]
        centerline_connect = centerline_sequence[3::clause_length]
        centerline_coeff = np.stack(
            [centerline_sequence[k::clause_length] for k in range(4, clause_length)], axis=1)

        results['centerline_sequence'] = centerline_sequence
        results['centerline_coord'] = centerline_coord
        results['centerline_label'] = centerline_label
        results['centerline_connect'] = centerline_connect
        results['centerline_coeff'] = centerline_coeff
        results['n_control'] = self.n_control


        return results


@TRANSFORMS.register_module()
class TransformAV2OrderedBzLane2Graph_new(object):
    def __init__(self, n_control=3, orderedDFS=True):
        self.order = orderedDFS
        self.n_control = n_control

    def __call__(self, results):
        centerlines = results['center_lines']
        nodes, nodes_adj = centerlines.export_node_adj()  # get nodes and adj


        scene_graph = AV2OrderedBzSceneGraph_new([centerlines.all_nodes], [centerlines.adj],
                                                 [centerlines.points_in_between_nodes],
                                                 self.n_control)
 

        scene_sentance, scene_sentance_list = scene_graph.sequelize_new(
            orderedDFS=self.order)
        centerline_sequence = sentance2bzseq2(
            scene_sentance_list, centerlines.pc_range, centerlines.dx, centerlines.bz_pc_range, centerlines.bz_nx)
        clause_length = 4 + 2*(self.n_control-2)
        if len(centerline_sequence) % clause_length != 0:
            centerline_sequence = centerline_sequence[:(
                centerline_sequence//clause_length*clause_length)]
        centerline_coord = np.stack(
            [centerline_sequence[::clause_length], centerline_sequence[1::clause_length]], axis=1)
        centerline_label = centerline_sequence[2::clause_length]
        centerline_connect = centerline_sequence[3::clause_length]
        centerline_coeff = np.stack(
            [centerline_sequence[k::clause_length] for k in range(4, clause_length)], axis=1)

        results['centerline_sequence'] = centerline_sequence
        results['centerline_coord'] = centerline_coord
        results['centerline_label'] = centerline_label
        results['centerline_connect'] = centerline_connect
        results['centerline_coeff'] = centerline_coeff
        results['n_control'] = self.n_control
        # results['token'] = results['sample_idx']

        return results
