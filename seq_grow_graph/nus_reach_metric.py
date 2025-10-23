# Copyright (c) OpenMMLab. All rights reserved.
import tempfile
from os import path as osp
from typing import Dict, List, Optional, Sequence, Tuple, Union

import mmengine
import numpy as np
import pyquaternion
import torch
from mmengine import Config, load
from mmengine.evaluator import BaseMetric
from mmengine.logging import MMLogger
from nuscenes.eval.detection.config import config_factory
from nuscenes.eval.detection.data_classes import DetectionConfig
from nuscenes.utils.data_classes import Box as NuScenesBox

from mmdet3d.models.layers import box3d_multiclass_nms
from mmdet3d.registry import METRICS
from mmdet3d.structures import (CameraInstance3DBoxes, LiDARInstance3DBoxes,
                                bbox3d2result, xywhr2xyxyr)


@METRICS.register_module()
class NuScenesReachMetric(BaseMetric):

    def __init__(self,
                 data_root: str,
                 ann_file: str,
                 metric: Union[str, List[str]] = 'bbox',
                 modality: dict = dict(use_camera=False, use_lidar=True),
                 prefix: Optional[str] = None,
                 format_only: bool = False,
                 jsonfile_prefix: Optional[str] = None,
                 eval_version: str = 'detection_cvpr_2019',
                 collect_device: str = 'cpu',
                 backend_args: Optional[dict] = None, 
                 grid_conf=None,
                 bz_grid_conf=None,
                 landmark_thresholds=[1, 3, 5, 8, 10],
                 reach_thresholds=[1, 2, 3, 4, 5],is_new=False,is_aaai=False
                 ) -> None:
        self.default_prefix = 'NuScenes metric'
        super(NuScenesReachMetric, self).__init__(
            collect_device=collect_device, prefix=prefix)
        if modality is None:
            modality = dict(
                use_camera=False,
                use_lidar=True,
            )
        self.ann_file = ann_file
        self.data_root = data_root
        self.modality = modality
        self.format_only = format_only
        if self.format_only:
            assert jsonfile_prefix is not None, 'jsonfile_prefix must be not '
            'None when format_only is True, otherwise the result files will '
            'be saved to a temp directory which will be cleanup at the end.'

        self.jsonfile_prefix = jsonfile_prefix
        self.backend_args = backend_args

        self.metrics = metric if isinstance(metric, list) else [metric]

        self.eval_version = eval_version
        self.eval_detection_configs = config_factory(self.eval_version)

        self.grid_conf = grid_conf
        self.bz_grid_conf = bz_grid_conf
        self.interval = grid_conf['xbound'][-1]
        self.landmark_thresholds = landmark_thresholds
        self.reach_thresholds = reach_thresholds
        self.is_new=is_new
        self.is_aaai=is_aaai

    def process(self, data_batch: dict, data_samples: Sequence[dict]) -> None:
        """Process one batch of data samples and predictions.

        The processed results should be stored in ``self.results``, which will
        be used to compute the metrics when all batches have been processed.

        Args:
            data_batch (dict): A batch of data from the dataloader.
            data_samples (Sequence[dict]): A batch of outputs from the model.
        """
        for data_sample in data_samples:
            result = dict()
            result['line_results'] = data_sample['line_results']
            result['token'] = data_sample['token']
            self.results.append(result)

    def compute_metrics(self, results: List[dict]) -> Dict[str, float]:
        """Compute the metrics from processed results.

        Args:
            results (List[dict]): The processed results of each batch.

        Returns:
            Dict[str, float]: The computed metrics. The keys are the names of
            the metrics, and the values are corresponding results.
        """
        logger: MMLogger = MMLogger.get_current_instance()

        classes = self.dataset_meta['classes']
        self.version = self.dataset_meta['version']
        # load annotations
        self.data_infos = load(
            self.ann_file, backend_args=self.backend_args)['infos']
        result_path, tmp_dir = self.format_results(results, classes,
                                                   self.jsonfile_prefix)

        metric_dict = {}

        if self.format_only:
            logger.info(
                f'results are saved in {osp.basename(self.jsonfile_prefix)}')
            return metric_dict

        for metric in self.metrics:
            ap_dict = self._evaluate_single(
                result_path, metric=metric, logger=logger)
            for result in ap_dict:
                metric_dict[result] = ap_dict[result]

        if tmp_dir is not None:
            tmp_dir.cleanup()
        return metric_dict

    def _evaluate_single(
            self,
            result_path, is_llm_result=False,
            metric='ar_reach',
            logger=None) -> Dict[str, float]:
        """Evaluation for a single model in nuScenes protocol.

        Args:
            result_path (str): Path of the result file.
            classes (List[str], optional): A list of class name.
                Defaults to None.
            result_name (str): Result name in the metric prefix.
                Defaults to 'pred_instances_3d'.

        Returns:
            Dict[str, float]: Dictionary of evaluation details.
        """
        if isinstance(metric, list):
            metric = metric[0]
        datatype, metric = metric.split('_')
        if datatype == 'ar':
            if self.is_aaai:
                from projects.SeqGrowGraph.seq_grow_graph.transforms import BzRoadnetReachDistEvalAAAI as RoadnetReachDistEval
            elif not self.is_new:
                from projects.SeqGrowGraph.seq_grow_graph.transforms import BzRoadnetReachDistEval as RoadnetReachDistEval
            else:
                from projects.SeqGrowGraph.seq_grow_graph.transforms import BzRoadnetReachDistEvalNew as RoadnetReachDistEval
        else:
            assert False, "Not implement error"
        
        metrics = RoadnetReachDistEval(result_path=result_path, 
                                           data_root=self.data_root,
                                           grid_conf=self.grid_conf,
                                           bz_grid_conf=self.bz_grid_conf,
                                           landmark_thresholds=self.landmark_thresholds,
                                           reach_thresholds=self.reach_thresholds, 
                                           logger=logger,
                                           val_pkl=self.data_infos
                                           )

        # record metrics
        detail = dict()
        metric_prefix = f'line_results_NuScenes'
        landmark_precision = metrics['landmark_precision']
        landmark_recall = metrics['landmark_recall']
        landmark_fscore = metrics['landmark_f_score']
        reach_precision = metrics['reach_precision']
        reach_recall = metrics['reach_recall']
        reach_fscore = metrics['reach_f_score']
        for i, thres in enumerate(self.landmark_thresholds):
            thres_str = float('{:.1f}'.format(thres * self.interval))
            detail['{}/LP_{}'.format(metric_prefix, thres_str)] = float('{:.4f}'.format(landmark_precision[i]))
            detail['{}/LR_{}'.format(metric_prefix, thres_str)] = float('{:.4f}'.format(landmark_recall[i]))
            detail['{}/LF_{}'.format(metric_prefix, thres_str)] = float('{:.4f}'.format(landmark_fscore[i]))
        detail['{}/mLP'.format(metric_prefix)] = float('{:.4f}'.format(landmark_precision.mean()))
        detail['{}/mLR'.format(metric_prefix)] = float('{:.4f}'.format(landmark_recall.mean()))
        detail['{}/mLF'.format(metric_prefix)] = float('{:.4f}'.format(landmark_fscore.mean()))
        for i, thres in enumerate(self.reach_thresholds):
            thres_str = float('{:.1f}'.format(thres * self.interval))
            detail['{}/RP_{}'.format(metric_prefix, thres_str)] = float('{:.4f}'.format(reach_precision[i]))
            detail['{}/RR_{}'.format(metric_prefix, thres_str)] = float('{:.4f}'.format(reach_recall[i]))
            detail['{}/RF_{}'.format(metric_prefix, thres_str)] = float('{:.4f}'.format(reach_fscore[i]))
        detail['{}/mRP'.format(metric_prefix)] = float('{:.4f}'.format(reach_precision.mean()))
        detail['{}/mRR'.format(metric_prefix)] = float('{:.4f}'.format(reach_recall.mean()))
        detail['{}/mRF'.format(metric_prefix)] = float('{:.4f}'.format(reach_fscore.mean()))
        return detail

    def format_results(
        self,
        results: List[dict],
        classes: Optional[List[str]] = None,
        jsonfile_prefix: Optional[str] = None
    ) -> Tuple[dict, Union[tempfile.TemporaryDirectory, None]]:
        """Format the mmdet3d results to standard NuScenes json file.

        Args:
            results (List[dict]): Testing results of the dataset.
            classes (List[str], optional): A list of class name.
                Defaults to None.
            jsonfile_prefix (str, optional): The prefix of json files. It
                includes the file path and the prefix of filename, e.g.,
                "a/b/prefix". If not specified, a temp file will be created.
                Defaults to None.

        Returns:
            tuple: Returns (result_dict, tmp_dir), where ``result_dict`` is a
            dict containing the json filepaths, ``tmp_dir`` is the temporal
            directory created for saving json files when ``jsonfile_prefix`` is
            not specified.
        """
        assert isinstance(results, list), 'results must be a list'

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None

        result_files = self._format_line(results, jsonfile_prefix)
        return result_files, tmp_dir


    def _format_line(self, results, jsonfile_prefix=None):
        """Convert the results to the standard format.

        Args:
            results (list[dict]): Testing results of the dataset.
            jsonfile_prefix (str): The prefix of the output jsonfile.
                You can specify the output directory/filename by
                modifying the jsonfile_prefix. Default: None.

        Returns:
            str: Path of the output json file.
        """
        nusc_annos = {}

        print('Start to convert detection format...')
        for sample_id, res in enumerate(mmengine.track_iter_progress(results)):
            if 'pred_node_lists' in res['line_results']:
                node_list = res['line_results']['pred_node_lists']
            else:
                node_list = res['line_results']['line_seqs']
                
            sample_token = res['token']
            nusc_annos[sample_token] = node_list
        nusc_submissions = {
            'meta': self.modality,
            'results': nusc_annos,
        }

        mmengine.mkdir_or_exist(jsonfile_prefix)
        res_path = osp.join(jsonfile_prefix, 'results_nusc.json')
        print('Results writes to', res_path)
        mmengine.dump(nusc_submissions, res_path)
        return res_path


def output_to_nusc_box(
        detection: dict) -> Tuple[List[NuScenesBox], Union[np.ndarray, None]]:
    """Convert the output to the box class in the nuScenes.

    Args:
        detection (dict): Detection results.

            - bboxes_3d (:obj:`BaseInstance3DBoxes`): Detection bbox.
            - scores_3d (torch.Tensor): Detection scores.
            - labels_3d (torch.Tensor): Predicted box labels.

    Returns:
        Tuple[List[:obj:`NuScenesBox`], np.ndarray or None]: List of standard
        NuScenesBoxes and attribute labels.
    """
    bbox3d = detection['bboxes_3d']
    scores = detection['scores_3d'].numpy()
    labels = detection['labels_3d'].numpy()
    attrs = None
    if 'attr_labels' in detection:
        attrs = detection['attr_labels'].numpy()

    box_gravity_center = bbox3d.gravity_center.numpy()
    box_dims = bbox3d.dims.numpy()
    box_yaw = bbox3d.yaw.numpy()

    box_list = []

    if isinstance(bbox3d, LiDARInstance3DBoxes):
        # our LiDAR coordinate system -> nuScenes box coordinate system
        nus_box_dims = box_dims[:, [1, 0, 2]]
        for i in range(len(bbox3d)):
            quat = pyquaternion.Quaternion(axis=[0, 0, 1], radians=box_yaw[i])
            velocity = (*bbox3d.tensor[i, 7:9], 0.0)
            # velo_val = np.linalg.norm(box3d[i, 7:9])
            # velo_ori = box3d[i, 6]
            # velocity = (
            # velo_val * np.cos(velo_ori), velo_val * np.sin(velo_ori), 0.0)
            box = NuScenesBox(
                box_gravity_center[i],
                nus_box_dims[i],
                quat,
                label=labels[i],
                score=scores[i],
                velocity=velocity)
            box_list.append(box)
    elif isinstance(bbox3d, CameraInstance3DBoxes):
        # our Camera coordinate system -> nuScenes box coordinate system
        # convert the dim/rot to nuscbox convention
        nus_box_dims = box_dims[:, [2, 0, 1]]
        nus_box_yaw = -box_yaw
        for i in range(len(bbox3d)):
            q1 = pyquaternion.Quaternion(
                axis=[0, 0, 1], radians=nus_box_yaw[i])
            q2 = pyquaternion.Quaternion(axis=[1, 0, 0], radians=np.pi / 2)
            quat = q2 * q1
            velocity = (bbox3d.tensor[i, 7], 0.0, bbox3d.tensor[i, 8])
            box = NuScenesBox(
                box_gravity_center[i],
                nus_box_dims[i],
                quat,
                label=labels[i],
                score=scores[i],
                velocity=velocity)
            box_list.append(box)
    else:
        raise NotImplementedError(
            f'Do not support convert {type(bbox3d)} bboxes '
            'to standard NuScenesBoxes.')

    return box_list, attrs


def lidar_nusc_box_to_global(
        info: dict, boxes: List[NuScenesBox], classes: List[str],
        eval_configs: DetectionConfig) -> List[NuScenesBox]:
    """Convert the box from ego to global coordinate.

    Args:
        info (dict): Info for a specific sample data, including the calibration
            information.
        boxes (List[:obj:`NuScenesBox`]): List of predicted NuScenesBoxes.
        classes (List[str]): Mapped classes in the evaluation.
        eval_configs (:obj:`DetectionConfig`): Evaluation configuration object.

    Returns:
        List[:obj:`DetectionConfig`]: List of standard NuScenesBoxes in the
        global coordinate.
    """
    box_list = []
    for box in boxes:
        # Move box to ego vehicle coord system
        lidar2ego = np.array(info['lidar_points']['lidar2ego'])
        box.rotate(
            pyquaternion.Quaternion(matrix=lidar2ego, rtol=1e-05, atol=1e-07))
        box.translate(lidar2ego[:3, 3])
        # filter det in ego.
        cls_range_map = eval_configs.class_range
        radius = np.linalg.norm(box.center[:2], 2)
        det_range = cls_range_map[classes[box.label]]
        if radius > det_range:
            continue
        # Move box to global coord system
        ego2global = np.array(info['ego2global'])
        box.rotate(
            pyquaternion.Quaternion(matrix=ego2global, rtol=1e-05, atol=1e-07))
        box.translate(ego2global[:3, 3])
        box_list.append(box)
    return box_list


def cam_nusc_box_to_global(
    info: dict,
    boxes: List[NuScenesBox],
    attrs: np.ndarray,
    classes: List[str],
    eval_configs: DetectionConfig,
    camera_type: str = 'CAM_FRONT',
) -> Tuple[List[NuScenesBox], List[int]]:
    """Convert the box from camera to global coordinate.

    Args:
        info (dict): Info for a specific sample data, including the calibration
            information.
        boxes (List[:obj:`NuScenesBox`]): List of predicted NuScenesBoxes.
        attrs (np.ndarray): Predicted attributes.
        classes (List[str]): Mapped classes in the evaluation.
        eval_configs (:obj:`DetectionConfig`): Evaluation configuration object.
        camera_type (str): Type of camera. Defaults to 'CAM_FRONT'.

    Returns:
        Tuple[List[:obj:`NuScenesBox`], List[int]]: List of standard
        NuScenesBoxes in the global coordinate and attribute label.
    """
    box_list = []
    attr_list = []
    for (box, attr) in zip(boxes, attrs):
        # Move box to ego vehicle coord system
        cam2ego = np.array(info['images'][camera_type]['cam2ego'])
        box.rotate(
            pyquaternion.Quaternion(matrix=cam2ego, rtol=1e-05, atol=1e-07))
        box.translate(cam2ego[:3, 3])
        # filter det in ego.
        cls_range_map = eval_configs.class_range
        radius = np.linalg.norm(box.center[:2], 2)
        det_range = cls_range_map[classes[box.label]]
        if radius > det_range:
            continue
        # Move box to global coord system
        ego2global = np.array(info['ego2global'])
        box.rotate(
            pyquaternion.Quaternion(matrix=ego2global, rtol=1e-05, atol=1e-07))
        box.translate(ego2global[:3, 3])
        box_list.append(box)
        attr_list.append(attr)
    return box_list, attr_list


def global_nusc_box_to_cam(info: dict, boxes: List[NuScenesBox],
                           classes: List[str],
                           eval_configs: DetectionConfig) -> List[NuScenesBox]:
    """Convert the box from global to camera coordinate.

    Args:
        info (dict): Info for a specific sample data, including the calibration
            information.
        boxes (List[:obj:`NuScenesBox`]): List of predicted NuScenesBoxes.
        classes (List[str]): Mapped classes in the evaluation.
        eval_configs (:obj:`DetectionConfig`): Evaluation configuration object.

    Returns:
        List[:obj:`NuScenesBox`]: List of standard NuScenesBoxes in camera
        coordinate.
    """
    box_list = []
    for box in boxes:
        # Move box to ego vehicle coord system
        ego2global = np.array(info['ego2global'])
        box.translate(-ego2global[:3, 3])
        box.rotate(
            pyquaternion.Quaternion(matrix=ego2global, rtol=1e-05,
                                    atol=1e-07).inverse)
        # filter det in ego.
        cls_range_map = eval_configs.class_range
        radius = np.linalg.norm(box.center[:2], 2)
        det_range = cls_range_map[classes[box.label]]
        if radius > det_range:
            continue
        # Move box to camera coord system
        cam2ego = np.array(info['images']['CAM_FRONT']['cam2ego'])
        box.translate(-cam2ego[:3, 3])
        box.rotate(
            pyquaternion.Quaternion(matrix=cam2ego, rtol=1e-05,
                                    atol=1e-07).inverse)
        box_list.append(box)
    return box_list


def nusc_box_to_cam_box3d(
    boxes: List[NuScenesBox]
) -> Tuple[CameraInstance3DBoxes, torch.Tensor, torch.Tensor]:
    """Convert boxes from :obj:`NuScenesBox` to :obj:`CameraInstance3DBoxes`.

    Args:
        boxes (:obj:`List[NuScenesBox]`): List of predicted NuScenesBoxes.

    Returns:
        Tuple[:obj:`CameraInstance3DBoxes`, torch.Tensor, torch.Tensor]:
        Converted 3D bounding boxes, scores and labels.
    """
    locs = torch.Tensor([b.center for b in boxes]).view(-1, 3)
    dims = torch.Tensor([b.wlh for b in boxes]).view(-1, 3)
    rots = torch.Tensor([b.orientation.yaw_pitch_roll[0]
                         for b in boxes]).view(-1, 1)
    velocity = torch.Tensor([b.velocity[0::2] for b in boxes]).view(-1, 2)

    # convert nusbox to cambox convention
    dims[:, [0, 1, 2]] = dims[:, [1, 2, 0]]
    rots = -rots

    boxes_3d = torch.cat([locs, dims, rots, velocity], dim=1).cuda()
    cam_boxes3d = CameraInstance3DBoxes(
        boxes_3d, box_dim=9, origin=(0.5, 0.5, 0.5))
    scores = torch.Tensor([b.score for b in boxes]).cuda()
    labels = torch.LongTensor([b.label for b in boxes]).cuda()
    nms_scores = scores.new_zeros(scores.shape[0], 10 + 1)
    indices = labels.new_tensor(list(range(scores.shape[0])))
    nms_scores[indices, labels] = scores
    return cam_boxes3d, nms_scores, labels
