# Copyright (c) OpenMMLab. All rights reserved.
from logging import root
import os
from collections import OrderedDict
from os import path as osp
from select import select
from typing import List, Tuple, Union

import numpy as np
from nuscenes.nuscenes import NuScenes
from pyquaternion import Quaternion

from nuscenes.map_expansion.map_api import NuScenesMap
import mmengine
import copy

from shapely.strtree import STRtree
from .nuscenes_pon_split import TRAIN_SCENES, VAL_SCENES


STATIC_CLASSES = ['drivable_area', 'ped_crossing', 'walkway', 'carpark_area']

LOCATIONS = ['boston-seaport', 'singapore-onenorth', 'singapore-queenstown',
             'singapore-hollandvillage']


nus_categories = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                  'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                  'barrier')

nus_attributes = ('cycle.with_rider', 'cycle.without_rider',
                  'pedestrian.moving', 'pedestrian.standing',
                  'pedestrian.sitting_lying_down', 'vehicle.moving',
                  'vehicle.parked', 'vehicle.stopped', 'None')

color_paltte = np.random.randint(255, size=(500, 3))


def create_nuscenes_infos(root_path,
                          info_prefix,
                          version='v1.0-trainval',
                          max_sweeps=10,
                          map_size=[(-70, 70),(-70, 70)],
                          resolution=0.5,
                          map_vis=False):
    """Create info file of nuscene dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        root_path (str): Path of the data root.
        info_prefix (str): Prefix of the info file to be generated.
        version (str, optional): Version of the data.
            Default: 'v1.0-trainval'.
        max_sweeps (int, optional): Max number of sweeps.
            Default: 10.
    """

    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)

    available_vers = ['v1.0-trainval', 'v1.0-test', 'v1.0-mini']
    assert version in available_vers
    train_scenes = TRAIN_SCENES
    val_scenes = VAL_SCENES

    # filter existing scenes.
    available_scenes = get_available_scenes(nusc)
    available_scene_names = [s['name'] for s in available_scenes]
    train_scenes = list(
        filter(lambda x: x in available_scene_names, train_scenes))
    val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
    train_scenes = set([
        available_scenes[available_scene_names.index(s)]['token']
        for s in train_scenes
    ])
    val_scenes = set([
        available_scenes[available_scene_names.index(s)]['token']
        for s in val_scenes
    ])

    map_data = { location : load_map_data(root_path, location) 
                for location in LOCATIONS }
    
    my_map_apis = { location : NuScenesMap(root_path, location) 
             for location in LOCATIONS }

    all_centers = {location : my_map_apis[location].discretize_centerlines(resolution)
             for location in LOCATIONS}
    

    centerlines_tokens = {}
    lane_or_laneconnector = {}
    for location in LOCATIONS:
        centerlines_tokens[location] = []
        scene_map_api = my_map_apis[location]
        all_lines = scene_map_api.lane + scene_map_api.lane_connector
        lane_list = ['lane' for i in range(len(scene_map_api.lane))]
        lane_connector_list = ['lane_connector' for i in range(len(scene_map_api.lane_connector))]
        lane_or_laneconnector[location] = lane_list+lane_connector_list
        all_lines_tokens = []
        for lin in all_lines:
            all_lines_tokens.append(lin['token'])
        
        centerlines_tokens[location] = all_lines_tokens

    test = 'test' in version
    if test:
        print('test scene: {}'.format(len(train_scenes)))
    else:
        print('train scene: {}, val scene: {}'.format(
            len(train_scenes), len(val_scenes)))
    train_nusc_infos, val_nusc_infos = _fill_trainval_infos(
        nusc, all_centers, centerlines_tokens, lane_or_laneconnector, my_map_apis, train_scenes, val_scenes, test, max_sweeps=max_sweeps, map_size=map_size, resolution=resolution, vis=map_vis)

    metadata = dict(version=version)
    if test:
        print('test sample: {}'.format(len(train_nusc_infos)))
        data = dict(infos=train_nusc_infos, metadata=metadata)
        info_path = osp.join(root_path,
                             '{}_infos_pon_test.pkl'.format(info_prefix))
        mmengine.dump(data, info_path)
    else:
        print('train sample: {}, val sample: {}'.format(
            len(train_nusc_infos), len(val_nusc_infos)))
        data = dict(infos=train_nusc_infos, metadata=metadata)
        info_path = osp.join(root_path,
                             '{}_infos_pon_train.pkl'.format(info_prefix))
        mmengine.dump(data, info_path)
        data['infos'] = val_nusc_infos
        info_val_path = osp.join(root_path,
                                 '{}_infos_pon_val.pkl'.format(info_prefix))
        mmengine.dump(data, info_val_path)


def get_available_scenes(nusc):
    """Get available scenes from the input nuscenes class.

    Given the raw data, get the information of available scenes for
    further info generation.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.

    Returns:
        available_scenes (list[dict]): List of basic information for the
            available scenes.
    """
    available_scenes = []
    print('total scene num: {}'.format(len(nusc.scene)))
    for scene in nusc.scene:
        scene_token = scene['token']
        scene_rec = nusc.get('scene', scene_token)
        sample_rec = nusc.get('sample', scene_rec['first_sample_token'])
        sd_rec = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
        has_more_frames = True
        scene_not_exist = False
        while has_more_frames:
            lidar_path, boxes, _ = nusc.get_sample_data(sd_rec['token'])
            lidar_path = str(lidar_path)
            if os.getcwd() in lidar_path:
                # path from lyftdataset is absolute path
                lidar_path = lidar_path.split(f'{os.getcwd()}/')[-1]
                # relative path
            if not mmengine.is_filepath(lidar_path):
                scene_not_exist = True
                break
            else:
                break
        if scene_not_exist:
            continue
        available_scenes.append(scene)
    print('exist scene num: {}'.format(len(available_scenes)))
    return available_scenes



def load_map_data(dataroot, location):

    # Load the NuScenes map object
    nusc_map = NuScenesMap(dataroot, location)
    

    map_data = OrderedDict()
    for layer in STATIC_CLASSES:
        
        # Retrieve all data associated with the current layer
        records = getattr(nusc_map, layer)
        polygons = list()

        # Drivable area records can contain multiple polygons
        if layer == 'drivable_area':
            for record in records:

                # Convert each entry in the record into a shapely object
                for token in record['polygon_tokens']:
                    poly = nusc_map.extract_polygon(token)
                    if poly.is_valid:
                        polygons.append(poly)
        else:
            for record in records:

                # Convert each entry in the record into a shapely object
                poly = nusc_map.extract_polygon(record['polygon_token'])
                if poly.is_valid:
                    polygons.append(poly)
        
        # Store as an R-Tree for fast intersection queries
        map_data[layer] = STRtree(polygons)
    
    return map_data

def _fill_trainval_infos(nusc,
                         nusc_all_centers,
                         centerline_tokens,
                         lane_or_laneconnector,
                         scene_map_api,
                         train_scenes,
                         val_scenes,
                         test=False,
                         max_sweeps=10,
                         map_size=[(-50, 50), (-50, 50)],
                         resolution=0.5,
                         vis=False):
    """Generate the train/val infos from the raw data.

    Args:
        nusc (:obj:`NuScenes`): Dataset class in the nuScenes dataset.
        train_scenes (list[str]): Basic information of training scenes.
        val_scenes (list[str]): Basic information of validation scenes.
        test (bool, optional): Whether use the test mode. In test mode, no
            annotations can be accessed. Default: False.
        max_sweeps (int, optional): Max number of sweeps. Default: 10.

    Returns:
        tuple[list[dict]]: Information of training set and validation set
            that will be saved to the info file.
    """
    train_nusc_infos = []
    val_nusc_infos = []
    frame_idx = 0
    for sample_idx, sample in enumerate(mmengine.track_iter_progress(nusc.sample)):
        lidar_token = sample['data']['LIDAR_TOP']
        sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        cs_record = nusc.get('calibrated_sensor',
                             sd_rec['calibrated_sensor_token'])
        pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
        lidar_path, boxes, _ = nusc.get_sample_data(lidar_token)
        log = nusc.get('log', nusc.get('scene', sample['scene_token'])['log_token'])
        location = log['location']
        try:
            mmengine.check_file_exist(lidar_path)
        except:
            print("lidar_loss  , ",lidar_path)


        info = {
            'lidar_path': lidar_path,
            'token': sample['token'],
            'prev': sample['prev'],
            'next': sample['next'],
            'frame_idx': frame_idx,
            'sweeps': [],
            'cams': dict(),
            'scene_token': sample['scene_token'],
            'lidar2ego_translation': cs_record['translation'],
            'lidar2ego_rotation': cs_record['rotation'],
            'ego2global_translation': pose_record['translation'],
            'ego2global_rotation': pose_record['rotation'],
            'location': location,
            'timestamp': sample['timestamp'],
            'center_lines':{'type':[], 'centerlines':[], 'centerline_ids':[], 'incoming_ids':[], 'outgoing_ids':[], 'start_point_idxs':[], 'end_point_idxs':[], 'start_end_point':[]},
        }

        if sample['next'] == '':
            frame_idx = 0
        else:
            frame_idx += 1

        l2e_r = info['lidar2ego_rotation']
        l2e_t = info['lidar2ego_translation']
        e2g_r = info['ego2global_rotation']
        e2g_t = info['ego2global_translation']
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

        for idx, center in enumerate(nusc_all_centers[location]):
            center_homo = np.hstack((copy.copy(center), np.ones((center.shape[0], 1))))
            e2g_mat = np.eye(4)
            e2g_mat[:3, :3] = e2g_r_mat
            e2g_mat[:3, -1] = e2g_t
            center_in_map_cor = (np.linalg.pinv(e2g_mat) @ center_homo.T).T[:, :3]  # global to ego (582, 3)
            in_center_x = np.logical_and(center_in_map_cor[:, 0] <= map_size[0][1], center_in_map_cor[:, 0] >= map_size[0][0])
            in_center_y = np.logical_and(center_in_map_cor[:, 1] <= map_size[1][1], center_in_map_cor[:, 1] >= map_size[1][0])
            in_center = np.logical_and(in_center_x, in_center_y)
            if_in_center = np.max(in_center)
            if if_in_center:
                info['center_lines']['type'].append(lane_or_laneconnector[location][idx])
                info['center_lines']['centerlines'].append(copy.copy(center_in_map_cor[in_center, :]))  # 593 -> 106
                info['center_lines']['centerline_ids'].append(centerline_tokens[location][idx])
                info['center_lines']['incoming_ids'].append(scene_map_api[location].get_incoming_lane_ids(info['center_lines']['centerline_ids'][-1]))
                info['center_lines']['outgoing_ids'].append(scene_map_api[location].get_outgoing_lane_ids(info['center_lines']['centerline_ids'][-1]))
                outgoing_num = len(info['center_lines']['outgoing_ids'][-1])
                incoming_num = len(info['center_lines']['incoming_ids'][-1])
                lane_record = scene_map_api[location].get_arcline_path(info['center_lines']['centerline_ids'][-1])

                if outgoing_num > 0:
                    try:
                        outgoing_lane_record = scene_map_api[location].get_arcline_path(info['center_lines']['outgoing_ids'][-1][0])
                        end_near_point = outgoing_lane_record[0]['start_pose']
                    except:
                        end_near_point = lane_record[0]['end_pose']
                else:
                    end_near_point = lane_record[0]['end_pose']

                if incoming_num > 0:
                    try:
                        incoming_lane_record = scene_map_api[location].get_arcline_path(info['center_lines']['incoming_ids'][-1][0])
                        start_near_point = incoming_lane_record[0]['end_pose']
                    except:
                        start_near_point = lane_record[0]['start_pose']

                else:
                    start_near_point = lane_record[0]['start_pose']


                start_end_near_point = np.array([start_near_point, end_near_point])  # (2, 3)
                start_end_near_homo = np.ones((2, 4))
                start_end_near_homo[:, :3] = start_end_near_point
                start_end_in_map_cor = (np.linalg.pinv(e2g_mat) @ start_end_near_homo.T).T[:, :3]

                start_point = start_end_in_map_cor[0]
                end_point = start_end_in_map_cor[1]
                if not ((start_point[0] <= map_size[0][1] and start_point[0] >= map_size[0][0]) and (
                        start_point[1] <= map_size[1][1] and start_point[1] >= map_size[1][0])):
                    start_point = info['center_lines']['centerlines'][-1][0]

                if not ((end_point[0] <= map_size[0][1] and end_point[0] >= map_size[0][0]) and (
                        end_point[1] <= map_size[1][1] and end_point[1] >= map_size[1][0])):
                    end_point = info['center_lines']['centerlines'][-1][-1]

                start_end_in_map_cor = np.stack((start_point, end_point))
                start_distance = np.linalg.norm(info['center_lines']['centerlines'][-1] - start_end_in_map_cor[0], ord=2, axis=1)
                end_distance = np.linalg.norm(info['center_lines']['centerlines'][-1] - start_end_in_map_cor[1], ord=2, axis=1)
                start_point_index = np.argmin(start_distance)
                end_point_index = np.argmin(end_distance)
                info['center_lines']['start_end_point'].append(start_end_in_map_cor)
                info['center_lines']['start_point_idxs'].append(start_point_index)
                info['center_lines']['end_point_idxs'].append(end_point_index)
                
        camera_types = [
            'CAM_FRONT',
            'CAM_FRONT_RIGHT',
            'CAM_FRONT_LEFT',
            'CAM_BACK',
            'CAM_BACK_LEFT',
            'CAM_BACK_RIGHT',
        ]
        for cam in camera_types:
            cam_token = sample['data'][cam]
            cam_path, _, cam_intrinsic = nusc.get_sample_data(cam_token)
            cam_info = obtain_sensor2top(nusc, cam_token, l2e_t, l2e_r_mat,
                                         e2g_t, e2g_r_mat, cam)
            cam_info.update(cam_intrinsic=cam_intrinsic)
            info['cams'].update({cam: cam_info})

        # obtain sweeps for a single key-frame
        sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        sweeps = []
        while len(sweeps) < max_sweeps:
            if not sd_rec['prev'] == '':
                sweep = obtain_sensor2top(nusc, sd_rec['prev'], l2e_t,
                                          l2e_r_mat, e2g_t, e2g_r_mat, 'lidar')
                sweeps.append(sweep)
                sd_rec = nusc.get('sample_data', sd_rec['prev'])
            else:
                break
        info['sweeps'] = sweeps
        # obtain annotation

        if sample['scene_token'] in train_scenes:
            train_nusc_infos.append(info)
        else:
            val_nusc_infos.append(info)
    return train_nusc_infos, val_nusc_infos


def obtain_sensor2top(nusc,
                      sensor_token,
                      l2e_t,
                      l2e_r_mat,
                      e2g_t,
                      e2g_r_mat,
                      sensor_type='lidar'):
    """Obtain the info with RT matric from general sensor to Top LiDAR.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.
        sensor_token (str): Sample data token corresponding to the
            specific sensor type.
        l2e_t (np.ndarray): Translation from lidar to ego in shape (1, 3).
        l2e_r_mat (np.ndarray): Rotation matrix from lidar to ego
            in shape (3, 3).
        e2g_t (np.ndarray): Translation from ego to global in shape (1, 3).
        e2g_r_mat (np.ndarray): Rotation matrix from ego to global
            in shape (3, 3).
        sensor_type (str, optional): Sensor to calibrate. Default: 'lidar'.

    Returns:
        sweep (dict): Sweep information after transformation.
    """
    sd_rec = nusc.get('sample_data', sensor_token)
    cs_record = nusc.get('calibrated_sensor',
                         sd_rec['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    data_path = str(nusc.get_sample_data_path(sd_rec['token']))
    if os.getcwd() in data_path:  # path from lyftdataset is absolute path
        data_path = data_path.split(f'{os.getcwd()}/')[-1]  # relative path
    sweep = {
        'data_path': data_path,
        'type': sensor_type,
        'sample_data_token': sd_rec['token'],
        'sensor2ego_translation': cs_record['translation'],
        'sensor2ego_rotation': cs_record['rotation'],
        'ego2global_translation': pose_record['translation'],
        'ego2global_rotation': pose_record['rotation'],
        'timestamp': sd_rec['timestamp']
    }
    l2e_r_s = sweep['sensor2ego_rotation']
    l2e_t_s = sweep['sensor2ego_translation']
    e2g_r_s = sweep['ego2global_rotation']
    e2g_t_s = sweep['ego2global_translation']

    # obtain the RT from sensor to Top LiDAR
    # sweep->ego->global->ego'->lidar
    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                  ) + l2e_t @ np.linalg.inv(l2e_r_mat).T
    sweep['sensor2lidar_rotation'] = R.T  # points @ R.T + T
    sweep['sensor2lidar_translation'] = T
    return sweep

