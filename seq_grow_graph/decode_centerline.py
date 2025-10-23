import torch
import mmcv
import os
import numpy as np
from math import factorial
from tqdm import tqdm
import cv2
import copy
import time
import warnings
import pdb
import argparse
import json
import bezier
import imageio
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


class EvalBzNode():
    def __init__(self, nodedict):
        self.coord = np.array(nodedict['coord'])
        self.type = [nodedict['sque_type']] if 'sque_type' in nodedict else None
        self.parents = []
        self.childs = []
        self.fork_from = None if nodedict['fork_from'] is None else nodedict['fork_from'] - 1
        self.merge_with = None if nodedict['merge_with'] is None else nodedict['merge_with'] - 1
        self.index = nodedict['sque_index'] - 1

    def __repr__(self) -> str:
        nodename = ''
        for name in self.type:
            nodename += name[0]
        return f"{nodename}_{self.index}"

    def __str__(self) -> str:
        nodename = ''
        for name in self.type:
            nodename += name[0]
        return f"{nodename}_{self.index}"
    
    
class EvalBzNodeNew():
    def __init__(self, nodedict):
        self.coord = np.array(nodedict['coord'])
        self.type = [nodedict['sque_type']] if 'sque_type' in nodedict else 'none'
        self.parents = []
        self.childs = []
  
        self.index = nodedict['sque_index'] 

    def __repr__(self) -> str:
        nodename = ''
        for name in self.type:
            nodename += name[0]
        return f"{nodename}_{self.index}"

    def __str__(self) -> str:
        nodename = ''
        for name in self.type:
            nodename += name[0]
        return f"{nodename}_{self.index}"


class EvalSuperBzNode():
    def __init__(self, nodechain, keypoints_perline=10, bezier_keys=50):
        self.nodechain = nodechain
        self.bezier_keys = bezier_keys
        self.chain_len = len(nodechain)
        self.__init_keypoints__(keypoints_perline)
        self.__init_start_end__()

    def __init_start_end__(self):
        if self.chain_len == 1:
            node, _ = self.nodechain[0]
            self.start_end = (node, node)
        else:
            node1, _ = self.nodechain[0]
            node2, _ = self.nodechain[-1]
            self.start_end = (node1, node2)

    def __init_keypoints__(self, keypoints_perline):
        keypoints = []
        diffs = []
        if self.chain_len == 1:
            node, _ = self.nodechain[0]
            keypoints.append(node.coord)
            self.keypoints = np.array(keypoints).astype(np.float32)
            diffs.append(np.array([1 / 2**0.5, 1 / 2**0.5]))
            self.diffs = np.array(diffs).astype(np.float32)
            return

        for i in range(1, self.chain_len):
            node, coeff = self.nodechain[i]
            last_node, _ = self.nodechain[i-1]

            fin_res = np.stack((last_node.coord, coeff, node.coord))
            curve = bezier.Curve(fin_res.T, degree=2)
            s_vals = np.linspace(0.0, 1.0, self.bezier_keys)
            key_idx = np.round(np.linspace(
                0, self.bezier_keys-1, keypoints_perline)).astype(np.int32)
            data_b = curve.evaluate_multi(s_vals).astype(np.float32).T
            keypoints.append(data_b[key_idx])
            diff = self.get_diff(data_b)
            diffs.append(diff[key_idx])

        self.keypoints = np.concatenate(keypoints, axis=0)
        self.diffs = np.concatenate(diffs, axis=0)

    @staticmethod
    def get_diff(data):
        def get_norm_diff(d):
            if np.linalg.norm(d) == 0.0:
                return d
            return d / np.linalg.norm(d)
        data_len = len(data)
        diffs = np.zeros(data.shape)
        for i in range(data_len):
            if i == 0:
                diff = data[i+1] - data[i]
                diffs[i] = get_norm_diff(diff)
                continue
            if i == data_len-1:
                diff = -data[i-1] + data[i]
                diffs[i] = get_norm_diff(diff)
                continue
            diff = data[i+1] - data[i-1]
            diffs[i] = get_norm_diff(diff)
        return diffs

    def __repr__(self) -> str:
        name = '|'
        for node, coeff in self.nodechain:
            name += str(node)+'->'
        name = name[:-2] + '|'
        return name


def dist_superbznode(snode1: EvalSuperBzNode, snode2: EvalSuperBzNode):
    pc1 = snode1.keypoints
    pc2 = snode2.keypoints
    diff1 = snode1.diffs
    diff2 = snode2.diffs
    dist = cdist(pc1, pc2, 'euclidean')
    diff = diff1 @ diff2.T
    diff_penalty = np.tan(-diff) + np.tan(1) + 1
    dist = dist * diff_penalty
    dist1 = np.min(dist, axis=0)
    dist2 = np.min(dist, axis=1)
    dist1 = dist1.mean(-1)
    dist2 = dist2.mean(-1)
    return (dist1 + dist2) / 2


class EvalMapBzGraph():
    def __init__(self, map_token, nodelist, bezier_keys=50, pixels_step=1, use_pixels=False,front_camera_only=False):
        self.token = map_token
        self.roots = []
        self.bezier_keys = bezier_keys
        key_pixels = [np.zeros((0, 2))]
        self.use_pixels = use_pixels
        self.front_camera_only=front_camera_only

        seqnodelen = len(nodelist)
        nodelen = nodelist[-1]['sque_index'] if seqnodelen > 0 else 0
        graph_nodelist = [None for _ in range(nodelen)]
        for i in range(seqnodelen):
            node = EvalBzNode(nodelist[i])
            if nodelist[i]['sque_type'] == 'continue':
                node.parents.append(
                    (graph_nodelist[node.index-1], nodelist[i]['coeff']))
                graph_nodelist[node.index] = node
                if graph_nodelist[node.index-1]  is not None:
                    graph_nodelist[node.index-1].childs.append(
                    (graph_nodelist[node.index], nodelist[i]['coeff']))
                if use_pixels:
                    pixels = self.init_pixels(
                        graph_nodelist[node.index-1], node, nodelist[i]['coeff'], pixels_step)
                    key_pixels.append(pixels)

            elif nodelist[i]['sque_type'] == 'merge':
                if graph_nodelist[node.index] is not None and graph_nodelist[node.index].index == node.index:
                    graph_nodelist[node.index].type.append('merge')
                    if node.merge_with < node.index and node.merge_with >= 0:
                        graph_nodelist[node.index].childs.append(
                            (graph_nodelist[node.merge_with], nodelist[i]['coeff']))
                        graph_nodelist[node.merge_with].parents.append(
                            (graph_nodelist[node.index], nodelist[i]['coeff']))
                        if use_pixels:
                            pixels = self.init_pixels(
                                graph_nodelist[node.index], graph_nodelist[node.merge_with], nodelist[i]['coeff'], pixels_step)
                            key_pixels.append(pixels)
                else:
                    if node.merge_with < node.index and node.merge_with >= 0:
                        node.childs.append(
                            (graph_nodelist[node.merge_with], nodelist[i]['coeff']))
                        graph_nodelist[node.merge_with].parents.append(
                            (node, nodelist[i]['coeff']))
                        if use_pixels:
                            pixels = self.init_pixels(
                                node, graph_nodelist[node.merge_with], nodelist[i]['coeff'], pixels_step)
                            key_pixels.append(pixels)
                    graph_nodelist[node.index] = node
            elif nodelist[i]['sque_type'] == 'fork':
                if graph_nodelist[node.index] is not None and graph_nodelist[node.index].index == node.index:
                    graph_nodelist[node.index].type.append('fork')
                    if node.fork_from < node.index and node.fork_from >= 0:
                        graph_nodelist[node.index].parents.append(
                            (graph_nodelist[node.fork_from], nodelist[i]['coeff']))
                        graph_nodelist[node.fork_from].childs.append(
                            (graph_nodelist[node.index], nodelist[i]['coeff']))
                        if use_pixels:
                            pixels = self.init_pixels(
                                graph_nodelist[node.fork_from], graph_nodelist[node.index], nodelist[i]['coeff'], pixels_step)
                            key_pixels.append(pixels)
                else:
                    if node.fork_from < node.index and node.fork_from >= 0:
                        node.parents.append(
                            (graph_nodelist[node.fork_from], nodelist[i]['coeff']))
                        graph_nodelist[node.fork_from].childs.append(
                            (node, nodelist[i]['coeff']))
                        if use_pixels:
                            pixels = self.init_pixels(
                                graph_nodelist[node.fork_from], node, nodelist[i]['coeff'], pixels_step)
                            key_pixels.append(pixels)
                    graph_nodelist[node.index] = node
            elif nodelist[i]['sque_type'] == 'start':
                graph_nodelist[node.index] = node
                self.roots.append(graph_nodelist[node.index])
        # for i in range(nodelen):
        #     parentslen = len(graph_nodelist[i].parents)
        #     for j in range(parentslen):
        #         graph_nodelist[i].parents[j].childs.append(graph_nodelist[i])
        self.key_pixels = np.concatenate(key_pixels, axis=0)
        self.graph_nodelist = graph_nodelist

    def init_pixels(self, node1, node2, coeff, pixels_step):
        fin_res = np.stack((node1.coord, coeff, node2.coord))
        curve = bezier.Curve(fin_res.T, degree=2)
        s_vals = np.linspace(0.0, 1.0, self.bezier_keys)
        # key_idx = np.round(np.linspace(0, self.bezier_keys-1, keypoints_perline)).astype(np.int32)
        data_b = curve.evaluate_multi(s_vals).astype(np.float32).T
        curve_dist = np.diff(data_b, axis=0)
        curve_dist = np.sum(curve_dist**2, axis=1) ** 0.5
        curve_len = np.sum(curve_dist)
        curve_dist = np.cumsum(curve_dist)
        if curve_len < pixels_step:
            return np.stack((node1.coord, node2.coord))
        curve_dist_std = np.arange(0, curve_len, pixels_step)
        curve_dist_cost = cdist(curve_dist_std[:, None], curve_dist[:, None])
        _, curve_dist_idx = linear_sum_assignment(curve_dist_cost)
        pixels = data_b[curve_dist_idx]
        return pixels



        
    
    
    def visualization(self, nx, path, aux_name, name, scale=5, res=None, bitmap=None):
    
        image = np.zeros([int(nx[1]) * scale, int(nx[0]) * scale, 3])
        # image = np.zeros([int(nx[1]) * scale, int(nx[0]) * scale, 3])
        point_color_map = {"start": (0, 0, 125), 'fork': (
            0, 255, 0), "continue": (0, 255, 255), "merge": (255, 0, 0)}
        for idx, node in enumerate(self.graph_nodelist):
            if 'start' in node.type:
                cv2.circle(image, node.coord * scale, int(scale**1.5),
                           color=point_color_map['start'], thickness=-1)
            else:
                if len(node.childs) > 0:
                    cv2.circle(image, node.coord * scale, int(scale **
                               1.7), color=(125, 0, 0), thickness=-1)
                else:
                    cv2.circle(image, node.coord * scale, int(scale **
                               1.7), color=(0, 125, 0), thickness=-1)

            cv2.putText(image, "%.2d" % node.index, node.coord * scale + np.array([-10, 10]), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 204, 0), 2, cv2.LINE_AA)

            for cnode, coeff in node.childs:
                fin_res = np.stack(
                    (node.coord * scale, coeff * scale, cnode.coord * scale))
                curve = bezier.Curve(fin_res.T, degree=2)
                s_vals = np.linspace(0.0, 1.0, 50)
                data_b = curve.evaluate_multi(s_vals).T
                data_b = data_b.astype(int)
                cv2.polylines(image, [data_b], False,
                              color=(0, 161, 244), thickness=2)
                arrowline = data_b[24:26, :].copy()
                diff = arrowline[1] - arrowline[0]
                if np.prod(arrowline[1] == arrowline[0]):
                    continue
                diff = diff / np.linalg.norm(diff) * 3
                arrowline[1] = arrowline[0] + diff
                cv2.arrowedLine(image, arrowline[0], arrowline[1],
                                color=(49, 78, 255), thickness=2, tipLength=5)
        if self.use_pixels:
            for pti, pt in enumerate(self.key_pixels):
                cv2.circle(image, pt.astype(int) * scale, 2,
                           color=(0, 255, 0), thickness=-1)
        if res is not None:
            cv2.putText(image, str(res),  np.array([0, int(nx[1]) * scale-10]), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2, cv2.LINE_AA)
        save_dir = os.path.join("vis",path)
        os.makedirs(save_dir, exist_ok=True)

        cv2.imwrite(os.path.join(
            save_dir, f"{name}_{self.token}_{aux_name}.png"), image)

    def better_visualization(self, nx, path, aux_name, name, scale=10, bitmap=None, puttext=True, pixels_step=40):
        if bitmap is not None:
            image = bitmap
            image = cv2.resize(image, (int(nx[0]) * scale, int(nx[1]) * scale))
            image = np.repeat(image[:, :, None], 3, axis=-1)
        else:
            image = np.zeros([int(nx[1]) * scale, int(nx[0]) * scale, 3])
        point_color_map = {"start": (185, 107, 146), 'fork': (
            200, 204, 144), "continue": (59, 92, 255), "end": (100, 188, 171)}

        for idx, node in enumerate(self.graph_nodelist):
            coord = node.coord * scale
            coord_tp = (coord[0], coord[1])
            if len(node.parents) == 0:
                cv2.circle(image, coord_tp, int(scale**1.4),
                           color=point_color_map['start'], thickness=-1)
                cv2.circle(image, coord_tp, int(scale**1.4),
                           color=(0, 0, 0), thickness=3)
            else:
                if len(node.childs) == 0:
                    cv2.circle(image, coord_tp, int(scale**1.4),
                               color=point_color_map['end'], thickness=-1)
                    cv2.circle(image, coord_tp, int(scale**1.4),
                               color=(0, 0, 0), thickness=3)
                else:
                    if len(node.childs) == 1 and len(node.parents) == 1:
                        cv2.circle(image, coord_tp, int(scale**1.4),
                                   color=point_color_map['continue'], thickness=-1)
                        cv2.circle(image, coord_tp, int(scale**1.4),
                                   color=(0, 0, 0), thickness=3)
                    else:
                        cv2.circle(image, coord_tp, int(scale**1.4),
                                   color=point_color_map['fork'], thickness=-1)
                        cv2.circle(image, coord_tp, int(scale**1.4),
                                   color=(0, 0, 0), thickness=3)

        for idx, node in enumerate(self.graph_nodelist):
            # if 'start' in node.type:
            #     cv2.circle(image, node.coord * scale, int(scale**1.4), color=point_color_map['start'], thickness=-1)
            # else:
            #     if len(node.childs) > 0:
            #         cv2.circle(image, node.coord * scale, int(scale**1.4), color=(125, 0, 0), thickness=-1)
            #     else:
            #         cv2.circle(image, node.coord * scale, int(scale**1.4), color=(0, 125, 0), thickness=-1)
            for cnode, coeff in node.childs:
                if len(node.childs) > 1 or len(cnode.parents) > 1:
                    color = point_color_map['fork']
                else:
                    color = point_color_map['continue']
                fin_res = np.stack(
                    (node.coord * scale, coeff * scale, cnode.coord * scale))
                curve = bezier.Curve(fin_res.T, degree=2)
                s_vals = np.linspace(0.0, 1.0, 50)
                data_b = curve.evaluate_multi(s_vals).T
                data_b = data_b.astype(int)
                cv2.polylines(image, [data_b], False, color=color, thickness=5)

                curve_dist = np.diff(data_b, axis=0)
                curve_dist = np.sum(curve_dist**2, axis=1) ** 0.5
                curve_len = np.sum(curve_dist)
                curve_dist = np.cumsum(curve_dist)
                if curve_len > pixels_step:
                    curve_dist_std = np.arange(0, curve_len, pixels_step)
                    curve_dist_cost = cdist(
                        curve_dist_std[:, None], curve_dist[:, None])
                    _, curve_dist_idx = linear_sum_assignment(curve_dist_cost)
                    curve_dist_idx = curve_dist_idx[curve_dist_idx > 1]
                    for curve_idx in curve_dist_idx:
                        arrowline = data_b[curve_idx-1:curve_idx+1, :].copy()
                        diff = arrowline[1] - arrowline[0]
                        if np.prod(arrowline[1] == arrowline[0]):
                            continue
                        diff = diff / np.linalg.norm(diff) * 3
                        arrowline[1] = arrowline[0] + diff
                        cv2.arrowedLine(image, arrowline[0], arrowline[1],
                                        color=color, thickness=5, tipLength=5)
        if puttext:
            for idx, node in enumerate(self.graph_nodelist):
                coord = node.coord * scale
                text_shape = np.array([45, 20]) * 0.7
                text_shape = text_shape.astype(int)
                if coord[0] > int(nx[0]) * scale - text_shape[0]:
                    coord[0] = int(nx[0]) * scale - text_shape[0]
                else:
                    if coord[0] < text_shape[0]//2:
                        coord[0] = 0
                    else:
                        coord[0] = coord[0] - text_shape[0]//2
                if coord[1] < text_shape[1]:
                    coord[1] = text_shape[1]
                else:
                    if coord[1] < int(nx[0]) * scale - text_shape[1]//2:
                        coord[1] = coord[1] + text_shape[1]//2
                cv2.putText(image, "%.2d" % node.index, coord, cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 0), 3, cv2.LINE_AA)

        save_dir = f"vis/{path}/"
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
    

        cv2.imwrite(os.path.join(
            save_dir, f"{name}_{self.token}_{aux_name}.png"), image)

    def better_videolization(self, nx, path, aux_name, name, scale=10, bitmap=None, puttext=True, pixels_step=40):
        save_dir = f"vis/gif_{path}/"
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)

        img_dir = os.path.join(save_dir, f"{name}_{self.token}_{aux_name}")
        if not os.path.exists(img_dir):
            os.mkdir(img_dir)

        frames = []

        if bitmap is not None:
            image = bitmap
            image = cv2.resize(image, (int(nx[0]) * scale, int(nx[1]) * scale))
            image = np.repeat(image[:, :, None], 3, axis=-1)
        else:
            image = np.zeros([int(nx[1]) * scale, int(nx[0]) * scale, 3])
        point_color_map = {"start": (185, 107, 146), 'fork': (
            200, 204, 144), "continue": (59, 92, 255), "end": (100, 188, 171)}
        frame_cnt = 0
        used_nodes = {}
        for idx, node in enumerate(self.graph_nodelist):
            used_nodes[node.index] = None
            if len(node.parents) == 0:
                cv2.circle(image, node.coord * scale, int(scale**1.4),
                           color=point_color_map['start'], thickness=-1)
                cv2.circle(image, node.coord * scale,
                           int(scale**1.4), color=(0, 0, 0), thickness=3)
                # puttext
                coord = node.coord * scale
                text_shape = np.array([45, 20]) * 0.7
                text_shape = text_shape.astype(int)
                if coord[0] > int(nx[0]) * scale - text_shape[0]:
                    coord[0] = int(nx[0]) * scale - text_shape[0]
                else:
                    if coord[0] < text_shape[0]//2:
                        coord[0] = 0
                    else:
                        coord[0] = coord[0] - text_shape[0]//2
                if coord[1] < text_shape[1]:
                    coord[1] = text_shape[1]
                else:
                    if coord[1] < int(nx[0]) * scale - text_shape[1]//2:
                        coord[1] = coord[1] + text_shape[1]//2
                cv2.putText(image, "%.2d" % node.index, coord, cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 0), 3, cv2.LINE_AA)
                # puttext end
                cv2.imwrite(os.path.join(img_dir, f"{frame_cnt}.png"), image)
                frames.append(cv2.cvtColor(cv2.imread(os.path.join(
                    img_dir, f"{frame_cnt}.png")), cv2.COLOR_BGR2RGB))
                frame_cnt += 1
            else:
                if len(node.childs) == 0:
                    cv2.circle(image, node.coord * scale, int(scale**1.4),
                               color=point_color_map['end'], thickness=-1)
                    cv2.circle(image, node.coord * scale,
                               int(scale**1.4), color=(0, 0, 0), thickness=3)
                else:
                    if len(node.childs) == 1 and len(node.parents) == 1:
                        cv2.circle(image, node.coord * scale, int(scale**1.4),
                                   color=point_color_map['continue'], thickness=-1)
                        cv2.circle(image, node.coord * scale,
                                   int(scale**1.4), color=(0, 0, 0), thickness=3)

                    else:
                        cv2.circle(image, node.coord * scale, int(scale**1.4),
                                   color=point_color_map['fork'], thickness=-1)
                        cv2.circle(image, node.coord * scale,
                                   int(scale**1.4), color=(0, 0, 0), thickness=3)
            for cnode, coeff in node.childs:
                if cnode.index not in used_nodes:
                    continue
                if len(node.childs) > 1 or len(cnode.parents) > 1:
                    lanecolor = point_color_map['fork']
                else:
                    lanecolor = point_color_map['continue']
                fin_res = np.stack(
                    (node.coord * scale, coeff * scale, cnode.coord * scale))
                curve = bezier.Curve(fin_res.T, degree=2)
                s_vals = np.linspace(0.0, 1.0, 50)
                data_b = curve.evaluate_multi(s_vals).T
                data_b = data_b.astype(int)
                cv2.polylines(image, [data_b], False,
                              color=lanecolor, thickness=5)

                curve_dist = np.diff(data_b, axis=0)
                curve_dist = np.sum(curve_dist**2, axis=1) ** 0.5
                curve_len = np.sum(curve_dist)
                curve_dist = np.cumsum(curve_dist)
                if curve_len > pixels_step:
                    curve_dist_std = np.arange(0, curve_len, pixels_step)
                    curve_dist_cost = cdist(
                        curve_dist_std[:, None], curve_dist[:, None])
                    _, curve_dist_idx = linear_sum_assignment(curve_dist_cost)
                    curve_dist_idx = curve_dist_idx[curve_dist_idx > 1]
                    for curve_idx in curve_dist_idx:
                        arrowline = data_b[curve_idx-1:curve_idx+1, :].copy()
                        diff = arrowline[1] - arrowline[0]
                        if np.prod(arrowline[1] == arrowline[0]):
                            continue
                        diff = diff / np.linalg.norm(diff) * 3
                        arrowline[1] = arrowline[0] + diff
                        cv2.arrowedLine(image, arrowline[0], arrowline[1],
                                        color=lanecolor, thickness=5, tipLength=5)
                # puttext
                coord = node.coord * scale
                text_shape = np.array([45, 20]) * 0.7
                text_shape = text_shape.astype(int)
                if coord[0] > int(nx[0]) * scale - text_shape[0]:
                    coord[0] = int(nx[0]) * scale - text_shape[0]
                else:
                    if coord[0] < text_shape[0]//2:
                        coord[0] = 0
                    else:
                        coord[0] = coord[0] - text_shape[0]//2
                if coord[1] < text_shape[1]:
                    coord[1] = text_shape[1]
                else:
                    if coord[1] < int(nx[0]) * scale - text_shape[1]//2:
                        coord[1] = coord[1] + text_shape[1]//2
                cv2.putText(image, "%.2d" % node.index, coord, cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 0), 3, cv2.LINE_AA)
                # puttext end
                # puttext
                coord = cnode.coord * scale
                text_shape = np.array([45, 20]) * 0.7
                text_shape = text_shape.astype(int)
                if coord[0] > int(nx[0]) * scale - text_shape[0]:
                    coord[0] = int(nx[0]) * scale - text_shape[0]
                else:
                    if coord[0] < text_shape[0]//2:
                        coord[0] = 0
                    else:
                        coord[0] = coord[0] - text_shape[0]//2
                if coord[1] < text_shape[1]:
                    coord[1] = text_shape[1]
                else:
                    if coord[1] < int(nx[0]) * scale - text_shape[1]//2:
                        coord[1] = coord[1] + text_shape[1]//2
                cv2.putText(image, "%.2d" % cnode.index, coord, cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 0), 3, cv2.LINE_AA)
                # puttext end
                cv2.imwrite(os.path.join(img_dir, f"{frame_cnt}.png"), image)
                frames.append(cv2.cvtColor(cv2.imread(os.path.join(
                    img_dir, f"{frame_cnt}.png")), cv2.COLOR_BGR2RGB))
                frame_cnt += 1
            for pnode, coeff in node.parents:
                if pnode.index not in used_nodes:
                    continue
                if len(pnode.childs) > 1 or len(node.parents) > 1:
                    lanecolor = point_color_map['fork']
                else:
                    lanecolor = point_color_map['continue']
                fin_res = np.stack(
                    (pnode.coord * scale, coeff * scale, node.coord * scale))
                curve = bezier.Curve(fin_res.T, degree=2)
                s_vals = np.linspace(0.0, 1.0, 50)
                data_b = curve.evaluate_multi(s_vals).T
                data_b = data_b.astype(int)
                cv2.polylines(image, [data_b], False,
                              color=lanecolor, thickness=5)

                curve_dist = np.diff(data_b, axis=0)
                curve_dist = np.sum(curve_dist**2, axis=1) ** 0.5
                curve_len = np.sum(curve_dist)
                curve_dist = np.cumsum(curve_dist)
                if curve_len > pixels_step:
                    curve_dist_std = np.arange(0, curve_len, pixels_step)
                    curve_dist_cost = cdist(
                        curve_dist_std[:, None], curve_dist[:, None])
                    _, curve_dist_idx = linear_sum_assignment(curve_dist_cost)
                    curve_dist_idx = curve_dist_idx[curve_dist_idx > 1]
                    for curve_idx in curve_dist_idx:
                        arrowline = data_b[curve_idx-1:curve_idx+1, :].copy()
                        diff = arrowline[1] - arrowline[0]
                        if np.prod(arrowline[1] == arrowline[0]):
                            continue
                        diff = diff / np.linalg.norm(diff) * 3
                        arrowline[1] = arrowline[0] + diff
                        cv2.arrowedLine(image, arrowline[0], arrowline[1],
                                        color=lanecolor, thickness=5, tipLength=5)
                # puttext
                coord = node.coord * scale
                text_shape = np.array([45, 20]) * 0.7
                text_shape = text_shape.astype(int)
                if coord[0] > int(nx[0]) * scale - text_shape[0]:
                    coord[0] = int(nx[0]) * scale - text_shape[0]
                else:
                    if coord[0] < text_shape[0]//2:
                        coord[0] = 0
                    else:
                        coord[0] = coord[0] - text_shape[0]//2
                if coord[1] < text_shape[1]:
                    coord[1] = text_shape[1]
                else:
                    if coord[1] < int(nx[0]) * scale - text_shape[1]//2:
                        coord[1] = coord[1] + text_shape[1]//2
                cv2.putText(image, "%.2d" % node.index, coord, cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 0), 3, cv2.LINE_AA)
                # puttext end
                # puttext
                coord = pnode.coord * scale
                text_shape = np.array([45, 20]) * 0.7
                text_shape = text_shape.astype(int)
                if coord[0] > int(nx[0]) * scale - text_shape[0]:
                    coord[0] = int(nx[0]) * scale - text_shape[0]
                else:
                    if coord[0] < text_shape[0]//2:
                        coord[0] = 0
                    else:
                        coord[0] = coord[0] - text_shape[0]//2
                if coord[1] < text_shape[1]:
                    coord[1] = text_shape[1]
                else:
                    if coord[1] < int(nx[0]) * scale - text_shape[1]//2:
                        coord[1] = coord[1] + text_shape[1]//2
                cv2.putText(image, "%.2d" % pnode.index, coord, cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (0, 0, 0), 3, cv2.LINE_AA)
                # puttext end
                cv2.imwrite(os.path.join(img_dir, f"{frame_cnt}.png"), image)
                frames.append(cv2.cvtColor(cv2.imread(os.path.join(
                    img_dir, f"{frame_cnt}.png")), cv2.COLOR_BGR2RGB))
                frame_cnt += 1
        imageio.mimsave(os.path.join(
            img_dir, f"{name}_{self.token}_{aux_name}.gif"), frames, fps=3)

    @staticmethod
    def ptwise_bfs(query_node, max_node):
        queue = [[(query_node, None)]]
        res = []
        while len(queue) > 0:
            nodechain = queue.pop(0)
            if len(nodechain) == max_node:
                res.append(EvalSuperBzNode(nodechain))
                continue
            node, _ = nodechain[-1]
            if len(node.childs) == 0:
                # res.append(SuperNode(nodechain))
                continue
            for cnode, coeff in node.childs:
                queue.append(nodechain + [(cnode, coeff)])
        return res

    def get_nodechains_dpt(self, max_node):
        res = []
        if max_node < 1:
            return res
        if max_node == 1:
            for node in self.graph_nodelist:
                res += self.ptwise_bfs(node, max_node)
            return res
        for node_num in range(2, max_node+1):
            for node in self.graph_nodelist:
                res += self.ptwise_bfs(node, node_num)
        return res


class EvalSeq2Graph(EvalMapBzGraph):
    def __init__(self, map_token, seq,pc_range, dx, bz_pc_range, bz_dx, bezier_keys=50, pixels_step=1, use_pixels=False,front_camera_only=False):
        self.token = map_token
        self.roots = []
        self.bezier_keys = bezier_keys
        key_pixels = [np.zeros((0, 2))]
        self.use_pixels = use_pixels
        self.front_camera_only=front_camera_only
        self.split_connect=571
        self.split_node=572
        
        seq=list(seq)
        split_node_idxs=[i for i,s in enumerate(seq) if s==self.split_node]
        graph_nodelist=[None for i in range(len(split_node_idxs))]
        split_node_idxs=[-1]+split_node_idxs
        

        for i in range(len(split_node_idxs)-1):
            node_seq=seq[split_node_idxs[i]+1:split_node_idxs[i+1]]
            if len(node_seq)<3:
                continue
            node=EvalBzNodeNew({
                'coord': [node_seq[0],node_seq[1]],
                'sque_index': node_seq[2]
            })
            if node.index<0 or node.index>=len(graph_nodelist):
                continue
            
            if self.split_connect in node_seq:
                split_connect_idx=node_seq.index(self.split_connect)
            else:
                split_connect_idx=len(node_seq) #没有分割符就默认是father的关系
            if split_connect_idx>3:
                stop_idx=(split_connect_idx-3)//3*3
                father_seqs=node_seq[3:3+stop_idx]
                father_seqs=np.array(father_seqs).reshape(-1,3)
                for father_seq in father_seqs:
                    father_idx=father_seq[0]
                    if father_idx>=0 and father_idx<len(graph_nodelist) and graph_nodelist[father_idx] is not None:
                        
                        coeff=father_seq[1:3]
                        coeff = coeff * bz_dx[:2] + bz_pc_range[:2]
                        coeff = ((coeff - pc_range[:2]) / dx[:2]).astype(int)
                        node.parents.append((graph_nodelist[father_idx],coeff))
                        graph_nodelist[father_idx].childs.append((node,coeff))
            
            if len(node_seq)>(split_connect_idx+1):
                stop_idx=(len(node_seq)-(split_connect_idx+1))//3*3
                child_seqs=node_seq[split_connect_idx+1:split_connect_idx+1+stop_idx]
                child_seqs=np.array(child_seqs).reshape(-1,3)
                for child_seq in child_seqs:
                    child_idx=child_seq[0]
                    if child_idx>=0 and child_idx <len(graph_nodelist) and graph_nodelist[child_idx] is not None:
                    
                        coeff=child_seq[1:3]
                        coeff = coeff * bz_dx[:2] + bz_pc_range[:2]
                        coeff = ((coeff - pc_range[:2]) / dx[:2]).astype(int)
                        node.childs.append((graph_nodelist[child_idx],coeff))
                        graph_nodelist[child_idx].parents.append((node,coeff))
            graph_nodelist [node.index]=node
            
            
        self.graph_nodelist = [node for node in graph_nodelist if node is not None]
        

    def visualization(self, nx, path, aux_name, name, scale=5, res=None, bitmap=None):
        if bitmap is not None:
            image = bitmap
            image = cv2.resize(image, (int(nx[0]) * scale, int(nx[1]) * scale))
            image = np.repeat(image[:, :, None], 3, axis=-1)
        else:
            image = np.zeros([int(nx[1]) * scale, int(nx[0]) * scale, 3])
        point_color_map = {"start": (0, 0, 125), 'fork': (
            0, 255, 0), "continue": (0, 255, 255), "merge": (255, 0, 0)}
        for idx, node in enumerate(self.graph_nodelist):
            if node is None:
                continue
            if len(node.parents) == 0 :
                cv2.circle(image, node.coord * scale, int(scale**1.5),
                           color=point_color_map['start'], thickness=-1)
            else:
                if len(node.childs) > 0:
                    cv2.circle(image, node.coord * scale, int(scale **
                               1.7), color=(125, 0, 0), thickness=-1)
                else:
                    cv2.circle(image, node.coord * scale, int(scale **
                               1.7), color=(0, 125, 0), thickness=-1)

            cv2.putText(image, "%.2d" % node.index, node.coord * scale + np.array([-10, 10]), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (255, 204, 0), 2, cv2.LINE_AA)

            for cnode, coeff in node.childs:
                fin_res = np.stack(
                    (node.coord * scale, coeff * scale, cnode.coord * scale))
                curve = bezier.Curve(fin_res.T, degree=2)
                s_vals = np.linspace(0.0, 1.0, 50)
                data_b = curve.evaluate_multi(s_vals).T
                data_b = data_b.astype(int)
                cv2.polylines(image, [data_b], False,
                              color=(0, 161, 244), thickness=2)
                arrowline = data_b[24:26, :].copy()
                diff = arrowline[1] - arrowline[0]
                if np.prod(arrowline[1] == arrowline[0]):
                    continue
                diff = diff / np.linalg.norm(diff) * 3
                arrowline[1] = arrowline[0] + diff
                cv2.arrowedLine(image, arrowline[0], arrowline[1],
                                color=(49, 78, 255), thickness=2, tipLength=5)
        if self.use_pixels:
            for pti, pt in enumerate(self.key_pixels):
                cv2.circle(image, pt.astype(int) * scale, 2,
                           color=(0, 255, 0), thickness=-1)
        if res is not None:
            cv2.putText(image, str(res),  np.array([0, int(nx[1]) * scale-10]), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2, cv2.LINE_AA)
        save_dir = os.path.join("vis",path)
        os.makedirs(save_dir, exist_ok=True)

        cv2.imwrite(os.path.join(
            save_dir, f"{name}_{self.token}_{aux_name}.png"), image)     
 

class EvalMapBzGraphAAAI(EvalSeq2Graph):
# class EvalMapBzGraphAAAI(EvalMapBzGraph):
    def __init__(self, map_token, seq,pc_range, dx, bz_pc_range, bz_dx, bezier_keys=50, pixels_step=1, use_pixels=False,front_camera_only=False):
        self.token = map_token
        self.roots = []
        self.bezier_keys = bezier_keys
        key_pixels = [np.zeros((0, 2))]
        self.use_pixels = use_pixels
        self.front_camera_only=front_camera_only
        self.end_v = 573
        self.vertex_id_start=200
        self.connect_start=250
        self.coeff_start=300


    

        seq=list(seq)
        if self.end_v in seq:
            split=seq.index(self.end_v)
        else:
            split=len(seq)
        split=split//2*2
        vert_seqs=np.array(seq[:split]).reshape(-1,2)
        self.graph_nodelist=[
            EvalBzNodeNew({
                'coord': [pos[0],pos[1]],
                'sque_index': i
            })
            for i,pos in enumerate(vert_seqs)]
        
        edge_seqs=seq[split+1:]
        split_node_idxs=[i for i,s in enumerate(edge_seqs) if s==self.connect_start]
        if len(split_node_idxs)!=0:
            
            split_node_idxs=[-1]+split_node_idxs

            for i,node in enumerate(self.graph_nodelist):
                if  i+1<len(split_node_idxs) and split_node_idxs[i]+1<len(edge_seqs):
                    edge_seq=edge_seqs[split_node_idxs[i]+1:split_node_idxs[i+1]]
                    edge_seq=edge_seq[:len(edge_seq)//3*3]
                    if len(edge_seq)!=0:
                        
                        edge_seq=np.array(edge_seq).reshape(-1,3)
                        for edge in edge_seq:
                            child_idx=edge[0]
                            if child_idx<len(self.graph_nodelist) and child_idx>=0:
                                coeff=edge[1:3]-self.coeff_start
                                coeff = coeff * bz_dx[:2] + bz_pc_range[:2]
                                coeff = ((coeff - pc_range[:2]) / dx[:2]).astype(int)
                                node.childs.append((self.graph_nodelist[child_idx],coeff))
                                self.graph_nodelist[child_idx].parents.append((node,coeff))
        
                    
                    

class EvalGraphDptDist():
    def __init__(self, dists) -> None:
        self.dists = dists
        self.max_dpt = len(dists)

    def __str__(self) -> str:
        name = ''
        for i in range(len(self.dists)):
            name += "dpt %d: %.3f " % (i, self.dists[i])
        return name

    def __repr__(self) -> str:
        name = ''
        for i in range(self.dists):
            name += "dpt %d: %.3f" % self.dists[i]
        return name

    def __add__(self, dist2):
        if self.max_dpt != dist2.max_dpt:
            return
        add_dists = [0 for _ in range(self.max_dpt)]
        for i in range(self.max_dpt):
            if np.isnan(dist2.dists[i]):
                continue
            add_dists[i] = self.dists[i] + dist2.dists[i]
        return EvalGraphDptDist(add_dists)

    def __truediv__(self, factor):
        div_dists = [0 for _ in range(self.max_dpt)]
        for i in range(self.max_dpt):
            div_dists[i] = self.dists[i] / factor
        return EvalGraphDptDist(div_dists)

    def __iadd__(self, dist2):
        if self.max_dpt != dist2.max_dpt:
            return
        for i in range(self.max_dpt):
            if np.isnan(dist2.dists[i]):
                continue
            self.dists[i] = self.dists[i] + dist2.dists[i]
        return self

    def __itruediv__(self, factor):
        for i in range(self.max_dpt):
            self.dists[i] = self.dists[i] / factor
        return self


def seq2bznodelist(seq, n_control):
    """"n control = 3"""
    length = 4 + 2*(n_control-2)
    seq = np.array(seq).reshape(-1, length)
    node_list = []
    # type_idx_map = {'start': 0, 'continue': 1, 'fork': 2, 'merge': 3}
    idx_type_map = {0: 'start', 1: 'continue', 2: "fork", 3: 'merge'}
    idx = 0
    epsilon = 2
    for i in range(len(seq)):
        node = {'sque_index': None,
                'sque_type': None,
                'fork_from': None,
                'merge_with': None,
                'coord': None,
                'coeff': [],
                }
        label = seq[i][2]
        if label > 3 or label < 0:
            label = 1

        node['coord'] = [seq[i][0], seq[i][1]]
        if label == 3:  # merge
            node['sque_type'] = idx_type_map[label]
            node['sque_index'] = idx
            node['merge_with'] = seq[i][3]
            node['coeff'] = np.array([seq[i][j] for j in range(4, length)])

        elif label == 2:  # fork
            node['sque_type'] = idx_type_map[label]
            node['fork_from'] = seq[i][3]
            node['coeff'] = np.array([seq[i][j] for j in range(4, length)])

            last_coord = np.array([seq[i - 1][0], seq[i - 1][1]])
            coord = np.array([seq[i][0], seq[i][1]])
            tmp = sum((coord - last_coord) ** 2)
            if tmp < epsilon:  # split fork
                node['sque_index'] = idx
            else:
                idx = idx + 1
                node['sque_index'] = idx
        elif label == 1:  # continue
            node['sque_type'] = idx_type_map[label]
            node['coeff'] = np.array([seq[i][j] for j in range(4, length)])
            idx = idx + 1
            node['sque_index'] = idx

        else:
            node['sque_type'] = idx_type_map[label]
            idx = idx + 1
            node['sque_index'] = idx

        node_list.append(node)

    return node_list


def seq2plbznodelist(seq, coeffs):
    """"n control = 3"""
    length = 4
    seq = np.array(seq).reshape(-1, length)
    node_list = []
    # type_idx_map = {'start': 0, 'continue': 1, 'fork': 2, 'merge': 3}
    idx_type_map = {0: 'start', 1: 'continue', 2: "fork", 3: 'merge'}
    idx = 0
    epsilon = 2
    coeff_idx = 0
    for i in range(len(seq)):
        node = {'sque_index': None,
                'sque_type': None,
                'fork_from': None,
                'merge_with': None,
                'coord': None,
                'coeff': [],
                }
        label = seq[i][2]
        if idx == 0:
            label = 0
        if label > 3 or label < 0:
            label = 1

        node['coord'] = [seq[i][0], seq[i][1]]
        if label == 3:  # merge
            node['sque_type'] = idx_type_map[label]
            node['sque_index'] = idx
            node['merge_with'] = seq[i][3]
            node['coeff'] = coeffs[coeff_idx]
            coeff_idx += 1

        elif label == 2:  # fork
            node['sque_type'] = idx_type_map[label]
            node['fork_from'] = seq[i][3]
            node['coeff'] = coeffs[coeff_idx]
            coeff_idx += 1

            last_coord = np.array([seq[i - 1][0], seq[i - 1][1]])
            coord = np.array([seq[i][0], seq[i][1]])
            tmp = sum((coord - last_coord) ** 2)
            if tmp < epsilon:  # split fork
                node['sque_index'] = idx
            else:
                idx = idx + 1
                node['sque_index'] = idx
        elif label == 1:  # continue
            node['sque_type'] = idx_type_map[label]
            node['coeff'] = coeffs[coeff_idx]
            coeff_idx += 1
            idx = idx + 1
            node['sque_index'] = idx

        else:
            node['sque_type'] = idx_type_map[label]
            node['coeff'] = coeffs[coeff_idx]
            coeff_idx += 1
            idx = idx + 1
            node['sque_index'] = idx

        node_list.append(node)

    return node_list
