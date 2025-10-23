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
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment
from pyquaternion import Quaternion
from typing import List
import numbers


class EvalNode():
    def __init__(self, nodedict):
        self.coord = np.array(nodedict['coord'])
        self.type = [nodedict['sque_type']]
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


class EvalSuperNode():
    def __init__(self, nodechain, keypoints_perline=5):
        self.nodechain = nodechain
        self.chain_len = len(nodechain)
        self.__init_keypoints__(keypoints_perline)

    def __init_keypoints__(self, keypoints_perline):
        keypoints = []
        if self.chain_len == 1:
            keypoints.append(self.nodechain[0].coord)
            self.keypoints = np.array(keypoints).astype(np.float32)
            return

        for i in range(self.chain_len-1):
            node = self.nodechain[i]
            next_node = self.nodechain[i+1]

            diff = next_node.coord - node.coord
            x_linspace = np.linspace(
                0, diff[0], keypoints_perline+1)[:-1] + node.coord[0]
            y_linspace = np.linspace(
                0, diff[1], keypoints_perline+1)[:-1] + node.coord[1]
            keypoints.append(np.stack([x_linspace, y_linspace], axis=0).transpose(
                1, 0).astype(np.float32))
        self.keypoints = np.concatenate(keypoints, axis=0)

    def __repr__(self) -> str:
        name = '|'
        for node in self.nodechain:
            name += str(node)+'->'
        name = name[:-2] + '|'
        return name


class EvalMapGraph():
    def __init__(self, map_token, nodelist):
        self.token = map_token
        self.roots = []

        seqnodelen = len(nodelist)
        nodelen = nodelist[-1]['sque_index'] if seqnodelen > 0 else 0
        graph_nodelist = [None for _ in range(nodelen)]
        for i in range(seqnodelen):
            node = EvalNode(nodelist[i])
            if nodelist[i]['sque_type'] == 'continue':
                node.parents.append(graph_nodelist[node.index-1])
                graph_nodelist[node.index] = node
            elif nodelist[i]['sque_type'] == 'merge':
                if graph_nodelist[node.index] is not None and graph_nodelist[node.index].index == node.index:
                    graph_nodelist[node.index].type.append('merge')
                    if node.merge_with < node.index and node.merge_with >= 0:
                        graph_nodelist[node.index].childs.append(
                            graph_nodelist[node.merge_with])
                else:
                    if node.merge_with < node.index and node.merge_with >= 0:
                        node.childs.append(graph_nodelist[node.merge_with])
                    graph_nodelist[node.index] = node
            elif nodelist[i]['sque_type'] == 'fork':
                if graph_nodelist[node.index] is not None and graph_nodelist[node.index].index == node.index:
                    graph_nodelist[node.index].type.append('fork')
                    if node.fork_from < node.index and node.fork_from >= 1:
                        graph_nodelist[node.index].parents.append(
                            graph_nodelist[node.fork_from])
                else:
                    if node.fork_from < node.index and node.fork_from >= 1:
                        node.parents.append(graph_nodelist[node.fork_from])
                    graph_nodelist[node.index] = node
            elif nodelist[i]['sque_type'] == 'start':
                graph_nodelist[node.index] = node
                self.roots.append(graph_nodelist[node.index])
        for i in range(nodelen):
            parentslen = len(graph_nodelist[i].parents)
            for j in range(parentslen):
                graph_nodelist[i].parents[j].childs.append(graph_nodelist[i])
        self.graph_nodelist = graph_nodelist

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

            for cnode in node.childs:
                cv2.arrowedLine(image, node.coord * scale, cnode.coord * scale,
                                color=(0, 161, 244), thickness=2, tipLength=0.1)
        if res is not None:
            cv2.putText(image, str(res),  np.array([0, int(nx[1]) * scale-10]), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2, cv2.LINE_AA)
        save_dir = f"vis/{path}/"
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        cv2.imwrite(os.path.join(
            save_dir, f"{name}_{self.token}_{aux_name}.png"), image)

    @staticmethod
    def ptwise_bfs(query_node, max_node):
        queue = [[query_node]]
        res = []
        while len(queue) > 0:
            nodechain = queue.pop(0)
            if len(nodechain) == max_node:
                res.append(EvalSuperNode(nodechain))
                continue
            if len(nodechain[-1].childs) == 0:
                # res.append(SuperNode(nodechain))
                continue
            for cnode in nodechain[-1].childs:
                queue.append(nodechain + [cnode])
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


def seq2nodelist(seq):
    seq = np.array(seq).reshape(-1, 4)
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
                'coord': None}
        label = seq[i][2]
        if label > 3 or label < 0:
            label = 1

        node['coord'] = [seq[i][0], seq[i][1]]
        if label == 3:  # merge
            node['sque_type'] = idx_type_map[3]
            node['sque_index'] = idx
            node['merge_with'] = seq[i][3]

        elif label == 2:  # fork
            node['sque_type'] = idx_type_map[2]
            node['fork_from'] = seq[i][3]

            last_coord = np.array([seq[i - 1][0], seq[i - 1][1]])
            coord = np.array([seq[i][0], seq[i][1]])
            tmp = sum((coord - last_coord) ** 2)
            if tmp < epsilon:  # split fork
                node['sque_index'] = idx
            else:
                idx = idx + 1
                node['sque_index'] = idx

        else:
            node['sque_type'] = idx_type_map[label]
            idx = idx + 1
            node['sque_index'] = idx

        node_list.append(node)

    return node_list
