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
from scipy.spatial.distance import cdist
from scipy.optimize import linear_sum_assignment


class EvalBzPlNode():
    def __init__(self, nodedict):
        self.coord = np.array(nodedict['coord'])
        self.type = [nodedict['sque_type']]
        self.parents = []
        self.childs = []
        self.fork_from = None if nodedict['fork_from'] is None else nodedict['fork_from'] - 1
        self.merge_with = None if nodedict['merge_with'] is None else nodedict['merge_with'] - 1
        self.index = nodedict['sque_index'] - 1
        self.subg_index = nodedict['subg_index'] - 1

    def __repr__(self) -> str:
        nodename = ''
        for name in self.type:
            nodename += name[0]
        return f"{self.subg_index}_{nodename}_{self.index}"

    def __str__(self) -> str:
        nodename = ''
        for name in self.type:
            nodename += name[0]
        return f"{self.subg_index}_{nodename}_{self.index}"


class EvalSuperBzPlNode():
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


def dist_superbznode(snode1: EvalSuperBzPlNode, snode2: EvalSuperBzPlNode):
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


class EvalMapBzPlGraph():
    def __init__(self, map_token, nodelists):
        self.token = map_token
        self.roots = []
        subglen = len(nodelists)
        graph_nodelists = [[] for nodelist in nodelists]
        for gi, nodelist in enumerate(nodelists):
            seqnodelen = len(nodelist)
            assert seqnodelen > 0
            nodelen = nodelist[-1]['sque_index']
            graph_nodelists[gi] = [EvalBzPlNode(
                nodelist[0])] + [None for _ in range(nodelen-1)]

        for gi, nodelist in enumerate(nodelists):
            seqnodelen = len(nodelist)
            assert seqnodelen > 0
            nodelen = nodelist[-1]['sque_index']
            for i in range(seqnodelen):
                node = EvalBzPlNode(nodelist[i])
                if nodelist[i]['sque_type'] == 'continue':
                    node.parents.append(
                        (graph_nodelists[gi][node.index-1], nodelist[i]['coeff']))
                    graph_nodelists[gi][node.index] = node
                    graph_nodelists[gi][node.index-1].childs.append(
                        (graph_nodelists[gi][node.index], nodelist[i]['coeff']))
                elif nodelist[i]['sque_type'] == 'merge':
                    if graph_nodelists[gi][node.index] is not None and graph_nodelists[gi][node.index].index == node.index:
                        graph_nodelists[gi][node.index].type.append('merge')
                        if node.merge_with < subglen and node.merge_with >= 0:
                            graph_nodelists[gi][node.index].childs.append(
                                (graph_nodelists[node.merge_with][0], nodelist[i]['coeff']))
                            graph_nodelists[node.merge_with][0].parents.append(
                                (graph_nodelists[gi][node.index], nodelist[i]['coeff']))
                    else:
                        if node.merge_with < subglen and node.merge_with >= 0:
                            node.childs.append(
                                (graph_nodelists[node.merge_with][0], nodelist[i]['coeff']))
                            graph_nodelists[node.merge_with][0].parents.append(
                                (node, nodelist[i]['coeff']))
                        graph_nodelists[gi][node.index] = node
                elif nodelist[i]['sque_type'] == 'fork':
                    if graph_nodelists[gi][node.index] is not None and graph_nodelists[gi][node.index].index == node.index:
                        graph_nodelists[gi][node.index].type.append('fork')
                        if node.fork_from < subglen and node.fork_from >= 0:
                            graph_nodelists[gi][node.index].parents.append(
                                (graph_nodelists[node.fork_from][0], nodelist[i]['coeff']))
                            graph_nodelists[node.fork_from][0].childs.append(
                                (node, nodelist[i]['coeff']))
                    else:
                        if node.fork_from < subglen and node.fork_from >= 0:
                            node.parents.append(
                                (graph_nodelists[node.fork_from][0], nodelist[i]['coeff']))
                            graph_nodelists[node.fork_from][0].childs.append(
                                (node, nodelist[i]['coeff']))
                        graph_nodelists[gi][node.index] = node
                elif nodelist[i]['sque_type'] == 'start':
                    self.roots.append(graph_nodelists[gi][0])
        # for i in range(nodelen):
        #     parentslen = len(graph_nodelist[i].parents)
        #     for j in range(parentslen):
        #         graph_nodelist[i].parents[j].childs.append(graph_nodelist[i])
        self.graph_nodelists = graph_nodelists

        graph_nodelist = []
        for subg_i in range(len(graph_nodelists)):
            graph_nodelist = graph_nodelist + graph_nodelists[subg_i]
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
        for graph_nodelist in self.graph_nodelists:
            for idx, node in enumerate(graph_nodelist):
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

                cv2.putText(image, "%.2d-%.2d" % (node.subg_index, node.index), node.coord * scale + np.array([-10, 10]), cv2.FONT_HERSHEY_SIMPLEX,
                            0.7, (255, 204, 0), 2, cv2.LINE_AA)

                for cnode, coeff in node.childs:
                    fin_res = np.stack(
                        (node.coord * scale, coeff * scale, cnode.coord * scale))
                    curve = bezier.Curve(fin_res.T, degree=2)
                    s_vals = np.linspace(0.0, 1.0, 50)
                    data_b = curve.evaluate_multi(s_vals).T
                    data_b = data_b.astype(np.int)
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
        if res is not None:
            cv2.putText(image, str(res),  np.array([0, int(nx[1]) * scale-10]), cv2.FONT_HERSHEY_SIMPLEX,
                        0.7, (0, 255, 0), 2, cv2.LINE_AA)
        save_dir = f"vis/{path}/"
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
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
        for graph_nodelist in self.graph_nodelists:
            for idx, node in enumerate(graph_nodelist):
                if len(node.parents) == 0:
                    cv2.circle(image, node.coord * scale, int(scale**1.4),
                               color=point_color_map['start'], thickness=-1)
                    cv2.circle(image, node.coord * scale,
                               int(scale**1.4), color=(0, 0, 0), thickness=3)
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

        for graph_nodelist in self.graph_nodelists:
            for idx, node in enumerate(graph_nodelist):
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
                    data_b = data_b.astype(np.int)
                    cv2.polylines(image, [data_b], False,
                                  color=color, thickness=5)

                    curve_dist = np.diff(data_b, axis=0)
                    curve_dist = np.sum(curve_dist**2, axis=1) ** 0.5
                    curve_len = np.sum(curve_dist)
                    curve_dist = np.cumsum(curve_dist)
                    if curve_len > pixels_step:
                        curve_dist_std = np.arange(0, curve_len, pixels_step)
                        curve_dist_cost = cdist(
                            curve_dist_std[:, None], curve_dist[:, None])
                        _, curve_dist_idx = linear_sum_assignment(
                            curve_dist_cost)
                        curve_dist_idx = curve_dist_idx[curve_dist_idx > 1]
                        for curve_idx in curve_dist_idx:
                            arrowline = data_b[curve_idx -
                                               1:curve_idx+1, :].copy()
                            diff = arrowline[1] - arrowline[0]
                            if np.prod(arrowline[1] == arrowline[0]):
                                continue
                            diff = diff / np.linalg.norm(diff) * 3
                            arrowline[1] = arrowline[0] + diff
                            cv2.arrowedLine(image, arrowline[0], arrowline[1],
                                            color=color, thickness=5, tipLength=5)
        if puttext:
            for graph_nodelist in self.graph_nodelists:
                for idx, node in enumerate(graph_nodelist):
                    coord = node.coord * scale
                    text_shape = np.array([110, 20]) * 0.7
                    text_shape = text_shape.astype(np.int)
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
                    cv2.putText(image, "%.2d-%.2d" % (node.subg_index, node.index), coord, cv2.FONT_HERSHEY_SIMPLEX,
                                0.7, (0, 0, 0), 3, cv2.LINE_AA)
        #         if len(node.parents)==0:
        #             cv2.circle(image, node.coord * scale, int(scale**1.4), color=point_color_map['start'], thickness=-1)
        #             cv2.circle(image, node.coord * scale, int(scale**1.4), color=(0,0,0), thickness=3)
        #         else:
        #             if len(node.childs) == 1:
        #                 cv2.circle(image, node.coord * scale, int(scale**1.4), color=point_color_map['continue'], thickness=-1)
        #                 cv2.circle(image, node.coord * scale, int(scale**1.4), color=(0,0,0), thickness=3)
        #             elif len(node.childs) > 1:
        #                 cv2.circle(image, node.coord * scale, int(scale**1.4), color=point_color_map['fork'], thickness=-1)
        #                 cv2.circle(image, node.coord * scale, int(scale**1.4), color=(0,0,0), thickness=3)
        #             else:
        #                 cv2.circle(image, node.coord * scale, int(scale**1.4), color=point_color_map['end'], thickness=-1)
        #                 cv2.circle(image, node.coord * scale, int(scale**1.4), color=(0,0,0), thickness=3)
        # if res is not None:
        #     cv2.putText(image, str(res),  np.array([0, int(nx[1]) * scale-10]), cv2.FONT_HERSHEY_SIMPLEX,
        #            0.7, (0, 255,0), 2, cv2.LINE_AA)
        save_dir = f"vis/{path}/"
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        cv2.imwrite(os.path.join(
            save_dir, f"{name}_{self.token}_{aux_name}.png"), image)

    @staticmethod
    def ptwise_bfs(query_node, max_node):
        queue = [[(query_node, None)]]
        res = []
        while len(queue) > 0:
            nodechain = queue.pop(0)
            if len(nodechain) == max_node:
                res.append(EvalSuperBzPlNode(nodechain))
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


def seq2bzplnodelist(seqs, n_control):
    """"n control = 3"""
    length = 4 + 2*(n_control-2)
    node_lists = []
    idx_type_map = {0: 'start', 1: 'continue', 2: "fork", 3: 'merge'}
    epsilon = 2
    for subg_i, seq in enumerate(seqs):
        node_list = []
        idx = 0
        seq = np.array(seq).reshape(-1, length)
        for i in range(len(seq)):
            node = {'subg_index': subg_i+1,
                    'sque_index': None,
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
        node_lists.append(node_list)
    return node_lists
