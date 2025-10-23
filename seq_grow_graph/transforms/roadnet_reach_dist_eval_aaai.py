import torch
import mmcv
import os
import numpy as np
from math import factorial
from tqdm import tqdm
import cv2
import imageio
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
import multiprocessing as mp
from mmengine.logging import print_log

from nuscenes.map_expansion.map_api import NuScenesMap
from nuscenes.map_expansion.bitmap import BitMap
from projects.SeqGrowGraph.seq_grow_graph.core.centerline import (
    PryCenterLine,
    EvalBzNode,
    EvalSuperBzNode,
    EvalGraphDptDist,
    seq2bznodelist,
    convert_coeff_coord,
    dist_superbznode,
    av2seq2bznodelist,
)
from projects.SeqGrowGraph.seq_grow_graph.decode_centerline import EvalMapBzGraphAAAI
from projects.SeqGrowGraph.seq_grow_graph.transforms.loading import (
    TransformLaneGraph,
    LoadAV2OrderedBzCenterline_new,
    TransformAV2OrderedBzLane2Graph_new,
    LoadNusClearOrderedBzCenterline,
    LoadNusOrderedBzCenterline,LoadNusOrderedBzCenterline
)



def dist_supernode(snode1: EvalSuperBzNode, snode2: EvalSuperBzNode):
    pc1 = snode1.keypoints
    pc2 = snode2.keypoints
    diff1 = snode1.diffs
    diff2 = snode2.diffs
    dist = cdist(pc1, pc2, "euclidean")
    diff = diff1 @ diff2.T
    diff_penalty = np.tan(-diff) + np.tan(1) + 1
    dist = dist * diff_penalty
    dist1 = np.min(dist, axis=0)
    dist2 = np.min(dist, axis=1)
    dist1 = dist1.mean(-1)
    dist2 = dist2.mean(-1)
    return (dist1 + dist2) / 2


def get_distance(gt_nodechains, pred_nodechains):
    cost = np.zeros((len(gt_nodechains), len(pred_nodechains)))
    for i in range(len(gt_nodechains)):
        for j in range(len(pred_nodechains)):
            cost[i][j] = dist_supernode(gt_nodechains[i], pred_nodechains[j])
    gt_ind, pred_ind = linear_sum_assignment(cost)
    dist = cost[gt_ind, pred_ind].mean(-1)
    return dist


def node_match(gt_nodes, pred_nodes, threshold):
    gt_node_poses = np.concatenate(
        [np.zeros((0, 2))] + [gt_node.keypoints for gt_node in gt_nodes], axis=0
    )
    pred_node_poses = np.concatenate(
        [np.zeros((0, 2))] + [pred_node.keypoints for pred_node in pred_nodes], axis=0
    )
    gt_node_objs = [gt_node.nodechain[0][0] for gt_node in gt_nodes]
    pred_node_objs = [pred_node.nodechain[0][0] for pred_node in pred_nodes]
    node_dist = cdist(pred_node_poses, gt_node_poses)
    pred_dist = np.min(node_dist, axis=1)
    pred_ind_list = np.argmin(node_dist, axis=1)
    pred_ind_list[pred_dist > threshold] = -1

    pred_ind = {}
    gt_ind = {}
    for i in range(len(pred_ind_list)):
        if pred_ind_list[i] != -1:
            pred_ind[pred_node_objs[i]] = gt_node_objs[pred_ind_list[i]]
            if gt_node_objs[pred_ind_list[i]] not in gt_ind.keys():
                gt_ind[gt_node_objs[pred_ind_list[i]]] = [
                    pred_node_objs[i],
                ]
            else:
                gt_ind[gt_node_objs[pred_ind_list[i]]].append(pred_node_objs[i])
    return gt_ind, pred_ind


def get_reach_diagnose(gt_nodechains, pred_nodechains, threshold):
    cost = np.zeros((len(pred_nodechains), len(gt_nodechains)))
    for i in range(len(pred_nodechains)):
        for j in range(len(gt_nodechains)):
            cost[i][j] = dist_supernode(pred_nodechains[i], gt_nodechains[j])
    pred_dist = np.min(cost, axis=1)
    pred_ind = np.argmin(cost, axis=1)
    tp_mask = pred_dist < threshold
    tp = tp_mask.sum()
    fp = len(pred_nodechains) - tp
    pred_ind = np.unique(pred_ind[tp_mask])
    fn = len(gt_nodechains) - len(pred_ind)
    return tp, fp, fn


def eval_landmark(
    gt_nodegraph: EvalMapBzGraphAAAI, pred_nodegraph: EvalMapBzGraphAAAI, thresholds: List
):
    res_tps = []
    res_fps = []
    res_fns = []
    gt_nodes = gt_nodegraph.get_nodechains_dpt(1)
    pred_nodes = pred_nodegraph.get_nodechains_dpt(1)
    for threshold in thresholds:
        gt_ind, pred_ind = node_match(gt_nodes, pred_nodes, threshold)
        tp = len(pred_ind.keys())
        fp = len(pred_nodes) - len(pred_ind.keys())
        fn = len(gt_nodes) - len(gt_ind.keys())
        res_tps.append(tp)
        res_fps.append(fp)
        res_fns.append(fn)
    return ResArray(res_tps), ResArray(res_fps), ResArray(res_fns)


def eval_reach(
    gt_nodegraph: EvalMapBzGraphAAAI,
    pred_nodegraph: EvalMapBzGraphAAAI,
    thresholds: List,
    max_node_num=5,
):
    res_tps = []
    res_fps = []
    res_fns = []
    gt_nodes = gt_nodegraph.get_nodechains_dpt(1)
    pred_nodes = pred_nodegraph.get_nodechains_dpt(1)
    gt_nodechains = gt_nodegraph.get_nodechains_dpt(max_node_num)
    pred_nodechains = pred_nodegraph.get_nodechains_dpt(max_node_num)
    gt_reachable_nodes = [gt_nodechain.start_end for gt_nodechain in gt_nodechains]
    pred_reachable_nodes = [
        pred_nodechain.start_end for pred_nodechain in pred_nodechains
    ]
    gt_reachables = {}
    pred_reachables = {}
    for i, gt_reachable_node in enumerate(gt_reachable_nodes):
        if gt_reachable_node not in gt_reachables.keys():
            gt_reachables[gt_reachable_node] = [
                gt_nodechains[i],
            ]
        else:
            gt_reachables[gt_reachable_node].append(gt_nodechains[i])
    for i, pred_reachable_node in enumerate(pred_reachable_nodes):
        if pred_reachable_node not in pred_reachables.keys():
            pred_reachables[pred_reachable_node] = [
                pred_nodechains[i],
            ]
        else:
            pred_reachables[pred_reachable_node].append(pred_nodechains[i])
    for threshold in thresholds:
        tps = 0
        fps = 0
        fns = 0
        gt_ind, pred_ind = node_match(gt_nodes, pred_nodes, 5)
        for gt_reachable in gt_reachables.keys():
            gt_start, gt_end = gt_reachable
            pred_starts = []
            pred_ends = []
            if gt_start in gt_ind.keys():
                pred_starts = gt_ind[gt_start]
            if gt_end in gt_ind.keys():
                pred_ends = gt_ind[gt_end]
            if len(pred_starts) != 0 and len(pred_ends) != 0:
                for pred_start in pred_starts:
                    for pred_end in pred_ends:
                        pred_start_end = (pred_start, pred_end)
                        if pred_start_end in pred_reachables.keys():
                            tp, fp, fn = get_reach_diagnose(
                                gt_reachables[gt_reachable],
                                pred_reachables[pred_start_end],
                                threshold,
                            )
                            tps += tp
                            fps += fp
                            fns += fn
                        else:
                            fns += len(gt_reachables[gt_reachable])
        res_tps.append(tps)
        res_fps.append(fps)
        res_fns.append(fns)
    return ResArray(res_tps), ResArray(res_fps), ResArray(res_fns)


def eval_fscore(tp, fn, fp, beta):
    def f_func(precision, recall, beta):
        if precision + recall == 0:
            return precision * recall
        return (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)

    def precision(tp, fn, fp):
        if tp + fp == 0:
            return ResArray(0.0, tp.length)
        return tp / (tp + fp)

    def recall(tp, fn, fp):
        if tp + fn == 0:
            return ResArray(0.0, tp.length)
        return tp / (tp + fn)

    pr = precision(tp, fn, fp)
    re = recall(tp, fn, fp)
    fm = f_func(pr, re, beta)
    return pr, re, fm


class ResArray:
    def __init__(self, dists, length=None) -> None:
        if not isinstance(dists, List):
            if isinstance(dists, numbers.Number) and (length is not None):
                dists = [dists for _ in range(length)]
            else:
                return
        self.dists = dists
        self.length = len(dists)

    def mean(self):
        if len(self.dists) > 0:
            mean = sum(self.dists) / len(self.dists)
        else:
            mean = 0
        return mean

    def __getitem__(self, ind):
        return self.dists[ind]

    def __str__(self) -> str:
        name = ""
        for i in range(len(self.dists)):
            name += "res-%d: %.3f | " % (i, self.dists[i])
        return name[:-1]

    def __repr__(self) -> str:
        name = ""
        for i in range(len(self.dists)):
            name += "res-%d: %.3f | " % (i, self.dists[i])
        return name[:-1]

    def __add__(self, dist2):
        if not isinstance(dist2, ResArray):
            if isinstance(dist2, numbers.Number):
                dist2 = ResArray([dist2 for _ in range(self.length)])
            else:
                return
        if self.length != dist2.length:
            return
        add_dists = [0 for _ in range(self.length)]
        for i in range(self.length):
            if np.isnan(dist2.dists[i]):
                continue
            add_dists[i] = self.dists[i] + dist2.dists[i]
        return ResArray(add_dists)

    def __radd__(self, dist2):
        if not isinstance(dist2, ResArray):
            if isinstance(dist2, numbers.Number):
                dist2 = ResArray([dist2 for _ in range(self.length)])
            else:
                return
        if self.length != dist2.length:
            return
        add_dists = [0 for _ in range(self.length)]
        for i in range(self.length):
            if np.isnan(dist2.dists[i]):
                continue
            add_dists[i] = self.dists[i] + dist2.dists[i]
        return ResArray(add_dists)

    def __mul__(self, dist2):
        if not isinstance(dist2, ResArray):
            if isinstance(dist2, numbers.Number):
                dist2 = ResArray([dist2 for _ in range(self.length)])
            else:
                return
        if self.length != dist2.length:
            return
        mul_dists = [0 for _ in range(self.length)]
        for i in range(self.length):
            if np.isnan(dist2.dists[i]):
                continue
            mul_dists[i] = self.dists[i] * dist2.dists[i]
        return ResArray(mul_dists)

    def __rmul__(self, dist2):
        if not isinstance(dist2, ResArray):
            if isinstance(dist2, numbers.Number):
                dist2 = ResArray([dist2 for _ in range(self.length)])
            else:
                return
        if self.length != dist2.length:
            return
        mul_dists = [0 for _ in range(self.length)]
        for i in range(self.length):
            if np.isnan(dist2.dists[i]):
                continue
            mul_dists[i] = self.dists[i] * dist2.dists[i]
        return ResArray(mul_dists)

    def __truediv__(self, dist2):
        if not isinstance(dist2, ResArray):
            if isinstance(dist2, numbers.Number):
                dist2 = ResArray([dist2 for _ in range(self.length)])
            else:
                return
        if self.length != dist2.length:
            return
        div_dists = [0 for _ in range(self.length)]
        for i in range(self.length):
            if np.isnan(dist2.dists[i]):
                continue
            div_dists[i] = self.dists[i] / dist2.dists[i]
        return ResArray(div_dists)

    def __eq__(self, dist2):
        if not isinstance(dist2, ResArray):
            if isinstance(dist2, numbers.Number):
                dist2 = ResArray([dist2 for _ in range(self.length)])
            else:
                return
        if self.length != dist2.length:
            return
        equal = True
        for i in range(self.length):
            if np.isnan(dist2.dists[i]):
                continue
            equal = equal and (self.dists[i] == dist2.dists[i])
        return equal

    def __iadd__(self, dist2):
        if not isinstance(dist2, ResArray):
            if isinstance(dist2, numbers.Number):
                dist2 = ResArray([dist2 for _ in range(self.length)])
            else:
                return
        if self.length != dist2.length:
            return
        for i in range(self.length):
            if np.isnan(dist2.dists[i]):
                continue
            self.dists[i] = self.dists[i] + dist2.dists[i]
        return self

    def __itruediv__(self, dist2):
        if not isinstance(dist2, ResArray):
            if isinstance(dist2, numbers.Number):
                dist2 = ResArray([dist2 for _ in range(self.length)])
            else:
                return
        for i in range(self.length):
            self.dists[i] = self.dists[i] / dist2.dists[i]
        return self


def gen_dx_bx(xbound, ybound, zbound):
    dx = np.array([row[2] for row in [xbound, ybound, zbound]])
    bx = np.array([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])
    nx = np.floor(
        np.array([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]])
    )
    return dx, bx, nx


def get_geom(grid_conf):
    dx, bx, nx = gen_dx_bx(
        grid_conf["xbound"],
        grid_conf["ybound"],
        grid_conf["zbound"],
    )
    dx = dx
    bx = bx
    nx = nx
    pc_range = np.concatenate((bx - dx / 2.0, bx - dx / 2.0 + nx * dx))

    x = np.arange(pc_range[0], pc_range[3], 0.1)
    y = np.arange(pc_range[1], pc_range[4], 0.1)
    z = np.array([0.0])
    xx, yy, zz = np.meshgrid(x, y, z)
    points = np.stack([xx, yy, zz], axis=-1)
    ego_points = np.concatenate(
        (points, np.ones((points.shape[0], points.shape[1], points.shape[2], 1))),
        axis=-1,
    )
    return dx, bx, nx, pc_range, ego_points


def get_range(grid_conf):
    dx, bx, nx = gen_dx_bx(
        grid_conf["xbound"],
        grid_conf["ybound"],
        grid_conf["zbound"],
    )
    dx = dx
    bx = bx
    nx = nx
    pc_range = np.concatenate((bx - dx / 2.0, bx - dx / 2.0 + nx * dx))
    return dx, bx, nx, pc_range


def get_ego_pose(info):
    ego_trans = info["ego2global_translation"]
    ego_rot = info["ego2global_rotation"]
    ego2global = np.eye(4)
    ego2global[:3, :3] = Quaternion(ego_rot).rotation_matrix
    ego2global[:3, 3] = ego_trans
    return ego2global


def print_iteration(iterations, stop_sign, max_iters):
    while stop_sign.value != 1:
        out_str = ""
        for id, iter in enumerate(iterations):
            out_str += "proc-%d: %d/%d\t" % (id, iter.value, max_iters[id].value)
        # print(out_str, end='\r')


def sub_eval(
    q_r_tp,
    q_r_fn,
    q_r_fp,
    q_l_tp,
    q_l_fn,
    q_l_fp,
    dataset_range,
    max_dpt,
    landmark_thresholds,
    reach_thresholds,
    val_pkl,
    results,
    centerline_loader,
    centerline_transform,
    pc_range,
    dx,
    bz_pc_range,
    bz_dx,
    iteration,
):
    reach_tp = ResArray([0 for _ in range(len(reach_thresholds))])
    reach_fp = ResArray([0 for _ in range(len(reach_thresholds))])
    reach_fn = ResArray([0 for _ in range(len(reach_thresholds))])
    landmark_tp = ResArray([0 for _ in range(len(landmark_thresholds))])
    landmark_fp = ResArray([0 for _ in range(len(landmark_thresholds))])
    landmark_fn = ResArray([0 for _ in range(len(landmark_thresholds))])
    for si in range(dataset_range[0], dataset_range[1]):
        try:
            info = val_pkl[si]
            token = info["token"]
            if token in results:
                info = centerline_loader(info)
                info = centerline_transform(info)
                gt_sequence =   list(info['vert_sentence'])+[573]+list(info['edge_sentence'] )

                gt_nodegraph = EvalMapBzGraphAAAI(
                    token,
                    gt_sequence,
                    pc_range=pc_range,
                    dx=dx,
                    bz_pc_range=bz_pc_range,
                    bz_dx=bz_dx,
                )
                pred_sequence = results[token]
                # pred_nodelist = convert_coeff_coord(pred_nodelist, pc_range, dx, bz_pc_range, bz_dx)
                pred_nodegraph = EvalMapBzGraphAAAI(
                    token,
                    pred_sequence,
                    pc_range=pc_range,
                    dx=dx,
                    bz_pc_range=bz_pc_range,
                    bz_dx=bz_dx,
                )
                tp, fp, fn = eval_reach(
                    gt_nodegraph, pred_nodegraph, reach_thresholds, max_dpt
                )
                reach_tp += tp
                reach_fp += fp
                reach_fn += fn

                tp, fp, fn = eval_landmark(
                    gt_nodegraph, pred_nodegraph, landmark_thresholds
                )
                landmark_tp += tp
                landmark_fp += fp
                landmark_fn += fn
                iteration.value += 1
        except Exception as e:
            import traceback

            traceback.print_exc()
            print(e)
    q_r_tp.put(reach_tp)
    q_r_fp.put(reach_fp)
    q_r_fn.put(reach_fn)
    q_l_tp.put(landmark_tp)
    q_l_fp.put(landmark_fp)
    q_l_fn.put(landmark_fn)


def BzRoadnetReachDistEvalAAAI(
    result_path,
    data_root,
    grid_conf,
    bz_grid_conf,
    beta=1.0,
    max_dpt=5,
    num_proc=8,
    landmark_thresholds=[1, 3, 5, 8, 10],
    reach_thresholds=[1, 2, 3, 4, 5],
    val_pkl=None,
    logger=None,
):
    sample_length = len(val_pkl)
    dx, bx, nx, pc_range, ego_points = get_geom(grid_conf)
    bz_dx, bz_bx, bz_nx, bz_pc_range = get_range(bz_grid_conf)
    centerline_loader = LoadNusOrderedBzCenterline(grid_conf, bz_grid_conf)
    centerline_transform = TransformLaneGraph()

    with open(result_path) as f:
        results = json.load(f)["results"]

    reach_tp = ResArray([0 for _ in range(len(reach_thresholds))])
    reach_fp = ResArray([0 for _ in range(len(reach_thresholds))])
    reach_fn = ResArray([0 for _ in range(len(reach_thresholds))])
    landmark_tp = ResArray([0 for _ in range(len(landmark_thresholds))])
    landmark_fp = ResArray([0 for _ in range(len(landmark_thresholds))])
    landmark_fn = ResArray([0 for _ in range(len(landmark_thresholds))])

    q_r_tp = mp.Queue()
    q_r_fn = mp.Queue()
    q_r_fp = mp.Queue()
    q_l_tp = mp.Queue()
    q_l_fn = mp.Queue()
    q_l_fp = mp.Queue()

    dataset_split = np.linspace(0, sample_length, num_proc + 1).astype(np.int32)
    max_iterations = (
        dataset_split
        - np.concatenate(
            [
                np.array(
                    [
                        0,
                    ]
                ),
                dataset_split,
            ]
        )[:-1]
    )[1:]
    max_iterations = [mp.Value("i", max_iterations[pi]) for pi in range(num_proc)]
    iterations = [mp.Value("i", 0) for pi in range(num_proc)]
    stop_sign = mp.Value("i", 0)
    pool = [
        mp.Process(
            target=sub_eval,
            args=(
                q_r_tp,
                q_r_fn,
                q_r_fp,
                q_l_tp,
                q_l_fn,
                q_l_fp,
                dataset_split[pi : pi + 2],
                max_dpt,
                landmark_thresholds,
                reach_thresholds,
                val_pkl,
                results,
                centerline_loader,
                centerline_transform,
                pc_range,
                dx,
                bz_pc_range,
                bz_dx,
                iterations[pi],
            ),
        )
        for pi in range(num_proc)
    ]
    log_proc = mp.Process(
        target=print_iteration, args=(iterations, stop_sign, max_iterations)
    )
    for proc in pool:
        proc.start()
    log_proc.start()
    for proc in pool:
        proc.join()
    stop_sign.value = 1
    log_proc.join()
    for pi in range(num_proc):
        reach_tp += q_r_tp.get()
        reach_fn += q_r_fn.get()
        reach_fp += q_r_fp.get()
        landmark_tp += q_l_tp.get()
        landmark_fn += q_l_fn.get()
        landmark_fp += q_l_fp.get()
    assert q_r_tp.empty() and q_r_fn.empty() and q_r_fp.empty()
    reach_precision, reach_recall, reach_f_score = eval_fscore(
        reach_tp, reach_fn, reach_fp, beta
    )
    landmark_precision, landmark_recall, landmark_f_score = eval_fscore(
        landmark_tp, landmark_fn, landmark_fp, beta
    )

    linewidth = 20 + 15 * len(reach_thresholds)
    print_log("=" * linewidth, logger=logger)
    print_log(f"Landmark Precision || {landmark_precision}", logger=logger)
    print_log(f"Landmark Recall    || {landmark_recall}", logger=logger)
    print_log("-" * linewidth, logger=logger)
    print_log(f"Landmark F1 score  || {landmark_f_score}", logger=logger)
    print_log("-" * linewidth, logger=logger)
    print_log("R-Precision || %.3f" % landmark_precision.mean(), logger=logger)
    print_log("R-Recall    || %.3f" % landmark_recall.mean(), logger=logger)
    print_log("R-F1 score  || %.3f" % landmark_f_score.mean(), logger=logger)
    print_log("=" * linewidth, logger=logger)
    print_log(f"Reachable Precision || {reach_precision}", logger=logger)
    print_log(f"Reachable Recall    || {reach_recall}", logger=logger)
    print_log("-" * linewidth, logger=logger)
    print_log(f"Reachable F1 score  || {reach_f_score}", logger=logger)
    print_log("-" * linewidth, logger=logger)
    print_log("R-Precision || %.3f" % reach_precision.mean(), logger=logger)
    print_log("R-Recall    || %.3f" % reach_recall.mean(), logger=logger)
    print_log("R-F1 score  || %.3f" % reach_f_score.mean(), logger=logger)
    print_log("=" * linewidth, logger=logger)

    res = dict()
    res["landmark_precision"] = landmark_precision
    res["landmark_recall"] = landmark_recall
    res["landmark_f_score"] = landmark_f_score
    res["reach_precision"] = reach_precision
    res["reach_recall"] = reach_recall
    res["reach_f_score"] = reach_f_score

    return res
