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
import math
import bezier
from math import factorial
from projects.SeqGrowGraph.seq_grow_graph.core.centerline.structures.pryordered_bz_centerline import NusOrederedBzCenterLine

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

def convert_coeff_coord(nodelist, pc_range, dx, bz_pc_range, bz_dx):
    seqnodelen = len(nodelist)
    for i in range(seqnodelen):
        coeff = nodelist[i]['coeff']
        if len(coeff) < 1:
            continue
        coeff = coeff * bz_dx[:2] + bz_pc_range[:2]
        coeff = ((coeff - pc_range[:2]) / dx[:2]).astype(np.int64)
        nodelist[i]['coeff'] = coeff
    return nodelist


class AV2BzNode(object):
    def __init__(self, position, epsilon=0.00001):
        self.parents = []
        self.children = []
        self.position = position
        self.type = None
        self.coeff = []
        self.node_index = None
        self.sque_type = None
        self.merge_with_index = None
        self.fork_from_index = None
        self.sque_index = None
        self.sque_points = None
        self.epsilon = epsilon

    def set_parents(self, parents):
        self.parents = parents

    def set_children(self, children):
        self.children = children

    def set_coeff(self, coeff):
        self.coeff = coeff

    def set_type(self, type_):
        self.type = type_

    def __repr__(self):
        return f"Node_sque_index : {self.sque_index}, sque_type : {self.sque_type}, fork_from : {self.fork_from_index}, merge with : {self.merge_with_index}, coord : {self.position}\n"

    def __eq__(self, __o):
        if np.linalg.norm(np.array(self.position[:2]) - np.array(__o.position[:2])) < self.epsilon:
            return True
        return False


class AV2OrderedBzLaneGraph(object):
    def __init__(self, Nodes_list, nodes_adj, nodes_points):
        self.nodes_list = Nodes_list
        self.nodes_adj = nodes_adj
        self.nodes_points = nodes_points
        self.num = len(self.nodes_list)
        self.node_type_index = None

        for i, j in self.nodes_points.keys():
            if self.nodes_adj[i][j] == 1:
                continue
            else:
                raise Exception("nodes points and nodes adj not matched!")

    def get_start_nodes_idx_sorted(self):
        self.__type_gen()
        start_nodes_sorted = self.__nodes_sort(self.node_type_index['Start'] + self.node_type_index['Start_and_Fork'],
                                               self.start_nodes_sort_method)
        self.first_start_node = self.nodes_list[start_nodes_sorted[0]]
        self.start_nodes_idx_sorted = self.__nodes_sort(
            self.node_type_index['Start'] + self.node_type_index['Start_and_Fork'], self.start_nodes_sort_method)

    def __repr__(self):
        return f"Lane Graph: {self.num} nodes"

    def __eq__(self):
        raise Exception("No Implement!")

    def __check_nodes__(self):
        raise Exception("No Implement!")

    def __len__(self):
        return self.num

    def start_nodes_sort_method(self, nodes_indexes: list):
        nodes_index_list = [(node_index, self.nodes_list[node_index]) for node_index in nodes_indexes]
        nodes_index_list = sorted(nodes_index_list, key=lambda x: x[1].position[0])
        return [node[0] for node in nodes_index_list]
    
    def fork_nodes_sort_method(self, nodes_indexes: list):
        nodes_index_list = [(node_index, self.nodes_list[node_index]) for node_index in nodes_indexes]
        nodes_index_list = sorted(nodes_index_list, key=lambda x: x[1].position[0])
        return [node[0] for node in nodes_index_list]

    def __nodes_sort(self, nodes_indexes, method):
        return method(nodes_indexes)

    def __sequelize__(self):
        start_nodes_sorted = self.__nodes_sort(self.node_type_index['Start'] + self.node_type_index['Start_and_Fork'],
                                               self.start_nodes_sort_method)
        visted_nodes = [False for i in self.nodes_list]
        visted_count = [0 for i in self.nodes_list]
        result = []
        for start_node_index in start_nodes_sorted:
            result = result + self.__dfs_sequelize(start_node_index, visted_nodes, visted_count, self.nodes_adj)

        for visted in visted_nodes:
            if visted:
                continue
            else:
                raise Exception("Some node missing!")

        return result

    def __type_gen(self):
        self.node_type_index = {'Continue': [], 'Fork_and_Merge': [], 'EndPoint': [], 'Merge': [], 'Start': [],
                                'Fork': [], 'EndPoint_and_Merge': [], 'Start_and_Fork': []}
        for idx, node in enumerate(self.nodes_list):
            sum_b0 = np.sum(self.nodes_adj[idx] > 0)
            sum_s0 = np.sum(self.nodes_adj[idx] < 0)

            if sum_b0 == 0 and sum_s0 == 0:
                raise Exception('wrong node')

            elif sum_b0 > 1 and sum_s0 > 1:

                self.nodes_list[idx].type = 'Fork_and_Merge'
                self.node_type_index['Fork_and_Merge'].append(idx)


            elif sum_b0 == sum_s0 and sum_b0 == 1:
                self.nodes_list[idx].type = 'Continue'
                self.node_type_index['Continue'].append(idx)


            elif sum_b0 < sum_s0:
                if sum_s0 == 1 and sum_b0 == 0:
                    self.nodes_list[idx].type = 'EndPoint'
                    self.node_type_index['EndPoint'].append(idx)
                elif sum_b0 == 0:
                    self.nodes_list[idx].type = 'EndPoint_and_Merge'
                    self.node_type_index['EndPoint_and_Merge'].append(idx)
                else:
                    self.nodes_list[idx].type = 'Merge'
                    self.node_type_index['Merge'].append(idx)


            elif sum_b0 > sum_s0:
                if sum_b0 == 1:
                    self.nodes_list[idx].type = 'Start'
                    self.node_type_index['Start'].append(idx)
                elif sum_s0 == 0:
                    self.nodes_list[idx].type = 'Start_and_Fork'
                    self.node_type_index['Start_and_Fork'].append(idx)
                else:
                    self.nodes_list[idx].type = 'Fork'
                    self.node_type_index['Fork'].append(idx)
            else:
                raise Exception("Error on type assign!")


class AV2OrderedBzLaneGraph_new(object):
    def __init__(self, Nodes_list, nodes_adj, nodes_points):
        self.nodes_list = Nodes_list
        self.nodes_adj = nodes_adj
        self.nodes_points = nodes_points
        self.num = len(self.nodes_list)
        self.node_type_index = None

        for i, j in self.nodes_points.keys():
            if self.nodes_adj[i][j] == 1:
                continue
            else:
                raise Exception("nodes points and nodes adj not matched!")

    def get_start_nodes_idx_sorted(self):
        self.__type_gen()
        start_nodes_sorted = self.__nodes_sort(self.node_type_index['Start']+self.node_type_index['Start_with_back'],
                                               self.start_nodes_sort_method)
        self.first_start_node = self.nodes_list[start_nodes_sorted[0]]
        self.start_nodes_idx_sorted = self.__nodes_sort(
            self.node_type_index['Start']+self.node_type_index['Start_with_back'], self.start_nodes_sort_method)

    def __repr__(self):
        return f"Lane Graph: {self.num} nodes"

    def __eq__(self):
        raise Exception("No Implement!")

    def __check_nodes__(self):
        raise Exception("No Implement!")

    def __len__(self):
        return self.num

    def start_nodes_sort_method(self, nodes_indexes: list):
        nodes_index_list = [(node_index, self.nodes_list[node_index]) for node_index in nodes_indexes]
        nodes_index_list = sorted(nodes_index_list, key=lambda x: x[1].position[0])
        return [node[0] for node in nodes_index_list]

    def fork_nodes_sort_method(self, nodes_indexes: list):
        nodes_index_list = [(node_index, self.nodes_list[node_index]) for node_index in nodes_indexes]
        nodes_index_list = sorted(nodes_index_list, key=lambda x: x[1].position[0])
        return [node[0] for node in nodes_index_list]

    def __nodes_sort(self, nodes_indexes, method):
        return method(nodes_indexes)

    def __sequelize__(self):
        start_nodes_sorted = self.__nodes_sort(self.node_type_index['Start'] + self.node_type_index['Start_and_Fork'],
                                               self.start_nodes_sort_method)
        visted_nodes = [False for i in self.nodes_list]
        visted_count = [0 for i in self.nodes_list]
        result = []
        for start_node_index in start_nodes_sorted:
            result = result + self.__dfs_sequelize(start_node_index, visted_nodes, visted_count, self.nodes_adj)

        for visted in visted_nodes:
            if visted:
                continue
            else:
                raise Exception("Some node missing!")

        return result

    def __type_gen(self):
        self.node_type_index = {'Continue': [], 'Fork_and_Merge': [], 'EndPoint': [], 'Merge': [], 'Start': [],
                                'Fork': [], 'EndPoint_and_Merge': [], 'Start_with_back': []}
        for idx, node in enumerate(self.nodes_list):
            if max(self.nodes_adj[:, idx])==0:  # out degree == 0
                self.nodes_list[idx].type = 'Start'
                self.node_type_index['Start'].append(idx)
            elif sum(self.nodes_adj[:, idx])==1 and sum(self.nodes_adj[idx])>=1:  # out_degree == 1 and in_degree >= 1
                self.nodes_list[idx].type = 'Start_with_back'
                self.node_type_index['Start_with_back'].append(idx)

        # self.node_type_index = {'Continue': [], 'Fork_and_Merge': [], 'EndPoint': [], 'Merge': [], 'Start': [],
        #                         'Fork': [], 'EndPoint_and_Merge': [], 'Start_and_Fork': []}
        # for idx, node in enumerate(self.nodes_list):
        #     sum_b0 = np.sum(self.nodes_adj[idx] > 0)
        #     sum_s0 = np.sum(self.nodes_adj[idx] < 0)
        #
        #     if sum_b0 == 0 and sum_s0 == 0:
        #         raise Exception('wrong node')
        #
        #     elif sum_b0 > 1 and sum_s0 > 1:
        #
        #         self.nodes_list[idx].type = 'Fork_and_Merge'
        #         self.node_type_index['Fork_and_Merge'].append(idx)
        #
        #
        #     elif sum_b0 == sum_s0 and sum_b0 == 1:
        #         self.nodes_list[idx].type = 'Continue'
        #         self.node_type_index['Continue'].append(idx)
        #
        #
        #     elif sum_b0 < sum_s0:
        #         if sum_s0 == 1 and sum_b0 == 0:
        #             self.nodes_list[idx].type = 'EndPoint'
        #             self.node_type_index['EndPoint'].append(idx)
        #         elif sum_b0 == 0:
        #             self.nodes_list[idx].type = 'EndPoint_and_Merge'
        #             self.node_type_index['EndPoint_and_Merge'].append(idx)
        #         else:
        #             self.nodes_list[idx].type = 'Merge'
        #             self.node_type_index['Merge'].append(idx)
        #
        #
        #     elif sum_b0 > sum_s0:
        #         if sum_b0 == 1:
        #             self.nodes_list[idx].type = 'Start'
        #             self.node_type_index['Start'].append(idx)
        #         elif sum_s0 == 0:
        #             self.nodes_list[idx].type = 'Start_and_Fork'
        #             self.node_type_index['Start_and_Fork'].append(idx)
        #         else:
        #             self.nodes_list[idx].type = 'Fork'
        #             self.node_type_index['Fork'].append(idx)
        #     else:
        #         raise Exception("Error on type assign!")


class AV2OrderedBzSceneGraph(object):
    def __init__(self, Nodes_list: list, adj: list, nodes_points: list, ncontrol=3):
        self.node_list = Nodes_list
        self.adj = adj
        self.num = len(Nodes_list)
        self.subgraph = [AV2OrderedBzLaneGraph(i, j, k) for (i, j, k) in zip(self.node_list, self.adj, nodes_points)]
        self.ncontrol = ncontrol

    def set_coeff(self, subgraph, subgraphs_points):
        """write coeff in node list
               node.coeff"""
        for i, node in enumerate(subgraph):
            if node.sque_type == 'start':
                continue
            elif node.sque_type == 'continue':
                fin_res = get_bezier_coeff(subgraphs_points[(node.sque_index - 1, node.sque_index)][:, :2],
                                           n_control=self.ncontrol)
            elif node.sque_type == 'fork':
                if (node.sque_index, node.fork_from_index) in subgraphs_points.keys():
                    fin_res = get_bezier_coeff(subgraphs_points[(node.sque_index, node.fork_from_index)][:, :2],
                                               n_control=self.ncontrol)
                else:
                    fin_res = get_bezier_coeff(subgraphs_points[(node.fork_from_index, node.sque_index)][:, :2],
                                               n_control=self.ncontrol)
            elif node.sque_type == 'merge':
                if (node.merge_with_index, node.sque_index) in subgraphs_points.keys():
                    fin_res = get_bezier_coeff(subgraphs_points[(node.merge_with_index, node.sque_index)][:, :2],
                                               n_control=self.ncontrol)
                else:
                    fin_res = get_bezier_coeff(subgraphs_points[(node.sque_index, node.merge_with_index)][:, :2],
                                               n_control=self.ncontrol)
            node.coeff = np.squeeze(fin_res[1:-1])

    def __repr__(self):
        return f"scene graph: {self.num} subgraphs"

    def sort_node_adj(self, subgraph):
        """sort nodelist and adj in each subgraph again to get ordered dfs result"""
        adj = subgraph.nodes_adj
        nodes_list = subgraph.nodes_list
        nodes_points = subgraph.nodes_points
        x_list = [i.position[0] for i in nodes_list]
        x_new = sorted(x_list)
        idx_list_new = [x_list.index(i) for i in x_new]
        idx_list = [x_new.index(i) for i in x_list]
        adj_new = np.zeros([len(nodes_list), len(nodes_list)])
        nodes_points_new = {}
        for k, v in nodes_points.items():
            k_new = (idx_list[k[0]], idx_list[k[1]])
            nodes_points_new[k_new] = v
            adj_new[k_new[0]][k_new[1]] = 1
            adj_new[k_new[1]][k_new[0]] = -1
        subgraph.nodes_points = nodes_points_new
        subgraph.nodes_adj = adj_new

        nodes_list_new = []
        for i in idx_list_new:
            nodes_list_new.append(nodes_list[i])
        subgraph.nodes_list = nodes_list_new

    def sequelize_new(self, orderedDFS=False):
        """"pry search"""
        # sort nodelist and adj in each subgraph again to get ordered dfs result
        if orderedDFS:
            for subgraph in self.subgraph:
                self.sort_node_adj(subgraph)

        for subgraph in self.subgraph:
            subgraph.get_start_nodes_idx_sorted()

        # sort subgraphs by x coordinate of the first start node in each subgraph
        self.subgraphs_sorted = sorted(self.subgraph, key=lambda x: x.first_start_node.position[0])

        result = []
        result_list = []
        for idx, subgraph in enumerate(self.subgraphs_sorted):
            subgraph_scene_sentance, new_subgraphs_points_in_between_nodes = self.subgraph_sequelize(subgraph)
            # scale=10
            # pc_range = np.array([-48., -32., -10.,  48.,  32.,  10.])
            # image = np.zeros((int(pc_range[4] - pc_range[1]) * scale, int(pc_range[3] - pc_range[0]) * scale, 3)).astype(np.uint8)
            # index = 0
            # save_dir = f'vis/gif_av2_centerline_val_48x32_debug3/fbee355f-8878-31fa-8ac8-b9a45a3f130a_315967354759805000/'
            # if not os.path.exists(save_dir):
            #     os.mkdir(save_dir)
            # for node_key in new_subgraphs_points_in_between_nodes:
            #     centerline_points = new_subgraphs_points_in_between_nodes[node_key]
            #     centerline_points = centerline_points - pc_range[:3]
            #     centerline_points = centerline_points[:, :2]
            #     centerline_points = centerline_points * scale
            #     centerline_points = centerline_points.round().astype(np.int32)
            #     color=(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
            #     cv2.polylines(image, [centerline_points], False, color=color, thickness=3)
            #     cv2.putText(image, str(node_key), centerline_points.mean(axis=0).astype(np.int64), cv2.FONT_HERSHEY_SIMPLEX,
            #             0.3, (0, 255, 0), 1, cv2.LINE_AA)
            #     cv2.imwrite(os.path.join(save_dir, f"{index}.png"), image)
            #     index += 1

            self.set_coeff(subgraph_scene_sentance, new_subgraphs_points_in_between_nodes)
            result = result + [(idx, i) for i in subgraph_scene_sentance]  # Add sub graph id
            result_list.append(subgraph_scene_sentance)

        return result, result_list

    def subgraph_sequelize(self, subgraph):
        """pry subgragh search"""
        # subgraph.nodes_list  subgraph.nodes_adj
        nodes = subgraph.nodes_list
        adj = subgraph.nodes_adj
        nodes_points = subgraph.nodes_points

        start_nodes_idx_sorted = subgraph.start_nodes_idx_sorted

        def dfs(index, visited, subgraph_nodes, adj):
            if visited[index]:
                return

            visited[index] = True
            subgraph_nodes.append(index)
            for idx, i in enumerate(adj[index]):
                if adj[index][idx] == 1:
                    dfs(idx, visited, subgraph_nodes, adj)

        if nodes is None or adj is None:
            raise Exception("construction nodes & adj raw first!")

        subgraph_count = 0
        visted = [False for i in nodes]
        # subgraphs_nodes_ = []
        subgraphs_nodes = []
        for idx in start_nodes_idx_sorted:  # dfs every connected graph and save the idx of nodes
            subgraph_nodes = []
            if not visted[idx]:
                subgraph_count += 1
                dfs(idx, visted, subgraph_nodes, adj)
            subgraphs_nodes += subgraph_nodes

        if len(subgraphs_nodes) != len(nodes):
            raise Exception("len(subgraphs_nodes_) != len(nodes)! Check dfs!")

        new_subgraphs_points_in_between_nodes = {}

        sub_nodes = subgraphs_nodes
        subgraph_adj = np.zeros((len(sub_nodes), len(sub_nodes)), dtype=np.int64)
        _list = []
        for idx in sub_nodes:
            _list.append(nodes[idx])
        for i in range(len(sub_nodes) - 1):
            for j in range(i + 1, len(sub_nodes)):
                subgraph_adj[i][j] = adj[sub_nodes[i]][sub_nodes[j]]
                subgraph_adj[j][i] = -subgraph_adj[i][j]
                if subgraph_adj[i][j] == 1:
                    new_subgraphs_points_in_between_nodes[(i, j)] = nodes_points[
                        (sub_nodes[i], sub_nodes[j])]
                if subgraph_adj[i][j] == -1:
                    new_subgraphs_points_in_between_nodes[(j, i)] = nodes_points[
                        (sub_nodes[j], sub_nodes[i])]

        new_subgraphs_nodes = _list
        new_subgraphs_adj = subgraph_adj

        for idx, node in enumerate(new_subgraphs_nodes):
            node.sque_index = idx

        final_subgraph_list = self.get_node_type(new_subgraphs_nodes, new_subgraphs_adj)
        # vis_sub_scenegraph_new(final_subgraph_list, new_subgraphs_points_in_between_nodes)

        return final_subgraph_list, new_subgraphs_points_in_between_nodes

    def get_node_type(self, node, adj):
        for i in range(len(node)):  # start
            if min(adj[i]) > -1:
                node[i].sque_type = 'start'
            # else:
            #     node[i].sque_type = 'continue'
        split_nodes = []

        # new continue and fork
        for i in range(1, len(node)):  # continue
            if adj[i][i - 1] == -1:  # identify continue first
                node[i].sque_type = 'continue'

        for i in range(1, len(node)):  # fork
            father_idx = np.argwhere(adj[i] == -1)

            idx_ = father_idx[np.where(father_idx < i - 1)]
            for idx in idx_:
                if node[i].sque_type == None:
                    node[i].sque_type = "fork"
                    node[i].fork_from_index = idx
                else:  # if it has already been a continue or fork point, split it as a fork point
                    cp_fork = copy.deepcopy(node[i])
                    cp_fork.sque_type = 'fork'
                    cp_fork.fork_from_index = idx
                    split_nodes.append(cp_fork)

            child_idx = np.argwhere(adj[i] == 1)
            idx_ = child_idx[np.where(child_idx < i)]
            for idx in idx_:
                cp_merge = copy.deepcopy(node[i])
                cp_merge.sque_type = 'merge'
                cp_merge.merge_with_index = idx
                split_nodes.append(cp_merge)

        # for i in range(1, len(node)):  # merge
        #     child_idx = np.argwhere(adj[i] == 1)
        #     idx_ = child_idx[np.where(child_idx < i)]
        #     for idx in idx_:
        #         cp_merge = copy.deepcopy(node[i])
        #         cp_merge.sque_type = 'merge'
        #         cp_merge.merge_with_index = idx
        #         split_nodes.append(cp_merge)

        node_new = copy.deepcopy(node)
        for i, split_node in enumerate(split_nodes):
            position = split_node.sque_index + i + 1
            node_new.insert(position, split_node)

        return node_new

    def __len__(self):
        return self.num

    def lane_graph_split(self):
        raise Exception("No Implement!")

    def __getitem__(self, idx):
        return self.subgraph[idx]


class AV2OrderedBzSceneGraph_new(object):
    def __init__(self, Nodes_list: list, adj: list, nodes_points: list, ncontrol=3):
        self.node_list = Nodes_list
        self.adj = adj
        self.num = len(Nodes_list)
        if adj[0].shape[0] == 0:
            self.subgraph = []
        else:
            self.subgraph = [AV2OrderedBzLaneGraph_new(i, j, k) for (i, j, k) in zip(self.node_list, self.adj, nodes_points)]
        self.ncontrol = ncontrol

    def set_coeff(self, subgraph, subgraphs_points):
        """write coeff in node list
               node.coeff"""
        for i, node in enumerate(subgraph):
            if node.sque_type == 'start':
                continue
            elif node.sque_type == 'continue':
                fin_res = get_bezier_coeff(subgraphs_points[(node.sque_index - 1, node.sque_index)][:, :2],
                                           n_control=self.ncontrol)
            elif node.sque_type == 'fork':
                if (node.sque_index, node.fork_from_index) in subgraphs_points.keys():
                    fin_res = get_bezier_coeff(subgraphs_points[(node.sque_index, node.fork_from_index)][:, :2],
                                               n_control=self.ncontrol)
                else:
                    fin_res = get_bezier_coeff(subgraphs_points[(node.fork_from_index, node.sque_index)][:, :2],
                                               n_control=self.ncontrol)
            elif node.sque_type == 'merge':
                if (node.merge_with_index, node.sque_index) in subgraphs_points.keys():
                    fin_res = get_bezier_coeff(subgraphs_points[(node.merge_with_index, node.sque_index)][:, :2],
                                               n_control=self.ncontrol)
                else:
                    fin_res = get_bezier_coeff(subgraphs_points[(node.sque_index, node.merge_with_index)][:, :2],
                                               n_control=self.ncontrol)
            node.coeff = np.squeeze(fin_res[1:-1])

    def __repr__(self):
        return f"scene graph: {self.num} subgraphs"

    # TODO: change
    def sort_node_adj(self, subgraph):
        """sort nodelist and adj in each subgraph again to get ordered dfs result"""
        adj = subgraph.nodes_adj
        nodes_list = subgraph.nodes_list
        nodes_points = subgraph.nodes_points
        x_list = [i.position[0] for i in nodes_list]
        x_new = sorted(x_list)
        idx_list_new = [x_list.index(i) for i in x_new]
        idx_list = [x_new.index(i) for i in x_list]
        adj_new = np.zeros([len(nodes_list), len(nodes_list)])
        nodes_points_new = {}
        for k, v in nodes_points.items():
            k_new = (idx_list[k[0]], idx_list[k[1]])
            nodes_points_new[k_new] = v
            adj_new[k_new[0]][k_new[1]] = 1
            # adj_new[k_new[1]][k_new[0]] = -1
        subgraph.nodes_points = nodes_points_new
        subgraph.nodes_adj = adj_new

        nodes_list_new = []
        for i in idx_list_new:
            nodes_list_new.append(nodes_list[i])
        subgraph.nodes_list = nodes_list_new

    def sequelize_new(self, orderedDFS=False):
        """"pry search"""
        # sort nodelist and adj in each subgraph again to get ordered dfs result
        if orderedDFS:
            for subgraph in self.subgraph:
                self.sort_node_adj(subgraph)

        for subgraph in self.subgraph:
            subgraph.get_start_nodes_idx_sorted()

        # sort subgraphs by x coordinate of the first start node in each subgraph
        self.subgraphs_sorted = sorted(self.subgraph, key=lambda x: x.first_start_node.position[0])

        result = []
        result_list = []
        for idx, subgraph in enumerate(self.subgraphs_sorted):
            subgraph_scene_sentance, new_subgraphs_points_in_between_nodes = self.subgraph_sequelize(subgraph)
            # scale=10
            # pc_range = np.array([-48., -32., -10.,  48.,  32.,  10.])
            # image = np.zeros((int(pc_range[4] - pc_range[1]) * scale, int(pc_range[3] - pc_range[0]) * scale, 3)).astype(np.uint8)
            # index = 0
            # save_dir = f'vis/gif_av2_centerline_val_48x32_debug3/fbee355f-8878-31fa-8ac8-b9a45a3f130a_315967354759805000/'
            # if not os.path.exists(save_dir):
            #     os.mkdir(save_dir)
            # for node_key in new_subgraphs_points_in_between_nodes:
            #     centerline_points = new_subgraphs_points_in_between_nodes[node_key]
            #     centerline_points = centerline_points - pc_range[:3]
            #     centerline_points = centerline_points[:, :2]
            #     centerline_points = centerline_points * scale
            #     centerline_points = centerline_points.round().astype(np.int32)
            #     color=(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
            #     cv2.polylines(image, [centerline_points], False, color=color, thickness=3)
            #     cv2.putText(image, str(node_key), centerline_points.mean(axis=0).astype(np.int64), cv2.FONT_HERSHEY_SIMPLEX,
            #             0.3, (0, 255, 0), 1, cv2.LINE_AA)
            #     cv2.imwrite(os.path.join(save_dir, f"{index}.png"), image)
            #     index += 1

            self.set_coeff(subgraph_scene_sentance, new_subgraphs_points_in_between_nodes)
            result = result + [(idx, i) for i in subgraph_scene_sentance]  # Add sub graph id
            result_list.append(subgraph_scene_sentance)

        return result, result_list

    def subgraph_sequelize(self, subgraph):
        """pry subgragh search"""
        # subgraph.nodes_list  subgraph.nodes_adj
        nodes = subgraph.nodes_list
        adj = subgraph.nodes_adj
        nodes_points = subgraph.nodes_points

        start_nodes_idx_sorted = subgraph.start_nodes_idx_sorted

        def dfs(index, visited, subgraph_nodes, adj):
            if visited[index]:
                return

            visited[index] = True
            subgraph_nodes.append(index)
            for idx, i in enumerate(adj[index]):
                if adj[index][idx] == 1:
                    dfs(idx, visited, subgraph_nodes, adj)

        if nodes is None or adj is None:
            raise Exception("construction nodes & adj raw first!")

        subgraph_count = 0
        visted = [False for i in nodes]
        # subgraphs_nodes_ = []
        subgraphs_nodes = []
        for idx in start_nodes_idx_sorted:  # dfs every connected graph and save the idx of nodes
            subgraph_nodes = []
            if not visted[idx]:
                subgraph_count += 1
                dfs(idx, visted, subgraph_nodes, adj)
            subgraphs_nodes += subgraph_nodes

        if len(subgraphs_nodes) != len(nodes):
            raise Exception("len(subgraphs_nodes_) != len(nodes)! Check dfs!")

        new_subgraphs_points_in_between_nodes = {}

        sub_nodes = subgraphs_nodes
        subgraph_adj = np.zeros((len(sub_nodes), len(sub_nodes)), dtype=np.int64)
        _list = []
        for idx in sub_nodes:
            _list.append(nodes[idx])
        for i in range(len(sub_nodes)):
            for j in range(len(sub_nodes)):
                subgraph_adj[i][j] = adj[sub_nodes[i]][sub_nodes[j]]
                # subgraph_adj[j][i] = -subgraph_adj[i][j]
                if subgraph_adj[i][j] == 1:
                    new_subgraphs_points_in_between_nodes[(i, j)] = nodes_points[
                        (sub_nodes[i], sub_nodes[j])]
                # if subgraph_adj[i][j] == -1:
                #     new_subgraphs_points_in_between_nodes[(j, i)] = nodes_points[
                #         (sub_nodes[j], sub_nodes[i])]

        new_subgraphs_nodes = _list
        new_subgraphs_adj = subgraph_adj

        for idx, node in enumerate(new_subgraphs_nodes):
            node.sque_index = idx

        final_subgraph_list = self.get_node_type(new_subgraphs_nodes, new_subgraphs_adj)
        # vis_sub_scenegraph_new(final_subgraph_list, new_subgraphs_points_in_between_nodes)

        return final_subgraph_list, new_subgraphs_points_in_between_nodes

    def get_node_type(self, node, adj):

        for i in range(len(node)):  # start
            if max(adj[:,i]) < 1:
                node[i].sque_type = 'start'
            elif sum(adj[:, i])==1 and sum(adj[i])>=1:
                node[i].sque_type = 'start'
            # else:
            #     node[i].sque_type = 'continue'
        split_nodes = []

        # new continue and fork
        for i in range(1, len(node)):  # continue
            if adj[i-1][i] == 1:  # identify continue first
                node[i].sque_type = 'continue'

        for i in range(1, len(node)):  # fork
            father_idx = np.argwhere(adj[:,i] == 1)

            idx_ = father_idx[np.where(father_idx < i - 1)]
            for idx in idx_:
                if node[i].sque_type == None:
                    node[i].sque_type = "fork"
                    node[i].fork_from_index = idx
                else:  # if it has already been a continue or fork point, split it as a fork point
                    cp_fork = copy.deepcopy(node[i])
                    cp_fork.sque_type = 'fork'
                    cp_fork.fork_from_index = idx
                    split_nodes.append(cp_fork)

            child_idx = np.argwhere(adj[i] == 1)
            idx_ = child_idx[np.where(child_idx < i)]
            for idx in idx_:
                cp_merge = copy.deepcopy(node[i])
                cp_merge.sque_type = 'merge'
                cp_merge.merge_with_index = idx
                split_nodes.append(cp_merge)

        # for i in range(1, len(node)):  # merge
        #     child_idx = np.argwhere(adj[i] == 1)
        #     idx_ = child_idx[np.where(child_idx < i)]
        #     for idx in idx_:
        #         cp_merge = copy.deepcopy(node[i])
        #         cp_merge.sque_type = 'merge'
        #         cp_merge.merge_with_index = idx
        #         split_nodes.append(cp_merge)

        node_new = copy.deepcopy(node)
        for i, split_node in enumerate(split_nodes):
            position = split_node.sque_index + i + 1
            node_new.insert(position, split_node)

        return node_new

    def __len__(self):
        return self.num

    def lane_graph_split(self):
        raise Exception("No Implement!")

    def __getitem__(self, idx):
        return self.subgraph[idx]


class AV2OrederedBzCenterLine(NusOrederedBzCenterLine):
    def __init__(self, centerlines, token, grid_conf, bz_grid_conf, epsilon=1e-5):
        self.token = token

        self.centerlines = copy.deepcopy(centerlines['centerlines'])
        self.centerlines=[np.array(i) for i in self.centerlines]
        self.coeff = copy.deepcopy(centerlines)
        self.start_point_idxs = [0 for i in self.centerlines]
        self.end_point_idxs = [len(centerline)-1 for centerline in self.centerlines]
        self.all_nodes = None
        self.adj = None
        self.subgraphs_nodes = None
        self.points_in_between_nodes = None
        self.grid_conf = grid_conf
        self.bz_grid_conf = bz_grid_conf
        self.epsilon=epsilon
        dx, bx, nx = self.gen_dx_bx(self.grid_conf['xbound'],
                                    self.grid_conf['ybound'],
                                    self.grid_conf['zbound'],)
        self.dx = dx
        self.bx = bx
        self.nx = nx
        self.pc_range = np.concatenate((self.bx - self.dx / 2., self.bx - self.dx / 2. + self.nx * self.dx))

        bz_dx, bz_bx, bz_nx = self.gen_dx_bx(self.bz_grid_conf['xbound'],
                                    self.bz_grid_conf['ybound'],
                                    self.bz_grid_conf['zbound'],)
        self.bz_dx = bz_dx
        self.bz_bx = bz_bx
        self.bz_nx = bz_nx
        self.bz_pc_range = np.concatenate((bz_bx - bz_dx / 2., bz_bx - bz_dx / 2. + bz_nx * bz_dx))
        self.filter_bev()
    
    @staticmethod
    def gen_dx_bx(xbound, ybound, zbound):
        dx = np.array([row[2] for row in [xbound, ybound, zbound]])
        bx = np.array([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])
        nx = np.floor(np.array([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]]))
        return dx, bx, nx
    


    def filter_bev(self):
        aug_centerlines = []
        aug_start_point_idxs = []
        aug_end_point_idxs = []

        for i in range(len(self.centerlines)):
            centerline = self.centerlines[i]
            in_bev_x = np.logical_and(centerline[:, 0] < self.pc_range[3], centerline[:, 0] >= self.pc_range[0])
            in_bev_y = np.logical_and(centerline[:, 1] <= self.pc_range[4], centerline[:, 1] >= self.pc_range[1])
            in_bev_xy = np.logical_and(in_bev_x, in_bev_y)
            if not np.max(in_bev_xy):
                continue

            if np.min(in_bev_xy):
                aug_centerlines.append(centerline)
                aug_start_point_idxs.append(self.start_point_idxs[i])
                aug_end_point_idxs.append(self.end_point_idxs[i])

                continue
            segments=self.find_true_segments(in_bev_xy)
            for segment in segments:
                start,end=segment[0],segment[1]
                if end<=start:
                    continue
                new_in_bev_xy=np.zeros_like(in_bev_xy,dtype=bool)
                new_in_bev_xy[start:end+1]=True
            # start_point_idx = self.start_point_idxs[i]
            # end_point_idx = self.end_point_idxs[i]
            # aug_start_point = centerline[start_point_idx]
            # aug_end_point = centerline[end_point_idx]
            # aug_centerline = centerline[in_bev_xy,:]
            # aug_idxs = idxs[in_bev_xy]
            # if not start_point_idx in aug_idxs:
            #     aug_start_point = aug_centerline[0]
            # if not end_point_idx in aug_idxs:
            #     aug_end_point = aug_centerline[-1]
            # start_distance = np.linalg.norm(aug_centerline - aug_start_point, ord=2, axis=1)
            # start_point_idx = np.argmin(start_distance)
            # end_distance = np.linalg.norm(aug_centerline - aug_end_point, ord=2, axis=1)
            # end_point_idx = np.argmin(end_distance)
            
                aug_centerline = centerline[new_in_bev_xy,:]
                start_point_idx = 0
                end_point_idx = len(aug_centerline) - 1

                aug_centerlines.append(aug_centerline)
                aug_start_point_idxs.append(start_point_idx)
                aug_end_point_idxs.append(end_point_idx)

        self.centerlines = aug_centerlines
        self.start_point_idxs = aug_start_point_idxs
        self.end_point_idxs = aug_end_point_idxs



    def export_node_adj(self):
        # self.construct_nodes_adj_raw()
        self.construct_nodes_adj_raw_and_raw_points()  # self.adj_raw.shape:[27,27]  len(self.raw_points_in_between.keys()):27
        self.nodes_merge()
        # self.draw_graph()
        return self.all_nodes, self.adj

    def construct_nodes_adj_raw_and_raw_points(self):
        '''
        self.adj_raw : node[i]-->node[j], adj_raw[i][j]=1, adj_raw[j][i]=-1
        '''
        self.all_nodes_raw = []
        self.raw_points_in_between = {}
        self.adj_raw = np.zeros((2 * len(self.centerlines), 2 * len(self.centerlines)), dtype=np.int8)
        for idx, centerline in enumerate(self.centerlines):
            self.all_nodes_raw.append(AV2BzNode(centerline[self.start_point_idxs[idx]]))
            self.all_nodes_raw.append(AV2BzNode(centerline[self.end_point_idxs[idx]]))
            self.adj_raw[2 * idx, 2 * idx + 1] = 1
            self.adj_raw[2 * idx + 1, 2 * idx] = -1
            self.raw_points_in_between[(2 * idx, 2 * idx + 1)] = centerline[
                                                                 self.start_point_idxs[idx]:self.end_point_idxs[idx]+1]

class AV2OrederedRMcontinuedBzCenterLine(AV2OrederedBzCenterLine):
    def __init__(self, centerlines, token, grid_conf, bz_grid_conf, epsilon=1e-5):
        super(AV2OrederedRMcontinuedBzCenterLine, self).__init__(centerlines, token, grid_conf, bz_grid_conf, epsilon)

    def export_node_adj(self):
        # self.construct_nodes_adj_raw()
        self.construct_nodes_adj_raw_and_raw_points()  # self.adj_raw.shape:[27,27]  len(self.raw_points_in_between.keys()):27
        self.nodes_merge()
        # self.clear_straight_continue_nodes()
        self.clear_continue_nodes()
        self.draw_graph()
        
        return self.all_nodes, self.adj  
    
    def clear_continue_nodes(self):
        # =============================================
        # continue点之间直接拉成一条线
        # =============================================
        i=0
        while i <len(self.adj): 
            if sum (self.adj[i]==1) == 1 and sum (self.adj[i]==-1) == 1:
                to_node=np.where(self.adj[i]==1)[0][0]
                from_node=np.where(self.adj[i]==-1)[0][0]
                if not (isinstance(self.points_in_between_nodes[(from_node,i)],tuple) or isinstance(self.points_in_between_nodes[(i,to_node)],tuple)):
                        

                    if ( from_node,to_node) in self.points_in_between_nodes:
                        if isinstance(self.points_in_between_nodes[( from_node,to_node)],tuple):
                            self.points_in_between_nodes[( from_node,to_node)]=*self.points_in_between_nodes[( from_node,to_node)],np.vstack([self.points_in_between_nodes.pop((from_node,i))[:-1,:],self.points_in_between_nodes.pop((i,to_node))])
                        else:
                            self.points_in_between_nodes[( from_node,to_node)]=self.points_in_between_nodes[( from_node,to_node)],np.vstack([self.points_in_between_nodes.pop((from_node,i))[:-1,:],self.points_in_between_nodes.pop((i,to_node))])
        
                    else:
                        self.points_in_between_nodes[( from_node,to_node)]=np.vstack([self.points_in_between_nodes.pop((from_node,i))[:-1,:],self.points_in_between_nodes.pop((i,to_node))])
                    self.adj[from_node][to_node]=1
                    self.adj[to_node][from_node]=-1
                    for idx_i in range(len(self.adj)):
                        for idx_j in range(len(self.adj)):
                            if (idx_i>i or idx_j>i) and (idx_i,idx_j) in self.points_in_between_nodes:
                                if idx_i>i:
                                    new_idx_i=idx_i-1
                                else:  
                                    new_idx_i=idx_i
                                if idx_j>i:
                                    new_idx_j=idx_j-1
                                else:
                                    new_idx_j=idx_j
                                self.points_in_between_nodes[(new_idx_i,new_idx_j)]=self.points_in_between_nodes.pop((idx_i,idx_j))
                    self.adj = np.delete(self.adj, i, axis=0)
                    self.adj = np.delete(self.adj, i, axis=1)
                    self.all_nodes=self.all_nodes[:i]+self.all_nodes[i+1:]
                    continue
           
          
            i+=1
    
class AV2OrederedBzCenterLine_new(object):
    def __init__(self, centerlines, token, grid_conf, bz_grid_conf, epsilon=1e-5):
        self.token = token
        self.centerline_ids = copy.deepcopy(centerlines['centerline_ids'])
        self.incoming_ids = copy.deepcopy(centerlines['incoming_ids'])
        self.outgoing_ids = copy.deepcopy(centerlines['outgoing_ids'])
        self.centerlines = copy.deepcopy(centerlines['centerlines'])
        self.coeff = copy.deepcopy(centerlines)
        self.start_point_idxs = [0 for i in self.centerlines]
        self.end_point_idxs = [len(centerline) - 1 for centerline in self.centerlines]
        self.all_nodes = None
        self.adj = None
        self.subgraphs_nodes = None
        self.points_in_between_nodes = None
        self.grid_conf = grid_conf
        self.bz_grid_conf = bz_grid_conf
        self.epsilon = epsilon
        dx, bx, nx = self.gen_dx_bx(self.grid_conf['xbound'],
                                    self.grid_conf['ybound'],
                                    self.grid_conf['zbound'], )
        self.dx = dx
        self.bx = bx
        self.nx = nx
        self.pc_range = np.concatenate((self.bx - self.dx / 2., self.bx - self.dx / 2. + self.nx * self.dx))

        bz_dx, bz_bx, bz_nx = self.gen_dx_bx(self.bz_grid_conf['xbound'],
                                             self.bz_grid_conf['ybound'],
                                             self.bz_grid_conf['zbound'], )
        self.bz_dx = bz_dx
        self.bz_bx = bz_bx
        self.bz_nx = bz_nx
        self.bz_pc_range = np.concatenate((bz_bx - bz_dx / 2., bz_bx - bz_dx / 2. + bz_nx * bz_dx))
        self.filter_bev()

    @staticmethod
    def gen_dx_bx(xbound, ybound, zbound):
        dx = np.array([row[2] for row in [xbound, ybound, zbound]])
        bx = np.array([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])
        nx = np.floor(np.array([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]]))
        return dx, bx, nx

    def flip(self, type):
        if type not in ['horizontal', 'vertical']:
            return
        aug_centerlines = []
        for i in range(len(self.centerlines)):
            centerline = self.centerlines[i]
            if type == 'horizontal':
                centerline[:, 0] = -centerline[:, 0]
            else:
                centerline[:, 1] = -centerline[:, 1]
            aug_centerlines.append(centerline)
        self.centerlines = aug_centerlines

    def scale(self, scale_ratio):
        scaling_matrix = self._get_scaling_matrix(scale_ratio)
        aug_centerlines = []
        for i in range(len(self.centerlines)):
            centerline = self.centerlines[i]
            aug_centerline = centerline @ scaling_matrix.T
            aug_centerlines.append(aug_centerline)
        self.centerlines = aug_centerlines

    def rotate(self, rotation_matrix):
        aug_centerlines = []
        for i in range(len(self.centerlines)):
            centerline = self.centerlines[i]
            aug_centerline = centerline @ rotation_matrix.T
            aug_centerlines.append(aug_centerline)
        self.centerlines = aug_centerlines

    def filter_bev(self):
        aug_centerlines = []
        aug_centerline_ids = []
        aug_start_point_idxs = []
        aug_end_point_idxs = []
        aug_incoming_ids = []
        aug_outgoing_ids = []
        existing_ids = set()
        for i in range(len(self.centerlines)):
            centerline = self.centerlines[i]
            aug_id = self.centerline_ids[i]
            in_bev_x = np.logical_and(centerline[:, 0] < self.pc_range[3], centerline[:, 0] >= self.pc_range[0])
            in_bev_y = np.logical_and(centerline[:, 1] <= self.pc_range[4], centerline[:, 1] >= self.pc_range[1])
            in_bev_xy = np.logical_and(in_bev_x, in_bev_y)
            if not np.max(in_bev_xy):
                continue
            existing_ids.add(aug_id)
            if np.min(in_bev_xy):
                aug_centerlines.append(centerline)
                aug_centerline_ids.append(self.centerline_ids[i])
                aug_start_point_idxs.append(self.start_point_idxs[i])
                aug_end_point_idxs.append(self.end_point_idxs[i])
                aug_incoming_ids.append(self.incoming_ids[i])
                aug_outgoing_ids.append(self.outgoing_ids[i])
                continue

            # start_point_idx = self.start_point_idxs[i]
            # end_point_idx = self.end_point_idxs[i]
            # aug_start_point = centerline[start_point_idx]
            # aug_end_point = centerline[end_point_idx]
            # aug_centerline = centerline[in_bev_xy,:]
            # aug_idxs = idxs[in_bev_xy]
            # if not start_point_idx in aug_idxs:
            #     aug_start_point = aug_centerline[0]
            # if not end_point_idx in aug_idxs:
            #     aug_end_point = aug_centerline[-1]
            # start_distance = np.linalg.norm(aug_centerline - aug_start_point, ord=2, axis=1)
            # start_point_idx = np.argmin(start_distance)
            # end_distance = np.linalg.norm(aug_centerline - aug_end_point, ord=2, axis=1)
            # end_point_idx = np.argmin(end_distance)

            aug_centerline = centerline[in_bev_xy, :]
            start_point_idx = 0
            end_point_idx = len(aug_centerline) - 1

            aug_centerlines.append(aug_centerline)
            aug_centerline_ids.append(self.centerline_ids[i])
            aug_start_point_idxs.append(start_point_idx)
            aug_end_point_idxs.append(end_point_idx)
            aug_incoming_ids.append(self.incoming_ids[i])
            aug_outgoing_ids.append(self.outgoing_ids[i])
        for i in range(len(aug_centerlines)):
            incoming_id = aug_incoming_ids[i]
            aug_incoming_id = []
            for aug_id in incoming_id:
                if aug_id in existing_ids:
                    aug_incoming_id.append(aug_id)
            aug_incoming_ids[i] = aug_incoming_id

            outgoing_id = aug_outgoing_ids[i]
            aug_outgoing_id = []
            for aug_id in outgoing_id:
                if aug_id in existing_ids:
                    aug_outgoing_id.append(aug_id)
            aug_outgoing_ids[i] = aug_outgoing_id
        self.centerlines = aug_centerlines
        self.centerline_ids = aug_centerline_ids
        self.incoming_ids = aug_incoming_ids
        self.outgoing_ids = aug_outgoing_ids
        self.start_point_idxs = aug_start_point_idxs
        self.end_point_idxs = aug_end_point_idxs

        # scale=10
        # image = np.zeros((int(self.pc_range[4] - self.pc_range[1]) * scale, int(self.pc_range[3] - self.pc_range[0]) * scale, 3)).astype(np.uint8)
        # for i, centerline_points in enumerate(self.centerlines):
        #     centerline_points = centerline_points - self.pc_range[:3]
        #     centerline_points = centerline_points[:, :2]
        #     centerline_points = centerline_points * scale
        #     centerline_points = centerline_points.round().astype(np.int32)
        #     color=(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
        #     cv2.polylines(image, [centerline_points], False, color=color, thickness=3)
        #     cv2.putText(image, str(aug_centerline_ids[i]), centerline_points.mean(axis=0).astype(np.int64), cv2.FONT_HERSHEY_SIMPLEX,
        #             0.3, (0, 255, 0), 1, cv2.LINE_AA)
        # cv2.imwrite(f'vis/av2_centerline_val_48x32/{self.token}.png', image)

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

    def sub_graph_split(self):

        def dfs(index, visited, subgraph_nodes, adj):
            if visited[index]:
                return

            visited[index] = True
            subgraph_nodes.append(index)
            for idx, i in enumerate(adj[index]):
                if adj[index][idx] == 1:
                    dfs(idx, visited, subgraph_nodes, adj)

        if self.all_nodes is None or self.adj is None:
            raise Exception("construction nodes & adj raw first!")

        subgraph_count = 0
        visted = [False for i in self.all_nodes]
        subgraphs_nodes_ = []
        subgraphs_nodes = []
        for idx, node in enumerate(self.all_nodes):  # dfs every connected graph and save the idx of nodes
            subgraph_nodes = []
            if not visted[idx]:
                subgraph_count += 1
                dfs(idx, visted, subgraph_nodes, self.adj)
            subgraphs_nodes_.append(subgraph_nodes)

        for subgraph_node in subgraphs_nodes_:  # delete empty lists
            if len(subgraph_node) <= 1:
                continue
            else:
                subgraphs_nodes.append(subgraph_node)
        self.subgraphs_nodes = []
        self.subgraphs_adj = []
        self.subgraphs_points_in_between_nodes = [{} for i in subgraphs_nodes]
        for idx_, sub_nodes in enumerate(subgraphs_nodes):
            _list = []
            if len(sub_nodes) == 0:
                continue
            subgraph_adj = np.zeros((len(sub_nodes), len(sub_nodes)), dtype=np.int64)
            for idx in sub_nodes:
                _list.append(self.all_nodes[idx])
            for i in range(len(sub_nodes) - 1):
                for j in range(i + 1, len(sub_nodes)):
                    subgraph_adj[i][j] = self.adj[sub_nodes[i]][sub_nodes[j]]
                    subgraph_adj[j][i] = -subgraph_adj[i][j]
                    if subgraph_adj[i][j] == 1:
                        self.subgraphs_points_in_between_nodes[idx_][(i, j)] = self.points_in_between_nodes[
                            (sub_nodes[i], sub_nodes[j])]
                    if subgraph_adj[i][j] == -1:
                        self.subgraphs_points_in_between_nodes[idx_][(j, i)] = self.points_in_between_nodes[
                            (sub_nodes[j], sub_nodes[i])]

            self.subgraphs_nodes.append(_list)
            self.subgraphs_adj.append(subgraph_adj)

        # scale=10
        # image = np.zeros((int(self.pc_range[4] - self.pc_range[1]) * scale, int(self.pc_range[3] - self.pc_range[0]) * scale, 3)).astype(np.uint8)
        # index = 0
        # save_dir = f'vis/gif_av2_centerline_val_48x32_debug2/{self.token}/'
        # if not os.path.exists(save_dir):
        #     os.mkdir(save_dir)
        # for node_key in self.subgraphs_points_in_between_nodes[0]:
        #     centerline_points = self.subgraphs_points_in_between_nodes[0][node_key]
        #     centerline_points = centerline_points - self.pc_range[:3]
        #     centerline_points = centerline_points[:, :2]
        #     centerline_points = centerline_points * scale
        #     centerline_points = centerline_points.round().astype(np.int32)
        #     color=(np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
        #     cv2.polylines(image, [centerline_points], False, color=color, thickness=3)
        #     cv2.putText(image, str(node_key), centerline_points.mean(axis=0).astype(np.int64), cv2.FONT_HERSHEY_SIMPLEX,
        #             0.3, (0, 255, 0), 1, cv2.LINE_AA)
        #     cv2.imwrite(os.path.join(save_dir, f"{index}.png"), image)
        #     index += 1
        # import pdb;pdb.set_trace()

    def export_node_adj(self):
        self.construct_nodes_adj_raw_and_raw_points()  # self.adj_raw.shape:[27,27]  len(self.raw_points_in_between.keys()):27
        self.nodes_merge()

        return self.all_nodes, self.adj


    def construct_nodes_adj_raw_and_raw_points(self):
        '''
        self.adj_raw : node[i]-->node[j], adj_raw[i][j]=1, adj_raw[j][i]=-1
        '''
        self.all_nodes_raw = []
        self.raw_points_in_between = {}
        self.adj_raw = np.zeros((2 * len(self.centerlines), 2 * len(self.centerlines)), dtype=np.int8)
        for idx, centerline in enumerate(self.centerlines):
            self.all_nodes_raw.append(AV2BzNode(centerline[self.start_point_idxs[idx]], epsilon=self.epsilon))
            self.all_nodes_raw.append(AV2BzNode(centerline[self.end_point_idxs[idx]], epsilon=self.epsilon))
            self.adj_raw[2 * idx, 2 * idx + 1] = 1
            self.adj_raw[2 * idx + 1, 2 * idx] = -1
            self.raw_points_in_between[(2 * idx, 2 * idx + 1)] = centerline[
                                                                 self.start_point_idxs[idx]:self.end_point_idxs[
                                                                                                idx] + 1]

    def __if_start_lane(self, index):
        raise Exception("No Implemention")

    def __if_end_lane(self, index):
        raise Exception("No Implemention")

    def nodes_merge(self):
        '''
        merge same nodes in node list and adjcent matrix
        '''
        self.all_nodes = []
        nodes_raw_nodes_map = [None for i in self.all_nodes_raw]  # 54
        all_nodes_index = []
        picked_raw_nodes = []
        for idx, node in enumerate(self.all_nodes_raw):
            if idx in picked_raw_nodes:
                continue
            self.all_nodes.append(self.all_nodes_raw[idx])
            all_nodes_index.append(idx)
            nodes_raw_nodes_map[idx] = idx
            picked_raw_nodes.append(idx)
            for idx_j in range(idx + 1, len(self.all_nodes_raw)):
                if self.all_nodes_raw[idx] == self.all_nodes_raw[idx_j]:
                    picked_raw_nodes.append(idx_j)
                    nodes_raw_nodes_map[idx_j] = idx

        # len: self.all_nodes 22
        nodes_raw_nodes_map = np.array(nodes_raw_nodes_map, dtype=np.int64)
        nodes_raw_nodes_index_map = []
        for idx in range(len(nodes_raw_nodes_map)):
            nodes_raw_nodes_index_map.append(all_nodes_index.index(nodes_raw_nodes_map[idx]))
        nodes_raw_nodes_index_map = np.array(nodes_raw_nodes_index_map, dtype=np.int64)
        ## map raw points in between
        self.points_in_between_nodes = {}
        for i, j in self.raw_points_in_between:
            if nodes_raw_nodes_index_map[i] == nodes_raw_nodes_index_map[j]:
                continue
            self.points_in_between_nodes[(nodes_raw_nodes_index_map[i], nodes_raw_nodes_index_map[j])] = \
                self.raw_points_in_between[(i, j)]

        self.adj = np.zeros((len(self.all_nodes), len(self.all_nodes)), dtype=np.int64)

        for i, j in self.points_in_between_nodes.keys():
            self.adj[i][j] = 1
            # self.adj[j][i] = -1

