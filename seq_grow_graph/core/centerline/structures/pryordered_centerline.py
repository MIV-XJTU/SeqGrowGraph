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

class Node(object):
    def __init__(self, position):
        self.parents = []
        self.children = []
        self.position = position
        self.type = None
        # self.coeff = []
        self.node_index = None
        self.sque_type = None
        self.merge_with_index = None
        self.fork_from_index = None
        self.sque_index = None
        self.sque_points = None

    def set_parents(self, parents):
        self.parents = parents

    def set_children(self, children):
        self.children = children

    # def set_coeff(self, coeff):
    #     self.coeff = coeff

    def set_type(self, type_):
        self.type = type_

    def __repr__(self):
        return f"Node_sque_index : {self.sque_index}, Node_type : {self.type}, sque_type : {self.sque_type}, fork_from : {self.fork_from_index}, merge with : {self.merge_with_index}, coord : {self.position}\n"

    def __eq__(self, __o):
        if np.linalg.norm(np.array(self.position) - np.array(__o.position)) < 2.1:
            return True
        return False


class OrderedLaneGraph(object):
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


class OrderedSceneGraph(object):
    def __init__(self, Nodes_list: list, adj: list, nodes_points: list):
        self.node_list = Nodes_list
        self.adj = adj
        self.num = len(Nodes_list)
        self.subgraph = [OrderedLaneGraph(i, j, k) for (i, j, k) in zip(self.node_list, self.adj, nodes_points)]

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
        # pdb.set_trace()
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
        # # sort subgraphs by x coordinate of the first start node in each subgraph
        # self.subgraphs_sorted = sorted(self.subgraph, key=lambda x: x.first_start_node.position[0])
        #
        # # sort nodelist and adj in each subgraph again to get ordered dfs result
        # if orderedDFS:
        #     for subgraph in self.subgraphs_sorted:
        #         self.sort_node_adj(subgraph)

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
        subgraph_adj = np.zeros((len(sub_nodes), len(sub_nodes)), dtype=np.int)
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
            if len(father_idx) == 0 :
                continue
            idx_ = father_idx[np.where(father_idx < i-1)]
            for idx in idx_:
                if node[i].sque_type == None:
                    node[i].sque_type = "fork"
                    node[i].fork_from_index = idx
                else:  # if it has already been a continue or fork point, split it as a fork point
                    cp_fork = copy.deepcopy(node[i])
                    cp_fork.sque_type = 'fork'
                    cp_fork.fork_from_index = idx
                    split_nodes.append(cp_fork)


        for i in range(1, len(node)):  # merge
            child_idx = np.argwhere(adj[i] == 1)
            idx_ = child_idx[np.where(child_idx < i)]
            for idx in idx_:
                cp_merge = copy.deepcopy(node[i])
                cp_merge.sque_type = 'merge'
                cp_merge.merge_with_index = idx
                split_nodes.append(cp_merge)

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


class PryOrederedCenterLine(object):
    def __init__(self, centerlines, grid_conf):
        self.types = copy.deepcopy(centerlines['type'])
        self.centerline_ids = copy.deepcopy(centerlines['centerline_ids'])
        self.incoming_ids = copy.deepcopy(centerlines['incoming_ids'])
        self.outgoing_ids = copy.deepcopy(centerlines['outgoing_ids'])
        self.start_point_idxs = copy.deepcopy(centerlines['start_point_idxs'])  # 问题出在这里 有三条中心线 start_idx==end_idx 所以和segmentation对不上
        self.end_point_idxs = copy.deepcopy(centerlines['end_point_idxs'])
        self.centerlines = copy.deepcopy(centerlines['centerlines'])
        # self.coeff = results['centerlines']
        self.all_nodes = None
        self.adj = None
        self.subgraphs_nodes = None
        self.points_in_between_nodes = None
        self.grid_conf = grid_conf
        dx, bx, nx = self.gen_dx_bx(self.grid_conf['xbound'],
                                    self.grid_conf['ybound'],
                                    self.grid_conf['zbound'],)
        self.dx = dx
        self.bx = bx
        self.nx = nx
        self.pc_range = np.concatenate((self.bx - self.dx / 2., self.bx - self.dx / 2. + self.nx * self.dx))
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
                centerline[:,0] = -centerline[:,0]
            else:
                centerline[:,1] = -centerline[:,1]
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
        aug_types = []
        aug_centerlines = []
        aug_centerline_ids = []
        aug_start_point_idxs = []
        aug_end_point_idxs = []
        aug_incoming_ids = []
        aug_outgoing_ids = []
        for i in range(len(self.centerlines)):
            centerline = self.centerlines[i]
            idxs = np.arange(len(centerline))
            in_bev_x = np.logical_and(centerline[:, 0] < self.pc_range[3], centerline[:, 0] >= self.pc_range[0])
            in_bev_y = np.logical_and(centerline[:, 1] <= self.pc_range[4], centerline[:, 1] >= self.pc_range[1])
            in_bev_xy = np.logical_and(in_bev_x, in_bev_y)
            if not np.max(in_bev_xy):
                continue
            if np.min(in_bev_xy):
                aug_types.append(self.types[i])
                aug_centerlines.append(centerline)
                aug_centerline_ids.append(self.centerline_ids[i])
                aug_start_point_idxs.append(self.start_point_idxs[i])
                aug_end_point_idxs.append(self.end_point_idxs[i])
                aug_incoming_ids.append(self.incoming_ids[i])
                aug_outgoing_ids.append(self.outgoing_ids[i])
                continue

            start_point_idx = self.start_point_idxs[i]
            end_point_idx = self.end_point_idxs[i]
            aug_start_point = centerline[start_point_idx]
            aug_end_point = centerline[end_point_idx]
            aug_centerline = centerline[in_bev_xy,:]
            aug_idxs = idxs[in_bev_xy]
            if not start_point_idx in aug_idxs:
                aug_start_point = aug_centerline[0]
            if not end_point_idx in aug_idxs:
                aug_end_point = aug_centerline[-1]
            start_distance = np.linalg.norm(aug_centerline - aug_start_point, ord=2, axis=1)
            start_point_idx = np.argmin(start_distance)
            end_distance = np.linalg.norm(aug_centerline - aug_end_point, ord=2, axis=1)
            end_point_idx = np.argmin(end_distance)
            
            aug_types.append(self.types[i])
            aug_centerlines.append(aug_centerline)
            aug_centerline_ids.append(self.centerline_ids[i])
            aug_start_point_idxs.append(start_point_idx)
            aug_end_point_idxs.append(end_point_idx)
            aug_incoming_ids.append(self.incoming_ids[i])
            aug_outgoing_ids.append(self.outgoing_ids[i])
        self.types = aug_types
        self.centerlines = aug_centerlines
        self.centerline_ids = aug_centerline_ids
        self.incoming_ids = aug_incoming_ids
        self.outgoing_ids = aug_outgoing_ids
        self.start_point_idxs = aug_start_point_idxs
        self.end_point_idxs = aug_end_point_idxs

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
                if adj[index][idx] == 1 or adj[index][idx] == -1:
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
            subgraph_adj = np.zeros((len(sub_nodes), len(sub_nodes)), dtype=np.int)
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

    def export_node_adj(self):
        # self.construct_nodes_adj_raw()
        self.construct_nodes_adj_raw_and_raw_points()  # self.adj_raw.shape:[27,27]  len(self.raw_points_in_between.keys()):27
        self.nodes_merge()

        return self.all_nodes, self.adj

    def construct_nodes_adj_raw(self):
        '''
        self.adj_raw : node[i]-->node[j], adj_raw[i][j]=1, adj_raw[j][i]=-1
        '''
        self.all_nodes_raw = []
        self.adj_raw = np.zeros((2 * len(self.centerlines), 2 * len(self.centerlines)), dtype=np.int8)
        for idx, centerline in enumerate(self.centerlines):
            self.all_nodes_raw.append(Node(centerline[self.start_point_idxs[idx]]))
            self.all_nodes_raw.append(Node(centerline[self.end_point_idxs[idx]]))
            self.adj_raw[2 * idx, 2 * idx + 1] = 1
            self.adj_raw[2 * idx + 1, 2 * idx] = -1

    def construct_nodes_adj_raw_and_raw_points(self):
        '''
        self.adj_raw : node[i]-->node[j], adj_raw[i][j]=1, adj_raw[j][i]=-1
        '''
        self.all_nodes_raw = []
        self.raw_points_in_between = {}
        self.adj_raw = np.zeros((2 * len(self.centerlines), 2 * len(self.centerlines)), dtype=np.int8)
        for idx, centerline in enumerate(self.centerlines):
            self.all_nodes_raw.append(Node(centerline[self.start_point_idxs[idx]]))
            self.all_nodes_raw.append(Node(centerline[self.end_point_idxs[idx]]))
            self.adj_raw[2 * idx, 2 * idx + 1] = 1
            self.adj_raw[2 * idx + 1, 2 * idx] = -1
            self.raw_points_in_between[(2 * idx, 2 * idx + 1)] = centerline[
                                                                 self.start_point_idxs[idx] + 1:self.end_point_idxs[
                                                                     idx]]

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
        nodes_raw_nodes_map = np.array(nodes_raw_nodes_map, dtype=np.int)
        nodes_raw_nodes_index_map = []
        for idx in range(len(nodes_raw_nodes_map)):
            nodes_raw_nodes_index_map.append(all_nodes_index.index(nodes_raw_nodes_map[idx]))
        nodes_raw_nodes_index_map = np.array(nodes_raw_nodes_index_map, dtype=np.int)
        ## map raw points in between
        self.points_in_between_nodes = {}
        for i, j in self.raw_points_in_between:
            if nodes_raw_nodes_index_map[i] == nodes_raw_nodes_index_map[j]:
                continue
            self.points_in_between_nodes[(nodes_raw_nodes_index_map[i], nodes_raw_nodes_index_map[j])] = \
            self.raw_points_in_between[(i, j)]

        self.adj = np.zeros((len(self.all_nodes), len(self.all_nodes)), dtype=np.int)

        for i, j in self.points_in_between_nodes.keys():
            self.adj[i][j] = 1
            self.adj[j][i] = -1
