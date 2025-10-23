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


class BzNode(object):
    def __init__(self, position):
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

    def set_parents(self, parents):
        self.parents = parents

    def set_children(self, children):
        self.children = children

    def set_coeff(self, coeff):
        self.coeff = coeff

    def set_type(self, type_):
        self.type = type_

    def __repr__(self):
        return f"Node_sque_index : {self.sque_index}, Node_type : {self.type}, sque_type : {self.sque_type}, fork_from : {self.fork_from_index}, merge with : {self.merge_with_index}, coord : {self.position}\n"

    def __eq__(self, __o):
        if np.linalg.norm(np.array(self.position) - np.array(__o.position)) < 2.1:
            return True
        return False


class OrderedBzLaneGraph(object):
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


class Laneseq2Graph(object):
    def __init__(self, Nodes_list: list, adj: list, nodes_points: list, ncontrol=3):
        self.node_list = Nodes_list
        self.adj = adj
        self.num = len(Nodes_list)
        self.subgraph = [OrderedBzLaneGraph(i, j, k) for (i, j, k) in zip(self.node_list, self.adj, nodes_points)]
        self.ncontrol = ncontrol

    def set_coeff(self, subgraph, subgraphs_points):
        """write coeff in node list
               node.coeff"""
        for i, node in enumerate(subgraph):
            if node.sque_type=='start':
                continue
            elif node.sque_type=='continue':
                fin_res = get_bezier_coeff(subgraphs_points[(node.sque_index - 1, node.sque_index)][:,:2],
                                           n_control=self.ncontrol)
            elif node.sque_type=='fork':
                if (node.sque_index, node.fork_from_index) in subgraphs_points.keys():
                    fin_res = get_bezier_coeff(subgraphs_points[(node.sque_index, node.fork_from_index)][:, :2],
                                               n_control=self.ncontrol)
                else:
                    fin_res = get_bezier_coeff(subgraphs_points[(node.fork_from_index, node.sque_index)][:, :2],
                                               n_control=self.ncontrol)
            elif node.sque_type=='merge':
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
        x_list = [(abs(i.position[0]),abs(i.position[1])) for i in nodes_list]
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

        graph_nodes = []
        graph_betweens = []
        for idx, subgraph in enumerate(self.subgraphs_sorted):
            subgraphs_nodes, new_subgraphs_points_in_between_nodes = self.subgraph_sequelize(subgraph)
            graph_nodes.append(subgraphs_nodes)
            graph_betweens.append(new_subgraphs_points_in_between_nodes)
        return graph_nodes, graph_betweens

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
                    new_subgraphs_points_in_between_nodes[(i, j)] = np.squeeze(get_bezier_coeff(nodes_points[(sub_nodes[i], sub_nodes[j])][:,:2], n_control=self.ncontrol)[1:-1])
                if subgraph_adj[i][j] == -1:
                    new_subgraphs_points_in_between_nodes[(j, i)] = np.squeeze(get_bezier_coeff(nodes_points[(sub_nodes[j], sub_nodes[i])][:,:2], n_control=self.ncontrol)[1:-1])

        new_subgraphs_nodes = _list

        for idx, node in enumerate(new_subgraphs_nodes):
            node.sque_index = idx

        return new_subgraphs_nodes, new_subgraphs_points_in_between_nodes


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
