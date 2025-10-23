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
import random
from pyquaternion import Quaternion

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
    
    def get_nearest_node_idx(self):
        node_position=[(node.position[0]**2+node.position[1]**2) for node in self.nodes_list]
        self.nearest_node_index= np.argmin(node_position)

        self.nearest_node=self.nodes_list[np.argmin(node_position)]
        

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
        nodes_index_list = sorted(nodes_index_list, key=lambda x: abs(x[1].position[0]))  #!
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


class OrderedBzSceneGraph(object):
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
        x_list = [(abs(i.position[0]),abs(i.position[1])) for i in nodes_list]  #!
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

        result = []
        result_list = []
        for idx, subgraph in enumerate(self.subgraphs_sorted):
            subgraph_scene_sentance, new_subgraphs_points_in_between_nodes = self.subgraph_sequelize(subgraph)
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


class OrderedBzSceneGraphNew(OrderedBzSceneGraph):
    def __init__(self, Nodes_list: list, adj: list, nodes_points: list, ncontrol=3):
        super(OrderedBzSceneGraphNew, self).__init__(Nodes_list, adj, nodes_points)
        self.ncontrol = ncontrol


    
    
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
        
        # for subgraph in self.subgraph:
        #     subgraph.get_nearest_node_idx()

        # sort subgraphs by x coordinate of the first start node in each subgraph
        # self.subgraphs_sorted = sorted(self.subgraph, key=lambda x: (x.nearest_node.position[0]**2+x.nearest_node.position[1]**2))
        self.subgraphs_sorted = sorted(self.subgraph, key=lambda x: x.first_start_node.position[0])
        
        adj_list=[]
        result_list = []
        new_subgraphs_points_in_between_nodes_list=[]
        for idx, subgraph in enumerate(self.subgraphs_sorted):
            subgraph_scene_sentance, adj,new_subgraphs_points_in_between_nodes = self.subgraph_sequelize(subgraph)
            # subgraph_scene_sentance, adj,new_subgraphs_points_in_between_nodes = self.subgraph_sequelize_start_from_center(subgraph)
            # self.set_coeff(subgraph_scene_sentance, new_subgraphs_points_in_between_nodes) #?
            new_subgraphs_points_in_between_nodes_list.append(new_subgraphs_points_in_between_nodes)
            result_list.append(subgraph_scene_sentance)
            adj_list.append(adj)


        return result_list,adj_list,new_subgraphs_points_in_between_nodes_list
    
    
    def subgraph_sequelize_random(self, subgraph):
        """pry subgragh search"""
        # subgraph.nodes_list  subgraph.nodes_adj
        nodes = subgraph.nodes_list
        adj = subgraph.nodes_adj
        nodes_points = subgraph.nodes_points


        new_subgraphs_points_in_between_nodes = {}
        random_list = list(range(len(nodes)))
        random.shuffle(random_list)
        sub_nodes = random_list
        subgraph_adj = np.zeros((len(sub_nodes), len(sub_nodes)), dtype=int)
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
    
        return new_subgraphs_nodes,new_subgraphs_adj, new_subgraphs_points_in_between_nodes
    
    
    def subgraph_sequelize_start_from_center(self, subgraph):
        """pry subgragh search"""
        # subgraph.nodes_list  subgraph.nodes_adj
        nodes = subgraph.nodes_list
        adj = subgraph.nodes_adj
        nodes_points = subgraph.nodes_points

        def dfs(index, visited, subgraph_nodes, adj):
            if visited[index]:
                return

            visited[index] = True
            subgraph_nodes.append(index)
            for idx, i in enumerate(adj[index]):
                if adj[index][idx] == 1:
                    dfs(idx, visited, subgraph_nodes, adj)
            for idx, i in enumerate(adj[index]):
                if adj[index][idx] == -1:
                    dfs(idx, visited, subgraph_nodes, adj)
            

        if nodes is None or adj is None:
            raise Exception("construction nodes & adj raw first!")

 
        visted = [False for i in nodes]
        # subgraphs_nodes_ = []
        subgraphs_nodes = []
        center_node_index=subgraph.nearest_node_index

        dfs(center_node_index, visted, subgraphs_nodes, adj)
        
        if len(subgraphs_nodes) != len(nodes):
            raise Exception("len(subgraphs_nodes_) != len(nodes)! Check dfs!")

        new_subgraphs_points_in_between_nodes = {}

        sub_nodes = subgraphs_nodes
        subgraph_adj = np.zeros((len(sub_nodes), len(sub_nodes)), dtype=int)
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

        return new_subgraphs_nodes,new_subgraphs_adj, new_subgraphs_points_in_between_nodes
    
    
    def subgraph_sequelize_by_coord(self, subgraph):
        """pry subgragh search"""
        # subgraph.nodes_list  subgraph.nodes_adj
        nodes = subgraph.nodes_list
        adj = subgraph.nodes_adj
        nodes_points = subgraph.nodes_points
        new_subgraphs_points_in_between_nodes = {}
    
        coord_list=[i.position for i in  nodes]
        coord_list_sort=sorted(enumerate(coord_list), key=lambda x: (x[1][0],x[1][1]))
        
        
        sub_nodes = [i[0] for i in coord_list_sort]
        subgraph_adj = np.zeros((len(sub_nodes), len(sub_nodes)), dtype=int)
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
        return new_subgraphs_nodes,new_subgraphs_adj, new_subgraphs_points_in_between_nodes
    
    
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
        subgraph_adj = np.zeros((len(sub_nodes), len(sub_nodes)), dtype=int)
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


        return new_subgraphs_nodes,new_subgraphs_adj, new_subgraphs_points_in_between_nodes



def get_proj_mat(intrins, rots, trans):
    intrins=np.array(intrins)
    rots=np.array(rots)
    trans=np.array(trans)
    K = np.eye(4)
    K[:3, :3] = intrins
    R = np.eye(4)
    R[:3, :3] = rots.transpose(-1, -2)
    T = np.eye(4)
    T[:3, 3] = -trans
    RT = R @ T
    return K @ RT

def perspective(cam_coords, proj_mat):
    pix_coords = proj_mat @ cam_coords
    valid_idx = pix_coords[2, :] > 0
    # pix_coords = pix_coords[:, valid_idx]
    
    pix_coords = pix_coords[:2, :] / (pix_coords[2, :] + 1e-7)
    pix_coords = pix_coords.transpose(1, 0)
    return pix_coords,valid_idx



class NusOrederedBzCenterLine(object):
    def __init__(self, centerlines, grid_conf, bz_grid_conf,cam_intrinsic,token=None):
        self.types = copy.deepcopy(centerlines['type'])
        self.centerline_ids = copy.deepcopy(centerlines['centerline_ids'])
        self.incoming_ids = copy.deepcopy(centerlines['incoming_ids'])
        self.outgoing_ids = copy.deepcopy(centerlines['outgoing_ids'])
        self.start_point_idxs = copy.deepcopy(centerlines['start_point_idxs'])  
        self.end_point_idxs = copy.deepcopy(centerlines['end_point_idxs'])
        self.centerlines = copy.deepcopy(centerlines['centerlines'])
        self.coeff = copy.deepcopy(centerlines)
        self.token=token
        self.all_nodes = None
        self.adj = None
        self.subgraphs_nodes = None
        self.points_in_between_nodes = None
        self.grid_conf = grid_conf
        self.bz_grid_conf = bz_grid_conf
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
        self.vis_mask=None
        if cam_intrinsic is not None:
            self.vis_mask=self.get_visible_mask(cam_intrinsic, 1600, self.pc_range, 1)
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

    def get_visible_mask(self,instrinsics, image_width, extents, resolution):

        # Get calibration parameters
        fu, cu = instrinsics[0][0], instrinsics[0][2]

        # Construct a grid of image coordinates
        x1, z1, x2, z2 = extents[1],extents[0],extents[4],extents[3]
        x, z = np.arange(x1, x2, resolution), np.arange(z1, z2, resolution)
        ucoords = x / z[:, None] * fu + cu

        # Return all points which lie within the camera bounds
        mask=(ucoords >= 0) & (ucoords < image_width)

        return mask
    
    def are_trues_connected(self,arr):
        if not np.any(arr):  # 如果没有任何True, 直接返回True
            return True
        
        first_true_index = np.argmax(arr)  # 找到第一个True的索引
        last_true_index = len(arr) - np.argmax(arr[::-1]) - 1  # 找到最后一个True的索引
        
        # 获取第一个和最后一个True之间的子数组，并判断是否全为True
        return np.all(arr[first_true_index:last_true_index+1])
    
    def make_trues_connected(self,arr):
    # 找到第一个True的索引
        first_true_index = np.argmax(arr)
        
        # 如果数组中没有True，返回原始数组
        if not arr[first_true_index]:
            return arr
        
        # 从第一个True开始的子数组
        sub_array = arr[first_true_index:]
        
        # 找到第一个False的索引，如果没有False则该值为len(sub_array)
        first_false_after_true_index = np.argmax(~sub_array)
        
        # 从第一个False之后的所有元素都设为False
        sub_array[first_false_after_true_index:] = False
        arr[first_true_index:] = sub_array
        
        return arr
    
    def find_true_segments(self,arr):
        # 转换布尔数组为整数，并在开头插入0，以便处理边界情况
        int_arr = np.concatenate(([0], arr.astype(int), [0]))
        
        # 计算差分
        diff = np.diff(int_arr)
        
        # 找到开始（1表示从False到True的变化）和结束位（-1表示True到False的变化）
        starts = np.where(diff == 1)[0]
        ends = np.where(diff == -1)[0] - 1
        
        # 返回开始和结束索引对
        return list(zip(starts, ends))
    
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
            if self.vis_mask is not None:
                centerline_points_int=centerline.astype(int)
                for index,centerline_point_int in enumerate(centerline_points_int):
                    if in_bev_xy[index] and not self.vis_mask[centerline_point_int[0]-int(self.pc_range[0]),centerline_point_int[1]-int(self.pc_range[1])]:
                        in_bev_xy[index]=False
                
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
            # if not self.are_trues_connected(in_bev_xy):
            #     print(in_bev_xy)
            #     in_bev_xy=self.make_trues_connected(in_bev_xy[::-1])[::-1]
            
            segments=self.find_true_segments(in_bev_xy)
            for segment in segments:
                start,end=segment[0],segment[1]
                if end<=start:
                    continue
                new_in_bev_xy=np.zeros_like(in_bev_xy,dtype=bool)
                new_in_bev_xy[start:end+1]=True
                # print(new_in_bev_xy)
                
                start_point_idx = self.start_point_idxs[i]
                end_point_idx = self.end_point_idxs[i]
                aug_start_point = centerline[start_point_idx]
                aug_end_point = centerline[end_point_idx]
                aug_centerline = centerline[new_in_bev_xy,:]
                aug_idxs = idxs[new_in_bev_xy]

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

    def export_node_adj(self):
        # self.construct_nodes_adj_raw()
        self.construct_nodes_adj_raw_and_raw_points() 
        self.nodes_merge()

        return self.all_nodes, self.adj
    

    def construct_nodes_adj_raw(self):
        '''
        self.adj_raw : node[i]-->node[j], adj_raw[i][j]=1, adj_raw[j][i]=-1
        '''
        self.all_nodes_raw = []
        self.adj_raw = np.zeros((2 * len(self.centerlines), 2 * len(self.centerlines)), dtype=np.int8)
        for idx, centerline in enumerate(self.centerlines):
            self.all_nodes_raw.append(BzNode(centerline[self.start_point_idxs[idx]]))
            self.all_nodes_raw.append(BzNode(centerline[self.end_point_idxs[idx]]))
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
            self.all_nodes_raw.append(BzNode(centerline[self.start_point_idxs[idx]]))
            self.all_nodes_raw.append(BzNode(centerline[self.end_point_idxs[idx]]))
            self.adj_raw[2 * idx, 2 * idx + 1] = 1
            self.adj_raw[2 * idx + 1, 2 * idx] = -1
            self.raw_points_in_between[(2 * idx, 2 * idx + 1)] = centerline[
                                                                 self.start_point_idxs[idx]:self.end_point_idxs[idx]+1]

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
            self.adj[j][i] = -1

    def draw_graph(self, nx=(200,200),scale=5):
        grid_conf = dict(
        xbound=[-48.0, 48.0, 0.5],
        ybound=[-32.0, 32.0, 0.5],
        zbound=[-10.0, 10.0, 20.0],
        dbound=[4.0, 48.0, 1.0], )
        dx,_, _, pc_range, _ = get_geom(grid_conf)
        
        
        # 创建一个空白图像
        image = np.zeros((nx[0]* scale, nx[1] * scale,3))
        # 绘制点
        for i, node in enumerate(self.all_nodes):
            x, y ,z= (np.array(node.position)- pc_range[:3]) / dx
            cv2.circle(image, (int(x* scale), int(y* scale)), int(scale**1.5), (0, 125, 0), -1)
            # cv2.putText(image, str(i), (int(x) + 10, int(y) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        # 绘制线
        
        for (i, j), points_tuple in self.points_in_between_nodes.items():
            if not isinstance(points_tuple,tuple):
                points_tuple=(points_tuple,)
            for points in points_tuple:
                points = ((points- pc_range[:3]) / dx* scale).astype(int)[:,:2]
                cv2.polylines(image, [points], isClosed=False, color=(0, 161, 244), thickness=2)
        
                cv2.arrowedLine(image, points[len(points)//2-1], points[len(points)//2],
                                    color=(49, 78, 255), thickness=2, tipLength=2)

        
        path="tmp"
        os.makedirs(path, exist_ok=True)
        cv2.imwrite(os.path.join(path, f"{self.token}.png"), image)
       


class NusOrederedBzCenterLineRandomCam(NusOrederedBzCenterLine):
    def __init__(self, centerlines, grid_conf, bz_grid_conf,cams,chosen_cams,token=None):
        self.types = copy.deepcopy(centerlines['type'])
        self.centerline_ids = copy.deepcopy(centerlines['centerline_ids'])
        self.incoming_ids = copy.deepcopy(centerlines['incoming_ids'])
        self.outgoing_ids = copy.deepcopy(centerlines['outgoing_ids'])
        self.start_point_idxs = copy.deepcopy(centerlines['start_point_idxs'])  # 问题出在这里 有三条中心线 start_idx==end_idx 所以和segmentation对不上
        self.end_point_idxs = copy.deepcopy(centerlines['end_point_idxs'])
        self.centerlines = copy.deepcopy(centerlines['centerlines'])
        self.coeff = copy.deepcopy(centerlines)
        # self.start_point_idxs = [0 for i in self.centerlines]
        # self.end_point_idxs = [len(centerline)-1 for centerline in self.centerlines]
        self.token=token
        self.all_nodes = None
        self.adj = None
        self.subgraphs_nodes = None
        self.points_in_between_nodes = None
        self.grid_conf = grid_conf
        self.bz_grid_conf = bz_grid_conf
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
    

        self.cams=cams
        # self.chosen_cams=["CAM_FRONT"]
        self.chosen_cams=chosen_cams
        self.filter_bev()
    

    
    def filter_bev(self):
        aug_types = []
        aug_centerlines = []
        aug_centerline_ids = []
        aug_start_point_idxs = []
        aug_end_point_idxs = []
        aug_incoming_ids = []
        aug_outgoing_ids = []
        
        if len(self.chosen_cams)<6:
            Ps=[]
            for cam,cam_info in self.cams.items():
                if cam in self.chosen_cams:
                    intrin=cam_info["cam_intrinsic"]
                    rot=Quaternion(cam_info["sensor2ego_rotation"]).rotation_matrix
                    tran=cam_info["sensor2ego_translation"]
                    P = get_proj_mat(intrin, rot, tran)
                    Ps.append(P)

        for i in range(len(self.centerlines)):
            centerline = self.centerlines[i]
            idxs = np.arange(len(centerline))
            in_bev_x = np.logical_and(centerline[:, 0] < self.pc_range[3], centerline[:, 0] >= self.pc_range[0])
            in_bev_y = np.logical_and(centerline[:, 1] <= self.pc_range[4], centerline[:, 1] >= self.pc_range[1])
            in_bev_xy = np.logical_and(in_bev_x, in_bev_y)
            
            if len(self.chosen_cams)<6:
                pts_num=len(centerline)
                ones = np.ones((pts_num, 1))
                world_coords = np.concatenate([centerline,  ones], axis=1).transpose(1, 0)
                in_all_pv=np.zeros(pts_num,dtype=bool)
                for P in Ps:
                    pix_coords,valid_idx = perspective(world_coords, P)
                    in_pv=np.logical_and(pix_coords[:,0]<1600 , pix_coords[:,0]>0)
                    in_pv=np.logical_and(in_pv,valid_idx)
                    in_all_pv=np.logical_or(in_all_pv,in_pv)
                in_bev_xy=np.logical_and(in_bev_xy,in_all_pv)
            

                
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
            # if not self.are_trues_connected(in_bev_xy):
            #     print(in_bev_xy)
            #     in_bev_xy=self.make_trues_connected(in_bev_xy[::-1])[::-1]
            
            segments=self.find_true_segments(in_bev_xy)
            for segment in segments:
                start,end=segment[0],segment[1]
                if end<=start:
                    continue
                new_in_bev_xy=np.zeros_like(in_bev_xy,dtype=bool)
                new_in_bev_xy[start:end+1]=True
                # print(new_in_bev_xy)
                
                start_point_idx = self.start_point_idxs[i]
                end_point_idx = self.end_point_idxs[i]
                aug_start_point = centerline[start_point_idx]
                aug_end_point = centerline[end_point_idx]
                aug_centerline = centerline[new_in_bev_xy,:]
                aug_idxs = idxs[new_in_bev_xy]

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


def calculate_mse_with_straight_line(points):
    x = points[:, 0]
    y = points[:, 1]
    # 使用最小二乘法拟合直线
    A = np.vstack([x, np.ones(len(x))]).T
    m, c = np.linalg.lstsq(A, y, rcond=None)[0]
    # 计算拟合直线上的所有点
    fit_y = m * x + c
    # 计算均方误差
    mse = np.mean((y - fit_y) ** 2)
    return mse

import math

def calculate_angle_2d(A, B, C):
    '''
    A是中点; B和C都是起点
    '''
    # 向量 AB 和 AC
    AB = (B[0] - A[0], B[1] - A[1])
    AC = (C[0] - A[0], C[1] - A[1])
    
    # 计算点积 AB · AC
    dot_product = AB[0] * AC[0] + AB[1] * AC[1]
    
    # 计算 |AB| 和 |AC|
    magnitude_AB = math.sqrt(AB[0]**2 + AB[1]**2)
    magnitude_AC = math.sqrt(AC[0]**2 + AC[1]**2)
    
    # 计算夹角的余弦值
    cos_theta = dot_product / (magnitude_AB * magnitude_AC)
    
    # 为了避免浮点数计算误差导致cos_theta超出[-1, 1]范围
    cos_theta = max(-1.0, min(1.0, cos_theta))
    
    # 计算夹角（弧度制）
    theta = math.acos(cos_theta)
    
    # 将角度转换为角度制
    angle_in_degrees = math.degrees(theta)
    
    return angle_in_degrees


def gen_dx_bx(xbound, ybound, zbound):
    dx = np.array([row[2] for row in [xbound, ybound, zbound]])
    bx = np.array([row[0] + row[2] / 2.0 for row in [xbound, ybound, zbound]])
    nx = np.floor(
        np.array([(row[1] - row[0]) / row[2] for row in [xbound, ybound, zbound]])
    )
    return dx, bx, nx
def get_geom(grid_conf):
    dx, bx, nx = gen_dx_bx(grid_conf['xbound'],
                                    grid_conf['ybound'],
                                    grid_conf['zbound'],)
    dx = dx
    bx = bx
    nx = nx
    pc_range = np.concatenate((bx - dx / 2., bx - dx / 2. + nx * dx))
    x = np.arange(pc_range[0], pc_range[3], 0.1)
    y = np.arange(pc_range[1], pc_range[4], 0.1)
    z = np.array([0.])
    xx, yy, zz = np.meshgrid(x, y, z)
    points = np.stack([xx, yy, zz], axis=-1)
    ego_points = np.concatenate((points, np.ones((points.shape[0], points.shape[1], points.shape[2], 1))), axis=-1)
    return dx, bx, nx, pc_range, ego_points


import bisect 
from shapely.geometry import LineString, Point
def split_line_include_original_points(points, distance):
    """
    分割由点组成的线段，返回每个段的线段，每个段包括线上的所有原始点。
    
    :param points: 原始点的列表，表示原始线段 [(x1, y1), (x2, y2), ...]
    :param distance: 每隔多少单位进行分割
    :return: 分割后的线段列表，每个线段保存在线段上的点的集合
    """
    # 创建原始 LineString 对象
    original_line = LineString(points)
    percent_list=[]
    for point in points:
        percent_list.append(original_line.project(Point(point)))
    
    # 计算线段的总长度
    length = original_line.length
    num_splits = int(length // distance)
    split_points=[]
    segments = []
    split_percents=[i*distance  for i in range(1,num_splits+1)]
    split_percents.append(length)
    
    last_index=1
    split_point=points[0]
    split_points.append(split_point)
    for split_percent in split_percents:
        index = bisect.bisect_left(percent_list, split_percent)
        segment=np.vstack((np.array([split_point]),points[last_index:index]))
        
        
        split_point=original_line.interpolate(split_percent)
   
        split_point=[split_point.x,split_point.y,split_point.z]
        segment=np.vstack([segment,np.array([split_point])])
        segments.append(segment)
        split_points.append(split_point)
        last_index=index
    
    return split_points,segments



def divide_line_with_shapely(points, n):
    """
    分割由点组成的线段，返回每个段的线段，每个段包括线上的所有原始点。
    
    :param points: 原始点的列表，表示原始线段 [(x1, y1), (x2, y2), ...]
    :param n: 需要多少个点表达
    :return: 分割后的线段列表，每个线段保存在线段上的点的集合
    """
    # 创建原始 LineString 对象
    original_line = LineString(points)
    percent_list=[]
    for point in points:
        percent_list.append(original_line.project(Point(point)))
    
    # 计算线段的总长度
    length = original_line.length
    num_splits =n-1
    distance = length /num_splits
    split_points=[]
    segments = []
    split_percents=[i*distance  for i in range(1,num_splits+1)]
    
    last_index=1
    split_point=points[0]
    split_points.append(split_point)
    for split_percent in split_percents:
        index = bisect.bisect_left(percent_list, split_percent)
        segment=np.vstack((np.array([split_point]),points[last_index:index]))
        
        
        split_point=original_line.interpolate(split_percent)
   
        split_point=[split_point.x,split_point.y,split_point.z]
        segment=np.vstack([segment,np.array([split_point])])
        segments.append(segment)
        split_points.append(split_point)
        last_index=index
    
    return split_points,segments


class NusOrederedBzCenterLineIsometry(NusOrederedBzCenterLine):
    def __init__(self, centerlines, grid_conf, bz_grid_conf, cam_intrinsic,token,distance):
        self.distance=distance
        super().__init__(centerlines, grid_conf, bz_grid_conf, cam_intrinsic,token)
        
    def generate_continue_nodes(self):
        # =============================================
        # continue点之间直接拉成一条线，不能用2阶贝塞尔函数表达了，所以还需要等间距抽点
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
        
        
        for i in range(len(self.adj)):
            to_nodes=np.where(self.adj[i]>=1)[0]
            for to_node in to_nodes:
                centerlines=self.points_in_between_nodes[(i,to_node)]
                if not isinstance(centerlines,tuple):
                    centerlines=(centerlines,)
                for idx,centerline in enumerate(centerlines):
                    split_points,segments=split_line_include_original_points(centerline,self.distance)   
                    
                    split_points=split_points[1:-1]
                    if len(split_points)>0:
                        from_node=i
                        self.adj[from_node,to_node]=0
                        self.adj[to_node,from_node]=0
                        if isinstance(self.points_in_between_nodes[(i,to_node)],tuple):
                            self.points_in_between_nodes[(i,to_node)]=self.points_in_between_nodes[(i,to_node)][:idx]+self.points_in_between_nodes[(i,to_node)][idx+1:]
                            if len(self.points_in_between_nodes[(i,to_node)])==0:
                                self.points_in_between_nodes.pop((i,to_node))
                        else:
                            self.points_in_between_nodes.pop((i,to_node))
                            
                        
                        for j,split_point in enumerate(split_points):
                            last_idx=len(self.all_nodes)
                            self.all_nodes.append(BzNode(split_point))
                            # 新增一行全为0
                            num_columns = self.adj.shape[1]  # 获取列数
                            new_row = np.zeros((1, num_columns))  # 创建一个全为0的新行
                            self.adj = np.append(self.adj, new_row, axis=0)
                            # 新增一列全为0
                            num_rows = self.adj.shape[0]  # 获取行数
                            new_column = np.zeros((num_rows, 1))  # 创建一个全为0的新列
                            self.adj = np.append(self.adj, new_column, axis=1)
                            self.points_in_between_nodes[from_node,last_idx]=segments[j]
                            
                            self.adj[from_node,last_idx]=1
                            self.adj[last_idx,from_node]=-1
                            from_node=last_idx
                        self.adj[from_node][to_node]=1
                        self.adj[to_node][from_node]=-1
                        self.points_in_between_nodes[from_node,to_node]=segments[-1]                
                       
                            
        
       
   
    
    def export_node_adj(self):
        # self.construct_nodes_adj_raw()
        self.construct_nodes_adj_raw_and_raw_points()  # self.adj_raw.shape:[27,27]  len(self.raw_points_in_between.keys()):27
        self.nodes_merge()
        self.generate_continue_nodes()
        # self.draw_graph()
        # self.draw_graph_better()
        
        return self.all_nodes, self.adj          







class NusOrederedBzCenterLineEqualQuantity(NusOrederedBzCenterLine):
    def __init__(self, centerlines, grid_conf, bz_grid_conf, cam_intrinsic,token,n_insert_point):
        super().__init__(centerlines, grid_conf, bz_grid_conf, cam_intrinsic,token)
        self.n_insert_point=n_insert_point
        
        
    def generate_continue_nodes(self):
        # =============================================
        # continue点之间直接拉成一条线，不能用一阶贝塞尔函数表达了，所以还需要都用n个点表达，n是中间插入的点数
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
        
        for i in range(len(self.adj)):
            to_nodes=np.where(self.adj[i]>=1)[0]
            for to_node in to_nodes:
                centerlines=self.points_in_between_nodes[(i,to_node)]
                if not isinstance(centerlines,tuple):
                    centerlines=(centerlines,)
                for idx,centerline in enumerate(centerlines):
                    split_points,segments=divide_line_with_shapely(centerline,self.n_insert_point+2)  
                    
                    split_points=split_points[1:-1]
                    if len(split_points)>0:
                        from_node=i
                        self.adj[from_node,to_node]=0
                        self.adj[to_node,from_node]=0
                        if isinstance(self.points_in_between_nodes[(i,to_node)],tuple):
                            self.points_in_between_nodes[(i,to_node)]=self.points_in_between_nodes[(i,to_node)][:idx]+self.points_in_between_nodes[(i,to_node)][idx+1:]
                            if len(self.points_in_between_nodes[(i,to_node)])==0:
                                self.points_in_between_nodes.pop((i,to_node))
                        else:
                            self.points_in_between_nodes.pop((i,to_node))
                            
                        
                        for j,split_point in enumerate(split_points):
                            last_idx=len(self.all_nodes)
                            self.all_nodes.append(BzNode(split_point))
                            # 新增一行全为0
                            num_columns = self.adj.shape[1]  # 获取列数
                            new_row = np.zeros((1, num_columns))  # 创建一个全为0的新行
                            self.adj = np.append(self.adj, new_row, axis=0)
                            # 新增一列全为0
                            num_rows = self.adj.shape[0]  # 获取行数
                            new_column = np.zeros((num_rows, 1))  # 创建一个全为0的新列
                            self.adj = np.append(self.adj, new_column, axis=1)
                            self.points_in_between_nodes[from_node,last_idx]=segments[j]
                            
                            self.adj[from_node,last_idx]=1
                            self.adj[last_idx,from_node]=-1
                            from_node=last_idx
                        self.adj[from_node][to_node]=1
                        self.adj[to_node][from_node]=-1
                        self.points_in_between_nodes[from_node,to_node]=segments[-1]
                    
                    
    def export_node_adj(self):
        # self.construct_nodes_adj_raw()
        self.construct_nodes_adj_raw_and_raw_points()  # self.adj_raw.shape:[27,27]  len(self.raw_points_in_between.keys()):27
        self.nodes_merge()
        self.generate_continue_nodes()
        # self.draw_graph_better()
        
        return self.all_nodes, self.adj 
    
       
class NusOrederedRMcontinuedBzCenterLine(NusOrederedBzCenterLine):
    def __init__(self, centerlines, grid_conf, bz_grid_conf, cam_intrinsic,token='none'):
        super().__init__(centerlines, grid_conf, bz_grid_conf, cam_intrinsic,token)

    def draw_graph(self, nx=(200,200),scale=5):
        grid_conf = dict(
        xbound=[-48.0, 48.0, 0.5],
        ybound=[-32.0, 32.0, 0.5],
        zbound=[-10.0, 10.0, 20.0],
        dbound=[4.0, 48.0, 1.0], )
        dx,_, _, pc_range, _ = get_geom(grid_conf)
        
        
        # 创建一个空白图像
        image = np.zeros((nx[0]* scale, nx[1] * scale,3))
        # 绘制点
        for i, node in enumerate(self.all_nodes):
            x, y ,z= (np.array(node.position)- pc_range[:3]) / dx
            cv2.circle(image, (int(x* scale), int(y* scale)), int(scale**1.5), (0, 125, 0), -1)
            cv2.putText(image, str(i), (int(x*scale) + 10, int(y*scale) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        # 绘制线
        for (i, j), points_tuple in self.points_in_between_nodes.items():
            if not isinstance(points_tuple,tuple):
                points_tuple=(points_tuple,)
            for points in points_tuple:
                points = ((points- pc_range[:3]) / dx* scale).astype(int)[:,:2]
                cv2.polylines(image, [points], isClosed=False, color=(0, 161, 244), thickness=2)
        
                cv2.arrowedLine(image, points[len(points)//2-1], points[len(points)//2],
                                    color=(49, 78, 255), thickness=2, tipLength=2)
        
        path="tmp"
        os.makedirs(path, exist_ok=True)
        cv2.imwrite(os.path.join(path, f"tmp.png"), image)

    def draw_graph_better(self, nx=(200,200),scale=10):
        grid_conf = dict(
        xbound=[-48.0, 48.0, 0.5],
        ybound=[-32.0, 32.0, 0.5],
        zbound=[-10.0, 10.0, 20.0],
        dbound=[4.0, 48.0, 1.0], )
        dx,_, _, pc_range, _ = get_geom(grid_conf)
        
        
        # 创建一个空白图像
        image = np.ones((int(nx[0]* scale),int( nx[1] * scale),3))*255

        # 绘制线
        shift=np.array([2,5,0])
        for (i, j), points_tuple in self.points_in_between_nodes.items():
            if not isinstance(points_tuple,tuple):
                points_tuple=(points_tuple,)
            for points in points_tuple:
                points = ((points- pc_range[:3]+shift) / dx* scale).astype(int)[:,:2]
                if len(points)>10:
                    cv2.arrowedLine(image, points[len(points)//2-1], points[len(points)//2],color=(236,200,150), thickness=10, tipLength=2.6)
                cv2.polylines(image, [points], isClosed=False, color=(236,200,150), thickness=10)
  
                # 绘制点
        for i, node in enumerate(self.all_nodes):
            x, y ,z= (np.array(node.position)- pc_range[:3]+shift) / dx
            cv2.circle(image, (int(x* scale), int(y* scale)), int(scale**1.35), (182, 142,84), -1)
        
            # cv2.putText(image, str(i), (int(x*scale) + 10, int(y*scale) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        path="tmp"
        os.makedirs(path, exist_ok=True)
        cv2.imwrite(os.path.join(path, f"{self.token}.png"), image)
    
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
                

    
    def export_node_adj(self):
        # self.construct_nodes_adj_raw()
        self.construct_nodes_adj_raw_and_raw_points()  # self.adj_raw.shape:[27,27]  len(self.raw_points_in_between.keys()):27
        self.nodes_merge()
        self.clear_continue_nodes()
        # self.draw_graph()
        
        return self.all_nodes, self.adj  
    
class NusClearOrederedBzCenterLine(object):
    def __init__(self, centerlines, grid_conf, bz_grid_conf, clear=True):
        self.types = copy.deepcopy(centerlines['type'])
        self.centerline_ids = copy.deepcopy(centerlines['centerline_ids'])
        self.incoming_ids = copy.deepcopy(centerlines['incoming_ids'])
        self.outgoing_ids = copy.deepcopy(centerlines['outgoing_ids'])
        self.start_point_idxs = copy.deepcopy(centerlines['start_point_idxs'])  # 问题出在这里 有三条中心线 start_idx==end_idx 所以和segmentation对不上
        self.end_point_idxs = copy.deepcopy(centerlines['end_point_idxs'])
        self.centerlines = copy.deepcopy(centerlines['centerlines'])
        self.coeff = copy.deepcopy(centerlines)
        # self.start_point_idxs = [0 for i in self.centerlines]
        # self.end_point_idxs = [len(centerline)-1 for centerline in self.centerlines]
        self.all_nodes = None
        self.adj = None
        self.subgraphs_nodes = None
        self.points_in_between_nodes = None
        self.grid_conf = grid_conf
        self.bz_grid_conf = bz_grid_conf
        self.clear = clear
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

    def export_node_adj(self):
        # self.construct_nodes_adj_raw()
        self.construct_nodes_adj_raw_and_raw_points()  # self.adj_raw.shape:[27,27]  len(self.raw_points_in_between.keys()):27
        self.nodes_merge()
        if self.clear and len(self.all_nodes) > 0:
            self.node_clear_one()

        return self.all_nodes, self.adj

    def construct_nodes_adj_raw(self):
        '''
        self.adj_raw : node[i]-->node[j], adj_raw[i][j]=1, adj_raw[j][i]=-1
        '''
        self.all_nodes_raw = []
        self.adj_raw = np.zeros((2 * len(self.centerlines), 2 * len(self.centerlines)), dtype=np.int8)
        for idx, centerline in enumerate(self.centerlines):
            self.all_nodes_raw.append(BzNode(centerline[self.start_point_idxs[idx]]))
            self.all_nodes_raw.append(BzNode(centerline[self.end_point_idxs[idx]]))
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
            self.all_nodes_raw.append(BzNode(centerline[self.start_point_idxs[idx]]))
            self.all_nodes_raw.append(BzNode(centerline[self.end_point_idxs[idx]]))
            self.adj_raw[2 * idx, 2 * idx + 1] = 1
            self.adj_raw[2 * idx + 1, 2 * idx] = -1
            self.raw_points_in_between[(2 * idx, 2 * idx + 1)] = centerline[
                                                                 self.start_point_idxs[idx]:self.end_point_idxs[idx]+1]

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
        self.out_index = np.ones(len(self.all_nodes), dtype=np.int64) * -1
        self.in_index = np.ones(len(self.all_nodes), dtype=np.int64) * -1
        for i, j in self.raw_points_in_between:
            if nodes_raw_nodes_index_map[i] == nodes_raw_nodes_index_map[j]:
                continue
            self.points_in_between_nodes[(nodes_raw_nodes_index_map[i], nodes_raw_nodes_index_map[j])] = \
            self.raw_points_in_between[(i, j)]
            self.out_index[nodes_raw_nodes_index_map[i]] = nodes_raw_nodes_index_map[j]
            self.in_index[nodes_raw_nodes_index_map[j]] = nodes_raw_nodes_index_map[i]

        self.adj = np.zeros((len(self.all_nodes), len(self.all_nodes)), dtype=np.int64)

        for i, j in self.points_in_between_nodes.keys():
            self.adj[i][j] = 1
            self.adj[j][i] = -1
    
    def node_clear_one(self):
        adj = copy.deepcopy(self.adj)
        adj[adj==-1] = 0
        degree_out = adj.sum(axis=1) == 1
        degree_in = adj.sum(axis=0) == 1
        single_degree = np.logical_and(degree_out, degree_in)
        single_index = np.arange(len(single_degree))[single_degree]
        out_index = self.out_index[single_degree]
        in_index = self.in_index[single_degree]
        # delete loop
        no_loop = out_index != in_index
        single_index = single_index[no_loop]
        out_index = out_index[no_loop]
        in_index = in_index[no_loop]
        single_degree = np.zeros(single_degree.shape).astype(bool)
        single_degree[single_index] = True

        if not max(single_degree):
            return
        map_index = np.arange(len(single_degree))
        clear_ids = np.arange(len(single_degree))[single_degree]
        left_len = len(single_degree) - len(clear_ids)
        count = 0
        for k in range(len(map_index)):
            if k in clear_ids:
                map_index[k] = left_len
                left_len += 1
            else:
                map_index[k] = count
                count += 1
        left_len = len(single_degree) - len(clear_ids)
        clear_all_nodes = copy.deepcopy(self.all_nodes)
        for k in range(len(self.all_nodes)):
            clear_all_nodes[map_index[k]] = self.all_nodes[k]
        self.all_nodes = clear_all_nodes[:left_len]
        clear_points_in_between_nodes = {}
        for clear_id in clear_ids:
            out_points = None
            in_points = None
            new_out = None
            new_in = None
            for i,j in self.points_in_between_nodes.keys():
                if i == clear_id:
                    out_points = self.points_in_between_nodes[i, j]
                    new_out = j
                if j == clear_id:
                    in_points = self.points_in_between_nodes[i, j]
                    new_in = i
            new_between_points = np.concatenate([in_points[:-1], out_points], axis=0)
            if (new_in, clear_id) in self.points_in_between_nodes.keys():
                del self.points_in_between_nodes[new_in, clear_id]
            if (clear_id, new_out) in self.points_in_between_nodes.keys():
                del self.points_in_between_nodes[clear_id, new_out]
            self.points_in_between_nodes[new_in, new_out] = new_between_points
        for i,j in self.points_in_between_nodes.keys():
            clear_points_in_between_nodes[map_index[i], map_index[j]] = self.points_in_between_nodes[i, j]
        self.points_in_between_nodes = clear_points_in_between_nodes
        
        self.adj = np.zeros((len(self.all_nodes), len(self.all_nodes)), dtype=np.int64)
        for i, j in self.points_in_between_nodes.keys():
            self.adj[i][j] = 1
            self.adj[j][i] = -1


class PryMonoOrederedBzCenterLine(object):
    def __init__(self, centerlines, grid_conf, bz_grid_conf):
        self.types = copy.deepcopy(centerlines['type'])
        self.centerline_ids = copy.deepcopy(centerlines['centerline_ids'])
        self.incoming_ids = copy.deepcopy(centerlines['incoming_ids'])
        self.outgoing_ids = copy.deepcopy(centerlines['outgoing_ids'])
        self.start_point_idxs = copy.deepcopy(centerlines['start_point_idxs'])  # 问题出在这里 有三条中心线 start_idx==end_idx 所以和segmentation对不上
        self.end_point_idxs = copy.deepcopy(centerlines['end_point_idxs'])
        self.centerlines = copy.deepcopy(centerlines['centerlines'])
        self.coeff = copy.deepcopy(centerlines)
        # self.start_point_idxs = [0 for i in self.centerlines]
        # self.end_point_idxs = [len(centerline)-1 for centerline in self.centerlines]
        self.all_nodes = None
        self.adj = None
        self.subgraphs_nodes = None
        self.points_in_between_nodes = None
        self.grid_conf = grid_conf
        self.bz_grid_conf = bz_grid_conf
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

    def filter_fvcam(self, lidar2img, ego2lidar, img_size):
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

            center_line_homo = np.concatenate([centerline, np.ones((centerline.shape[0], 1))], axis=1).reshape(centerline.shape[0], 4, 1)
            coords = lidar2img @ ego2lidar @ center_line_homo
            coords = np.squeeze(coords, axis=-1)

            depth = coords[..., 2]
            on_img = (coords[..., 2] > 1e-5)
            coords[..., 2] = np.clip(coords[..., 2], 1e-5, 1e5)
            coords[..., 0:2] /= coords[..., 2:3]
            coords = coords[..., :2]
            h, w = img_size

            on_img = (on_img & (coords[..., 0] < w) 
                    & (coords[..., 0] >= 0) 
                    & (coords[..., 1] < h) 
                    & (coords[..., 1] >= 0))
            keep_bev = np.logical_and(on_img, in_bev_xy)

            if not np.max(keep_bev):
                continue
            if np.min(keep_bev):
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
            aug_centerline = centerline[keep_bev,:]
            aug_idxs = idxs[keep_bev]
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
            self.all_nodes_raw.append(BzNode(centerline[self.start_point_idxs[idx]]))
            self.all_nodes_raw.append(BzNode(centerline[self.end_point_idxs[idx]]))
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
            self.all_nodes_raw.append(BzNode(centerline[self.start_point_idxs[idx]]))
            self.all_nodes_raw.append(BzNode(centerline[self.end_point_idxs[idx]]))
            self.adj_raw[2 * idx, 2 * idx + 1] = 1
            self.adj_raw[2 * idx + 1, 2 * idx] = -1
            self.raw_points_in_between[(2 * idx, 2 * idx + 1)] = centerline[
                                                                 self.start_point_idxs[idx]:self.end_point_idxs[idx]+1]

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
            self.adj[j][i] = -1
