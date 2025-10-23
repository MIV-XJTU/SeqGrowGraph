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
        # return f"Node: type : {self.type}, {self.node_index}, position: {self.position}, parents: {self.parents}, children: {self.children}"
        # return f"Node_sque_index : {self.sque_index}, Node: type : {self.type}, sque_type : {self.sque_type}, fork_from : {self.fork_from_index}, merge with : {self.merge_with_index}, Points : {None if self.sque_points is None else True}\n"
        return f"Node_sque_index : {self.sque_index}, sque_type : {self.sque_type}, fork_from : {self.fork_from_index}, merge with : {self.merge_with_index}, coord : {self.position}\n"

    def __eq__(self, __o):
        if np.linalg.norm(np.array(self.position) - np.array(__o.position)) < 2.1:
            return True
        return False


class LaneGraph(object):
    def __init__(self, Nodes_list, nodes_adj, nodes_points):
        self.nodes_list = Nodes_list
        self.nodes_adj = nodes_adj
        self.nodes_points = nodes_points
        self.num = len(self.nodes_list)
        self.node_type_index = None
        self.__type_gen()
        start_nodes_sorted = self.__nodes_sort(self.node_type_index['Start'] + self.node_type_index['Start_and_Fork'], self.start_nodes_sort_method)
        self.first_start_node = self.nodes_list[start_nodes_sorted[0]]
        self.start_nodes_idx_sorted = self.__nodes_sort(self.node_type_index['Start'] + self.node_type_index['Start_and_Fork'], self.start_nodes_sort_method)
        for i, j in self.nodes_points.keys():
            if self.nodes_adj[i][j] == 1:
                continue
            else:
                raise Exception("nodes points and nodes adj not matched!")

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
        start_nodes_sorted = self.__nodes_sort(self.node_type_index['Start']+self.node_type_index['Start_and_Fork'], self.start_nodes_sort_method)
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

    def __dfs_sequelize(self, index, visted, visted_count, adj, last_index_input=None):
        if self.nodes_list[index].type == 'EndPoint':

            if self.nodes_list[last_index_input].type == 'Fork' or self.nodes_list[last_index_input].type == 'Start_and_Fork' or self.nodes_list[last_index_input].type == 'Fork_and_Merge':
                visted_count[last_index_input] += 1

            if visted_count[last_index_input] > 2:
                self.nodes_list[index].sque_type = 'fork'
                self.nodes_list[index].fork_from_index = last_index_input
                self.nodes_list[index].sque_points = self.nodes_points[(last_index_input, index)]

            else:
                self.nodes_list[index].sque_type = 'continue'
                self.nodes_list[index].sque_points = self.nodes_points[(last_index_input, index)]


            # self.nodes_list[index].sque_type = 'continue'

            last_index = np.array(range(len(self.nodes_list)))[adj[index] == -1]
            assert len(last_index) == 1 
            last_index = last_index[0]
            assert last_index == last_index_input
            self.nodes_list[index].sque_points = self.nodes_points[(last_index_input, index)]

            visted[index] = True
            visted_count[index] += 1
            return [(index, self.nodes_list[index])]
        if self.nodes_list[index].type == 'EndPoint_and_Merge':
            if visted_count[index] == 0:

                if self.nodes_list[last_index_input].type == 'Fork' or self.nodes_list[last_index_input].type == 'Start_and_Fork' or self.nodes_list[last_index_input].type == 'Fork_and_Merge':
                    visted_count[last_index_input] += 1
                if visted_count[last_index_input] > 2:
                    self.nodes_list[index].sque_type = 'fork'
                    self.nodes_list[index].fork_from_index = last_index_input
                    self.nodes_list[index].sque_points = self.nodes_points[(last_index_input, index)]

                else:
                    self.nodes_list[index].sque_type = 'continue'
                    self.nodes_list[index].sque_points = self.nodes_points[(last_index_input, index)]

                # self.nodes_list[index].sque_type = 'continue'
                # self.nodes_list[index].sque_points = self.nodes_points[(last_index_input, index)]

                visted[index] = True
                visted_count[index] += 1
                return [(index, self.nodes_list[index])]
            else:
                merge_split_node = copy.deepcopy(self.nodes_list[index])

                if self.nodes_list[last_index_input].type == "Fork" or self.nodes_list[last_index_input].type == "Start_and_Fork"  or self.nodes_list[last_index_input].type == "Fork_and_Merge":
                    visted_count[last_index_input] += 1
                    if visted_count[last_index_input] > 2:
                        merge_split_node.sque_type = 'fork_and_merge'
                        merge_split_node.fork_from_index = last_index_input
                    else:
                        merge_split_node.sque_type = 'merge'

                else:
                    merge_split_node.sque_type = 'merge'

                merge_split_node.merge_with_index = index

                merge_split_node.sque_points = self.nodes_points[(last_index_input, index)]

                visted[index] = True
                visted_count[index] += 1
                return [(None, merge_split_node)]
        if self.nodes_list[index].type == 'Merge':
            if visted_count[index] > 0:
                merge_split_node = copy.deepcopy(self.nodes_list[index])

                if self.nodes_list[last_index_input].type == "Fork" or self.nodes_list[last_index_input].type == "Start_and_Fork"  or self.nodes_list[last_index_input].type == "Fork_and_Merge":
                    visted_count[last_index_input] += 1
                    if visted_count[last_index_input] > 2:
                        merge_split_node.sque_type = 'fork_and_merge'
                        merge_split_node.fork_from_index = last_index_input
                    else:
                        merge_split_node.sque_type = 'merge'

                else:
                    merge_split_node.sque_type = 'merge'

                merge_split_node.sque_points = self.nodes_points[(last_index_input, index)]
                merge_split_node.merge_with_index = index
                visted[index] = True
                visted_count[index] += 1
                return [(None, merge_split_node)]
            else:
                if self.nodes_list[last_index_input].type == "Fork" or self.nodes_list[last_index_input].type == "Start_and_Fork"  or self.nodes_list[last_index_input].type == "Fork_and_Merge":
                    visted_count[last_index_input] += 1
                    if visted_count[last_index_input] > 2:
                        self.nodes_list[index].sque_type = 'fork'
                        self.nodes_list[index].fork_from_index = last_index_input
                    else:
                        self.nodes_list[index].sque_type = 'continue'
                else:
                    self.nodes_list[index].sque_type = 'continue'
                self.nodes_list[index].sque_points = self.nodes_points[(last_index_input, index)]
                visted[index] = True
                visted_count[index] += 1

                next_index = np.array(range(len(self.nodes_list)))[adj[index] == 1]
                assert len(next_index) == 1 
                next_index = next_index[0]
                return [(index, self.nodes_list[index])] + self.__dfs_sequelize(next_index, visted, visted_count, adj, index)
        if self.nodes_list[index].type == 'Fork_and_Merge':
            if visted_count[index] > 0:
               merge_split_node = copy.deepcopy(self.nodes_list[index])

               if self.nodes_list[last_index_input].type == "Fork" or self.nodes_list[last_index_input].type == "Start_and_Fork"  or self.nodes_list[last_index_input].type == "Fork_and_Merge":
                   visted_count[last_index_input] += 1
                   if visted_count[last_index_input] > 2:
                       merge_split_node.sque_type = 'fork_and_merge'
                       merge_split_node.fork_from_index = last_index_input
                   else:
                       merge_split_node.sque_type = 'merge'

               else:
                   merge_split_node.sque_type = 'merge'


            #    merge_split_node.sque_type = 'merge'
               merge_split_node.sque_points = self.nodes_points[(last_index_input, index)]
               merge_split_node.merge_with_index = index
               visted[index] = True
            #    visted_count[index] += 1
               return [(None, merge_split_node)]
        if visted[index]:
            return []
        if self.nodes_list[index].type == 'Start':

            self.nodes_list[index].sque_type = 'start'
            visted[index] = True
            visted_count[index] += 1
            next_index = np.array(range(len(self.nodes_list)))[adj[index] == 1]
            assert len(next_index) == 1 
            next_index = next_index[0]

            return [(index, self.nodes_list[index])] + self.__dfs_sequelize(next_index, visted, visted_count, adj, index)
            # except:
            #     raise Exception("Error flag 0")

        if self.nodes_list[index].type == 'Start_and_Fork':
            self.nodes_list[index].sque_type = 'start'
            visted[index] = True
            visted_count[index] += 1

            next_indexes = np.array(range(len(self.nodes_list)))[adj[index] == 1].tolist()
            result = [(index, self.nodes_list[index])]
            next_indexes_sorted = self.__nodes_sort(next_indexes, self.fork_nodes_sort_method)

            for next_index in next_indexes_sorted:
                result = result + self.__dfs_sequelize(next_index, visted, visted_count, adj, index)
            return result


        if self.nodes_list[index].type == 'Continue':
            visted[index] = True
            visted_count[index] += 1
            next_index = np.array(range(len(self.nodes_list)))[adj[index] == 1]
            last_index = np.array(range(len(self.nodes_list)))[adj[index] == -1]
            assert len(next_index) == 1 
            assert len(last_index) == 1 
            next_index = next_index[0]
            last_index = last_index[0]
            assert last_index == last_index_input


            if self.nodes_list[last_index_input].type == 'Fork' or self.nodes_list[last_index_input].type == 'Start_and_Fork' or self.nodes_list[last_index_input].type == 'Fork_and_Merge':
                visted_count[last_index_input] += 1
            if visted_count[last_index_input] > 2:
                self.nodes_list[index].sque_type = 'fork'
                self.nodes_list[index].fork_from_index = last_index_input
                self.nodes_list[index].sque_points = self.nodes_points[(last_index_input, index)]

            else:
                self.nodes_list[index].sque_type = 'continue'
                self.nodes_list[index].sque_points = self.nodes_points[(last_index_input, index)]

            return [(index, self.nodes_list[index])] + self.__dfs_sequelize(next_index, visted, visted_count, adj, index)
        if self.nodes_list[index].type == 'Fork' or self.nodes_list[index].type == 'Fork_and_Merge':
            visted[index] = True
            visted_count[index] += 1
            self.nodes_list[index].sque_type = 'continue'
            self.nodes_list[index].sque_points = self.nodes_points[(last_index_input, index)]
            next_indexes = np.array(range(len(self.nodes_list)))[adj[index] == 1].tolist()
            result = [(index, self.nodes_list[index])]
            next_indexes_sorted = self.__nodes_sort(next_indexes, self.fork_nodes_sort_method)
            for next_index in next_indexes_sorted:
                result = result + self.__dfs_sequelize(next_index, visted, visted_count, adj, index)
            return result

        raise Exception("No Implementation!")

        

    def __type_gen(self):
        self.node_type_index = {'Continue':[], 'Fork_and_Merge':[], 'EndPoint':[], 'Merge':[], 'Start':[], 'Fork':[], 'EndPoint_and_Merge':[], 'Start_and_Fork':[]}
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


            elif sum_b0 > sum_s0 :
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


class SceneGraph(object):
    def __init__(self, Nodes_list: list, adj: list, nodes_points: list):
        self.node_list = Nodes_list
        self.adj = adj
        self.num = len(Nodes_list)
        self.subgraph = [LaneGraph(i, j, k) for (i, j, k) in zip(self.node_list, self.adj, nodes_points)]
        
    def __repr__(self):
        return f"scene graph: {self.num} subgraphs"

    def sequelize_new(self):
        """"pry search"""
        self.subgraphs_sorted = sorted(self.subgraph, key=lambda x: x.first_start_node.position[0])

        result = []
        result_list = []
        image_list_all = []
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

        if len(subgraphs_nodes)!=len(nodes):
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
            if adj[i][i-1] == -1:  # identify continue first
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

        for i in range(1, len(node)): # merge
            child_idx = np.argwhere(adj[i]==1)
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


    def sequelize(self):
        self.subgraphs_sorted = sorted(self.subgraph, key=lambda x : x.first_start_node.position[0])
        serial_nodes = []
        result = []
        for idx, subgraph in enumerate(self.subgraphs_sorted):
            result = result + [(idx, *i) for i in subgraph.__sequelize__()]  # Add sub graph id

        ### index mapping

        map_dict = {}
        for idx, _ in enumerate(result):
            if result[idx][1] is None:
                pass
            else:
                map_dict[(result[idx][0], result[idx][1])] = idx
            result[idx][2].sque_index = idx

            if not result[idx][2].fork_from_index is None:
                result[idx][2].fork_from_index = map_dict[(result[idx][0], result[idx][2].fork_from_index)]
            if not result[idx][2].merge_with_index is None:
                result[idx][2].merge_with_index = map_dict[(result[idx][0], result[idx][2].merge_with_index)]
            serial_nodes.append(result[idx][2])
        
        return serial_nodes

    def __len__(self):
        return self.num

    def lane_graph_split(self):
        raise Exception("No Implement!")

    def __getitem__(self, idx):
        return self.subgraph[idx]


class LaneLine2NodesConverter(object):
    def __init__(self, results):
        self.results = results
        self.centerlines = results['center_lines']
        self.centerline_ids = self.centerlines['centerline_ids']
        self.incoming_ids = self.centerlines['incoming_ids']
        self.outgoing_ids = self.centerlines['outgoing_ids']
        self.start_point_idxs = self.centerlines['start_point_idxs']  # 问题出在这里 有三条中心线 start_idx==end_idx 所以和segmentation对不上
        self.end_point_idxs = self.centerlines['end_point_idxs']
        self.centerlines = self.centerlines['centerlines']
        # self.coeff = results['centerlines']
        self.all_nodes = None
        self.adj = None
        self.subgraphs_nodes = None
        self.points_in_between_nodes = None

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
                subgraph_count+=1
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
            for i in range(len(sub_nodes)-1):
                for j in range(i+1, len(sub_nodes)):
                    subgraph_adj[i][j] = self.adj[sub_nodes[i]][sub_nodes[j]]
                    subgraph_adj[j][i] = -subgraph_adj[i][j]
                    if subgraph_adj[i][j] == 1:
                        self.subgraphs_points_in_between_nodes[idx_][(i, j)] = self.points_in_between_nodes[(sub_nodes[i], sub_nodes[j])]
                    if subgraph_adj[i][j] == -1:
                        self.subgraphs_points_in_between_nodes[idx_][(j, i)] = self.points_in_between_nodes[(sub_nodes[j], sub_nodes[i])]

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
        self.adj_raw = np.zeros((2 * len(self.centerlines), 2 * len(self.centerlines) ), dtype=np.int8)
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
        self.adj_raw = np.zeros((2 * len(self.centerlines), 2 * len(self.centerlines) ), dtype=np.int8)
        for idx, centerline in enumerate(self.centerlines):
            self.all_nodes_raw.append(Node(centerline[self.start_point_idxs[idx]]))
            self.all_nodes_raw.append(Node(centerline[self.end_point_idxs[idx]]))
            self.adj_raw[2 * idx, 2 * idx + 1] = 1
            self.adj_raw[2 * idx + 1, 2 * idx] = -1
            self.raw_points_in_between[(2 * idx, 2 * idx + 1)] = centerline[self.start_point_idxs[idx]+1:self.end_point_idxs[idx]]


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
            for idx_j in range(idx+1, len(self.all_nodes_raw)):
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
            self.points_in_between_nodes[(nodes_raw_nodes_index_map[i], nodes_raw_nodes_index_map[j])] = self.raw_points_in_between[(i,j)]

        self.adj = np.zeros((len(self.all_nodes), len(self.all_nodes)), dtype = np.int)
        
        for i, j in self.points_in_between_nodes.keys():
            self.adj[i][j] = 1
            self.adj[j][i] = -1

def sentance2seq(sentance, pc_range, dx):
    """ for each node, seq: x, y, cls, IDX
    if type == start or continue , IDX = 0"""
    type_idx_map = {'start':0, 'continue':1, 'fork':2, 'merge':3}
    seq = []
    count = 0
    for idx, sub_sent in enumerate(sentance):
        for node in sub_sent:
            # node.position[0], node.position[1] = (node.position[0] - pc_range[0]) / dx[0], (node.position[1] - pc_range[1]) / dx[0]
            node.position = (node.position - pc_range[:3]) / dx
            if idx == 0:
                node.sque_index += 1
            else:
                node.sque_index = node.sque_index + sentance[idx-1][-1].sque_index + 1
            seq += node.position[:2].astype(int).tolist()  # x y
            seq.append(type_idx_map[node.sque_type])  # cls
            # IDX
            if node.sque_type == "fork":
                if idx == 0:
                    node.fork_from_index += 1
                else:
                    node.fork_from_index = node.fork_from_index + sentance[idx - 1][-1].sque_index + 1
                seq.append(node.fork_from_index)

            elif node.sque_type == "merge":
                if idx == 0:
                    node.merge_with_index += 1
                else:
                    node.merge_with_index = node.merge_with_index + sentance[idx - 1][-1].sque_index + 1
                seq.append(node.merge_with_index)

            else:
                seq.append(0)
    return  seq


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




def sentance2bzseqNew(sentance,adj_list,new_subgraphs_points_in_between_nodes_list,pc_range, dx, bz_pc_range, bz_nx,ncontrol):
    """ for each node, seq: x, y, cls, IDX
    if type == start or continue , IDX = 0"""

    seq_list = []
    
    for idx, (sub_sent,adj,new_subgraphs_points_in_between_nodes) in enumerate(zip(sentance,adj_list,new_subgraphs_points_in_between_nodes_list)):
        for node_i,node in enumerate(sub_sent):
        
            # node.position[0], node.position[1] = (node.position[0] - pc_range[0]) / dx[0], (node.position[1] - pc_range[1]) / dx[0]
            node.position = (node.position - pc_range[:3]) / dx
            if idx != 0:
          
                node.sque_index = node.sque_index + sentance[idx-1][-1].sque_index +1
            seq = node.position[:2].astype(int).tolist()  # x y
            seq.append(node.sque_index)  # cls
            father_idx = np.argwhere(adj[node_i] == -1)
            father_seq=[]
            if len(father_idx) != 0 :
    
                father_idx = father_idx[np.where(father_idx < node_i)]
                for f_idx in father_idx:
                    fin_res=get_bezier_coeff(new_subgraphs_points_in_between_nodes [(f_idx, node_i)][:, :2],
                                               n_control=ncontrol)
                    coeff=np.squeeze(fin_res[1:-1])
                    coeff=(coeff- bz_pc_range[:2]) / dx[:2]
                    coeff[0] = np.clip(coeff[0], 0, bz_nx[0]-1)
                    coeff[1] = np.clip(coeff[1], 0, bz_nx[1]-1)
                    coeff=coeff.astype(int)
                    if idx!=0:
                        father_seq.append([f_idx + sentance[idx-1][-1].sque_index+1 ,coeff])
                    else:
                        father_seq.append([f_idx ,coeff])
                        
            seq.append(father_seq)
            
            child_idx=np.argwhere(adj[node_i] == 1)
            child_seq=[]
            if len(child_idx)!=0:
                child_idx=child_idx[np.where(child_idx < node_i)]
                for c_idx in child_idx:
                    fin_res=get_bezier_coeff(new_subgraphs_points_in_between_nodes [(node_i, c_idx)][:, :2],
                                           n_control=ncontrol)
                    coeff=np.squeeze(fin_res[1:-1])
                    coeff=(coeff- bz_pc_range[:2]) / dx[:2]
                    coeff[0] = np.clip(coeff[0], 0, bz_nx[0]-1)
                    coeff[1] = np.clip(coeff[1], 0, bz_nx[1]-1)
                    coeff=coeff.astype(int)
                    if idx!=0:
                        child_seq.append([c_idx + sentance[idx-1][-1].sque_index+1 ,coeff])
                    else:
                        child_seq.append([c_idx  ,coeff])
            
            seq.append(child_seq)
            seq_list.append(seq)
       
    return  seq_list


def sentance2bzseq(sentance, pc_range, dx, bz_pc_range, bz_nx):
    """ for each node, seq: x, y, cls, IDX
    if type == start or continue , IDX = 0"""
    type_idx_map = {'start':0, 'continue':1, 'fork':2, 'merge':3}
    seq = []
    count = 0
    for idx, sub_sent in enumerate(sentance):
        for node in sub_sent:
            # node.position[0], node.position[1] = (node.position[0] - pc_range[0]) / dx[0], (node.position[1] - pc_range[1]) / dx[0]
            node.position = (node.position - pc_range[:3]) / dx
            if idx == 0:
                node.sque_index += 1
            else:
                node.sque_index = node.sque_index + sentance[idx-1][-1].sque_index + 1
            seq += node.position[:2].astype(int).tolist()  # x y
            seq.append(type_idx_map[node.sque_type])  # cls
            # IDX
            if node.sque_type == "fork":
                if idx == 0:
                    node.fork_from_index += 1
                else:
                    node.fork_from_index = node.fork_from_index + sentance[idx - 1][-1].sque_index + 1
                seq.append(node.fork_from_index)

            elif node.sque_type == "merge":
                if idx == 0:
                    node.merge_with_index += 1
                else:
                    node.merge_with_index = node.merge_with_index + sentance[idx - 1][-1].sque_index + 1
                seq.append(node.merge_with_index)

            else:
                seq.append(0)
            
            if len(node.coeff):
                node.coeff = (node.coeff - bz_pc_range[:2]) / dx[:2]
                node.coeff[0] = np.clip(node.coeff[0], 0, bz_nx[0]-1)
                node.coeff[1] = np.clip(node.coeff[1], 0, bz_nx[1]-1)
                node.coeff = node.coeff.astype(int)
                seq.append(node.coeff[0])
                seq.append(node.coeff[1])
            else:
                seq.append(0)
                seq.append(0)
    return  seq


def sentance2bzseq2(sentance, pc_range, dx, bz_pc_range, bz_nx):
    """ 
    use round in sequentialization
    for each node, seq: x, y, cls, IDX
    if type == start or continue , IDX = 0"""
    type_idx_map = {'start':0, 'continue':1, 'fork':2, 'merge':3}
    seq = []
    count = 0
    for idx, sub_sent in enumerate(sentance):
        for node in sub_sent:
            # node.position[0], node.position[1] = (node.position[0] - pc_range[0]) / dx[0], (node.position[1] - pc_range[1]) / dx[0]
            node.position = (node.position - pc_range[:3]) / dx
            if idx == 0:
                node.sque_index += 1
            else:
                node.sque_index = node.sque_index + sentance[idx-1][-1].sque_index + 1
            seq += node.position[:2].round().astype(int).tolist()  # x y
            seq.append(type_idx_map[node.sque_type])  # cls
            # IDX
            if node.sque_type == "fork":
                if idx == 0:
                    node.fork_from_index += 1
                else:
                    node.fork_from_index = node.fork_from_index + sentance[idx - 1][-1].sque_index + 1
                seq.append(node.fork_from_index)

            elif node.sque_type == "merge":
                if idx == 0:
                    node.merge_with_index += 1
                else:
                    node.merge_with_index = node.merge_with_index + sentance[idx - 1][-1].sque_index + 1
                seq.append(node.merge_with_index)

            else:
                seq.append(0)
            
            if len(node.coeff):
                node.coeff = (node.coeff - bz_pc_range[:2]) / dx[:2]
                node.coeff[0] = np.clip(node.coeff[0], 0, bz_nx[0]-1)
                node.coeff[1] = np.clip(node.coeff[1], 0, bz_nx[1]-1)
                node.coeff = node.coeff.round().astype(int)
                seq.append(node.coeff[0])
                seq.append(node.coeff[1])
            else:
                seq.append(0)
                seq.append(0)
    return  seq
