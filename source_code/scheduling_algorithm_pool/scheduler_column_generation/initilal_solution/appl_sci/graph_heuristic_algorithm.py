# Copyright (year) Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# coding: utf-8
import os
import math
import copy
import random
import numpy as np
import sys
from collections import defaultdict
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../..'))
from source_code.scheduling_algorithm_pool.scheduler_column_generation.initilal_solution.appl_sci.get_best_pick import \
    service_partition
from source_code.utility.result_check import calculate_local_traffic


def graph_heuristic(p, d, d_r, s_full, u_full, cut_anti_affinity, rounds, sever_cpu=96, sever_mem=256):
    """
    Control work flow of the heuristic algorithm

    :param p: affinity data
    :param d: container number of each service
    :param d_r: resource request of each services
    :param s_full: compatibility, but since we separate different compatible problem first, this is useless
    :param u_full: machine resource matrix
    :param cut_anti_affinity: anti-affinity rule of the problem
    :param rounds: how many rounds to repeat the heuristic algorithm(which is a randomized algorithm)
    :param sever_cpu: the heuristic algorithm requires to set the machine's cpu size first
    :param sever_mem: the heuristic algorithm requires to set the machine's memory size first
    :return: two variable, first is the schedule, second is the gained affinity of the schedule
    """

    # prepare some variables
    sever_type = [sever_cpu, sever_mem]
    psm_num = len(d)
    edge_num = len(p.keys())
    node_num = u_full.shape[0]
    service_num = len(d)
    service_info = [[None, None, None, None]] * psm_num
    edge_info = {}

    if len(p.keys()) == 0:
        return np.zeros([service_num, node_num], dtype=int), 0

    u_type_dict = defaultdict(list)
    type_2_node_dict = {}
    type_2_curr_index_dict = {}
    for node in range(node_num):
        u_type_dict[(u_full[node][0], u_full[node][1])].append(node)
    node_type_num = len(list(u_type_dict.keys()))
    node_info = [[None, None, None]] * node_type_num
    for i, key in enumerate(list(u_type_dict.keys())):
        node_info[i] = [key[0], key[1], len(u_type_dict[key])]
        type_2_node_dict[i] = u_type_dict[key]
        type_2_curr_index_dict[i] = 0

    part_dict = {}
    random_distributed_edge = [[None, None, None]] * edge_num
    # Some old-version code's problem, don't worry about this
    for psm in range(psm_num):
        service = psm
        service_info[service] = [d[service], d_r[service][0], d_r[service][1], service]
        part_dict[service] = [service]
    tmp_sum = 0
    for key in p.keys():
        service1 = key[0]
        service2 = key[1]
        edge_info[(service1, service2)] = p[key]
    for i, key in enumerate(list(edge_info.keys())):
        random_distributed_edge[i] = [key[0], key[1], tmp_sum]
        tmp_sum += p[key]
    random_distributed_edge.append([-1, -1, tmp_sum])
    random_distributed_edge = np.array(random_distributed_edge)
    random_distributed_edge[:, 2] = np.divide(random_distributed_edge[:, 2],  np.max(random_distributed_edge[:, 2]))

    service_info = np.array(service_info)
    node_info = np.array(node_info)
    random_distributed_edge = np.array(random_distributed_edge)

    # get the best-pick
    curr_reward = -1.0
    curr_res = None
    for curr_round in range(rounds):
        res, reward = service_partition(service_info, edge_info, node_info, random_distributed_edge,
                                        len(list(edge_info.keys())), copy.deepcopy(part_dict), sever_type,
                                        cut_anti_affinity)
        if reward > curr_reward:
            curr_reward = reward
            curr_res = res

    # rounding
    x_lp = np.zeros([psm_num, node_num])
    x_int = np.zeros([psm_num, node_num], dtype=int)
    for key in curr_res.keys():
        pattern = curr_res[key][0]
        deploy_num = curr_res[key][1]
        node_type = curr_res[key][2]

        for _ in range(deploy_num):
            node = type_2_node_dict[node_type][type_2_curr_index_dict[node_type]]
            type_2_curr_index_dict[node_type] += 1
            x_lp[:, node] = pattern

    graph_heuristic_rounding(x_lp, x_int, d, d_r, u_full)

    return x_int, calculate_local_traffic(x_int, d, p, print_flag=False)[0]


def graph_heuristic_rounding(x_lp, x_int, d, d_r, u_full):
    """
    The heuristic algorithm will give fractional results, rounds it the interger.

    :param x_lp: fractional schedule
    :param x_int: integer schedule that will pass to the algorithm
    :param d: number of containers of each service
    :param d_r: resource request matrix of each services
    :param u_full: total cpu resources of each machines
    :return:
    """

    not_zero_mat = [x_lp > 0]

    # obtaining randomized rounding values
    threshold_mat = np.zeros(x_lp.shape)
    threshold_mat[tuple(not_zero_mat)] = x_lp[tuple(not_zero_mat)] - np.floor(x_lp[tuple(not_zero_mat)])
    random_mat = np.random.rand(x_lp.shape[0], x_lp.shape[1])
    round_up_mat = np.array(np.zeros(x_lp.shape), dtype=bool)
    round_down_mat = np.array(np.zeros(x_lp.shape), dtype=bool)
    round_up_mat[tuple(not_zero_mat)] = (random_mat <= threshold_mat)[tuple(not_zero_mat)]
    round_down_mat[tuple(not_zero_mat)] = (random_mat > threshold_mat)[tuple(not_zero_mat)]

    round_up_mat = [round_up_mat]
    round_down_mat = [round_down_mat]

    # rounding
    x_int[tuple(round_up_mat)] = np.ceil(x_lp[tuple(round_up_mat)])
    x_int[tuple(round_down_mat)] = np.floor(x_lp[tuple(round_down_mat)])

    # maintain d (exceeding part)
    exceeding_services = list(((np.sum(x_int, axis=1) - np.array(d)) > 0).nonzero()[0])
    for service in exceeding_services:
        exceed_num = np.sum(x_int[service, :]) - d[service]
        possible_indices = x_int[service, :].nonzero()[0]
        for i in range(len(possible_indices)-1, -1, -1):
            if exceed_num > 0:
                node = possible_indices[i]
                del_num = min(exceed_num, x_int[service][node])
                exceed_num -= del_num
                x_int[service][node] -= del_num
            else:
                break

    # maintain resource
    deploy_cpu = np.dot(x_int.T, d_r[:, 0])
    deploy_mem = np.dot(x_int.T, d_r[:, 1])
    not_cool_nodes = ((deploy_cpu > u_full[:, 0] + 1e-9) + (deploy_mem > u_full[:, 1] + 1e-9)).nonzero()[0]
    for node in not_cool_nodes:
        possible_services = x_int[:, node].nonzero()[0]
        while np.dot(x_int[:, node], d_r[:, 0]) > u_full[node][0] + 1e-9 or \
                np.dot(x_int[:, node], d_r[:, 1]) > u_full[node][1] + 1e-9:
            index = possible_services[math.floor(len(possible_services) * random.random())]
            x_int[index, node] -= 1
