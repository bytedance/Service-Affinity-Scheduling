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

import numpy as np


def combine_cuts_results_in_X_index(cut_service_2_orig_service_list, cuts_x_list, cuts_q_list,
                                    q, x_int, u_free, machine_type_2_index_list, d_r, valid_cut_list):
    """
    Combine the schedule of different sub-problem as the solution of the entire cluster

    :param cut_service_2_orig_service_list: the mapping from the service index inside the cut to the service index in the entire cluster
    :param cuts_x_list: cuts_x_list[i] is a matrix that represents the schedule of containers to machines of the subproblem i
    :param cuts_q_list:cuts_q_list[i] is the machine list of the subproblem i
    :param q: the machine list of the entire problem
    :param x_int: the matrix that represents the schedule of containers to machines
    :param u_free: the free resource mat of the entire problem
    :param machine_type_2_index_list: the mapping from machine type to machines
    :param d_r: the resource request matrix of the services
    :param valid_cut_list: for some cuts, we might not need to combine the result (we will deal with these cuts later)
    :return: None
    """

    machine_type_num = len(q)
    machine_type_starter = [0 for _ in range(machine_type_num)]

    # 构造x_int矩阵
    for cut_id in valid_cut_list:
        cut_box_starter = 0
        for cut_machine_type in range(machine_type_num):
            for _ in range(cuts_q_list[cut_id][cut_machine_type]):
                # 处理cut的x[:, cut_box_starter]
                orig_box_id = machine_type_starter[cut_machine_type]
                orig_box_index = machine_type_2_index_list[cut_machine_type][orig_box_id]
                machine_type_starter[cut_machine_type] += 1
                for cut_service in range(cuts_x_list[cut_id].shape[0]):
                    orig_service = cut_service_2_orig_service_list[cut_id][cut_service]
                    x_int[orig_service][orig_box_index] = cuts_x_list[cut_id][cut_service][cut_box_starter]
                cut_box_starter += 1
    u_free -= (x_int.T @ d_r)


def get_new_service_mat_by_cut(cut_sets, cut_id, p, d, d_r, s_type, service_node_level_list, anti_affinity_list):
    """
    Get the data structures of a sub-problem by given the cut information.

    :param cut_sets: cut_sets[i] is the cut id of the service i
    :param cut_id: the sub-problem's cut id
    :param p: affinity data
    :param d: the number of containers of each services
    :param d_r: the request resource matrix of the services
    :param s_type: whether a service can be placed on a machine type
    :param service_node_level_list: the node-level names of each services
    :param anti_affinity_list: anti-affinity information, now we only support one anti-affinity rule
    :return: all related data structure of a sub-problem
    """

    # some number results
    cut_service_num = 0
    orig_service_num = len(d)
    resource_num = d_r.shape[1]
    machine_type_num = s_type.shape[0]

    # cut_service_mark_list marks the service that appears in the given cut
    cut_service_mark_list = -np.ones(orig_service_num, dtype=int)
    for service in range(orig_service_num):
        if cut_sets[service] == cut_id:
            cut_service_mark_list[service] = cut_service_num
            cut_service_num += 1

    # construct the mapping from cut service index to original service index
    cut_service_2_orig_service_list = -np.ones(cut_service_num, dtype=int)
    orig_service_2_cut_service_list = -np.ones(orig_service_num, dtype=int)
    cut_service_node_level_list = ['INVALID_NODE_LEVEL' for _ in range(cut_service_num)]

    for service in range(orig_service_num):
        if cut_service_mark_list[service] != -1:
            cut_service_index = cut_service_mark_list[service]
            cut_service_2_orig_service_list[cut_service_index] = service
            orig_service_2_cut_service_list[service] = cut_service_index

    # construct cut's affinity data(cut_p), container number(cut_d), compatibility data(cut_s), resource request(d_r)
    cut_p = {}
    cut_d = np.zeros(cut_service_num, dtype=int)
    cut_s = np.zeros([s_type.shape[0], cut_service_num], dtype=int)
    cut_d_r = np.zeros([cut_service_num, d_r.shape[1]])
    for cut_service in range(cut_service_num):
        orig_service = cut_service_2_orig_service_list[cut_service]
        cut_d[cut_service] = d[orig_service]
        for r in range(resource_num):
            cut_d_r[cut_service][r] = d_r[orig_service][r]
        for machine_type in range(machine_type_num):
            cut_s[machine_type][cut_service] = s_type[machine_type][orig_service]
        cut_service_node_level_list[cut_service] = service_node_level_list[orig_service]
    cut_s_type = cut_s

    # construct cut's anti-affinity data
    cut_anti_affinity = []
    for rule in anti_affinity_list:
        cut_rule = []
        for orig_service in rule:
            if cut_sets[orig_service] != cut_id:
                continue
            cut_service = orig_service_2_cut_service_list[orig_service]
            cut_rule.append(cut_service)
        if cut_rule:
            cut_anti_affinity.append(cut_rule)

    # construct cut's affinity data
    for key in p.keys():
        if cut_sets[round(key[0])] == cut_id and cut_sets[round(key[1])] == cut_id:
            service1 = orig_service_2_cut_service_list[key[0]]
            service2 = orig_service_2_cut_service_list[key[1]]
            cut_p[(service1, service2)] = p[key]

    return cut_p, cut_d, cut_d_r, cut_s_type, cut_service_node_level_list, cut_anti_affinity, \
           cut_service_2_orig_service_list, orig_service_2_cut_service_list
