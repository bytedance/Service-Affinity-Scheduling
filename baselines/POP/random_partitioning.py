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

import copy
import numpy as np
import os
import sys
APP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..')
sys.path.append(APP_PATH)
import random, math
from collections import defaultdict
from source_code.algorithm_selection.algorithm_selection import estimate_runtime
from source_code.data_splitting.non_affinity_partitioning import separate_not_in_link
from source_code.data_splitting.compatibility_partitioning import separate_different_node_level


def POP_pre_partitioning(cut_sets, p, d, service_node_level_list, max_time):
    """
    Roughly partitioning the services by compatibility and non-affinity
    """

    # Still, two generally useful partitioning can be applied - non-affinity and compatibility
    service_num = len(d)
    auxiliary_p = copy.deepcopy(p)
    separate_not_in_link(auxiliary_p, service_num, cut_sets, [0], -1)
    node_level_index_dict = separate_different_node_level(auxiliary_p, service_num, cut_sets, service_node_level_list,
                                                          [0], 100000)

    # re-indexing
    init_id = 0
    map_from_old_cut_index_2_new_cut_index = {}
    for key in node_level_index_dict.keys():
        map_from_old_cut_index_2_new_cut_index[node_level_index_dict[key]] = init_id
        init_id += 1
    for service in range(service_num):
        if cut_sets[service] != -1:
            cut_sets[service] = map_from_old_cut_index_2_new_cut_index[cut_sets[service]]

    # Dealing with irrelevant services
    max_cut_num = max(cut_sets) + 2
    for service in range(service_num):
        if cut_sets[service] == -1:
            cut_sets[service] = max_cut_num - 1

    # Estimate the runtime for each cut
    est_time_each_cut_dict = estimate_runtime(cut_sets, max_time, max_cut_num, p)

    return max_cut_num, est_time_each_cut_dict


def POP_client_partitioning(cut_sets, d, target_id, q, u_full, u_r_type, service_node_level_list,
                            machine_type_node_level_list, est_time, part_num=4):
    """
    POP's client and server partitioning techniques
    """
    # Get the cut's d and q
    mask_d = copy.deepcopy(np.array(d))
    mask_q = copy.deepcopy(np.array(q))
    mask_d[(cut_sets != target_id)] = 0
    service_node_level = service_node_level_list[np.array(cut_sets == target_id).nonzero()[0][0]]
    for machine_type in range(len(mask_q)):
        if service_node_level != machine_type_node_level_list[machine_type]:
            mask_q[machine_type] = 0

    # According to POP
    cut_d_dict = {}
    cut_q_dict = {}
    cut_u_full_dict = {}
    for part in range(part_num):
        cut_d_dict[part] = np.zeros(np.array(d).shape, dtype=int)
        cut_q_dict[part] = np.zeros(np.array(q).shape, dtype=int)
        cut_u_full_dict[part] = np.zeros(np.array(u_full).shape)

    # Randomly partitioning the d
    for service in mask_d.nonzero()[0]:
        for _ in range(d[service]):
            cut_d_dict[get_index(part_num)][service] += 1

    # Randomly partitioning the q
    for machine_type in mask_q.nonzero()[0]:
        for _ in range(q[machine_type]):
            cut_q_dict[get_index(part_num)][machine_type] += 1

    # Maintain u_full
    each_type_init = np.zeros(np.array(q).shape, dtype=int)
    curr = 0
    for machine_type in range(len(q)):
        each_type_init[machine_type] = curr
        curr += q[machine_type]
    for machine_type in mask_q.nonzero()[0]:
        for part in range(part_num):
            for _ in range(int(cut_q_dict[part][machine_type])):
                cut_u_full_dict[part][each_type_init[machine_type]][0] = u_r_type[machine_type][0]
                cut_u_full_dict[part][each_type_init[machine_type]][1] = u_r_type[machine_type][1]
                each_type_init[machine_type] += 1

    # Estimate runtime for each part
    est_part_max_time_dict = defaultdict(float)
    total_time = 0.0
    for part in range(part_num):
        total_time += len(cut_d_dict[part]) * cut_u_full_dict[part].shape[0]
    for part in range(part_num):
        est_part_max_time_dict[part] = math.ceil(len(cut_d_dict[part]) * cut_u_full_dict[part].shape[0] * est_time * 0.9
                                                 / total_time)

    return cut_d_dict, cut_q_dict, cut_u_full_dict, est_part_max_time_dict


def get_index(part_num):
    """
    uniformly partitioning into part_num parts
    """
    return random.randint(0, part_num-1)
