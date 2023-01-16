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

import os
import sys
APP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../..')
sys.path.append(APP_PATH)
from source_code.data_splitting.data_splitting import *
from source_code.scheduling_algorithm_pool.scheduler_column_generation.initilal_solution.appl_sci.graph_heuristic_algorithm import *
from source_code.utility.creates_and_combines import get_new_service_mat_by_cut


def graph_heuristic_for_init_column(d, p, d_r, s_full, u_full, s_type, service_node_level_list, anti_affinity_list,
                                    machine_type_node_level_list, machine_index_2_type_list):
    """
    This function construct a initial solution based on applied science 19's paper.

    :param d:
    :param p:
    :param d_r:
    :param s_full:
    :param u_full:
    :param s_type:
    :param service_node_level_list:
    :param anti_affinity_list:
    :param machine_type_node_level_list:
    :param machine_index_2_type_list:
    :return: a schedule of services to machines
    """

    # Some parameters related to the problem
    rounds = 20
    sever_cpu = min(u_full[:, 0])
    sever_mem = min(u_full[:, 1])

    # some number results
    service_num = len(d)
    node_num = u_full.shape[0]
    cut_sets = np.zeros(service_num, dtype=int)
    x_int = np.zeros([service_num, node_num], dtype=int)

    # compatibility partitioning
    node_level_index_dict = \
        separate_different_node_level(copy.deepcopy(p), service_num, cut_sets, service_node_level_list, [0], 1)

    # apply the heuristic algorithm for each compatible sub-problem
    get_traffic = 0.0
    for i, nl in enumerate(list(node_level_index_dict.keys())):
        nl_index = node_level_index_dict[nl]
        # get the input of the sub-problem
        cut_p, cut_d, cut_d_r, cut_s_type, cut_service_node_level_list, cut_anti_affinity, \
        cut_service_2_orig_service_list, orig_service_2_cut_service_list \
            = get_new_service_mat_by_cut(cut_sets, nl_index, p, d, d_r, s_type, service_node_level_list, anti_affinity_list)

        # assign machines to the sub-problem
        cut_node = []
        for node in range(node_num):
            if machine_type_node_level_list[machine_index_2_type_list[node]] == nl:
                cut_node.append(node)
        cut_s_full = s_full[cut_node, :][:, cut_service_2_orig_service_list]
        cut_u_full = copy.deepcopy(u_full[cut_node, :])

        # run the heuristic algorithm
        cut_x, traffic = graph_heuristic(cut_p, cut_d, cut_d_r, cut_s_full, cut_u_full, cut_anti_affinity, rounds,
                                         sever_cpu, sever_mem)

        get_traffic += traffic

        # store results
        two_dimension_slice(x_int, cut_service_2_orig_service_list, cut_node, cut_x)

    return x_int


def two_dimension_slice(arr, x, y, target):
    if (len(x), len(y)) != target.shape:
        print('Error when constructing applied science algorithm as the initial solution: two dimensional slice not match.')
    else:
        for index_i, i in enumerate(x):
            arr[i, y] = target[index_i, :]
