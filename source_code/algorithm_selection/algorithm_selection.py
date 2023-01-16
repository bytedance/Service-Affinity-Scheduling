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
APP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..')
sys.path.append(APP_PATH)
from source_code.utility.creates_and_combines import *
from source_code.scheduling_algorithm_pool.scheduler_column_generation.scheduler_column_generation import scheduler_column_generation
from source_code.scheduling_algorithm_pool.scheduler_mip.scheduler_mip import scheduler_mip
from source_code.algorithm_selection.gcn.classifier import get_method
from source_code.utility.result_check import *
from collections import defaultdict
import math, copy


def algorithm_selection_and_apply(p, u_r, q, d, d_r, s_type, cut_sets, max_cut_num, service_node_level_list,
                                  global_traffic, machine_type_node_level_list, u_free, machine_type_2_index_list,
                                  anti_affinity_list, max_time=60, method_in=None, ff_filter=0.02, heuri_init_flag=True,
                                  scale_rate=1.1, force_k8s_heuri=False):
    """
    algorithm selection component: select an appropriate algorithm for each sub-problem and apply the selected algorithm to it.

    :param p: affinity data
    :param u_r: resource matrix of each machine specification
    :param q: number of machines for each machine specification
    :param d: number of containers for each services
    :param d_r: resource request for each services
    :param s_type: compatibility matrix
    :param cut_sets: data splitting output, which represents the partitioning of the problem
    :param max_cut_num: total number of partitions of the problem
    :param service_node_level_list: the node-level of each services
    :param global_traffic: total affinity of the entire problem
    :param machine_type_node_level_list: the node-level of each machine specification
    :param u_free: free resources for each machines
    :param machine_type_2_index_list: the mapping from machine specification to the list of its machine indexes
    :param anti_affinity_list: new feature -- anti-affinity
    :param max_time: user-specific max runtime, in seconds
    :param method_in: user may give the method to solve the problem. If None, we use heuristic-gcn hybrid method to select the algorithm
    :param ff_filter: parameters related to the ff-filter
    :param heuri_init_flag: if True, we will calculate initial solution for the two exact algorithms. This may acceleration the algortihm
    :param scale_rate: we will assign scale_rate * total_resource_request resources of machines to the sub-problem with a total request resource of total_resource_request
    :param force_k8s_heuri: one of the initial solution for the two exact algorithm is an k8s+ algorithm, this algorithm is time-consuming, user can decide whether to calculate this initial solution
    :return:
        x_int: the schedule of the entire problem
    """

    # preparing some data structures
    service_num = len(d)
    box_num = sum(q)
    x_int = np.zeros([service_num, box_num], dtype=int)
    cuts_service_2_orig_service_dict = {}
    cuts_x_dict = {}
    cuts_q_dict = {}
    valid_cut_list = []

    # Estimate the runtime of each sub-problem
    est_time_out_dict = estimate_runtime(cut_sets, max_time, max_cut_num, p)

    # For each cut, select an appropriate algorithm and apply the selected algorithm to it.
    curr_remain_q = np.asarray(q.copy())
    for cut_id in range(max_cut_num - 1):
        # dealing with the sub-problem with cut id = cut_id

        # get the sub-problem's input
        cut_p, cut_d, cut_d_r, cut_s_type, cut_service_node_level_list, cut_anti_affinity, \
         cut_service_2_orig_service_list, rig_service_2_cut_service_list = \
         get_new_service_mat_by_cut(cut_sets, cut_id, p, d, d_r, s_type, service_node_level_list, anti_affinity_list)
        node_level = cut_service_node_level_list[0]

        print('\r\n')
        print('Dealing with sub-problem %d:, sub-problem\' total affinity %.2f%%, number of services %d.' %
              (cut_id, sum(cut_p.values())*100.0/sum(p.values()), len(cut_d)))

        # A heuristic-gcn hybrid algorithm selection
        if sum(cut_p.values()) < ff_filter * global_traffic:
            # FF-filter
            continue
        else:
            if method_in is None:
                # If the sub-problem pass the FF-filter, we use gcn classifier to select an appropriate algorithm
                method = get_method(cut_p, d_r, d)
            elif method_in == "Heuristic":
                avg_machine_num = get_avg_machine_num_by_cut(cut_d_r, cut_d, curr_remain_q, node_level,
                                                             machine_type_node_level_list, u_r, q)
                if sum(cut_d) / len(cut_d) < avg_machine_num:
                    method = "mip"
                else:
                    method = "cg"
            else:
                # If the developer assigns an algorithm
                method = method_in
        valid_cut_list.append(cut_id)

        # Apply the selected algorithm
        if method == "cg":
            # Get the sub-problem's machines
            total_cpu_demand = sum([cut_d_r[i][0] * cut_d[i] for i in range(len(cut_d))]) * 1.1
            total_mem_demand = sum([cut_d_r[i][1] * cut_d[i] for i in range(len(cut_d))]) * 1.1
            cpu_supply = 0
            mem_supply = 0
            cut_q = np.zeros_like(q)
            for i in range(len(q)):
                if node_level != machine_type_node_level_list[i]:
                    continue
                elif curr_remain_q[i] <= 3:
                    continue
                elif curr_remain_q[i] * u_r[i][0] + cpu_supply < total_cpu_demand or \
                        curr_remain_q[i] * u_r[i][1] + mem_supply < total_mem_demand:
                    cpu_supply += curr_remain_q[i] * u_r[i][0]
                    mem_supply += curr_remain_q[i] * u_r[i][1]
                    cut_q[i] = curr_remain_q[i]
                    curr_remain_q[i] = 0
                else:
                    q_use = max(np.ceil((total_cpu_demand - cpu_supply) / u_r[i][0]),
                                np.ceil((total_mem_demand - mem_supply) / u_r[i][1]))
                    cut_q[i] = q_use
                    curr_remain_q[i] -= q_use
                    break

            # Apply column generation algorithm
            lag = 20
            increase_gap = 1e-4
            iter_max = est_time_out_dict[cut_id]
            cut_x, _, traffic = scheduler_column_generation(cut_p, u_r, cut_q, cut_d_r, cut_d, cut_s_type, iter_max,
                                                            lag, increase_gap, global_traffic, cut_service_node_level_list,
                                                            machine_type_node_level_list, cut_anti_affinity)
        else:
            # Apply mip-based algorithm
            time_out = est_time_out_dict[cut_id]
            cut_x, cut_q, traffic = scheduler_mip(cut_p, u_r, curr_remain_q, cut_d, cut_d_r, global_traffic, node_level,
                                                  machine_type_node_level_list, cut_anti_affinity, time_out,
                                                  heuri_init_flag, scale_rate, force_k8s_heuri)

        # store the schedule of each sub-problem
        cuts_service_2_orig_service_dict[cut_id] = cut_service_2_orig_service_list
        cuts_x_dict[cut_id] = cut_x
        cuts_q_dict[cut_id] = cut_q

    # Combine results
    combine_cuts_results_in_X_index(cuts_service_2_orig_service_dict, cuts_x_dict, cuts_q_dict, q, x_int, u_free,
                                    machine_type_2_index_list, d_r, valid_cut_list)

    return x_int


def estimate_runtime(cut_sets, max_time, max_cut_num, p):
    est_time_out_dict = {}
    est_ratio = defaultdict(float)
    est_traffic_ratio = defaultdict(float)

    # Count the number of services for each cut, except the last
    for service in range(len(cut_sets)):
        cut_id = cut_sets[service]
        if cut_id != max_cut_num - 1:
            est_ratio[cut_id] += 1.0
    total_service_num = sum(est_ratio.values())
    for key in est_ratio.keys():
        est_ratio[key] = float(est_ratio[key]) / float(total_service_num)

    for key in p.keys():
        if cut_sets[key[0]] == cut_sets[key[1]] and cut_sets[key[0]] != max_cut_num - 1:
            est_traffic_ratio[cut_sets[key[0]]] += p[key]
    total_traffic = sum(est_traffic_ratio.values())
    for key in est_traffic_ratio.keys():
        est_traffic_ratio[key] = est_traffic_ratio[key] / total_traffic

    # Assign time
    for key in est_ratio.keys():
        r = 1.1
        est_time_out_dict[key] = math.ceil( (r * est_ratio[key] + (2 - r) * est_traffic_ratio[key]) / 2 * max_time)

    return est_time_out_dict


def get_avg_machine_num_by_cut(cut_d_r, cut_d, curr_remain_q, node_level, machine_type_node_level_list, u_r, q):
    total_cpu_demand = sum([cut_d_r[i][0] * cut_d[i] for i in range(len(cut_d))]) * 1.1
    total_mem_demand = sum([cut_d_r[i][1] * cut_d[i] for i in range(len(cut_d))]) * 1.1
    cpu_supply = 0
    mem_supply = 0
    tmp_cut_q = np.zeros_like(q)
    tmp_curr_remain_q = copy.deepcopy(curr_remain_q)
    for i in range(len(q)):
        if node_level != machine_type_node_level_list[i]:
            continue
        elif tmp_curr_remain_q[i] <= 3:
            continue
        elif tmp_curr_remain_q[i] * u_r[i][0] + cpu_supply < total_cpu_demand or \
                tmp_curr_remain_q[i] * u_r[i][1] + mem_supply < total_mem_demand:
            cpu_supply += tmp_curr_remain_q[i] * u_r[i][0]
            mem_supply += tmp_curr_remain_q[i] * u_r[i][1]
            tmp_cut_q[i] = tmp_curr_remain_q[i]
            tmp_curr_remain_q[i] = 0
        else:
            q_use = max(np.ceil((total_cpu_demand - cpu_supply) / u_r[i][0]),
                        np.ceil((total_mem_demand - mem_supply) / u_r[i][1]))
            tmp_cut_q[i] = q_use
            tmp_curr_remain_q[i] -= q_use
            break
    avg_machine = sum(tmp_cut_q) / len(np.array(tmp_cut_q).nonzero()[0])
    return avg_machine
