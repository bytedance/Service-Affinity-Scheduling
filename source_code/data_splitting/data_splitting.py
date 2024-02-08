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
from collections import defaultdict
import os
import sys
APP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..')
sys.path.append(APP_PATH)
from source_code.data_splitting.non_affinity_partitioning import separate_not_in_link
from source_code.data_splitting.master_affinity_partitioning import separate_non_master
from source_code.data_splitting.compatibility_partitioning import separate_different_node_level
from source_code.data_splitting.balanced_partitioning import separate_balanced_cuts
from source_code.data_splitting.random_partitioning import random_partitioning
import numpy as np
import math


def data_splitting_workflow(p, service_num, d, service_node_level_list, delete_ratio=None, K=None,
                            splitting_method="default"):
    """
    Control the workflow of data splitting, including four different data splitting.

    :param p: affinity data, including the edges and affinities data.
    :param service_num: the number of services
    :param d: the number of containers for each service
    :param service_node_level_list: the node-level name of each service
    :param delete_ratio: the master-affinity partitioning parameter, top (1-delete_Ratio) affinity service is the master services
    :param K: the loss-minimizing equally partitioning parameter, partitioning into K balanced parts
    :return cut_sets: cut_sets[i] is the cut_id of the service i
    :return max_cut_num: the number of different cuts
    """

    # prepare some variables
    cut_sets = np.zeros(service_num, dtype=int)
    auxiliary_p = copy.deepcopy(p)
    if delete_ratio is None:
        # Originally delete edges, now delete services
        delete_ratio = 0.001 * math.sqrt(service_num)
    if K is None:
        # Don't worry if K is None, we deal with it later
        pass

    # Multi-stage feature-oriented partitioning
    print("Proceed multi-stage data splitting...")
    if splitting_method == "default":
        separate_not_in_link(auxiliary_p, service_num, cut_sets, [0], -1)
        separate_non_master(auxiliary_p, service_num, d, delete_ratio, cut_sets, [0], -1)
        cut_node_level_2_index_dict = separate_different_node_level(auxiliary_p, service_num, cut_sets,
                                                                    service_node_level_list, [0], 100000)
        max_cut_num, cut_sets = separate_balanced_cuts(auxiliary_p, service_num, cut_sets, service_node_level_list, K)
    elif splitting_method == "kahip":
        pass
    elif splitting_method == "nopart":
        max_cut_num = 2
        cut_sets = np.zeros(service_num, dtype=int)  # cut_sets remain the same
    elif splitting_method == "randompart":
        max_cut_num, cut_sets = random_partitioning(cut_sets, service_num, K)
    else:
        raise ValueError\
            ("INVALID data splitting method name, please enter default/kahip/nopart/randompart as the param.")

    # By the way, re-indexing cut_sets
    cut_ids = {}
    for service in range(service_num):
        cut_ids[service] = True
    re_index_cut_sets(cut_sets, orig_cut_id_list=list(cut_ids.keys()))

    # Print out the overall results
    print("Multi-stage data splitting is finished, here's the overall situation:")
    print_overall_situation_after_data_splitting(p, d, cut_sets)

    return cut_sets, max_cut_num


def re_index_cut_sets(cut_sets, orig_cut_id_list):
    """
    This function will re-indexes the cut ids to the range of [0, max_cut_num - 1]
    """

    # Construct the mapping from original cut_id to new_cut_id
    max_cut_num = len(orig_cut_id_list)
    map_from_orig_id_2_new_id = {}
    curr_index = 0
    for cut_id in orig_cut_id_list:
        if cut_id == -1:
            map_from_orig_id_2_new_id[cut_id] = max_cut_num - 1
        else:
            map_from_orig_id_2_new_id[cut_id] = curr_index
            curr_index += 1

    # Change the indexes of cut_sets
    for service in range(cut_sets.shape[0]):
        cut_sets[service] = map_from_orig_id_2_new_id[cut_sets[service]]


def print_overall_situation_after_data_splitting(p, d, cut_sets):
    """
    Count the number of containers for each cut and the amount of affinity for each cut, then print them out.
    """
    each_cut_service_num = defaultdict(int)
    each_cut_container_num = defaultdict(int)
    each_cut_traffic = defaultdict(float)
    # Count the number of containers for each cut_id
    for service, cut_id in enumerate(list(cut_sets)):
        each_cut_service_num[cut_id] += 1
        each_cut_container_num[cut_id] += d[service]

    # Count the amount of traffic for each cut_id
    for key in p.keys():
        if cut_sets[key[0]] == cut_sets[key[1]]:
            cut_id = cut_sets[key[0]]
            each_cut_traffic[cut_id] += p[key]
    total_traffic = sum(p.values())

    for key in range(len(each_cut_service_num)):
        print(" cut_id = %d, the number of service = %d, the number of containers = %d, the amount of affinity = %.2f%%" %
              (key, each_cut_service_num[key], each_cut_container_num[key], each_cut_traffic[key] * 100.0 / total_traffic))
