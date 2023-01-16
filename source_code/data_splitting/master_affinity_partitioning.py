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


def separate_non_master(auxiliary_p, service_num, d, delete_ratio, cut_sets, deal_cut_id_list, target_cut_id,
                        max_master_service_num=None):
    """
    This function separate services which has is non-master, i.e., services that contributes very little to the total gained affinity

    :param auxiliary_p: affinity data.
    :param service_num: total number of services.
    :param d: number of containers for each service.
    :param delete_ratio: this defines what is master-services, i.e., services with top (1 - delete_ration) affinity is master services.
    :param cut_sets: cut_sets[i] == k represents that service i is categorized in the service set k.
    :param deal_cut_id_list: only partitioning services that are in cut sets whose cut id is in deal_cut_id_list.
    :param target_cut_id: set the non-master service to cut with cut_id = target_cut_id.
    :param max_master_service_num: this extends the definition of master-services, we hope that the number of master services may not exceed max_master_service_num.

    :return edge_weight_list:
    :return delete_edge_num:
    """

    # Construct data structure that is easier to process
    edge_num = len(auxiliary_p)
    edge_weight_list = []
    for key in auxiliary_p.keys():
        edge_weight_list.append([key[0], key[1], auxiliary_p[key]])
    edge_weight_list.sort(key=lambda x: x[2], reverse=False)
    total_weight = sum(auxiliary_p.values())
    # Count the degree of each service vertex
    degree_list = np.zeros(service_num, dtype=int)
    for key in auxiliary_p.keys():
        degree_list[key[0]] += 1
        degree_list[key[1]] += 1
    # Delete edges from the least weight to the largest weight
    curr_ratio = 0.0
    delete_edge_num = 0
    for edge in range(edge_num):
        if curr_ratio < delete_ratio:
            service1 = edge_weight_list[edge][0]
            service2 = edge_weight_list[edge][1]
            curr_ratio += edge_weight_list[edge][2] / total_weight

            degree_list[service1] -= 1
            degree_list[service2] -= 1
            delete_edge_num += 1
    related_service_num = sum(degree_list > 0)
    old_delete_edge_num = delete_edge_num
    if max_master_service_num is None:
        max_master_service_num = service_num
    for edge in range(old_delete_edge_num, edge_num):
        if related_service_num > max_master_service_num:
            service1 = edge_weight_list[edge][0]
            service2 = edge_weight_list[edge][1]
            degree_list[service1] -= 1
            degree_list[service2] -= 1
            delete_edge_num += 1
            related_service_num = sum(degree_list > 0)
    # Count the degree after the deletion of edges
    degree_list = np.zeros(service_num, dtype=int)
    separate_service_num = 0
    for edge in range(delete_edge_num, edge_num):
        degree_list[edge_weight_list[edge][0]] += 1
        degree_list[edge_weight_list[edge][1]] += 1
    delete_demand = 0
    # Count the non-master services
    for service in range(service_num):
        if degree_list[service] == 0 and cut_sets[service] in deal_cut_id_list:
            cut_sets[service] = target_cut_id
            delete_demand += d[service]
            separate_service_num += 1

    key_list = list(auxiliary_p.keys()).copy()
    for key in key_list:
        if cut_sets[key[0]] == target_cut_id or cut_sets[key[1]] == target_cut_id:
            auxiliary_p.pop(key)

    print("Data Splitting: master-affinity partitioning is finished, master service num %d,"
          " total service num %d, remain %d edges" % (service_num, sum(d), len(auxiliary_p)))
    return edge_weight_list, delete_edge_num
