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
import math
import copy
import random
import numpy as np


def service_partition(service_info_in, edge_info, node_info_in, random_distributed_edge, merge_num, part_dict, sever_type,
                      cut_anti_affinity):
    """
    Randomly partition the graph based on minimizing the weights.
    """
    # Prepare data
    service_info = copy.deepcopy(service_info_in)
    node_info = copy.deepcopy(node_info_in)

    for i in range(merge_num):
        # randomly pick an edge
        edge = pick_merged_edge(edge_info, random_distributed_edge)
        # merge the services
        merge_edge(service_info, edge_info, edge, part_dict, sever_type, cut_anti_affinity)

    # update the reward of the cut
    reward = calculate_reward(edge_info, service_info)

    res = get_complete_results(part_dict, node_info, service_info)

    return res, reward


def calculate_reward(edge_info, service_info):
    """
    Calculate the maximum possible gained affinity of the cut
    """

    return sum([(service_info[key[0]][3] == service_info[key[1]][3]) * edge_info[key] for key in edge_info.keys()])


def get_complete_results(part_dict, node_info, service_info):
    """
    Get the complete results based on the graph cuts.
    """

    # Corner case
    for key in part_dict.keys():
        if part_dict[key] is not None and len(part_dict[key]) == 1:
            # None is merged services, len == 1 is never-being-operated
            service_info[part_dict[key][0]] = -1
            part_dict[key] = None

    # Calculate ratio
    node_num = len(node_info)
    service_num = len(service_info)
    final_result = {}
    counter = 0
    for key in part_dict.keys():
        if part_dict[key] is not None:
            require_cpu, require_mem, deploy_num = get_part_require_resource(service_info, part_dict[key])

            need_deploy_num = deploy_num
            curr_node_type = 0
            while curr_node_type < node_num:
                if node_info[curr_node_type][2] <= 0:
                    curr_node_type += 1
                    continue
                if need_deploy_num < 1e-7:
                    break

                # If we cannot host more pods, continue
                if node_info[curr_node_type][1] < require_mem or node_info[curr_node_type][0] < require_cpu:
                    curr_node_type += 1
                    continue
                else:
                    # Calculate how many more pods can this pattern host
                    full_max = math.floor(min((node_info[curr_node_type][1] / require_mem),
                                              (node_info[curr_node_type][0] / require_cpu)))
                    full_ratio = min(full_max, need_deploy_num)
                    pattern = np.zeros(service_num)
                    pattern[part_dict[key]] = np.multiply(service_info[part_dict[key], 0], full_ratio / deploy_num)
                    pattern_num = min(math.floor(need_deploy_num / full_ratio), int(node_info[curr_node_type][2]))
                    real_deploy_num = pattern_num * full_max
                    # Store results
                    final_result[counter] = [pattern, pattern_num, curr_node_type]
                    counter += 1
                    node_info[curr_node_type][2] -= pattern_num
                    need_deploy_num -= real_deploy_num
                    curr_node_type += 1

    return final_result


def pick_merged_edge(edge_info, random_distributed_edge):
    """
    Randomly get two services to merge.
    """

    prob = random.random()
    target_index = bin_search(prob, random_distributed_edge)

    return random_distributed_edge[target_index][0], random_distributed_edge[target_index][1]


def check_can_merge(service1, service2, part1, part2, service_info, edge_info, part_dict, sever_type, cut_anti_affinity):
    """
    This function checks whether two cuts can be merged.
    """

    # Check whether the two cut are the same cut
    if part1 == part2:
        return False

    # Check whether the resource is less than the threshold resource
    unified_part = part_dict[part1] + part_dict[part2]
    require_cpu, require_mem, _ = get_part_require_resource(service_info, unified_part)
    if require_cpu > sever_type[0] or require_mem > sever_type[1]:
        return False

    if cut_anti_affinity:
        rule = cut_anti_affinity[0]
        if len(list(set(unified_part)&set(rule))) > 1:
            return False

    return True


def get_deploy_num(service_info, service_lists):
    """
    Each cut will be deployed by "piece", this function calculate one piece of the cut should contains how many containers
    """

    return round(min(np.array(service_info[service_lists, 0])))


def get_total_resource(service_info, service_lists):
    """
    Calculate the total cpu and memory of the cut's service
    """

    return np.sum(np.array(np.dot(service_info[service_lists, 1], service_info[service_lists, 0]))), \
            np.sum(np.array(np.dot(service_info[service_lists, 2], service_info[service_lists, 0])))


def get_part_require_resource(service_info, service_lists):
    """
    Calculate the request cpu and memory to deploy one "piece" of the cut's services
    """

    deploy_num = get_deploy_num(service_info, service_lists)
    total_cpu, total_mem = get_total_resource(service_info, service_lists)

    return total_cpu / deploy_num, total_mem / deploy_num, deploy_num


def merge_edge(service_info, edge_info, edge, part_dict, sever_type, cut_anti_affinity):
    """
    If we found two services that should be in the same cut, we merge it.
    """
    
    service1 = int(edge[0])
    service2 = int(edge[1])
    part1 = int(service_info[service1][3])
    part2 = int(service_info[service2][3])

    if check_can_merge(service1, service2, part1, part2, service_info, edge_info, part_dict, sever_type, cut_anti_affinity):
        # get the cut_id
        part_id = min(part1, part2)
        expired_part_id = max(part1, part2)

        # merge
        part_dict[part_id] += part_dict[expired_part_id]
        for service in part_dict[expired_part_id]:
            service_info[service][3] = part_id
        part_dict[expired_part_id] = None


def bin_search(prob, arr):
    """
    Binary search.
    """

    n = len(arr)
    if n == 0:
        return -1
    if n == 1:
        return 0

    s = 0
    e = n
    while s + 1 < e:
        mid = (s + e) // 2
        if prob < arr[mid][2]:
            e = mid
        else:
            s = mid
    return s
