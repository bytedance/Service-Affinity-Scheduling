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
import time
import numpy as np
from collections import defaultdict
import copy


def scheduler_first_fit_full(d_r, remain_demand, u_free, s_full, service_node_level_list, x_int, anti_affinity):
    """
    This function computes a first-fit schedule of a given input.
    Note that the values of the input matrix x_int will be changed

    :param d_r: resource mat for services
    :param remain_demand: the number of containers that needs to be deployed for each service
    :param u_free: free amount of resources for each machines
    :param s_full: whether a service container can be placed on a machine, s_full[machine, service] = 1 is allowed
    :param service_node_level_list: the node-level name of each service
    :param x_int: x_int[service, machine] is the number of containers at machine for service
    :param anti_affinity: anti affinity information
    :return: the first variable the deviation of x_int, that is x_after - x_before, the second variable is the free resource mat of machines
    """

    print('Proceed first-fit algorithm...')
    d = remain_demand
    start = time.time()
    u_free_cp = np.maximum(u_free, 0.0)
    x_before_ff = copy.deepcopy(x_int)
    flag = True

    # Deal with the anti-affinity constraint.
    # Now we only support one anti-affinity rule, can be extended.
    if anti_affinity:
        rule = anti_affinity[0]
        can_place_list = (x_int[rule, :].sum(axis=0) == 0).astype(int)
        tmp_rule = np.zeros(len(d), dtype=int)
        tmp_rule[rule] = 1
        merge_services = np.multiply(np.array(d) > 0, tmp_rule).nonzero()[0]
        if sum(d[merge_services]) > sum(can_place_list):
            flag = False
        else:
            pattern = (0.0, 0.0, 'tmp')
            tmp_merge_demand = {pattern: sum(d[merge_services])}
            deploy_by_first_fit(x_int, can_place_list, d, merge_services, u_free_cp, pattern, tmp_merge_demand)
            d[merge_services] = 0

    # First-fit scheduler: merge different specification for fast speed
    pattern_list = list(zip(d_r[:, 0], d_r[:, 1], service_node_level_list))
    merged_demand_dict = defaultdict(int)
    merged_services_dict = defaultdict(list)
    for index, service_type in enumerate(pattern_list):
        merged_demand_dict[service_type] += d[index]
        merged_services_dict[service_type].append(index)  # Data type is [service，demand]
    for pattern in merged_demand_dict.keys():
        if merged_demand_dict[pattern] == 0:
            continue
        can_place_list = get_can_place(pattern[0], pattern[1], u_free_cp, s_full[:, merged_services_dict[pattern][0]],
                                       merged_demand_dict[pattern])
        if sum(can_place_list) < merged_demand_dict[pattern]:
            flag = False
            continue
        deploy_by_first_fit(x_int, can_place_list, d, merged_services_dict[pattern], u_free_cp, pattern, merged_demand_dict)

    if not flag:
        print("First-fit error, cannot place all containers.")
    end = time.time()
    print("First-fit algorithm is finished, runtime = %.2f" % (end-start))
    return x_int - x_before_ff, u_free_cp


def get_can_place(cpu, mem, u_free, s_line, max_demand):
    """
    This function will computes the available machines for the given service.
    can_place_list[i] is the number of containers that the machine i can host.

    :param cpu: the service's CPU request
    :param mem: the service's memory request
    :param u_free: the free resource mat of machines
    :param s_line: whether the service can be placed on the machine
    :param max_demand: the number of containers that the service demands
    :return: return can_place_list
    """
    u_service_cpu = np.multiply(u_free[:, 0], s_line)
    u_service_mem = np.multiply(u_free[:, 1], s_line)
    if cpu == 0.0:
        c1 = max_demand * (u_service_cpu > 0)
    else:
        c1 = np.floor(u_service_cpu / cpu)
    if mem == 0.0:
        c2 = max_demand * (u_service_mem > 0)
    else:
        c2 = np.floor(u_service_mem / mem)
    r = np.array(c1 < c2)
    not_r = np.subtract(np.ones(len(r)), r)
    can_place_list = np.add(np.multiply(r, c1), np.multiply(not_r, c2))
    return can_place_list


def deploy_by_first_fit(x_ff, can_place_list, d, services_list, u_free, pattern, merged_demand_dict):
    """
    This function assign the containers to specific machines.
    """

    # maintain d, can_place_list, x_ff, u_free, deploy_plan
    deploy_plan = np.zeros(len(can_place_list), dtype=int)
    # get the non-zero elements of can_place_list
    non_zero_nodes = list(can_place_list.nonzero()[0])
    service_order = 0
    node_order = 0
    while service_order < len(services_list) and node_order < len(non_zero_nodes):
        # double-pointers to deploy the containers
        service = services_list[service_order]
        node = non_zero_nodes[node_order]
        can_place_in_node = can_place_list[node]
        if can_place_in_node > 0 and d[service] > 0:
            if d[service] > can_place_in_node:
                # One machine cannot host the containers
                place_num = can_place_in_node
                node_order += 1
            else:
                place_num = d[service]
                service_order += 1
            # maintain key variables
            x_ff[service, node] += place_num
            d[service] -= place_num
            deploy_plan[node] += place_num
            can_place_list[node] -= place_num
            merged_demand_dict[pattern] -= place_num
        elif can_place_in_node == 0:
            node_order += 1
        elif d[service] == 0:
            service_order += 1
    # maintain resource mat
    u_free[:, 0] -= (deploy_plan * pattern[0])
    u_free[:, 1] -= (deploy_plan * pattern[1])


def solve_remain_demands(d_r, d, x_int, u_free, s_full, service_node_level_list, anti_affinity):
    """
    If there are any container that is not scheduled yet, use first fit for the last try
    """
    print('\r\n')

    service_num = len(d)
    # Get the number of containers that is required to deploy
    need_to_deploy = np.zeros(service_num, dtype=int)
    for service in range(service_num):
        need_to_deploy[service] = (d[service] - np.sum(x_int[service, :]))

    remain_total_deploy = np.sum(need_to_deploy)
    print('Solve remain demands...The remaining demand number is %d.' % remain_total_deploy)
    if remain_total_deploy == 0:
        return x_int, u_free
    x_increment, u_free_cp = scheduler_first_fit_full(d_r, need_to_deploy, u_free, s_full, service_node_level_list, x_int,
                                                           anti_affinity)
    unsolve_demand = remain_total_deploy - np.sum(np.sum(x_increment))
    print('Cannot solve %d containers (if the number is 0, then it is cool, otherwise it is not cool).' %
          unsolve_demand)
    return x_int, u_free_cp
