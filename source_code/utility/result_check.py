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
from collections import defaultdict


def result_check(x_old, x_new, p, d, d_r, u_full, s_total, anti_affinity, global_traffic):
    """
    Check the scheduling results, including checking constraints, calculating objectives, calculate key indicators.

    :param x_old: the original schedule of containers to machines
    :param x_new: the computed schedule of containers to machines
    :param p: affinity data
    :param d: the number of containers for each services
    :param d_r: service's resource request mat
    :param u_full: machine's resource mat
    :param s_total: whether a service container can be placed on a machine
    :param anti_affinity: anti affinity constraint
    :param global_traffic: total affinity of the given cluster
    :return:
        old_affinity: the gained affinity from the original schedule
        old_affinity_ratio: the ratio of gained affinity to the total affinity in the original schedule
        new_affinity: the gained affinity from the optimized schedule
        new_affinity_ratio: the ratio of gained affinity to the total affinity in the optimized schedule
    """
    # print('Result checking...')
    u_free_new = u_full - x_new.T @ d_r
    check_basic_constrain(x_new, u_full, u_free_new, d_r, d, s_total, anti_affinity)
    get_cpu_mem_occupancy_ratio(u_full, u_free_new, x_old, x_new)
    old_affinity, old_affinity_ratio = calculate_local_traffic(x_old, d, p, prefix_str='Before the optimization: ', global_traffic=global_traffic)
    new_affinity, new_affinity_ratio = calculate_local_traffic(x_new, d, p, prefix_str='After the optimization: ', global_traffic=global_traffic)
    return old_affinity, old_affinity_ratio, new_affinity, new_affinity_ratio


def check_basic_constrain(x, u_full, u_free, d_r, d, s_total, anti_affinity):
    """
    Check the basic constraints and print out the results
    """

    print("Check basic constraints...")

    # Check the integer constraints
    if x.dtype == 'int64':
        print("1. [Good] The integer constraints.")
    else:
        print("1. [Bad] The integer constraints.!")

    # Check the resource constraints
    if np.min(u_free) >= 0.0:
        print("2. [Good] The resource constraints.")
    else:
        print("2. [Bad] The resource constraints!")
        print('Min resource:', np.min(u_free))

    # Check the demand constraints
    if x.sum(axis=1).tolist() == list(d):
        print("3. [Good] The demand constraints.")
    else:
        print("3. [Bad] The demand constraints!")

    # Check the compatibility constraints
    if np.isin(np.nonzero(x), np.nonzero(s_total.T)).all():
        print("4. [Good] The compatibility constraints.")
    else:
        print("4. [Bad] The compatibility constraints!")

    # Check the anti-affinity constraints
    flag = True
    for rule in anti_affinity:
        if max(x[rule, :].sum(axis=0)) > 1:
            flag = False
    if flag:
        print("5. [Good] The anti-affinity constraints.")
    else:
        print("5. [Bad] The anti-affinity constraints!")


def get_cpu_mem_occupancy_ratio(u_full, u_free, x_old, x_new):
    """
    Calculate the cpu usage and memory usage.
    """

    source_usage = (u_full - u_free) / u_full
    cpu_usage = source_usage[:, 0]
    mem_usage = source_usage[:, 1]
    before_opt_empty = np.where(x_old.sum(axis=0) == 0)[0].shape[0]
    after_opt_empty = np.where(x_new.sum(axis=0) == 0)[0].shape[0]

    cpu_mean = cpu_usage.sum()/(np.array(u_full).shape[0] - after_opt_empty)
    mem_mean = mem_usage.sum()/(np.array(u_full).shape[0] - after_opt_empty)

    print(f'Average CPU usage = {cpu_mean}, average memory usage = {mem_mean} \r\n'
          f'Emptied number of machines before and after the optimization{before_opt_empty, after_opt_empty}')


def calculate_local_traffic(x, d, p, prefix_str='', global_traffic=None, print_flag=True):
    """
    Calculate the gained affinity.

    :param x: the given schedule of containers to machines
    :param d: the number of containers for each service
    :param p: affinity data
    :param prefix_str: the string that is printed out
    :param global_traffic: the total affinity of the given cluster
    :return: two variables, first one is the amount of gained affinity, second the proportion the gained affinity to total affinity
    """

    if global_traffic is None:
        global_traffic = sum(p.values())
    if global_traffic == 0.0:
        return 0.0, 0.0
    proximity = 0.0
    for key in p.keys():
        psm1 = key[0]
        psm2 = key[1]
        psm1_deployed = x[psm1, :] / d[psm1]
        psm2_deployed = x[psm2, :] / d[psm2]
        r = np.array(psm1_deployed < psm2_deployed)
        not_r = np.subtract(np.ones(len(r)), r)
        local_traffic = p[key] * np.sum(np.add(np.multiply(r, psm1_deployed), np.multiply(not_r, psm2_deployed)))
        proximity += local_traffic
    if print_flag:
        print(prefix_str + 'The gained affinity = %.3f%%, absolute value = %.5f' % (proximity*100.0/global_traffic, proximity))
    return proximity, proximity*100.0/sum(p.values())


def get_schedule_by_optimizing_x(x_int, service_index_2_container_index_dict, machine_ip_list,
                                container_index_2_container_name_list):
    """
    Get a comprehensive schedule by x_int

    :param x_int: optimized schedule in index form
    :param service_index_2_container_index_dict: the mapping from service to container name list
    :param machine_ip_list: the mapping from machine index to machine IP
    :return:
        schedule: schedule[xxx] is a list of containers that are deployed on the machine with IP = xxx
    """

    schedule = defaultdict(list)
    for machine in range(x_int.shape[1]):
        non_zeros_services = list(np.array(x_int[:, machine]).nonzero()[0])
        ip = machine_ip_list[machine]
        for service in non_zeros_services:
            for _ in range(x_int[service][machine]):
                cont_name = container_index_2_container_name_list[service_index_2_container_index_dict[service].pop()]
                schedule[ip].append(cont_name)
    return schedule

