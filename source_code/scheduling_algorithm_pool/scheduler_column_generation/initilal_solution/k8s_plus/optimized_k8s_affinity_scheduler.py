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
import random


"""
An optimized k8s simulator as the initial solution, i.e., an online heuristic algorithm
"""


def run_simulated_k8s(cut_d, d_r, u_full, s_full, p, anti_affinity_list):
    """
    Workflow controller of the simulated k8s+ algorithm
    """

    print('Proceed k8s+ simulation...')

    d = copy.deepcopy(cut_d)
    service_num = len(d)
    node_num = s_full.shape[0]

    # Since we simulate a k8s process, we need to shuffle the containers to determine the order of online scheduling
    if anti_affinity_list:
        rule = anti_affinity_list[0]
    else:
        rule = []
    counter = 0
    shuffle_order = [-1] * (sum(d) - sum(np.array(d)[rule]))
    for service in range(service_num):
        if d[service] == 0 or (service in rule):
            continue
        shuffle_order[counter: counter + d[service]] = [service] * d[service]
        counter += d[service]
    random.shuffle(shuffle_order)
    shuffle_order = rule + shuffle_order
    u_free = copy.deepcopy(u_full)
    x_curr = np.zeros([service_num, node_num], dtype=int)

    # simulation of k8s, an optimized scoring function
    for i in range(len(shuffle_order)):
        service = shuffle_order[i]

        # filter machines that do not satisfied the constraints
        good_nodes = filter_nodes(service, u_free, d_r, s_full, rule, x_curr)

        if len(good_nodes) == 0:
            continue

        # scoring according to the affinity
        node = scoring_nodes(service, good_nodes, x_curr, p, d)

        # write to the values
        deploy(service, u_free, d_r, node, x_curr)
    return x_curr, u_free


def filter_nodes(service, u_free, d_r, s_full, rule, x_curr):
    """
    Filter machines according to the constraints, including resource constraints, compatibility constraints etc.
    """

    tol = 1e-9

    filter1 = np.greater(u_free[:, 0], d_r[service][0]-tol)
    filter2 = np.greater(u_free[:, 1], d_r[service][1]-tol)
    filtered_mid = np.multiply(s_full[:, service], np.multiply(filter1, filter2))
    if rule and (service in rule):
        filter0 = (x_curr[rule, :].sum(axis=0) == 0)
        filtered = np.multiply(filtered_mid, filter0)
    else:
        filtered = filtered_mid

    return list(filtered.nonzero()[0])


def scoring_nodes(service, good_nodes, x_curr, p, d):
    aff_list = calculate_affinity(service, good_nodes, x_curr, p, d)
    return good_nodes[np.argmax(aff_list)]


def calculate_affinity(service, good_nodes, x_curr, p, d):
    """
    Part of the scoring function, calculate the affinity to each machine
    """

    aff_list = np.zeros(len(good_nodes))
    for key in p.keys():
        service1 = key[0]
        service2 = key[1]
        if d[service1] == 0 or d[service2] == 0:
            continue
        if service1 == service or service2 == service:
            service1_deployed = x_curr[service1, good_nodes] / d[service1]
            service2_deployed = x_curr[service2, good_nodes] / d[service2]
            tmp = np.zeros(len(good_nodes))
            if service1 == service:
                valid = list(service2_deployed.nonzero()[0])
                if len(valid) != 0:
                    tmp[valid] = integrate_affinity_function(service1_deployed[valid], service2_deployed[valid])
            else:
                valid = list(service1_deployed.nonzero()[0])
                if len(valid) != 0:
                    tmp[valid] = integrate_affinity_function(service2_deployed[valid], service1_deployed[valid])
            aff_list += p[key] * tmp
    return aff_list


def integrate_affinity_function(service_source_deploy_list, service_target_deploy_list):
    """
    Scoring function, concerning the affinity. The more affinity we could gained, the higher score the machine it is.
    """
    # return np.divide(1, 1 + np.square(
    #                     np.subtract(1, np.divide(service_source_deploy_list, service_target_deploy_list))
    # ))
    return np.subtract(1, np.divide(service_source_deploy_list, service_target_deploy_list) )


def deploy(service, u_free, d_r, node, x_curr):
    """
    Update the data structure
    """
    u_free[node][0] -= d_r[service][0]
    u_free[node][1] -= d_r[service][1]
    x_curr[service][node] += 1
