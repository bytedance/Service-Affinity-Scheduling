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

from gurobipy import *
import numpy as np
import os
import sys
APP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../..')
sys.path.append(APP_PATH)
from source_code.scheduling_algorithm_pool.scheduler_column_generation.initilal_solution.heuristic.\
    initial_column_generation import data_processing_init_column
from source_code.scheduling_algorithm_pool.scheduler_column_generation.column_generation_loops import\
    column_generation_loop_controller
from source_code.scheduling_algorithm_pool.scheduler_column_generation.rounding.post_processing_rounding import \
    multi_dimensional_carry_based_rounding
from source_code.utility.result_check import calculate_local_traffic


def scheduler_column_generation(p, u_r, q, d_r, d, s_type, iter_max, lag, increase_gap,
                                global_flow, service_node_level_list,
                                machine_type_node_level_list, anti_affinity_list):
    """
    Workflow controller of the column generation algorithm, including initial column generation, multiple-iterations, and rounding.

    :param p: affinity data
    :param u_r: resource matrix for each machine specification
    :param q: number of machines for each machine specification
    :param d_r: resource request for each services
    :param d: number of containers for each service
    :param s_type: compatibility matrix
    :param iter_max: maximum iteration number for column generation algorithms
    :param lag: terminate condition parameters, if the obj. didn't improve for more than tol in lag iterations, we terminate
    :param increase_gap:terminate condition parameters, see (param: lag)
    :param global_flow: total affinity
    :param service_node_level_list: node-level list of each services
    :param machine_type_node_level_list: node-level of each machine specification
    :param anti_affinity_list: new feature - anti-affinity 
    :return:
    """
    print("Proceed column generation algorithm...")

    # new features: anti-affinity
    if anti_affinity_list:
        rule = anti_affinity_list[0]
    else:
        rule = []

    # Old issues from old version of code
    d_t = []
    for service in range(len(d)):
        d_t.append([service])

    # acceleration for column generation, finding initial columns
    useful_machine_type_list = get_useful_machine_type(s_type)
    heuri_a, heuri_mid = data_processing_init_column(p, d, q, d_r, u_r, d_t, s_type, service_node_level_list,
                                                     machine_type_node_level_list, rule)

    # main-body of column generation algorithm, iterating for multiple times with master and pricing problem solving
    final_a, final_y, object_list = column_generation_loop_controller(d_t, p, d, q, d_r, u_r, s_type,
                                                                      heuri_a, heuri_mid, iter_max, lag, increase_gap,
                                                                      useful_machine_type_list, global_flow, rule)

    # Post-processing of column generation algorithm: rounding the fractional solution to an integer solution
    x_int, U_r_full = multi_dimensional_carry_based_rounding(d_r, d, q, u_r, final_a, final_y, d_t, p, s_type,
                                                             max_iter=1, random_power=1)

    # Calculate the gained affinity of the column generation solving
    traffic_cg, _ = calculate_local_traffic(x_int, d, p)

    return x_int, U_r_full, traffic_cg


def get_useful_machine_type(s_type):
    """
    Some machines can never be used due to compatibility constraints, these are useless machines, filter them out.

    :param s_type: compatibility matrix for machine specification
    :return:
        useful_machine_type_list: list of machine specifications that can be used for column generation
    """

    machine_type_num = s_type.shape[0]
    useful_machine_type_list = np.zeros(machine_type_num, dtype=int)
    s_machines = np.sum(s_type, axis=1)
    for machine_type in range(machine_type_num):
        if s_machines[machine_type] > 0:
            useful_machine_type_list[machine_type] = 1
    return useful_machine_type_list

