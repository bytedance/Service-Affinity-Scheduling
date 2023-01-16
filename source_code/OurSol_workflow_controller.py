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

import time, copy, json
import numpy as np
from collections import defaultdict
import os
import sys
APP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.append(APP_PATH)
from source_code.utility.preprocess_data import read_from_json_stream
from source_code.data_splitting.data_splitting import data_splitting_workflow
from source_code.algorithm_selection.algorithm_selection import algorithm_selection_and_apply
from source_code.scheduling_algorithm_pool.scheduler_first_fit.scheduler_first_fit import solve_remain_demands
from source_code.utility.result_check import result_check, get_schedule_by_optimizing_x


def OurSol_workflow_controller(input_path, output_path, max_time=60, method_in=None, print_flag="1"):
    """
    This function controls the workflow of the entire solution, including data processing, data splitting, algorithm selection etc.

    :param input_path: the path of the input file
    :param output_path: the path of the output file
    :param max_time: maximum runtime
    :param method_in: the user can assign the method to solve the problem, if not assigned, we will use an heuristic-gcn hybrid algorithm selection to select the method
    :return:
        schedule: schedule[xxx] is a list of containers that are deployed on the machine with IP = xxx
    """
    if print_flag != "1":
        sys.stdout = open(os.devnull, 'w')

    start_time = time.time()
    print('--- Proceed RASA algorithm, input file is %s ---\r\n' % input_path)

    # Process the file and construct the related data structure
    with open(input_path, 'r', encoding='utf-8', ) as fp:
        input_json_stream = json.load(fp)
    x_old, p, d, d_r, s_type, s_full, u_r_type, u_r_full, q, service_index_2_service_name_list, \
     service_name_2_service_index_dict, service_node_level_list, machine_index_2_machine_ip_list, \
     machine_type_index_2_machine_index_dict, machine_index_2_machine_type_index_list, \
     machine_type_index_2_node_level_list, anti_affinity_list, global_traffic, \
     service_index_2_container_index_dict, container_index_2_container_name_list = \
     read_from_json_stream(input_json_stream)

    # Data splitting
    service_num = len(d)
    cut_sets, max_cut_num = data_splitting_workflow(p, service_num, d, service_node_level_list)

    # Algorithm selection and apply the selected algorithm to the sub-problem
    u_free = copy.deepcopy(u_r_full)
    x_int = algorithm_selection_and_apply(p, u_r_type, q, d, d_r, s_type, cut_sets, max_cut_num, service_node_level_list,
                                          global_traffic, machine_type_index_2_node_level_list, u_free,
                                          machine_type_index_2_machine_index_dict, anti_affinity_list,
                                          max_time=max_time, method_in=method_in)

    # Some containers might not be scheduled yet, solve them with first fit
    x_int, u_free_cp = solve_remain_demands(d_r, d, x_int, u_free, s_full, service_node_level_list, anti_affinity_list)

    # Result check
    print('\r\n')
    print('----- Computation is finished(OurSol), now result checking... -----')
    print('Checking for input: %s ' % input_path)
    old_affinity, old_affinity_ratio, new_affinity, new_affinity_ratio = \
        result_check(x_old, x_int, p, d, d_r, u_r_full, s_full, anti_affinity_list, global_traffic)

    # Construct a new schedule and store it in the given file
    schedule = get_schedule_by_optimizing_x(x_int, service_index_2_container_index_dict,
                                            machine_index_2_machine_ip_list, container_index_2_container_name_list)
    with open(output_path, 'w', encoding='utf-8', ) as fp:
        json.dump(schedule, fp, indent=True)
    # print(schedule)

    end_time = time.time()
    print('Total runtime: %.3f seconds' % (end_time-start_time))

    if print_flag != "1":
        sys.stdout = sys.__stdout__
        print("OurSol: gained affinity %.2f%%, runtime = %.2f seconds" % (new_affinity_ratio, end_time - start_time))

    return schedule, new_affinity_ratio, end_time-start_time


# # Debug
# if __name__ == '__main__':
#     # file_path = '../dataset/M1.json'
#     file_path = '../dataset/M2.json'
#     # file_path = '../dataset/M3.json'
#     # file_path = '../dataset/M4.json'
#
#     max_time = 30
#     # method_in = None
#     # method_in = 'cg'
#     method_in = 'mip'
#     # method_in = 'heuristic'
#
#     output_path = '../output/OurSol_output_testing.json'
#
#     OurSol_workflow_controller(file_path, output_path, max_time=max_time, method_in=method_in)
#
#     pass


if __name__ == '__main__':
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    if len(sys.argv) > 3:
        max_time = int(sys.argv[3])
    else:
        max_time = 60
    if len(sys.argv) > 4:
        method_in = sys.argv[4]
        if method_in == "null":
            method_in = None
    else:
        method_in = None
    if len(sys.argv) > 5:
        print_flag = sys.argv[5]
    else:
        print_flag = "1"

    OurSol_workflow_controller(input_path, output_path, max_time, method_in, print_flag)
