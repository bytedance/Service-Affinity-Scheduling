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
import os
import sys
APP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..')
sys.path.append(APP_PATH)
from baselines.POP.random_partitioning import POP_pre_partitioning, POP_client_partitioning
from baselines.POP.MIP_model import MIP_model
from source_code.utility.preprocess_data import read_from_json_stream
from source_code.scheduling_algorithm_pool.scheduler_first_fit.scheduler_first_fit import solve_remain_demands
from source_code.utility.result_check import result_check, get_schedule_by_optimizing_x


def POP_workflow_controller(input_path, output_path, max_time=60, part_num=4, print_flag="1"):
    """
    This function uses POP algorithm to compute the schedule of containers to machines, aiming to maximize service affinity.

    :param input_path: the path of input file
    :param output_path: the path of output file
    :param max_time: we allow user to tune the runtime of POP algorithm(but only roughly) via this parameter
    :param part_num: the number of partitions that POP uses
    :return:
        schedule: schedule[xxx] is a list of containers that are deployed on the machine with IP = xxx
    """
    if print_flag != "1":
        sys.stdout = open(os.devnull, 'w')

    start_time = time.time()
    print('--- Proceed POP algorithm, input file is %s ---\r\n' % input_path)

    # Process the file and construct the related data structure
    with open(input_path, 'r', encoding='utf-8', ) as fp:
        input_json_stream = json.load(fp)
    x_old, p, d, d_r, s_type, s_full, u_r_type, u_r_full, q, service_index_2_service_name_list, \
     service_name_2_service_index_dict, service_node_level_list, machine_index_2_machine_ip_list, \
     machine_type_index_2_machine_index_dict, machine_index_2_machine_type_index_list, \
     machine_type_index_2_node_level_list, anti_affinity_list, global_traffic, \
     service_index_2_container_index_dict, container_index_2_container_name_list = \
     read_from_json_stream(input_json_stream)

    # pre-partitioning the problem
    service_num = len(d)
    cut_sets = np.zeros(service_num, dtype=int)
    max_cut_num, est_time_each_cut_dict = POP_pre_partitioning(cut_sets, p, d, service_node_level_list, max_time)

    # For each compatibility set: randomly partitioning into part_num parts, i.e., POP-part_num algorithm
    x_int = np.zeros([service_num, u_r_full.shape[0]], dtype=int)
    for target_id in range(max_cut_num - 1):
        # Get the POP partitioning data: including client and servers
        cut_d_dict, cut_q_dict, cut_u_full_dict, est_part_max_time_dict = \
            POP_client_partitioning(cut_sets, d, target_id, q, u_r_full, u_r_type, service_node_level_list,
                                    machine_type_index_2_node_level_list, est_time_each_cut_dict[target_id], part_num)

        # Apply MIP algorithm
        for key in cut_d_dict.keys():
            # Run MIP algorithm
            x_increment = MIP_model(p, cut_d_dict[key], d_r, cut_u_full_dict[key], [],
                                    time_out=est_part_max_time_dict[key])

            # Store results
            x_int = x_int + x_increment

    # Post-processing: get u_free
    u_free = u_r_full - x_int.T @ d_r

    # Some containers might not be scheduled yet, solve them with first fit
    x_int, u_free_cp = solve_remain_demands(d_r, d, x_int, u_free, s_full, service_node_level_list, anti_affinity_list)

    # Result check
    print('\r\n')
    print('----- Computation is finished(POP), now result checking... -----')
    print('Checking for input: %s ' % input_path)
    old_affinity, old_affinity_ratio, new_affinity, new_affinity_ratio = \
        result_check(x_old, x_int, p, d, d_r, u_r_full, s_full, anti_affinity_list, global_traffic)

    # Construct a new schedule and store it in the given file
    schedule = get_schedule_by_optimizing_x(x_int, service_index_2_container_index_dict,
                                            machine_index_2_machine_ip_list, container_index_2_container_name_list)
    with open(output_path, 'w', encoding='utf-8', ) as fp:
        json.dump(schedule, fp, indent=True)

    end_time = time.time()
    print('Total runtime: %.3f seconds' % (end_time-start_time))

    if print_flag != "1":
        sys.stdout = sys.__stdout__
        print("POP: gained affinity %.2f%%, runtime = %.2f seconds" % (new_affinity_ratio, end_time - start_time))
        # print("%.12f, %.8f" % (new_affinity, end_time - start_time))

    return schedule, new_affinity_ratio, end_time-start_time

#
# # Debug
# if __name__ == '__main__':
#     # file_path = '../../dataset/M1.json'
#     # file_path = '../../dataset/M2.json'
#     file_path = '../../dataset/M3.json'
#     # file_path = '../../dataset/M4.json'
#
#     max_time = 10
#     part_num = 4
#
#     output_path = '../../output/POP_output_testing.json'
#
#     POP_workflow_controller(file_path, output_path, max_time, part_num)
#
#     pass

if __name__ == '__main__':
    input_path = sys.argv[1]
    output_path = sys.argv[2]
    max_time = int(sys.argv[3])
    part_num = int(sys.argv[4])

    if len(sys.argv) > 5:
        print_flag = sys.argv[5]
    else:
        print_flag = "1"

    POP_workflow_controller(input_path, output_path, max_time, part_num, print_flag)
