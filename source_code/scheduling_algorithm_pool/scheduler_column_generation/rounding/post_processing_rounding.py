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
import math
import time
import random

"""
Multiple-phase rounding algorithm for column generation
"""


def TakeOne(ele):
    return ele[0]


def TakeContribution(ele):
    return ele[1]


def TakeThird(ele):
    return ele[2]


def MaintainingPatternContribution(machine_type_num, pattern_mat, d_t, p, d_t_count, y_continuous,
                                   pattern_contribution_list):
    before_total_cost = 0
    total_pattern_num = 0
    for machine_type in range(machine_type_num):
        pattern_num = pattern_mat[machine_type].shape[0]
        total_pattern_num = total_pattern_num + pattern_num
        for pattern in range(pattern_num):
            cost = 0
            for key in p.keys():
                psm1 = key[0]
                psm2 = key[1]
                psm1_deployed = 0
                psm2_deployed = 0
                for i in d_t[psm1]:
                    psm1_deployed = psm1_deployed + pattern_mat[machine_type][pattern][i]
                for j in d_t[psm2]:
                    psm2_deployed = psm2_deployed + pattern_mat[machine_type][pattern][j]
                cost = cost + p[key] * min(psm1_deployed / d_t_count[psm1],
                                           psm2_deployed / d_t_count[psm2])
            pattern_contribution_list.append([machine_type, pattern, cost, y_continuous[machine_type][pattern]])
            before_total_cost = before_total_cost + cost * y_continuous[machine_type][pattern]
    return before_total_cost, total_pattern_num


def CalculateYintCost(machine_type_num, pattern_mat, y_int, pattern_contribution_list, pattern_rank):
    ycost = 0
    for machine_type in range(machine_type_num):
        pattern_num = pattern_mat[machine_type].shape[0]
        for pattern in range(pattern_num):
            ycost = ycost + y_int[machine_type][pattern] * \
                    pattern_contribution_list[pattern_rank[machine_type][pattern]][2]
    return ycost


def CalculateXIndexCost(box_num, p, d_t, d_t_count, x_int):
    costx = 0
    for key in p.keys():
        psm1 = key[0]
        psm2 = key[1]
        for box in range(box_num):
            psm1_deployed = 0
            psm2_deployed = 0
            for i in d_t[psm1]:
                psm1_deployed = psm1_deployed + x_int[i, box]
            for j in d_t[psm2]:
                psm2_deployed = psm2_deployed + x_int[j, box]
            costx = costx + p[key] * min(psm1_deployed / d_t_count[psm1],
                                         psm2_deployed / d_t_count[psm2])
    return costx


def Phase1Rounding(machine_type_num, item_num, total_pattern_num, pattern_mat, pattern_contribution_list,
                   carry, pattern_rank, tol, y_continuous, y_int, alpha, beta, basic_factor):
    carry_round_start_time = time.time()
    up_count = 0
    down_count = 0
    effective_count = 0
    non_zeros = 0
    for machine_type in range(machine_type_num):
        pattern_num = pattern_mat[machine_type].shape[0]
        for pattern in range(pattern_num):
            if y_continuous[machine_type][pattern] > tol:
                non_zeros = non_zeros + 1

            if abs(y_continuous[machine_type][pattern] - round(y_continuous[machine_type][pattern])) < tol:
                y_int[machine_type][pattern] = round(y_continuous[machine_type][pattern])
            else:
                # print(y_continuous[machine_type][pattern])
                if pattern_contribution_list[pattern_rank[machine_type][pattern]][2] > tol:
                    effective_count = effective_count + 1

                carry_round_up = copy.deepcopy(carry)
                carry_round_down = copy.deepcopy(carry)
                # If rounding up, calculate the f(carry)
                dev = -(math.ceil(y_continuous[machine_type][pattern]) - y_continuous[machine_type][pattern])
                for item in range(item_num):
                    carry_round_up[item] = carry_round_up[item] + dev * pattern_mat[machine_type][pattern][item]
                # If rounding down, calculate the f(carry)
                dev = y_continuous[machine_type][pattern] - math.floor(y_continuous[machine_type][pattern])
                for item in range(item_num):
                    carry_round_down[item] = carry_round_down[item] + dev * pattern_mat[machine_type][pattern][item]
                # Calculate the second moment
                sec_moment_up = 0
                sec_moment_down = 0
                up_avg = np.sum(carry_round_up) / float(item_num)
                down_avg = np.sum(carry_round_down) / float(item_num)
                carry_round_dev_up = 0
                carry_round_dev_down = 0
                for item in range(item_num):
                    sec_moment_up = sec_moment_up + carry_round_up[item] * carry_round_up[item]
                    sec_moment_down = sec_moment_down + carry_round_down[item] * carry_round_down[item]
                    carry_round_dev_up = carry_round_dev_up + (carry_round_up[item] - up_avg) * (
                            carry_round_up[item] - up_avg)
                    carry_round_dev_down = carry_round_dev_down + \
                                           (carry_round_down[item] - down_avg) * (carry_round_down[item] - down_avg)
                up_factor = alpha * sec_moment_up + (1 - alpha) * carry_round_dev_up
                down_factor = alpha * sec_moment_down + (1 - alpha) * carry_round_dev_down
                # up_factor = up_factor - (beta + 0.989) * abs(up_factor - down_factor) * \
                #             float(pattern_rank[machine_type][pattern]) / total_pattern_num
                this_rand = random.random()
                prob_up = pow(float(pattern_rank[machine_type][pattern]) / total_pattern_num, beta * basic_factor)
                # /total_pattern_num is protected, since if total_pattern_num is 0, there is no for loop.
                if this_rand < prob_up or up_factor < down_factor:
                    # if up_factor < down_factor:
                    up_count = up_count + 1
                    y_int[machine_type][pattern] = math.ceil(y_continuous[machine_type][pattern])
                else:
                    down_count = down_count + 1
                    y_int[machine_type][pattern] = math.floor(y_continuous[machine_type][pattern])
            dev = y_continuous[machine_type][pattern] - y_int[machine_type][pattern]
            for item in range(item_num):
                carry[item] = carry[item] + dev * pattern_mat[machine_type][pattern][item]
    carry_round_end_time = time.time()
    return up_count, down_count, effective_count, non_zeros, carry_round_start_time, carry_round_end_time


def Phase2MaintainingPhysicalMachineConstraints(machine_type_num, item_num, pattern_mat,
                                                max_physical_machine_mat, d_t, d_t_count,
                                                p, y_int, carry):
    phase2_start_time = time.time()
    total_deleted = 0
    for machine_type in range(machine_type_num):
        deployed_at_n = 0
        pattern_num = pattern_mat[machine_type].shape[0]
        for pattern in range(pattern_num):
            deployed_at_n = deployed_at_n + y_int[machine_type][pattern]
        if deployed_at_n > max_physical_machine_mat[machine_type]:
            # Sort the patterns for machine type n by the contribution for proximity
            phase2_pattern_contribution_list = []
            for pattern in range(pattern_num):
                pattern_contribution = 0
                for key in p.keys():
                    psm1 = key[0]
                    psm2 = key[1]
                    psm1_deployed = 0
                    psm2_deployed = 0
                    for i in d_t[psm1]:
                        psm1_deployed = psm1_deployed + pattern_mat[machine_type][pattern][i]
                    for j in d_t[psm2]:
                        psm2_deployed = psm2_deployed + pattern_mat[machine_type][pattern][j]
                    pattern_contribution = pattern_contribution + p[key] * \
                                           min(psm1_deployed / d_t_count[psm1],
                                               psm2_deployed / d_t_count[psm2])
                phase2_pattern_contribution_list.append([pattern, pattern_contribution])

            phase2_pattern_contribution_list.sort(key=TakeContribution)
            # Sorted by contribution is done, now delete instance
            traverse_index = 0
            while deployed_at_n > max_physical_machine_mat[machine_type]:
                deleted_index = phase2_pattern_contribution_list[traverse_index][0]
                if y_int[machine_type][deleted_index] > 0:
                    deleted_instance = min([deployed_at_n - max_physical_machine_mat[machine_type],
                                            y_int[machine_type][deleted_index]])
                    total_deleted = total_deleted + deleted_instance
                    y_int[machine_type][deleted_index] = y_int[machine_type][deleted_index] - deleted_instance
                    deployed_at_n = deployed_at_n - deleted_instance
                    for item in range(item_num):
                        carry[item] = carry[item] + deleted_instance * pattern_mat[machine_type][deleted_index][item]
                traverse_index = (traverse_index + 1) % pattern_num
    phase2_end_time = time.time()
    return phase2_start_time, phase2_end_time, total_deleted


def Phase3ConvertXindex(machine_type_num, item_num, resource_num, machine_class_starter,
                        y_int, pattern_mat, tmp_box_resource_mat, x_int, psm_resource_mat):
    convert_start_time = time.time()
    for machine_type in range(machine_type_num):
        box = machine_class_starter[machine_type]
        pattern_num = pattern_mat[machine_type].shape[0]
        for pattern in range(pattern_num):
            # y_int is guaranteed to be integers
            for pattern_l_instance_in_machine_n in range(round(y_int[machine_type][pattern])):
                for item in range(item_num):
                    # guarantee that pattern_mat is also integers
                    x_int[item][box] = round(pattern_mat[machine_type][pattern][item])
                    for resource in range(resource_num):
                        tmp_box_resource_mat[box][resource] = tmp_box_resource_mat[box][resource] - \
                                                              x_int[item][box] * psm_resource_mat[item][resource]
                box = box + 1
    convert_end_time = time.time()
    return convert_start_time, convert_end_time


def Phase4MaintainNonNegCarry(item_num, resource_num, tol, x_int, carry,
                              tmp_box_resource_mat, psm_resource_mat):
    phase4_start_time = time.time()
    total_deleted = 0
    for item in range(item_num):
        deleted_index = 0
        # print('%d.'% item)
        while carry[item] < -tol:
            if x_int[item, deleted_index] > tol:
                deleted_num = min([round(abs(carry[item])), x_int[item, deleted_index]])
                x_int[item, deleted_index] = x_int[item, deleted_index] - deleted_num
                carry[item] = carry[item] + deleted_num
                total_deleted = total_deleted + deleted_num
                for resource in range(resource_num):
                    tmp_box_resource_mat[deleted_index][resource] = tmp_box_resource_mat[deleted_index][resource] + \
                                                                    deleted_num * psm_resource_mat[item][resource]
            deleted_index = deleted_index + 1
    phase4_end_time = time.time()
    return phase4_start_time, phase4_end_time, total_deleted


def Phase5Reassignment(machine_type_num, item_num, resource_num, box_num, tol,
                       carry, machine_class_starter, node_level_mat, tmp_box_resource_mat,
                       psm_resource_mat, x_int):
    phase5_start_time = time.time()
    counter_failed = 0
    for item in range(item_num):
        if abs(carry[item]) < tol:
            continue

        # For every psm i, if we successfully reallocate i's carry, then flag turn True
        flag = False
        # Iterate through all machines so as to find the greedy and naive solution of "carry"
        for machine_type in range(machine_type_num):
            box = machine_class_starter[machine_type]
            if machine_type + 1 == machine_type_num:
                next_box = box_num
            else:
                next_box = machine_class_starter[machine_type + 1]
            while box < next_box:
                if abs(carry[item]) < tol:
                    flag = True
                    break

                if node_level_mat[machine_type][item] == 0:
                    box = box + 1
                    continue

                resource_flag = True
                for resource in range(resource_num):
                    if tmp_box_resource_mat[box][resource] < psm_resource_mat[item][resource]:
                        resource_flag = False
                # Assign several instance of psm i on machine k
                if resource_flag:
                    # Determine the assigned number
                    max_allowed_assigned = round(carry[item])
                    for resource in range(resource_num):
                        if psm_resource_mat[item][resource] == 0.0:
                            continue
                        if math.floor(tmp_box_resource_mat[box][resource] / float(psm_resource_mat[item][resource])) \
                                < max_allowed_assigned:
                            max_allowed_assigned = math.floor(tmp_box_resource_mat[box][resource]
                                                              / float(psm_resource_mat[item][resource]))
                    # The assignment process
                    for resource in range(resource_num):
                        tmp_box_resource_mat[box][resource] = tmp_box_resource_mat[box][resource] - \
                                                              max_allowed_assigned * psm_resource_mat[item][resource]
                    x_int[item][box] = x_int[item][box] + max_allowed_assigned
                    carry[item] = carry[item] - max_allowed_assigned
                box = box + 1
        # If failed even go through the machines
        if flag is False:
            counter_failed = counter_failed + 1
    phase5_end_time = time.time()
    return phase5_start_time, phase5_end_time, counter_failed


def multi_dimensional_carry_based_rounding(psm_resource_mat, pod_demand_mat, max_physical_machine_mat,
                                           physical_machine_resource_mat, pattern_mat, y_continuous, d_t, p,
                                           node_level_mat, alpha=0.3, basic_factor=10.0, max_iter=4, tol=0.0001,
                                           satis_threshold=0.03, exp_ratio=1.5, random_power=2, beta=1.0):
    """
    Multi-Dimensional Carry-Based Rounding Strategy
    For column generation model
    Maintaining a multi-dimensional "carry" vector
    Random version
    Iterative
    """

    print('Proceed rounding...')
    # variables preparation
    machine_type_num = physical_machine_resource_mat.shape[0]  # Machine type number
    item_num = len(pod_demand_mat)  # item number
    psm_num = len(d_t)
    resource_num = physical_machine_resource_mat.shape[1]  # Resource number
    box_num = sum(max_physical_machine_mat)  # Machine number
    pattern_contribution_list = []
    d_t_count = np.zeros(psm_num)
    pattern_rank = []

    if item_num == 0 or psm_num == 0 or resource_num == 0 or box_num == 0 or machine_type_num == 0:
        print('Column generation rounding error, #psm or #resource or #machine = 0')
        return None, None

    for machine_type in range(machine_type_num):
        pattern_num = pattern_mat[machine_type].shape[0]
        for pattern in range(pattern_num):
            if math.isinf(y_continuous[machine_type][pattern]) or math.isnan(y_continuous[machine_type][pattern]):
                print('Column generation rounding error, # machine type = %d, #pattern = %d, illegal values: inf or nan'
                      % (machine_type, pattern))
                return None, None

    for psm in range(psm_num):
        d_t_count[psm] = sum(pod_demand_mat[k] for k in d_t[psm])
        if d_t_count[psm] == 0:
            print('Column generation rounding error, some service # containers = 0, need pre-solve')
            return None, None

    before_total_cost, total_pattern_num = MaintainingPatternContribution(machine_type_num, pattern_mat, d_t, p,
                                                                          d_t_count, y_continuous,
                                                                          pattern_contribution_list)
    pattern_contribution_list.sort(key=TakeThird)
    for machine_type in range(machine_type_num):
        pattern_rank.append(np.zeros(pattern_mat[machine_type].shape[0], dtype=int))
    for i in range(total_pattern_num):
        machine_type = pattern_contribution_list[i][0]
        pattern = pattern_contribution_list[i][1]
        pattern_rank[machine_type][pattern] = i

    BOX_RESOURCE_MAT = np.zeros([box_num, resource_num])
    machine_class_starter = np.zeros(machine_type_num, dtype=int)
    start_index = 0
    for machine_type in range(machine_type_num):
        machine_class_starter[machine_type] = start_index
        for resource in range(resource_num):
            BOX_RESOURCE_MAT[start_index:start_index + max_physical_machine_mat[machine_type], resource: resource + 1] \
                = physical_machine_resource_mat[machine_type][resource] * np.ones(
                [max_physical_machine_mat[machine_type], 1])
        start_index = start_index + max_physical_machine_mat[machine_type]

    y_int = copy.deepcopy(y_continuous)
    x_int_best = np.zeros([item_num, box_num], dtype=int)
    init_beta = beta
    round_count = 0
    break_flag = False
    for derandom_count in range(max_iter):
        if break_flag:
            break
        for random_count in range(random_power):
            if break_flag:
                break

            round_count = round_count + 1
            x_int = np.zeros([item_num, box_num], dtype=int)

            # Phase 1 rounding:
            carry = np.zeros(item_num)
            up_count, down_count, effective_count, non_zeros, carry_round_start_time, carry_round_end_time = \
                Phase1Rounding(machine_type_num, item_num, total_pattern_num, pattern_mat, pattern_contribution_list,
                               carry, pattern_rank, tol, y_continuous, y_int, alpha, beta, basic_factor)
            carry_abs = abs(carry)

            # Phase 2 maximum maintaining physical machine number
            phase2_start_time, phase2_end_time, total_deleted = \
                Phase2MaintainingPhysicalMachineConstraints(machine_type_num, item_num, pattern_mat,
                                                            max_physical_machine_mat, d_t, d_t_count,
                                                            p, y_int, carry)

            # Phase 3: Convert the current solution to the x_{i,k} indexed solution
            tmp_box_resource_mat = copy.deepcopy(BOX_RESOURCE_MAT)
            convert_start_time, convert_end_time = Phase3ConvertXindex(machine_type_num, item_num, resource_num,
                                                                       machine_class_starter, y_int, pattern_mat,
                                                                       tmp_box_resource_mat, x_int, psm_resource_mat)

            # Phase 4: Maintaining the non-negativity of carry
            carry = pod_demand_mat - np.sum(x_int, axis=1)
            phase4_start_time, phase4_end_time, total_deleted = \
                Phase4MaintainNonNegCarry(item_num, resource_num, tol, x_int, carry,
                                          tmp_box_resource_mat, psm_resource_mat)

            # Phase 5: Second, assign "carry" vector in a naive way
            # Now the carries are all positive
            # "Assign" means reassigning the demands
            print('Column generation rounding, failed number of containers = ', round_count)

            x_int_best = copy.deepcopy(x_int)

        # Determine beta in here
        if derandom_count >= math.ceil(max_iter * 3.0 / 4.0):
            if derandom_count == math.ceil(max_iter * 3.0 / 4.0):
                beta = init_beta
            beta = beta / exp_ratio
        else:
            beta = beta * exp_ratio
        satis_threshold = satis_threshold * 1.1

    return x_int_best, BOX_RESOURCE_MAT
