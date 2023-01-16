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

import os
import sys
APP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../..')
sys.path.append(APP_PATH)
from source_code.scheduling_algorithm_pool.scheduler_column_generation.models.pricing import pricing_model
from gurobipy import *
import time


def single_specification_solving(pi_1, pi_2, d_r, U_R, p, d, d_t, s, a_ori, index, rule):
    """
    This function solves the sub-problem of the column generation for a single specification,
        produces the new pattern for the given machine specification

    :param pi_1: dual variables, see column generation model
    :param pi_2: dual variables, see column generation model
    :param d_r: resource request for each service
    :param U_R: resource matrix for each machine specification
    :param p: affinity data
    :param d: container number for each service
    :param d_t: remaining issue of old version code, never mind this
    :param s: compatibility matrix
    :param a_ori: current pattern matrix
    :param index: index of the given machine specification
    :param rule: anti-affinity rule
    :return:
        1. new pattern matrix
        2. gained affinity matrix of the new patterns
        3. runtime of this solving
        4. runtime of the pricing modeling and solving
    """
    # Get the model and solving
    model_solve_start_time = time.time()
    a_gen_, mid_gen_, model = pricing_model(pi_1, pi_2, d_r, U_R, p, d, d_t, s, rule)
    model_solve_end_time = time.time()
    num_sol = model.SolCount

    # Deal with the results of the pricing model, get new patterns
    all_a_gen = []  # store new patterns
    all_mid_gen = []  # store the gained affinity of the new patterns
    if model.getObjective().getValue() > 0:
        # Only if the objective of the pricing model is greater than 0.0
        for idx_solution in range(num_sol):
            # Search through all solution that we have found, add them to the pattern matrix
            traffic = 1e-6
            model.setParam(GRB.Param.SolutionNumber, idx_solution)
            if model.PoolObjVal > 0:
                tmp_a_gen = []
                tmp_mid_gen = {}
                for i in range(len(d)):
                    tmp_a_gen.append(round(a_gen_[i].Xn))
                # get the gained affinity
                for (t_1, t_2) in p:
                    traffic += p[(t_1, t_2)] * mid_gen_[(t_1, t_2)].Xn
                    tmp_mid_gen[(t_1, t_2)] = mid_gen_[(t_1, t_2)].Xn
                # If the obtained pattern is repeated(already existed in the pattern matrix, ignore it)
                if traffic > 0.0 and tuple(tmp_a_gen) not in a_ori[index]:
                    all_a_gen.append(tuple(tmp_a_gen))
                    all_mid_gen.append(tmp_mid_gen)
                else:
                    pass
            else:
                break
        return all_a_gen, all_mid_gen, model_solve_end_time-model_solve_start_time, model.runTime
    else:
        return None, None, 0.0, 0


def single_processing(pi_1, pi_2, d_r, U_R_sub, p, d, U_R_total, d_t, s_total, a_ori, signal_local_list, add_num,
                      a_dict, mid_dict, rule):
    """
    At each iteration, the column generation needs to solve the pricing problem for each machine specification, this can
     be done via multi-processing for acceleration. This is the workflow controller for one processing.

    :param pi_1: the dual variable, see the pricing model of column generation algorithm
    :param pi_2: the dual variable, see the pricing model of column generation algorithm
    :param d_r: resource request matrix of each services
    :param U_R_sub: the machine specification indexes that are assigned to this process
    :param p: affinity data
    :param d: number of containers for each service
    :param U_R_total: resource matrix of all machine specifications
    :param d_t: issues that are remaining from old version of code
    :param s_total: compatibility matrix
    :param a_ori: pattern matrix
    :param signal_local_list: flag that tells whether a sub-problem is converges
    :param add_num: record the number of new patterns for each machine specification
    :param a_dict: matrix that stores the new patterns
    :param mid_dict: matrix that stores tha gained affinity matrix of the new patterns
    :param rule: anti-affinity
    :return:
        total runtime of this process
    """
    run_time = 0.0
    # U_R_sub is the machine specification indexes that are assigned to this process
    for index in U_R_sub:
        a_gen, mid_gen, model_solve_duration_time, opt_time = \
            single_specification_solving(pi_1, pi_2[index], d_r, U_R_total[index, :], p, d, d_t, s_total[index, :],
                                         a_ori, index, rule)
        run_time += model_solve_duration_time
        if mid_gen:
            signal_local_list[index] = True  # New patterns are produced
            add_num[index] = len(a_gen)  # number of new patterns
            a_dict[index] = a_gen  # store the new patterns
            mid_tmp = dict({})  # store the gained affinity of new patterns
            for l in range(len(mid_gen)):
                for (t_1, t_2) in p:
                    mid_tmp[(t_1, t_2, len(a_ori[index]) + l, index)] = mid_gen[l][(t_1, t_2)]
            mid_dict.update(mid_tmp)  # update the gained affinity dictionary
    return run_time


def multi_processing(pi_1, pi_2, d_r, p, d, U_R_total, d_t, s_total, master_model, y, d_dict, phy_dict, a_ori, mid,
                     do_sub_problem_machine_index, rule):
    """
    At each iteration, the column generation needs to solve the pricing problem for each machine specification, this can
         be done via multi-processing for acceleration. This is the workflow controller for multi-processing.

    :param pi_1: the dual variable, see the pricing model of column generation algorithm
    :param pi_2: the dual variable, see the pricing model of column generation algorithm
    :param d_r: resource request matrix of each services
    :param p: affinity data
    :param d: number of containers for each service
    :param U_R_total: resource matrix of all machine specifications
    :param d_t: issues that are remaining from old version of code
    :param s_total: compatibility matrix
    :param master_model: the model of the master column generation problem
    :param y: decision variable
    :param d_dict: never mind, this is useless
    :param phy_dict: machines dictionary
    :param a_ori: pattern dictionary
    :param mid: gained affinity dictionary
    :param do_sub_problem_machine_index: only deal with these machine specification
    :param rule: anti-affinity
    :return:
    """
    n_item = len(d)

    # partitioning the machine specification
    # each group of machine specification will be solved by an independent process
    a_dict = {}  # new pattern dict
    mid_dict = {}  # gained affinity of new pattern dict
    signal_local_list = list([False] * U_R_total.shape[0])  # whether a specification of machine is converged
    add_num = list([0] * U_R_total.shape[0])  # number of new patterns for each machine specification
    model_solve_dur_time = 0.0
    time_str = ""
    # Solve the sub-problems
    # ===== By experiment: multi-processing is not very helpful for Python, we cancel the multi-processing now. =====
    for i in do_sub_problem_machine_index:
        tmp_model_solve_dur_time = \
            single_processing(pi_1, pi_2, d_r, [i], p, d, U_R_total, d_t, s_total, a_ori, signal_local_list, add_num,
                              a_dict, mid_dict, rule)
        time_str += "[%d,%.2f]," % (i, tmp_model_solve_dur_time)
        model_solve_dur_time += tmp_model_solve_dur_time

    signal = (sum(signal_local_list) > 0)  # Whether the sub-problem is converged
    if not signal:
        pass
    signal_list = []
    add_num_list = []

    for i in range(U_R_total.shape[0]):
        signal_list.append(signal_local_list[i])
        add_num_list.append(add_num[i])

    for index, U_R in enumerate(U_R_total):
        if index in a_dict:
            a_ori[index].extend(a_dict[index])
            # Add new patterns for the master problem
            for l in range(add_num_list[index]):
                col = Column()
                for i in range(n_item):
                    col.addTerms(a_ori[index][len(a_ori[index]) - add_num_list[index] + l][i], d_dict[i])
                col.addTerms(1, phy_dict[index])
                # update the gained affinity of the new columns
                for (t_1, t_2) in p:
                    mid[(t_1, t_2, len(a_ori[index]) - add_num_list[index] + l, index)] = mid_dict[
                        (t_1, t_2, len(a_ori[index]) - add_num_list[index] + l, index)]
                # add new variables
                y[index, len(a_ori[index]) - add_num_list[index] + l] = master_model.addVar(
                    obj=-quicksum(p[t_1, t_2] * mid[t_1, t_2, len(a_ori[index]) - add_num_list[index] + l, index]
                                  for (t_1, t_2) in p).getValue(), lb=0, vtype=GRB.INTEGER,
                    name="y(%s, %s)" % (index, len(a_ori[index]) - add_num_list[index] + l), column=col)
                master_model.update()

    return signal, signal_list, add_num_list, master_model, y, d_dict, phy_dict, a_ori, mid
