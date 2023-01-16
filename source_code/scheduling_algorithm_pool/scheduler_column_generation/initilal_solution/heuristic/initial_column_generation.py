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

from collections import defaultdict
import numpy as np
from gurobipy import *
import copy
import os
import sys
APP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../../..')
sys.path.append(APP_PATH)
from source_code.utility.data_extension import *
from source_code.scheduling_algorithm_pool.scheduler_column_generation.initilal_solution.appl_sci.graph_heuristic_scheduler import graph_heuristic_for_init_column


def heuristic_mip(d, d_t, p, q, d_r, U_R, s, b_n, w_r, returnAllOptvars=False, x_start=None, threads=None, MIPGap=0.01,
                  MIPFocus=None, verbose=True, Seed=None, TimeLimit=None, Presolve=None, ImproveStartTime=None,
                  VarBranch=None, Cuts=None, tune=False, TuneCriterion=None, TuneJobs=None, TuneTimeLimit=None,
                  TuneTrials=None, tune_foutpref=None, Nonconvex=False):
    """
    The model that calculate a feasible schedule, for the heuristic initial solution.
    """

    model = Model('heuristic_mip_model')
    model.setParam('OutputFlag', 0)
    if verbose:
        model.Params.OutputFlag = 0
    if threads is not None:
        model.Params.Threads = threads
    if MIPGap is not None:
        model.Params.MIPGap = MIPGap  # default = 1e-4, try 1e-2
    if MIPFocus is not None:
        model.Params.MIPFocus = MIPFocus
    if Seed is not None:
        model.Params.Seed = Seed
    if TimeLimit is not None:
        model.Params.TimeLimit = TimeLimit
    if Presolve is not None:
        model.Params.Presolve = Presolve
    if ImproveStartTime is not None:
        model.Params.ImproveStartTime = ImproveStartTime
    if VarBranch is not None:
        model.Params.VarBranch = VarBranch
    if Cuts is not None:
        model.Params.Cuts = Cuts
    # tune params
    if tune and TuneCriterion is not None:
        model.Params.TuneCriterion = TuneCriterion
    if tune and TuneJobs is not None:
        model.Params.TuneJobs = TuneJobs
    if tune and TuneTimeLimit is not None:
        model.Params.TuneTimeLimit = TuneTimeLimit
    if tune and TuneTrials is not None:
        model.Params.TuneTrials = TuneTrials

    n_item = d_r.shape[0]
    n_r = d_r.shape[1]

    a = model.addVars(tuplelist([i for i in range(n_item)]), lb=0, vtype=GRB.INTEGER, name="a")
    mid = model.addVars(tuplelist([(t_1, t_2) for (t_1, t_2) in p]), lb=0, ub=1.0, vtype=GRB.CONTINUOUS, name="mid")
    model.update()
    objective1 = LinExpr([(p[t_1, t_2], mid[t_1, t_2]) for (t_1, t_2) in p])
    objective = objective1
    model.setObjective(objective, GRB.MAXIMIZE)
    for r in range(n_r):
        model.addConstr(LinExpr([(d_r[i, r], a[i]) for i in range(n_item)]) <= U_R[r], "resource_%s" % r)

    for (t_1, t_2) in p:
        model.addConstr(mid[t_1, t_2] <= quicksum(a[i] for i in d_t[t_1]) / sum(d[i] for i in d_t[t_1]),
                        name="mid_cons2_1(%s, %s)" % (t_1, t_2))
        model.addConstr(mid[t_1, t_2] <= quicksum(a[j] for j in d_t[t_2]) / sum(d[j] for j in d_t[t_2]),
                        name="mid_cons2_2(%s, %s)" % (t_1, t_2))

    for i in range(n_item):
        model.addConstr(a[i] <= int(s[i] * b_n[i]), name="d_%s" % i)
        # model.addConstr(a[i] <= int(s[i] * d[i]), name="d_%s" % i)

    if tune == True:
        model.tune()
        if tune_foutpref is None:
            tune_foutpref = 'tune'
        for i in range(model.tuneResultCount):
            model.getTuneResult(i)
            model.write(tune_foutpref + str(i) + '.prm')
    model.optimize()

    return a, mid, model


def graph_heuristic_init_column(d, d_t, p, q, d_r, U_r_type, s_type, item_node_level_list, machine_type_node_level_list,
                                rule):
    """
    This function controls the workflow of the applied science 19's algorithm for the initial columns of column
    generation algorithm, mainly concerns how to get patterns(or columns) from the schedule of the algorithm.
    """

    u_full, machine_index_2_type_index, machine_class_starter = extend_machine_type_to_box(U_r_type, q)
    s_full, _, _ = extend_s_type_to_total(s_type, q)

    pattern_mat = defaultdict(list)
    if u_full.shape[0] == 0:
        return pattern_mat
    x_int = graph_heuristic_for_init_column(d, p, d_r, s_full, u_full, s_type, item_node_level_list, [rule],
                                            machine_type_node_level_list, machine_index_2_type_index)

    # get pattern matrix by the schedule of applied science 19's heuristic algorithm
    for node in range(x_int.shape[1]):
        P = x_int[:, node]
        node_type = machine_index_2_type_index[node]
        if np.sum(P) > 0 and (tuple(P) not in pattern_mat[node_type]):
            pattern_mat[node_type].append(tuple(P))
    return pattern_mat


def heuristic_generate_init_column(d, d_t, p, q, d_r, U_R_total, s, rule, w_r):
    """
    An heuristic initial solution.
    """

    d_tmp = copy.deepcopy(d)
    d_t_tmp = copy.deepcopy(d_t)
    p_tmp = copy.deepcopy(p)
    q_tmp = copy.deepcopy(q)
    d_r_tmp = copy.deepcopy(d_r)
    U_R_total_tmp = copy.deepcopy(U_R_total)  # machine_type_num
    s_tmp = copy.deepcopy(s)  # machine_type_num
    pattern_mat = defaultdict(list)  # To store the pattern of each machine specification
    psm_num = len(d_t_tmp)
    item_num = len(d)
    machine_type_num = len(q)

    # construct \sum_p_{i,g}, represents the possible gained affinity of each service pair
    sig_p = np.zeros(item_num)
    for (t1, t2) in p_tmp.keys():
        for i in d_t_tmp[t1]:
            sig_p[i] += p_tmp[(t1, t2)]
        for j in d_t_tmp[t2]:
            sig_p[j] += p_tmp[(t1, t2)]
    remain_d = [d_tmp[index] for index in range(item_num)]
    un_solve_d = np.zeros(item_num, dtype=int)
    remain_q = [q_tmp[index] for index in range(machine_type_num)]
    total_remain_d = sum(d_tmp)
    total_remain_q = sum(q_tmp)

    psm_counter = 5 * np.ones(item_num, dtype=int)
    node_counter = 3 * np.ones(machine_type_num, dtype=int)
    while total_remain_d > 0 and total_remain_q > 0:
        sig_p_and_d = [sig_p[item] * remain_d[item] for item in range(item_num)]
        machine_type_sort_list = np.argsort(-np.array(remain_q)).tolist()
        remain_d_sort_list = np.argsort(-np.array(sig_p_and_d)).tolist()

        get_item_num = min(np.count_nonzero(np.array(remain_d)), 80)
        top_item = -1
        for index in range(item_num):
            item = remain_d_sort_list[index]
            if remain_d[item] > 0:
                top_item = item
                break
        # Get the machine specification that is to be dealt
        deal_machine_type = -1
        for index in range(machine_type_num):
            if round(s[machine_type_sort_list[index]][top_item]) == 1:
                deal_machine_type = machine_type_sort_list[index]
                break
        if deal_machine_type == -1 or remain_q[deal_machine_type] <= 0:
            un_solve_d[top_item] = d_tmp[top_item]
            total_remain_d -= remain_d[top_item]
            remain_d[top_item] = 0
            continue
        # Get the service list to be dealt, with node-level concerning
        deal_item_list = np.zeros(item_num, dtype=int)
        deal_good_list = []
        curr_get_item = 0
        for index in range(item_num):
            if curr_get_item == get_item_num:
                break
            item = remain_d_sort_list[index]
            if round(s[deal_machine_type][item]) == 1 and remain_d[item] > 0:
                # Deal with anti-affinity
                if (item in rule) and len(list(set(rule) & set(deal_good_list))) == 0:
                    deal_item_list[item] = 1
                    deal_good_list.append(item)

        # make sure that each machine specification got its machine
        num = remain_q[deal_machine_type]
        for item in range(item_num):
            if deal_item_list[item] == 1:
                num = min(num, remain_d[item])
        # construct b_n that represent one pattern of schedule
        b_n = [0 for _ in range(item_num)]
        for index in range(item_num):
            if deal_item_list[index] == 1:
                b_n[index] = math.floor(remain_d[index] / num)

        # MIP modeling and solving
        a, mid, model = heuristic_mip(d_tmp, d_t_tmp, p_tmp, remain_q, d_r_tmp, U_R_total_tmp[deal_machine_type, :],
                                      s_tmp[deal_machine_type, :], b_n, w_r)
        P = [0 for _ in range(item_num)]
        a_gen_ = model.getAttr('X', a)
        num = float("inf")
        for i in range(item_num):
            P[i] = round(a_gen_[i])
            if P[i] > 0:
                num = min(num, int(remain_d[i] / P[i]))

        if num == 0:
            print('Initial solution error Occur: num is 0.')

        for i in range(item_num):
            if P[i] > 0:
                psm_counter[i] -= 1
            if psm_counter[i] == 0:
                remain_d[i] = 0
            else:
                remain_d[i] = remain_d[i] - num * P[i]
        node_counter[deal_machine_type] -= 1
        if node_counter[deal_machine_type] == 0:
            remain_q[deal_machine_type] = 0
        else:
            remain_q[deal_machine_type] -= num
        pattern_mat[deal_machine_type].append(tuple(P))
        total_remain_d = sum(remain_d)
        total_remain_q = sum(remain_q)
    return pattern_mat


def data_processing_init_column(p, d, q, d_r, U_r_type, d_t, s_type, item_node_level_list,
                                machine_type_node_level_list, rule):
    """
    This function is the controller of the initial column generation, now we concerns with diagonal columns,
     applied science 19's columns, and heuristic algorithm columns
    """

    machine_type_num = len(q)
    a = defaultdict(set)

    # A heuristic algorithm as the initial columns
    a_gen = heuristic_generate_init_column(d, d_t, p, q, d_r, U_r_type, s_type, rule, w_r=1)
    for i in range(machine_type_num):
        if i in a_gen:
            for pattern_tmp in a_gen[i]:
                if tuple(pattern_tmp) not in a[i]:
                    a[i].add(pattern_tmp)

    # Applied science 19's algorithm as the initial columns, a graph based algorithm
    del a_gen
    q_in = []
    for i in q:
        q_in.append(math.floor(1.5*i))
    a_gen = graph_heuristic_init_column(d, d_t, p, q_in, d_r, U_r_type, s_type, item_node_level_list,
                                        machine_type_node_level_list, rule)
    for machine_type in range(machine_type_num):
        if machine_type in a_gen:
            for pattern in a_gen[machine_type]:
                if tuple(pattern) not in a[machine_type]:
                    a[machine_type].add(pattern)

    # Now get the final initial column list
    for i in range(machine_type_num):
        if not a[i]:
            a[i] = []
    for key in a.keys():
        a[key] = list(a[key])
    mid = defaultdict(float)
    for n in range(machine_type_num):
        # print('\t Machine type index %d, Initial column number %d.' % (n, len(a[n])))
        for l in range(len(a[n])):
            tmp_pattern = a[n][l]
            for (t_1, t_2) in p:
                local_t_1 = sum(tmp_pattern[k] for k in d_t[t_1])
                local_t_2 = sum(tmp_pattern[k] for k in d_t[t_2])
                local_rate_t_1 = float(local_t_1 / sum(d[k] for k in d_t[t_1]))
                local_rate_t_2 = float(local_t_2 / sum(d[k] for k in d_t[t_2]))
                mid[(t_1, t_2, l, n)] = min(local_rate_t_1, local_rate_t_2)
    print("Initial columns number for each machine specification:", [len(a[n]) for n in range(machine_type_num)])
    return a, mid
