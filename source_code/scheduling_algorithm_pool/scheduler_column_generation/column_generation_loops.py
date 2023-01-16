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
import time
import numpy as np
import os
import sys
APP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../..')
sys.path.append(APP_PATH)
from source_code.scheduling_algorithm_pool.scheduler_column_generation.column_generation_one_iteration import *
from source_code.scheduling_algorithm_pool.scheduler_column_generation.models.master_model import *


def column_generation_loop_controller(d_t, p, d, Q, d_r, U_R_total, s_total, a, mid, iter_max, lag, tol,
                                      useful_machine_type_list, global_flow, rule):
    """
    This function maintains the multiple iteration of column generation sub-problem solving, a sub-problem including a master problem and a pricing problem.

    :param d_t: issues that remains from old version of code
    :param p: affinity data
    :param d: container number of each services
    :param Q: the number of machines for each machine specification
    :param d_r: resource request of each services
    :param U_R_total: resource matrix of each machine specification
    :param s_total: compatibility matrix
    :param a: pattern matrix
    :param mid: gained affinity of each pattern
    :param iter_max: maximum number of the column generation iteration
    :param lag: terminate condition parameters, if the obj. didn't improve for more than tol in lag iterations, we terminate
    :param tol: terminate condition parameters, see (param: lag)
    :param useful_machine_type_list: we only solve the pricing problem for these machine specifications
    :param global_flow: total affinity
    :param rule: anti-affinity
    :return:
        a_master: pattern matrix for the master problem
        y_master: decision variables for the master problem
        -1.0: issues from old version of code, can ignore this
    """
    signal = True  # signal to terminate the column generation iteration
    epoch = 0  # iteration index
    time_start_model = time.time()
    y, d_dict, phy_dict, master_model = master_problem_model(p, a, mid, d, Q)  # master problem modelling
    object_list = []

    # Some sub-problem can be ignored if the sub-problem's machine number is small
    do_sub_problem_machine_index = []
    for index in range(U_R_total.shape[0]):
        if useful_machine_type_list[index] == 0 or Q[index] < 0.015 * sum(Q):
            continue
        else:
            do_sub_problem_machine_index.append(index)

    # Multiple iteration of column generation, workflow controller
    while signal:
        time_start = time.time()
        master_relax = master_model.relax()
        master_relax.params.method = 2
        master_relax.params.DualReductions = 0  # get an accurate model_status
        master_relax.optimize()
        object_list.append(master_relax.getObjective().getValue())
        # Calculate the dual values
        pi_1 = {}
        for ss in master_relax.getConstrs()[:len(d)]:
            pi_1[int(
                str(ss).split("_")[-1].replace(">", "").replace("(", "").replace(")", "").split(",")[0])] = ss.Pi

        pi_2 = {}
        for ss in master_relax.getConstrs()[len(d):len(d) + len(Q)]:
            pi_2[int(
                str(ss).split("_")[-1].replace(">", "").replace("(", "").replace(")", "").split(",")[0])] = ss.Pi

        # One iteration to solve the master and pricing problem
        signal, signal_list, add_num_list, master_model, y, d_dict, phy_dict, a_ori, mid = \
            multi_processing(pi_1, pi_2, d_r, p, d, U_R_total, d_t, s_total, master_model, y, d_dict, phy_dict, a, mid,
                             do_sub_problem_machine_index, rule,)
        epoch = epoch + 1
        time_final = time.time()
        print("Column generation algorithm: One iteration (%d) is done, runtime = %.2f seconds." % (epoch, time_final - time_start))

        # store results of this iteration
        if lag > len(object_list) > 2 and abs(object_list[-1] - object_list[0]) > 0.01:
            if abs(object_list[-1] - object_list[0]) / abs(object_list[-1]) < tol:
                signal = False
        elif len(object_list) > 2 and abs(object_list[-1] - object_list[0]) > 0.01:  # 避免python的精度损失
            if abs(object_list[-1] - object_list[-lag]) / abs(object_list[-1]) < tol:
                signal = False

        if epoch >= iter_max:
            signal = False

    # Post processing of column generation, sort out patterns
    a_master = {}
    for key in a.keys():
        if signal_list[key]:
            a_master[key] = np.array([list(x) for x in a[key][:-add_num_list[key]]])
        else:
            a_master[key] = np.array([list(x) for x in a[key]])

    y_lp = defaultdict(float)
    for gurobi_var in master_relax.getVars():
        var_name = gurobi_var.VarName
        var_value = float(gurobi_var.X)
        row = int(var_name[1:].split(",")[0].replace("(", ""))
        columns = int(var_name[1:].split(",")[1].replace(")", ""))
        y_lp[row, columns] = var_value

    y_master = {}
    for key in a.keys():
        if signal_list[key]:
            y_master[key] = np.array([y_lp[key, kk] for kk in range(len(a[key]) - add_num_list[key])])
        else:
            y_master[key] = np.array([y_lp[key, kk] for kk in range(len(a[key]))])

    time_end = time.time()
    print('Column generation algorithm is finished, total runtime = %.2f seconds.' % (time_end - time_start_model))
    return a_master, y_master, -1.0
