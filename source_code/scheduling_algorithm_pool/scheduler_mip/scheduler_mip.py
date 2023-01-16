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
import time
import copy
from gurobipy import Model, GRB, quicksum
import os
import sys
APP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../..')
sys.path.append(APP_PATH)
from source_code.scheduling_algorithm_pool.scheduler_column_generation.initilal_solution.appl_sci.\
    graph_heuristic_scheduler import graph_heuristic_for_init_column
from source_code.scheduling_algorithm_pool.scheduler_column_generation.initilal_solution.k8s_plus.\
    optimized_k8s_affinity_scheduler import run_simulated_k8s
from source_code.utility.result_check import calculate_local_traffic

"""
An optimized mixture integer programming for RASA.
"""


def num_host_require(total_cpu, total_mem, num_layer, q_remain, U_R_total):
    """
    Determine the number of hosts that is required for the sub-problem
    """
    total_cpu /= num_layer
    total_mem /= num_layer
    U_R_single = np.repeat(U_R_total, q_remain // num_layer, axis=0)
    avai_cpu = avai_mem = 0
    for num_host in range(1, len(U_R_single) + 1):
        avai_cpu += U_R_single[num_host - 1, 0]
        avai_mem += U_R_single[num_host - 1, 1]
        if avai_cpu > total_cpu * 1.1 and avai_mem > total_mem * 1.1:
            return num_host
    return 10 ** 9


def scheduler_mip(cut_p, U_R_total, q_remain, d, d_r, global_flow, item_node_level, pm_node_level_list,
                  cut_anti_affinity, time_out=None, heuri_init_flag=True, scale_rate=1.1, force_k8s_heuri=False):
    """
    Building the MIP model and then the solving.
    """

    print("Proceed MIP algorithm...")
    if global_flow == 0.0:
        return np.zeros([len(d), 0], dtype=int), np.zeros(len(q_remain), dtype=int), 0.0

    edges = {}
    for (i, j), flow in cut_p.items():
        edges[(i, j)] = flow / global_flow

    q_mask = np.zeros_like(q_remain, dtype=int)
    for i in range(len(q_mask)):
        if item_node_level == pm_node_level_list[i]:
            q_mask[i] = 1

    item_num = len(d)
    item_this_island = list(range(item_num))

    # Further optimize the MIP model
    total_local_flow_rate = 0
    num_pm_type = len(U_R_total)
    cut_q = np.zeros_like(q_remain)
    cut_x = [[] for _ in range(num_pm_type)]

    # s1. separate different layers
    total_cpu = sum([d_r[i, 0] * d[i] for i in item_this_island])
    total_mem = sum([d_r[i, 1] * d[i] for i in item_this_island])
    left = 1
    right = num_layer = 1  # at most 50 layers
    while left <= right:
        mid = (left + right) // 2
        num_host = num_host_require(total_cpu, total_mem, mid, q_remain * q_mask, U_R_total)
        if num_host == 10 ** 9 or num_host * item_num <= 2000 or num_host <= 15:  # 控制每一层的整数变量数
            num_layer = mid
            right = mid - 1
        else:
            left = mid + 1

    # d_nodelevel is the number of containers for each services
    d_nodelevel1 = d // num_layer  # 前num_layer - 1
    d_nodelevel2 = d_nodelevel1 + d % num_layer

    # q: the number of machines for each specification of machines
    # q_remain: the remaining number of machines for each specification of machines
    # q_used_layer: the number of machines for each layer
    # print('Layer number: ', num_layer)
    for layer_idx in range(num_layer):
        if layer_idx == num_layer - 1:
            d = d_nodelevel2
        else:
            d = d_nodelevel1

        # more acceleration：compress the number of containers (harm: nearly 0)
        if layer_idx == 0:  # First layer
            U_R_single = np.repeat(U_R_total, q_remain * q_mask // num_layer, axis=0)
        elif layer_idx == num_layer - 1:  # Last layer
            U_R_single = np.repeat(U_R_total, q_remain * q_mask, axis=0)
        else:  # Middle layer
            q_remain -= q_used_layer
            cut_q += q_used_layer
            total_local_flow_rate += -model.ObjVal * 100
            continue

        avai_cpu = avai_mem = 0
        for num_host_init in range(1, sum(q_remain * q_mask) + 1):
            avai_cpu += U_R_single[num_host_init - 1, 0]
            avai_mem += U_R_single[num_host_init - 1, 1]
            if avai_cpu > total_cpu * scale_rate and avai_mem > total_mem * scale_rate:
                break

        for num_host in range(num_host_init, sum(q_remain * q_mask) + 1):
            model = Model()
            model.setParam('OutputFlag', 0)
            model.setParam("Symmetry", 2)
            x = model.addVars(item_this_island, num_host, vtype=GRB.INTEGER)
            u = model.addVars(item_this_island, num_host)
            v = model.addVars(edges.keys(), num_host, obj={(i1, i2, k): -edges[i1, i2] for i1, i2 in edges for k in range(num_host)})

            # Now setting the initial solution for the MIP model
            if heuri_init_flag:
                # Setting the initial solution for the MIP model: applied science 19
                gh_d = copy.deepcopy(np.array(d)[item_this_island])
                gh_edges = copy.deepcopy(edges)
                gh_d_r = copy.deepcopy(np.array(d_r)[item_this_island, :])
                gh_u_full = copy.deepcopy(U_R_single[0:num_host, :])
                gh_s_type = np.ones([num_host, len(gh_d)], dtype=int)
                gh_s_full = np.ones([num_host, sum(gh_d)], dtype=int)
                gh_item_node_level_list = ['Self_Construct_Node_Level'] * len(gh_d)
                gh_machine_type_node_level_list = ['Self_Construct_Node_Level'] * num_host
                gh_machine_index_2_type_index = [i for i in range(num_host)]
                # 需要保证一个psm只有一个item。
                gh_x = graph_heuristic_for_init_column(gh_d, gh_edges, gh_d_r, gh_s_full, gh_u_full, gh_s_type,
                                                       gh_item_node_level_list, cut_anti_affinity,
                                                       gh_machine_type_node_level_list, gh_machine_index_2_type_index)
                gh_obtained = calculate_local_traffic(gh_x, gh_d, gh_edges)[0]
                print('MIP initial solution - applied science 19: %.2f%%' % (100.0 * gh_obtained))
                # Setting the initial solution for the MIP model: k8s+
                if sum(gh_d) > 10000:
                    k8s_obtained = 0.0
                    k8s_x = gh_x
                else:
                    if force_k8s_heuri:
                        k8s_x = np.zeros([len(gh_d), gh_u_full.shape[0]], dtype=int)
                    else:
                        k8s_x, k8s_u_free = run_simulated_k8s(gh_d, gh_d_r, gh_u_full, gh_s_full, gh_edges, cut_anti_affinity)
                    k8s_obtained = calculate_local_traffic(k8s_x, gh_d, gh_edges)[0]
                print('MIP initial solution - k8s+ : %.2f%%' % (100.0 * k8s_obtained))
                # update model
                if k8s_obtained > gh_obtained:
                    init_x = k8s_x
                else:
                    init_x = gh_x
                for i in item_this_island:
                    for j in range(num_host):
                        x[i, j].start = init_x[i][j]
            model.update()

            # Building model: adding constraints
            for t in item_this_island:
                model.addLConstr(quicksum(x.select(t, "*")) <= d[t])
            for k in range(num_host):
                for r in range(2):
                    model.addLConstr(quicksum(x[t, k] * d_r[t, r] for t in item_this_island) <= U_R_single[k][r])
            for i in item_this_island:
                for k in range(num_host):
                    model.addLConstr(u[i, k] == x[i, k] / d[i])
            for i1, i2 in edges:
                for k in range(num_host):
                    model.addLConstr(v[i1, i2, k] <= u[i1, k])
                    model.addLConstr(v[i1, i2, k] <= u[i2, k])
            # extra anti-affinity constraint
            if cut_anti_affinity:
                rule = cut_anti_affinity[0]
                for k in range(num_host):
                    model.addLConstr(quicksum(x[t, k] for t in rule) <= 1)

            if time_out is not None:
                model.setParam("TimeLimit", time_out)
            else:
                model.setParam("TimeLimit", 60)

            def early_stop(model, where):
                tick = time.time()
                if where == GRB.Callback.MIPSOL:
                    new_bst = -model.cbGet(GRB.Callback.MIPSOL_OBJBST)
                    old_bst = model._last_incumbent
                    bnd = -model.cbGet(GRB.Callback.MIPSOL_OBJBND)
                    model._abs_gap = bnd - new_bst
                    model._relative_gap = (bnd - new_bst) / (new_bst + 1e-6)
                    if (new_bst - old_bst) > 0.0005:
                        # 超过0.05%的改进
                        print(f"callback: best={new_bst * 100:.3f}%, bound={bnd * 100:.3f}%, relative gap="
                              f"{model._relative_gap * 100:.3f}%, absolute gap={model._abs_gap * 100:.3f}% x {model._num_layer} = {model._abs_gap * 100 * model._num_layer:.3f}")
                        model._last_incumbent = new_bst
                        model._tick = time.time()
                if where == GRB.Callback.POLLING:
                    if model._abs_gap * (model._num_layer) < 0.0025 and time.time() - model._start_tick > 10:
                        print(f"callback: terminate：gap={model._abs_gap * model._num_layer}，time={time.time() - model._start_tick}")
                        model.terminate()
                    if time.time() - model._tick > 60 and model._relative_gap < 0.3:
                        print("callback: terminate, long time without improvements.")
                        model.terminate()
                model._cb_time += time.time() - tick

            model._last_incumbent = 0
            model._tick = time.time()
            model._cb_time = 0
            model._abs_gap = 1
            model._relative_gap = 1
            if num_layer > 1:
                model._num_layer = (num_layer - 1) if layer_idx == 0 else 1
            else:
                model._num_layer = 1
            model._start_tick = time.time()
            model.optimize(early_stop)
            print(f"callback time = {model._cb_time}")

            if model.status not in [GRB.INFEASIBLE, GRB.INF_OR_UNBD]:
                print(f"layer: {layer_idx + 1}/{num_layer}，gained affinity {-model.ObjVal * 100:.5f}%")
                total_local_flow_rate += -model.ObjVal * 100
                print(f"gained affinity: {total_local_flow_rate:.3f}% / {sum(edges.values()) * 100:.3f}%")
                # Deal with machines
                x_pm = 0
                if layer_idx == 0:
                    # First layer
                    q_used_layer = np.zeros_like(q_remain)
                    used = 0
                    for i, num in enumerate(q_remain * q_mask // num_layer):
                        if used + num <= num_host:
                            q_used_layer[i] = num
                            used += num
                            for _ in range(num):
                                for _ in range(max(1, num_layer - 1)):
                                    cut_x[i].append([round(x[pod, x_pm].x) for pod in item_this_island])
                                x_pm += 1
                        else:
                            q_used_layer[i] = num_host - used
                            for _ in range(num_host - used):
                                for _ in range(max(1, num_layer - 1)):
                                    cut_x[i].append([round(x[pod, x_pm].x) for pod in item_this_island])
                                x_pm += 1
                            break
                    assert x_pm == num_host
                    q_remain -= q_used_layer
                    cut_q += q_used_layer
                else:
                    # Last layer
                    for i, num in enumerate(q_remain * q_mask):
                        if num <= num_host:
                            num_host -= num
                            q_remain[i] = 0
                            cut_q[i] += num
                            for _ in range(num):
                                cut_x[i].append([round(x[pod, x_pm].x) for pod in item_this_island])
                                x_pm += 1
                        else:
                            q_remain[i] -= num_host
                            cut_q[i] += num_host
                            for _ in range(num_host):
                                cut_x[i].append([round(x[pod, x_pm].x) for pod in item_this_island])
                                x_pm += 1
                            break
            break  # No more machines is needed.

    pms = []
    for i in range(num_pm_type):
        pms.extend(cut_x[i])
    cut_x = np.asarray(pms).T

    assert cut_x.shape[1] == sum(cut_q)
    return cut_x, cut_q, total_local_flow_rate / 100 * global_flow
