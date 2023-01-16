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
from gurobipy import Model, GRB, quicksum, tuplelist, LinExpr
import os
import sys
APP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../..')
sys.path.append(APP_PATH)


def MIP_model(p, d, d_r, u_full, anti_affinity_list, time_out=60):
    """
    MIP model for POP algorithm

    :param p: affinity data
    :param d: container number of each services
    :param d_r: resource request for each services
    :param u_full: resource matrix of machines
    :param anti_affinity_list: new-feature -- anti-affinity
    :param time_out: maximum runtime
    :return:
        x_increment: the incrementation of x_int
    """

    model = Model('Pure MILP')

    service_num = len(d)
    machine_num = u_full.shape[0]

    # Pre-processing to reduce redundant
    aux_p = p.copy()
    for key in p.keys():
        if d[key[0]] == 0 or d[key[1]] == 0:
            aux_p.pop(key)
    p = aux_p
    valid_service_list = (np.array(d) > 0).nonzero()[0]
    valid_machine_list = (np.array(u_full[:, 0]) > 0).nonzero()[0]
    print("MIP for POP: #edge = %d, #services = %d, #machines = %d" % (len(p), len(valid_service_list),
                                                                       len(valid_machine_list)))

    # Add variables
    x = model.addVars(valid_service_list, valid_machine_list, lb=0, vtype=GRB.INTEGER, name='x')
    mid = model.addVars(tuplelist([(t_1, t_2, k) for (t_1, t_2) in p for k in valid_machine_list]), lb=0, ub=1.0,
                        vtype=GRB.CONTINUOUS, name="mid")
    v = model.addVars(tuplelist([(t_1, t_2) for (t_1, t_2) in p]), lb=0, ub=1.0, vtype=GRB.CONTINUOUS, name="v")

    # Add constraints
    # demand constraints
    for service in valid_service_list:
        model.addConstr(quicksum(x[service, k] for k in valid_machine_list) <= d[service], "demand_%d" % service)
    # resource constraints
    for machine in valid_machine_list:
        model.addConstr(quicksum(x[i, machine] * d_r[i][0] for i in valid_service_list) <= u_full[machine][0], "cpu_%d" % machine)
        model.addConstr(quicksum(x[i, machine] * d_r[i][1] for i in valid_service_list) <= u_full[machine][1], "mem_%d" % machine)
    # Don't worry about compatibility
    # Anti-affinity constraints
    if anti_affinity_list:
        rule = anti_affinity_list[0]
        for machine in range(machine_num):
            model.addConstr(quicksum(x[i, machine] for i in rule) <= 1, "anti_affinity_at_machine_%d" % machine)

    # Calculate gained affinity
    for (t1, t2) in p:
        for machine in valid_machine_list:
            model.addConstr(mid[t1, t2, machine] <= quicksum(x[service, machine] for service in [t1])/d[t1],
                            "%d_%d_%d_1" % (t1, t2, machine))
            model.addConstr(mid[t1, t2, machine] <= quicksum(x[service, machine] for service in [t2])/d[t2],
                            "%d_%d_%d_2" % (t1, t2, machine))
        model.addConstr(v[t1, t2] <= quicksum(mid[t1, t2, machine] for machine in valid_machine_list),
                        "%d_%d" % (t1, t2))

    # Add objective -- gained affinity
    traffic_obj = LinExpr([(p[(t1, t2)], v[t1, t2]) for (t1, t2) in p])
    model.setObjective(traffic_obj, GRB.MAXIMIZE)

    # Update model parameters
    model.setParam("Presolve", 0)  # pre solve takes too much time
    model.setParam("TimeLimit", time_out)  # set TimeOut
    model.Params.MIPGap = 0.001  # set Gap
    model.update()

    # Solving
    model.optimize()

    if model.status in [GRB.INFEASIBLE, GRB.INF_OR_UNBD]:
        print('MIP solving error, Infeasible or Out of Bound')
    x_increment = np.zeros([service_num, machine_num], dtype=int)
    for service in valid_service_list:
        for machine in valid_machine_list:
            x_increment[service][machine] = x[service, machine].getAttr("x")
    return x_increment
