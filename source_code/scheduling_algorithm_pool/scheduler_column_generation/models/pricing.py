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
import time

"""
Column generation algorithm: the modelling of the sub-problem.
"""


def pricing_model(pi_1, pi_2, d_r, U_R, p, d, d_t, s, rule, returnAllOptvars=False, x_start=None, threads=None,
                  MIPGap=0.01, MIPFocus=None, verbose=True, Seed=None, TimeLimit=0.125, Presolve=None,
                  ImproveStartTime=None, VarBranch=None, Cuts=None, tune=False, TuneCriterion=None, TuneJobs=None,
                  TuneTimeLimit=None, TuneTrials=None, tune_foutpref=None, Nonconvex=False):
    """
    This function models the sub-problem of the column generation

    :param pi_1: dual variables, see column generation model
    :param pi_2: dual variables, see column generation model
    :param d_r: resource request of each services
    :param U_R: resource of each machine specifications
    :param p: affinity data
    :param d: number of containers for each services
    :param d_t: issues from the old version of the code
    :param s: compatibility matrix
    :return :
        a: decision values concerning with the pattern matrix
        mid: possible gained affinity matrix
        model: sub-problem model
    """
    # solution output setting
    model = Model('pricing_model')
    model.setParam('OutputFlag', 0)
    if verbose:
        model.Params.OutputFlag = 0
    # run params (see the user guide of gurobi 9.0.3 for tuning method)
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

    # Add variables
    a = model.addVars(tuplelist([i for i in range(n_item)]), lb=0, vtype=GRB.INTEGER, name="a")
    mid = model.addVars(tuplelist([(t_1, t_2) for (t_1, t_2) in p]), lb=0, ub=1.0, vtype=GRB.CONTINUOUS, name="mid")

    # Add objective function
    objective_1 = LinExpr([(pi_1[i], a[i]) for i in range(n_item)])
    objective_2 = LinExpr([(p[t_1, t_2], mid[t_1, t_2]) for (t_1, t_2) in p])
    objective = objective_1 + objective_2# - pi_2
    model.setObjective(objective, GRB.MAXIMIZE)

    # Add constraints
    for r in range(n_r):
        model.addConstr(quicksum((d_r[i, r] * a[i] for i in range(n_item))) <= U_R[r], "resource_%s" %r)

    for (t_1, t_2) in p:
        model.addConstr(mid[t_1, t_2] <= quicksum(a[i] for i in d_t[t_1])/sum(d[i] for i in d_t[t_1]),
                        name="mid_cons2_1(%s, %s)" % (t_1, t_2))
        model.addConstr(mid[t_1, t_2] <= quicksum(a[j] for j in d_t[t_2])/sum(d[j] for j in d_t[t_2]),
                        name="mid_cons2_2(%s, %s)" % (t_1, t_2))
    if rule:
        model.addConstr(quicksum(a[t] for t in rule) <= 1, "anti_affinity")

    # Tune the model
    if tune == True:
        model.tune()
        if tune_foutpref is None:
            tune_foutpref = 'tune'
        for i in range(model.tuneResultCount):
            model.getTuneResult(i)
            model.write(tune_foutpref + str(i) + '.prm')
    model.update()
    model.optimize()
    return a, mid, model
