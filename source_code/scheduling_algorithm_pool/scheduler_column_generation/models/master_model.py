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

"""
Column generation algorithm: the modelling of the master-problem.
"""


def master_problem_model(p, a, mid, d, q, returnAllOptvars=False, x_start=None, threads=None, MIPGap=0.01, MIPFocus=None,
                         verbose=True, Seed=None, TimeLimit=None, Presolve=None, ImproveStartTime=None, VarBranch=None,
                         Cuts=None, tune=False, TuneCriterion=None, TuneJobs=None, TuneTimeLimit=None, TuneTrials=None,
                         tune_foutpref=None, Nonconvex=False):
    """
    :param p: affinity data
    :param a: a[n][l] is a list of service_num slots, which is the l-th pattern of to machine specification n
    :param mid: mid[i,j,l,n] the affinity of service i and j in the l-th pattern to machine specification n
    :param d: the number of containers of each services
    :param q: the number of machines of each specification
    :return:
        y：variables of the master problem
        d_dict：never mind, never used, remaining issues from the old version of code
        phy_dict：never mind, never used, remaining issues from the old version of code
        master_model：the model of the master problem
    """

    # define model
    master_model = Model('master_model')
    master_model.setParam('OutputFlag', 0)
    # solution output setting
    if verbose:
        master_model.Params.OutputFlag = 0
    # run params (see the user guide of gurobi 9.0.3 for tuning method)
    if threads is not None:
        master_model.Params.Threads = threads
    if MIPGap is not None:
        master_model.Params.MIPGap = MIPGap  # default = 1e-4, try 1e-2
    if MIPFocus is not None:
        master_model.Params.MIPFocus = MIPFocus
    if Seed is not None:
        master_model.Params.Seed = Seed
    if TimeLimit is not None:
        master_model.Params.TimeLimit = TimeLimit
    if Presolve is not None:
        master_model.Params.Presolve = Presolve
    if ImproveStartTime is not None:
        master_model.Params.ImproveStartTime = ImproveStartTime
    if VarBranch is not None:
        master_model.Params.VarBranch = VarBranch
    if Cuts is not None:
        master_model.Params.Cuts = Cuts
    # tune params
    if tune and TuneCriterion is not None:
        master_model.Params.TuneCriterion = TuneCriterion
    if tune and TuneJobs is not None:
        master_model.Params.TuneJobs = TuneJobs
    if tune and TuneTimeLimit is not None:
        master_model.Params.TuneTimeLimit = TuneTimeLimit
    if tune and TuneTrials is not None:
        master_model.Params.TuneTrials = TuneTrials

    n_item = len(d)
    n_phy = len(q)

    # Add the decision variables y[n,l]
    y = {}
    for n in range(n_phy):
        for l in range(len(a[n])):
            y[n, l] = master_model.addVar(lb=0, vtype=GRB.INTEGER, name="y(%s, %s)" % (n, l))

    # Add the objective function
    objective = LinExpr([(p[t_1, t_2] * mid[t_1, t_2, l, n], y[n, l]) for (t_1, t_2) in p for n in range(n_phy) for l in
                         range(len(a[n]))])
    master_model.setObjective(-objective, GRB.MINIMIZE)

    # Add the constraints
    d_dict = {}
    for i in range(n_item):
        d_dict[i] = master_model.addConstr(
            quicksum(round(a[n][l][i]) * y[n, l] for n in range(n_phy) for l in range(len(a[n]))) <= d[i], "d_%s" % i)

    phy_dict = {}
    for n in range(n_phy):
        phy_dict[n] = master_model.addConstr(quicksum(y[n, l] for l in range(len(a[n]))) <= q[n], "physical_machine_%s" % (n))

    if tune == True:
        master_model.tune()
        if tune_foutpref is None:
            tune_foutpref = 'tune'
        for i in range(master_model.tuneResultCount):
            master_model.getTuneResult(i)
            master_model.write(tune_foutpref + str(i) + '.prm')
    
    # Update the models
    master_model.update()

    return y, d_dict, phy_dict, master_model
