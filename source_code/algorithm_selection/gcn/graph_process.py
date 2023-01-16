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

import torch as th
import dgl
import numpy as np
from copy import deepcopy


def get_node(unique_node, u_or_v):
    """
    re-indexing the services
    """

    node_map = deepcopy(unique_node)
    e_list = []
    for source in u_or_v:
        idx = np.where(node_map == source)[0][0]
        e_list.append(idx)
    return e_list


def generate_graph(edge_list, feature, weight):
    """
    generate the tensor data structure of the sub-problem
    """
    u, v = zip(*edge_list)

    all_node = np.array(list(u)+list(v))
    unique_node = np.unique(all_node)
    source_list = get_node(unique_node, u)
    dire_list = get_node(unique_node, v)

    # Construct the affinity topology of the sub-problem
    g = dgl.graph((source_list, dire_list))
    # Add service info
    g.ndata['x'] = th.from_numpy(np.array(feature, dtype=np.float32)[unique_node])
    # Add weight(or affinity) info
    w = []
    for edge in edge_list:
        w.append(weight[edge])
    g.edata['w'] = th.tensor(w)
    return g
