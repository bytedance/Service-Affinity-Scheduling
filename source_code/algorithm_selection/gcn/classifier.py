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
import torch
import dgl
import os
import sys
APP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../..')
sys.path.append(APP_PATH)
from source_code.algorithm_selection.gcn.model import GCNClassifier
from source_code.algorithm_selection.gcn.graph_process import generate_graph


def get_method(cut_p, d_r, d):
    """
    This function uses the trained gcn model to select an appropriate algorithm for the sub-problem

    :param cut_p: sub-problem's affinity data
    :param d_r: request resource matrix of the services in the sub-problem
    :param d: list of the number of containers of each services in the sub-problem
    :return:
        method: a selected label in {"cg", "mip"}
    """

    # Load the trained model
    model = GCNClassifier(3, 256, 2)
    try:
        model.load_state_dict(torch.load('algorithm_selection/gcn/trained_model'))
    except:
        model.load_state_dict(torch.load('../source_code/algorithm_selection/gcn/trained_model'))
    model.eval()

    # generate feature matrix of the given sub-problem
    edge_key_list = []
    for edge in cut_p.keys():
        edge_key_list.append(edge)
    feature = []
    for i in range(len(d)):
        feature.append(np.append(d_r[i], d[i]).tolist())
    g = generate_graph(edge_key_list, feature, cut_p)
    g = dgl.add_self_loop(g)

    # predict the selected algorithm
    with torch.no_grad():
        pred = torch.softmax(model(g), 1)
        pred = torch.max(pred, 1)[1].view(-1)
        get_label = pred.detach().cpu().numpy().tolist()[0]

        if get_label == 1:
            method = "cg"
        else:
            method = "mip"

    return method
