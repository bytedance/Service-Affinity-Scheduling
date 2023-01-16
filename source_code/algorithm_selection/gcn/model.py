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

import dgl
import dgl.nn.pytorch as dglnn
import torch.nn as nn
import torch.nn.functional as F


class GCNClassifier(nn.Module):
    def __init__(self, in_dim, hidden_dim, n_classes):
        super(GCNClassifier, self).__init__()
        self.conv1 = dglnn.GraphConv(in_dim, hidden_dim, weight=True)
        self.conv2 = dglnn.GraphConv(hidden_dim, hidden_dim, weight=True)
        self.classify = nn.Linear(hidden_dim, n_classes)

    def forward(self, g):
        h = g.ndata['x']  # [N, 1]
        # apply relu and activate function
        h = F.relu(self.conv1(g, h))
        h = F.relu(self.conv2(g, h))
        with g.local_scope():
            g.ndata['h'] = h
            # readout the graph representation
            hg = dgl.mean_nodes(g, 'h')
            return self.classify(hg)
