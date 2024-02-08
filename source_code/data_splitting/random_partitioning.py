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

from collections import deque, defaultdict
import random, time
import numpy as np


def random_partitioning(cut_sets, service_num, K):
    """
    This function computes the random partitioning of the graph
    """
    random.seed(time.time())
    max_cut_num = K + 1

    for service in range(service_num):
        if cut_sets[service] == 0:
            cut_id = random.randint(0, K - 1)
            cut_sets[service] = cut_id

    return max_cut_num, cut_sets
