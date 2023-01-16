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


def separate_balanced_cuts(auxiliary_p, service_num, cut_sets, service_level_list, K):
    """
    This function computes the balanced graph cut of each large connected components

    :param auxiliary_p: edges and weights
    :param service_num: the number of total services
    :param cut_sets: records each service is assign to which cut
    :param service_level_list: the node-level of each service
    :param K: the balanced graph partitioning parameter, partitioning into K parts
    :return
        max_cut_num: the number of cuts
        cut_sets: the partitioning of services
    """
    max_service_num = 180

    # Breadth first search algorithm
    def dfs(i, visited, adjacent, islands):
        visited[i] = True
        islands[-1].append(i)
        for j in adjacent[i]:
            if not visited[j]:
                dfs(j, visited, adjacent, islands)

    # get the amount of affinity inside a connect component(or island)
    def get_island_flow(island, adjacent):
        if len(island) == 1:
            return 0

        total_flow = 0
        island = set(island)
        for service1 in island:
            for service2 in adjacent[service1]:
                if service2 in island:
                    total_flow += auxiliary_p.get((service1, service2), 0)
        return total_flow

    # Recognize different connect components
    visited = [False] * service_num
    adjacent = defaultdict(lambda: set())
    for i, j in auxiliary_p:
        adjacent[i].add(j)
        adjacent[j].add(i)
    islands = []
    for i in range(service_num):
        if visited[i]:
            continue
        islands.append([])
        dfs(i, visited, adjacent, islands)

    # Equally partitioning while minimizing the loss between parts
    islands.sort(key=lambda x: len(x))
    split_islands = []
    loss3 = 0  # loss due to this partitioning
    while islands and len(islands[-1]) >= max_service_num:
        island = set(islands.pop())
        edges = []
        for i, j in auxiliary_p:
            if i in island and j in island:
                edges.append((i, j))
        if K is None:
            # if sum(d[pod] for service in island for pod in dt[service]) < 1000:
            # if sum(d[service] for service in island) < 1000:
            #     K = 1
            # else:
            #     K = 1
            K = len(island) // max_service_num + 1
        print("Data splitting: in balanced partitioning, from %d services, %d edges to %d subgraph" %
              (len(island), len(edges), K))

        min_loss = 1
        best_cuts = None
        # for iter_id in range(25000 if K > 1 else 1):
        for iter_id in range(len(edges) if K > 1 else 1):
            cuts = [[] for _ in range(K)]
            qs = [deque([i]) for i in random.sample(list(island), K)]  # bloom
            belong = [-1] * service_num
            while sum(len(que) for que in qs):
                for cut_id, que in enumerate(qs):
                    while que:
                        nd = que.popleft()  # bloom
                        if belong[nd] >= 0:
                            continue
                        cuts[cut_id].append(nd)
                        belong[nd] = cut_id
                        for next_nd in adjacent[nd]:
                            if belong[next_nd] >= 0:
                                continue
                            que.append(next_nd)
                        break
            loss = 0
            for v1, v2 in edges:
                if belong[v1] != belong[v2]:
                    loss += auxiliary_p[v1, v2]
            freq = [len(cut) for cut in cuts]

            if max(freq) < len(island) / K + 50 and min(freq) > max(freq) / 2:
                if loss < min_loss:
                    # print(f"\t 第{iter_id}次迭代：loss={loss}，分割方案={[len(cut) for cut in cuts]}")
                    min_loss = loss
                    best_cuts = cuts
        for sub_cut in best_cuts:
            split_islands.append(sub_cut)
        loss3 += min_loss
    print("Data splitting: the loss due to balanced partitioning is %.2f%%" % (loss3 * 100.0))
    islands.extend(split_islands)
    islands.sort(key=lambda x: get_island_flow(x, adjacent), reverse=True)

    # Deal with cut_sets
    cut_sets = np.zeros(len(cut_sets), dtype=int)
    max_cut_num = 0
    merge_id_l5 = {}
    merge_id_0 = None
    for island in islands:
        if len(island) == 1:
            if merge_id_0 is None:
                merge_id_0 = max_cut_num
                max_cut_num += 1
            cut_sets[island[0]] = merge_id_0
        elif len(island) <= 5:
            node_level = service_level_list[island[0]]
            if node_level not in merge_id_l5:
                merge_id_l5[node_level] = max_cut_num
                max_cut_num += 1
            for service in island:
                cut_sets[service] = merge_id_l5[node_level]
        else:
            for service in island:
                cut_sets[service] = max_cut_num
            max_cut_num += 1

    return max_cut_num, cut_sets
