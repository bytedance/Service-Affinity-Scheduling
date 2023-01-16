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



def separate_different_node_level(auxiliary_p, service_num, cut_sets, service_node_level_list, deal_cut_id_list, init_id):
    """
    This function categorize services to different sets, according to their node-level(node-level indicates the machines that the service is allowed to be placed)

    :param auxiliary_p: edges and weights
    :param service_num: total number of services.
    :param cut_sets: cut_sets[i] == k represents that service i is categorized in the service set k.
    :param service_node_level_list: the node-level of each service.
    :param deal_cut_id_list: only partitioning services that are in cut sets whose cut id is in deal_cut_id_list.
    :param init_id: we will re-index the different node-level service set, this is the starting number of the re-indexing.
    :return node_level_index_dict: mapping from the node-level to the correspond service set's cut id.
    """
    counter = 0
    node_level_index_dict = {}

    for service in range(service_num):
        if cut_sets[service] in deal_cut_id_list:
            nl = service_node_level_list[service]

            if nl not in node_level_index_dict.keys():
                node_level_index_dict[nl] = init_id + counter
                counter += 1

            cut_sets[service] = node_level_index_dict[nl]

    # Separating edges
    key_list = list(auxiliary_p.keys()).copy()
    for key in key_list:
        if cut_sets[key[0]] != cut_sets[key[1]]:
            auxiliary_p.pop(key)

    print("Data splitting: compatibility partitioning, into %d parts." % len(list(node_level_index_dict.keys())))

    return node_level_index_dict
