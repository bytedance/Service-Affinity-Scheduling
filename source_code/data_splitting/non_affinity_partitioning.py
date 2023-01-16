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


def separate_not_in_link(auxiliary_p, service_num, cut_sets, deal_cut_id_list, target_cut_id):
    """
    This function separate services which has no affinity relations with other services, i.e., non-affinity services.

    :param auxiliary_p: affinity data, need to pop if an edge if separated
    :param service_num: total number of services.
    :param cut_sets: cut_sets[i] == k represents that service i is categorized in the service set k.
    :param deal_cut_id_list: only partitioning services that are in cut sets whose cut id is in deal_cut_id_list.
    :param target_cut_id: set the non-affinity service to cut with cut_id = target_cut_id.

    :return: None
    """
    service_in_link = -np.ones(service_num, dtype=int)
    separate_psm_num = 0
    for key in auxiliary_p.keys():
        key0 = key[0]
        key1 = key[1]
        service_in_link[key0] = -2
        service_in_link[key1] = -2
    for service in range(service_num):
        if service_in_link[service] == -1 and cut_sets[service] in deal_cut_id_list:
            cut_sets[service] = target_cut_id
            separate_psm_num += 1
    print('Data Splitting: Non-master partitioning finished, separating out %d non-affinity services.'
          % separate_psm_num)
