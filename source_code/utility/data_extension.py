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


def extend_machine_type_to_box(U_r_type, q):
    """
    This function extends the resource matrix of each machine specification to each machine

    :param U_r_type: resource matrix of each machine specification
    :param q: the number of machines for each machine specification
    :return: full resource matrix of each machines, the mapping from machine index to machine specification index, the first machine index of each specification of machine
    """
    box_num = sum(q)
    machine_type_num = U_r_type.shape[0]
    resource_num = U_r_type.shape[1]

    BOX_RESOURCE_MAT = np.zeros([box_num, resource_num])
    box_to_type_list = -np.ones(box_num, dtype=int)
    machine_class_starter = np.zeros(machine_type_num, dtype=int)
    start_index = 0

    for machine_type in range(machine_type_num):
        machine_class_starter[machine_type] = start_index
        for resource in range(2):
            BOX_RESOURCE_MAT[start_index:start_index + q[machine_type], resource: resource + 1] \
                = U_r_type[machine_type][resource] * np.ones([q[machine_type], 1])
        box_to_type_list[start_index:start_index + q[machine_type]] = machine_type * np.ones(q[machine_type], dtype=int)
        start_index = start_index + q[machine_type]
    return BOX_RESOURCE_MAT, box_to_type_list, machine_class_starter


def extend_s_type_to_total(s_type, q):
    """
    This function extends the compatibility matrix of each machine specification to each machine

    :param s_type: compatibility matrix of each machine specification
    :param q: the number of machines for each machine specification
    :return: full compatibility matrix of each machines, the mapping from machine index to machine specification index, the first machine index of each specification of machine
    """

    box_num = sum(q)
    machine_type_num = s_type.shape[0]
    item_num = s_type.shape[1]

    s_total = np.zeros([box_num, item_num], dtype=int)
    box_to_type_list = -np.ones(box_num, dtype=int)
    machine_class_starter = np.zeros(machine_type_num, dtype=int)
    box = 0
    for machine_type in range(machine_type_num):
        machine_class_starter[machine_type] = box
        for _ in range(q[machine_type]):
            s_total[box] = s_type[machine_type]
            box_to_type_list[box] = machine_type
            box += 1
    return s_total, box_to_type_list, machine_class_starter
