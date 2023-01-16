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

from collections import defaultdict
import hashlib, random, json
import numpy as np


def read_from_json_stream(input_json_stream):
    """
    This function reads from the input json stream, then construct related python objects.

    :param input_json_stream: input json data stream.

    :return x_old: original assignment of services to machines, x[i][k] represents the number of service i's container
        to machine j.
    :return p: affinity data, which is a dictionary, p[(i1, i2)] represents the affinity quantity between service i1
        and i2.
    :return d: the number of containers for each service, d[i] is the number of containers for service i.
    :return d_r: resource of each service's container, d_r[i][0] is the request CPU for a container of service i,
        d_r[i][1] is the request memory for a container of service i.
    :return s_type: whether a service container can be placed on a type of machine, s_type[i][k] = 1 represents that
        the service i's container can be placed on the k-th type of machine, vice verse.
    :return s_full: whether a service container can be placed on a machine, s_full[i][k] == 1 represents that the
        service i's container can be placed on the machine k.
    :return u_r_type: the total resource of machine types.
    :return u_r_full: the total resource of all machines.
    :return q: the number of machines for each machine type.
    :return service_index_2_service_name_list: mapping from service index to service name.
    :return service_name_2_service_index_dict: mapping from service name to service index.
    :return service_node_level_list: mapping from service index to service's node level (P.S. node level is a indicator
        of a service or machines type, service containers can only be placed on machines with the same node level)
    :return machine_index_2_machine_ip_list: mapping from machine index to machine IP(or machine name).
    :return machine_type_index_2_machine_index_dict: mapping from machine type index to machine index (one to multiple).
    :return machine_index_2_machine_type_index_list: mapping from machine index to machine type index.
    :return machine_type_index_2_node_level_list: mapping from machine type to the machine node level.
    :return anti_affinity_list: anti affinity relations, services in the same list are not allowed to placed on the
        same machine.
    :return global_traffic: total affinity quantity of the entire cluster.
    :return status: status of the data preprocess function, True is normal and success, False means there is errors.
    :return status_description: the description of the data preprocessing status.
    """
    # status = True
    # status_description = "Success"

    container_index_2_container_name_list = []
    container_name_2_container_index_dict = {}
    service_index_2_service_name_list = []
    service_name_2_service_index_dict = {}
    machine_index_2_machine_ip_list = []
    machine_ip_2_machine_index_dict = {}

    # Deal with "ServiceList" objects
    service_num = 0
    container_num = 0
    service_index_2_container_index_dict = defaultdict(list)
    container_index_2_service_index_list = []
    node_level_2_service_index_dict = defaultdict(list)
    raw_machine_ip_2_node_level_dict = defaultdict(dict)
    service_node_level_list = []
    d = []
    d_r = []
    every_machine_has_node_level = {}
    # try:
    service_list = input_json_stream["ServiceList"]
    for service_dict in service_list:
        # print(service_dict)
        # For convenience, get key values
        service_name = service_dict["Service"]
        request_cpu = float(service_dict["RequestCPU"])
        request_mem = float(service_dict["RequestMem"])
        container_name_list = service_dict["ContainerList"]
        compatible_machine_ips = service_dict["CompatibleMachines"]

        # Indexing and registering services
        service_index = service_num
        service_name_2_service_index_dict[service_name] = service_index
        service_index_2_service_name_list.append(service_name)

        # resource parameters
        d.append(len(container_name_list))
        d_r.append([request_cpu, request_mem])

        # node level parameters
        bytes_stream = bytes(str(compatible_machine_ips), 'utf-8')
        node_level = hashlib.sha256(bytes_stream).hexdigest()
        node_level_2_service_index_dict[node_level].append(service_index)
        service_node_level_list.append(node_level)
        if compatible_machine_ips == "*":
            every_machine_has_node_level[node_level] = True
        else:
            for machine_ip in compatible_machine_ips:
                raw_machine_ip_2_node_level_dict[machine_ip][node_level] = True

        service_num += 1

        # Indexing and registering containers
        for container_name in container_name_list:
            container_index = container_num
            container_name_2_container_index_dict[container_name] = container_index
            container_index_2_container_name_list.append(container_name)

            service_index_2_container_index_dict[service_index].append(container_index)
            container_index_2_service_index_list.append(service_index)

            container_num += 1
    # except:
    #     status = False
    #     status_description = "Input file format error: possibly errors in \"ServiceList\" key."

    # Deal with "MachineList" objects
    machine_num = 0
    u_r_full = []
    # try:
    for machine_dict in input_json_stream["MachineList"]:
        machine_ip = machine_dict["MachineIP"]
        machine_cpu = float(machine_dict["TotalCPU"])
        machine_mem = float(machine_dict["TotalMem"])
        # deployed_container_names = machine_dict["InitialDeployingContainers"]

        # Indexing and registering machine
        machine_index = machine_num
        machine_index_2_machine_ip_list.append(machine_ip)
        machine_ip_2_machine_index_dict[machine_ip] = machine_index

        # Add resource
        u_r_full.append([machine_cpu, machine_mem])
        machine_num += 1
    # except:
    #     status = False
    #     status_description = "Input file format error: possibly errors in \"MachineList\" key."

    # Determine machine's node level
    node_level_2_machine_index_dict = defaultdict(list)
    machine_index_2_node_level_list = [None] * machine_num
    # try:
    for machine_ip in machine_ip_2_machine_index_dict.keys():
        for node_level in every_machine_has_node_level.keys():
            raw_machine_ip_2_node_level_dict[machine_ip][node_level] = True
    # After the last step, the node-level is basically maintained, check here.
    for machine_index in range(machine_num):
        machine_ip = machine_index_2_machine_ip_list[machine_index]
        possible_node_level_list = list(raw_machine_ip_2_node_level_dict[machine_ip].keys())
        node_level = possible_node_level_list[random.randint(0, len(possible_node_level_list) - 1)]
        node_level_2_machine_index_dict[node_level].append(machine_index)
        machine_index_2_node_level_list[machine_index] = node_level
    # except:
    #     status = False
    #     status_description = "Input file format error: possibly errors in \"CompatibleMachines\"."

    # Deal with MACHINE TYPE problem
    machine_type_num = 0
    machine_type_name_2_machine_type_index_dict = {}
    machine_type_index_2_machine_type_name_list = []
    machine_type_index_2_machine_index_dict = defaultdict(list)
    machine_type_index_2_node_level_list = []
    node_level_2_machine_type_index_list = defaultdict(list)
    machine_index_2_machine_type_index_list = []
    u_r_type = []
    q = [0] * machine_num
    for machine_index in range(machine_num):
        # Indexing and registering machine type
        machine_type_name = str(u_r_full[machine_index]) + "/" + str(machine_index_2_node_level_list[machine_index])
        if machine_type_name in machine_type_name_2_machine_type_index_dict.keys():
            machine_type_index = machine_type_name_2_machine_type_index_dict[machine_type_name]
        else:
            machine_type_index = machine_type_num
            machine_type_name_2_machine_type_index_dict[machine_type_name] = machine_type_index
            machine_type_index_2_machine_type_name_list.append(machine_type_name)
            node_level = machine_index_2_node_level_list[machine_index]
            machine_type_index_2_node_level_list.append(node_level)
            node_level_2_machine_type_index_list[node_level].append(machine_type_index)
            u_r_type.append(u_r_full[machine_index])
            machine_type_num += 1

        q[machine_type_index] += 1

        machine_type_index_2_machine_index_dict[machine_type_index].append(machine_index)
        machine_index_2_machine_type_index_list.append(machine_type_index)
    q = q[0:machine_type_num]

    # Deal with x matrix and s matrix
    x_old = np.zeros([service_num, machine_num], dtype=int)
    s_type = np.zeros([machine_type_num, service_num], dtype=int)
    s_full = np.zeros([machine_num, service_num], dtype=int)
    # try:
    # Deal with x_old
    for machine_dict in input_json_stream["MachineList"]:
        machine_index = machine_ip_2_machine_index_dict[machine_dict["MachineIP"]]
        deployed_container_names = machine_dict["InitialDeployingContainers"]
        for container_name in deployed_container_names:
            service_index = container_index_2_service_index_list[
                container_name_2_container_index_dict[container_name]]
            x_old[service_index][machine_index] += 1

    # Deal with s_full and s_type
    each_machine_avail_item = [[] for _ in range(machine_type_num)]
    array_item_node_level_list = np.array(service_node_level_list)
    for machine_type in range(machine_type_num):
        each_machine_avail_item[machine_type] = list(np.where([array_item_node_level_list ==
                                                               machine_type_index_2_node_level_list[machine_type]])[
                                                         1])
    for machine_type in range(machine_type_num):
        s_type[machine_type, each_machine_avail_item[machine_type]] = 1
        for node in machine_type_index_2_machine_index_dict[machine_type]:
            s_full[node, each_machine_avail_item[machine_type]] = 1
    # except:
    #     status = False
    #     status_description = "Input file format error: possibly errors in \"InitialDeployingContainers\"."

    # Deal with "TrafficList" objects
    p = {}
    global_traffic = 0.0
    # try:
    for traffic_json in input_json_stream["TrafficList"]:
        key = (int(service_name_2_service_index_dict[traffic_json["Service1"]]),
               int(service_name_2_service_index_dict[traffic_json["Service2"]]))
        p[key] = traffic_json["Traffic"]
    if "real_global_traffic" in input_json_stream.keys():
        global_traffic = input_json_stream["real_global_traffic"]
    else:
        global_traffic = sum(p.values())
    # except:
    #     status = False
    #     status_description = "Input file format error: possibly errors in \"TrafficList\"."

    # Deal with anti_affinity
    anti_affinity_list = []

    d_r = np.array(d_r)
    u_r_full = np.array(u_r_full)
    u_r_type = np.array(u_r_type)

    L = 100000.0
    d_r = d_r * L
    u_r_type = u_r_type * L
    u_r_full = u_r_full * L

    return x_old, p, d, d_r, s_type, s_full, u_r_type, u_r_full, q, service_index_2_service_name_list, \
           service_name_2_service_index_dict, service_node_level_list, machine_index_2_machine_ip_list, \
           machine_type_index_2_machine_index_dict, machine_index_2_machine_type_index_list, \
           machine_type_index_2_node_level_list, anti_affinity_list, global_traffic, \
           service_index_2_container_index_dict, container_index_2_container_name_list


# # Debug code
# if __name__ == "__main__":
#     json_path = "../../dataset/M3.json"
#     with open(json_path, 'r', encoding='utf-8', ) as fp:
#         json_stream = json.load(fp)
#
#     x_old, p, d, d_r, s_type, s_full, u_r_type, u_r_full, q, service_index_2_service_name_list, \
#     service_name_2_service_index_dict, service_node_level_list, machine_index_2_machine_ip_list, \
#     machine_type_index_2_machine_index_dict, machine_index_2_machine_type_index_list, \
#     machine_type_index_2_node_level_list, anti_affinity_list, global_traffic, status, status_description \
#         = read_from_json_stream(json_stream)
