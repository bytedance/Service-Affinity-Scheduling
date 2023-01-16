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

import os
import sys
APP_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.append(APP_PATH)
from source_code.OurSol_workflow_controller import OurSol_workflow_controller

# Deal with the shell interface

# Print out information for reference
print('----- Welcome to use RASA algorithm -----')
print('  This algorithm is designed for solving cloud resource allocation problem concerning with service affinity.')
print('  You are request to give \r\n'
      '1. the path of the input file;\r\n'
      '2. the path of output file;\r\n'
      '3. maximum runtime in seconds.')
print('Separating them with space, and press enter to run the algorithm.')
print('')
print('  An example is as follow:')
print('../dataset/M1.json ../output/M1_OurSol_result.json 60')
print('')
print('  And then press enter, the algorithm will give an optimized resource allocation plan that optimizing service affinity.')
print('  The log of executing the algorithm will be printed out in the console, in the last part of the log, there will be ')
print('information concerning key indicators of the results, including gained affinity, basic constraint checking etc.')
print('  Now give it a try:')

try:
    # Get input
    input_str = input('')
    print('\r\n')
    input_list = input_str.split(' ')
    print(input_list)
    input_path = input_list[-3]
    output_path = input_list[-2]
    max_runtime = int(input_list[-1])
except:
    print("Sorry, input format error!")
else:
    # Input is good
    print('Input path:', input_path)
    print('Output path:', output_path)
    print('Maximum runtime(can only roughly control) in seconds:', max_runtime)
    OurSol_workflow_controller(input_path, output_path, max_time=max_runtime)

