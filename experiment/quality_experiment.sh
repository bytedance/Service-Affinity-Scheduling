#!/bin/bash

echo "This shell will run the following algorithms for the RASA problem:"
echo "1. OurSol: A solver-heuristic hybrid approach;"
echo "2. ApplSci19: A min-weight graph cut based heuristic algorithm;"
echo "3. K8s+: A simulated k8s scheduler, with scoring function concerning with the service affinity;"
echo "4. POP: A solver-based algorithm with randomized partitioning techniques"

echo "Enter three parameters, separated with space:"
read -p "input_file_path, output_file_path, maximum_runtime: " input_path output_path maximum_runtime
echo "Received input file path: $input_path"
echo "Received output file path: $output_path"
echo "Receive maximum runtime: $maximum_runtime"

echo "####################### Run OurSol ########################"
oursol=`python ../source_code/OurSol_workflow_controller.py $input_path $output_path $maximum_runtime null 0`
echo $oursol

echo "####################### Run ApplSci19 ########################"
applsci19=`python ../baselines/ApplSci19/ApplSci19_workflow_controller.py $input_path $output_path 0`
echo $applsci19

echo "####################### Run k8s+ ########################"
k8splus=`python ../baselines/K8s_plus/K8s_plus_workflow_controller.py $input_path $output_path 0`
echo $k8splus

echo "####################### Run POP ########################"
k8splus=`python ../baselines/POP/POP_workflow_controller.py $input_path $output_path $maximum_runtime 4 0`
echo $k8splus
