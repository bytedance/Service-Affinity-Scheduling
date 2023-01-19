#!/bin/bash

echo "This shell will run the following algorithm selection techniques for the RASA problem:"
echo "1. GCN-Based: A heuristic-GCN hybrid approach;"
echo "2. only CG: Only uses column generation (CG) based algorithm;"
echo "3. only MIP: Only uses mixture integer programming (MIP) based algorithm;"

echo "Enter three parameters, separated with space:"
read -p "input_file_path, output_file_path, maximum_runtime: " input_path output_path maximum_runtime
echo "Received input file path: $input_path"
echo "Received output file path: $output_path"
echo "Receive maximum runtime: $maximum_runtime"

echo "####################### Run GCN-Based ########################"
oursol=`python ../source_code/OurSol_workflow_controller.py $input_path $output_path $maximum_runtime null 0`
echo $oursol

echo "####################### Run CG ########################"
cg=`python ../source_code/OurSol_workflow_controller.py $input_path $output_path $maximum_runtime cg 0`
echo $cg

echo "####################### Run MIP ########################"
mip=`python ../source_code/OurSol_workflow_controller.py $input_path $output_path $maximum_runtime mip 0`
echo $mip