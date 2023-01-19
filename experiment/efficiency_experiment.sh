#!/bin/bash

echo "This shell will run one of the following assigned algorithm for RASA with different time-out parameters:"
echo "1. OurSol: A solver-heuristic hybrid approach;"
echo "2. ApplSci19: A min-weight graph cut based heuristic algorithm;"
echo "3. K8s+: A simulated k8s scheduler, with scoring function concerning with the service affinity;"
echo "4. POP: A solver-based algorithm with randomized partitioning techniques"

echo "Enter three parameters, separated with space:"
echo "For time_range, three parameters are required, that is the minimum time, maximum time, and increment."
echo "An example is the time_range parameters input is 2 10 3, then we will run the algorithm with maximum runtime of 2, 2+3, 2+3*2 (2, 5, 8), respectively."
read -p "[input_file_path], [algorithm_index({1,2,3,4})], [time_range(3 numbers, min_time, max_time and increment)]: " input_path algorithm_index mintime maxtime increment

echo "Received input file path: $input_path"
echo "Received algorithm index: $algorithm_index"
echo "Receive minimum runtime: $mintime"
echo "Receive maximum runtime: $maxtime"
echo "Receive incrementation of time: $increment"

output_path="../output/efficiency_tmp.json"

if [ $algorithm_index == "1" ]
then
  echo "####################### Run OurSol ########################"
  for (( i=$mintime;i<=$maxtime;i=i+$increment ))
  do
   oursol=`python ../source_code/OurSol_workflow_controller.py $input_path $output_path $i null 0`
   echo $oursol
  done
elif [ $algorithm_index == "2" ]
then
  echo "####################### Run ApplSci19 ########################"
  applsci19=`python ../baselines/ApplSci19/ApplSci19_workflow_controller.py $input_path $output_path 0`
  echo $applsci19
elif [ $algorithm_index == "3" ]
then
  echo "####################### Run k8s+ ########################"
  k8splus=`python ../baselines/K8s_plus/K8s_plus_workflow_controller.py $input_path $output_path 0`
  echo $k8splus
elif [ $algorithm_index == "4" ]
then
  echo "####################### Run POP ########################"
  for (( i=$mintime;i<=$maxtime;i=i+$increment ))
  do
   pop=`python ../baselines/POP/POP_workflow_controller.py $input_path $output_path $i 4 0`
   echo $pop
  done
else
   echo "The input index of algorithm is invalid, should be in {1, 2, 3, 4}."
fi