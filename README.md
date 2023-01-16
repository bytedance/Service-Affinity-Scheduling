## Introduction

This repository contains the implementation of the algorithm to **resource allocation with service affinity (RASA)** 
problem. To be specific, we consider the constrained optimization problem that aims to find a mapping from containers
to machines, in a way that maximize the affinity between services. If we could benefit from collocating two containers
on the same machine, or the same group of machines, then we say the services of the two containers have an affinity 
relation, which is denoted as **service affinity**. Optimizing service affinity could boost application performance and
reduce resource usage. 

In this repository, we provide a novel solver-heuristic hybrid approach to solve the problem of RASA, in a way that 
ensuring both computational efficiency and near-optimality. This is especially helpful for problem in large industry 
scale.

## Requirements

To make sure that you can run the code successfully, please check the following requirements.

#### System requirement

Debian 9 (and other Unix-like system) are desired.

The code was also tested on MacOS system (v12 or higher).

#### Installing packages

- Python: the code was tested on python 3.7. Higher version can be compatible, too.

- numpy: the code was tested on numpy 1.23.4.
  
- gurobipy: the code was test on gurobipy 9.5.1.
  **To use the Gurobi as the mathematical programming solver, a Gurobi licensed is required. If you are an academic 
  user, you can apply for an [academic license](https://www.gurobi.com/academia/academic-program-and-licenses). If you 
  are a non-academic user, you can apply for an 
  [evaluation license](https://www.gurobi.com/downloads/request-an-evaluation-license/). After obtaining the license, 
  you can refer to [Gurobi documentation](https://www.gurobi.com/documentation/quickstart.html) for installation of 
  the software and the license.**

- torch: the code was tested on torch 1.11.0

- dgl: the code was tested on dgl 0.9.0

## Datasets

The format of input data can be referred in
[research-cluster-dataset](https://github.com/bytedance/research-cluster-dataset), in which container data, machine data 
and affinity data are provided. Four example input files are given in the `dataset` directory.

## Run the code

To illustrate how to run the code and baselines, we will `dataset\M1.json` as the example of the input file.

#### Run RASA algorithm
To run our solver-heuristic hybrid algorithm, you can follow the following steps:

1) Make sure that you have installed related packages as required in
   [Requirements](#Requirements).
2) Open a terminal, change the directory to `source_code`.
3) The `main.py` in `source_code` directory is the entry of our approach. Enter `python main.py` in the terminal .
4) A few paragraphs will be printed on the terminal, indicating how to enter the related parameters. You are required
to give 3 parameters, separated by space.
   1. The path of the input file (The path can be a global path or a related path to `source_code\` );
   2. The path of the output file;
   3. The maximum runtime (we can only roughly control it).
  An example is `../dataset/M1.json ../output/M1_OurSol_result.json 60`.
      
5) After given the parameters, the algorithm starts and finally print out key indicators. And the schedule will also
be stored on the given output file path.

#### Run experiments
The three baselines(ApplSci19, k8s+, POP) are given in the `baselines` directory. The entries for these algorithms are 
file with suffix of `_workflow_controller.py`. An example to run these baselines are as follows(Run ApplSci19's 
algorithm): enter `python ApplSci19_workflow_controller.py ../../dataset/M1.json ../../output/applSci19_M1_result.json` 
in the terminal.

Furthermore, we give a more convenient way to run the experiment. In the `experiment` directory, three shell files are 
given, use `bash shell_file_name.sh` to automatically run the corresponding experiments and get the results. See the 
following content:

###### Gained affinity comparison
To run the gained affinity comparison of our approach and three baselines, just `bash gained_affinity_comparison.sh` in 
the `experiment` directory. The shell console will require you to enter three parameters: 1. input file path; 2.output 
file path; 3. maximum runtime.  These three parameters are all compulsory. An example is 
`../dataset/M1.json ../output/shell_M1_test.json 60`, which will take `M1.json` as the input and compare their gained 
affinity within 60 seconds.

###### Data splitting comparison
To be added.

###### Algorithm selection comparison
To run the algorithm selection comparison of GCN-Based, only CG and only MIP, just `bash algorithm_selection_comparison.sh`
in the `experiment` directory. The shell console will require you to enter three parameters: 1. input file path; 
2.output file path; 3. maximum runtime.  These three parameters are all compulsory. An example is 
`../dataset/M1.json ../output/shell_M1_test.json 60`, which will take `M1.json` as the input and compare their gained 
affinity within 60 seconds.

## Security

If you discover a potential security issue in this project, or think you may
have discovered a security issue, we ask that you notify Bytedance Security 
via our [security center](https://security.bytedance.com/src) or
[vulnerability reporting email](sec@bytedance.com).

Please do **not** create a public GitHub issue.

## License

This project is licensed under the [Apache-2.0 License](LICENSE).
