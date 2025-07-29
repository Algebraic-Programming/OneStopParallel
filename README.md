# one-stop-parallel

This repository, one-stop-parallel, is copyright by the
Computing Systems Laboratory, Zurich Research Center, Huawei Technologies
Switzerland AG.

Data and tools are distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the applicable licenses
for the specific language governing permissions and limitations.

## Description

This project aims to develop scheduling algorithms for parallel computing systems based on the Bulk Synchronous Parallel (BSP) model. The algorithms optimize the allocation of tasks to processors, taking into account factors such as load balancing, memory constraints and communication overhead. 



## Tools

All tools in this directory are licensed under the Apache License, Version 2.0
(the "License"); you may not use the tools except in compliance with the
License. You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0


# Command line tool

The main purpose of this file is to provide a command-line interface for users to execute some of the scheduling algorithms implemented in this project. It allows users to input an instance consisting of a computational DAG and machine parameters and execute the desired scheduling algorithm. For further instructions ./main can be invoked without an parameters.

For example, to run a greedy bsp algortihm on an example instance, the follwing command can be executed (relative to this folder).
```bash
./build/main -g examples/instances/instance_bicgstab.txt -m examples/instances/p4_g3_l5.txt --GreedyBsp
```

# Sankey visualization

The tool provides a visualzation for BspSchedules. For more details, [see here.](third/SankeyPlots/README.md)

# Dot visualization

The folder tools contains a python script to generate a representation of a BspSchedule based on graphviz. The input is a BspSchedule saved in the .dot format. Schedules in the .dot format can, for example, be generated with the command line tool adding the flag "-d". The python script is invoked with the location of the input file, e.g.,
```bash
python /tools/plot_graphviz.py tool/instance_bicgstab_p4_g3_l5_GreedyBsp_schedule.dot
```
The output is generated in the same folder and has the same name as the input where the file ending is changed from .dot to .gv.

## Quickstart

Compile the project using the following commands:

```bash
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```
