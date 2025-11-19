# OneStopParallel

This repository, OneStopParallel, is copyright by the
Computing Systems Laboratory, Zurich Research Center, Huawei Technologies
Switzerland AG.

Data and tools are distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
CONDITIONS OF ANY KIND, either express or implied. See the applicable licenses
for the specific language governing permissions and limitations.

## Description

This project aims to develop scheduling algorithms for parallel computing systems based on the Bulk Synchronous Parallel (BSP) model. The algorithms optimize the allocation of tasks to processors, taking into account factors such as load balancing, memory constraints and communication overhead. 


## Tools

All tools in this repository are licensed under the [Apache License, Version 2.0](http://www.apache.org/licenses/LICENSE-2.0).


## Command-Line Tool

A command-line interface is provided to execute scheduling algorithms.  
Users can input an instance (computational DAG + machine parameters) and run the desired scheduler.

For example, to run a **Greedy BSP** algorithm on an example instance:

```bash
./build/apps/osp -g data/spaa/tiny/instance_bicgstab.hdag -m data/machine_params/p3.arch --GreedyBsp
```

To see available options:

```bash
./build/apps/osp
```


## Visualizations

### Sankey Visualization
BSP schedules can be visualized using Sankey diagrams.  
For details, see [SankeyPlots README](third/SankeyPlots/README.md).

### Graphviz Visualization
The folder tools contains a python script to generate a representation of a BspSchedule based on graphviz. The input is a BspSchedule saved in the .dot format. Schedules in the .dot format can, for example, be generated with the command line tool adding the flag "-d". The python script is invoked with the location of the input file, e.g.,
```bash
python3 tools/graphviz_visualization/plot_graphviz.py tools/graphviz_visualization/instance_bicgstab_p4_g3_l5_GreedyBsp_schedule.dot
```
The output is generated in the same folder and has the same name as the input where the file ending is changed from .dot to .gv.


## Quickstart

### Build

```bash
mkdir -p build && cd build
cmake ..
make -j$(nproc)
```

### Install

The project supports CMake installation. From the build directory:

```bash
make install
```


## CMake Options and Build Types

When configuring with CMake, several options and build types influence the build:

### Build Types (`CMAKE_BUILD_TYPE`)

- **Debug**  
  Adds debug symbols (`-g`) and enables strict warnings.  
  ```bash
  cmake -DCMAKE_BUILD_TYPE=Debug ..
  ```

- **Release**  
  Optimized build with `-O3 -DNDEBUG`.  
  ```bash
  cmake -DCMAKE_BUILD_TYPE=Release ..
  ```

- **RelWithDebInfo**  
  Optimized build with debug symbols (`-O3 -g`).  
  ```bash
  cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo ..
  ```

- **Library**  
  Installs only the header-only library (apps/tests are excluded).  
  Useful if you just want to consume the library in another project.  
  ```bash
  cmake -DCMAKE_BUILD_TYPE=Library ..
  ```

If no build type is given, CMake applies default optimizations and warnings.

---

### Options

- **`BUILD_TESTS` (default: ON)**  
  Build and run the test suite.  
  Automatically forced to `OFF` if `CMAKE_BUILD_TYPE=Library`.  
  Requires Boost (graph + unit_test_framework) and OpenMP.  
  ```bash
  cmake -DBUILD_TESTS=OFF ..
  ```

## Optional Functionality
Some algorithms and executables are only enabled with following optional dependencies:
  - [Boost (≥ 1.71)]
  - [Eigen3 (≥ 3.4)](https://eigen.tuxfamily.org/)
  - [OpenMP](https://www.openmp.org/)
  - [COPT](https://github.com/huawei-noah/COPT)

