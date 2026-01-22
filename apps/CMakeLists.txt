# Copyright 2024 Huawei Technologies Co., Ltd.

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# @author Toni Boehnlein, Benjamin Lozes, Pal Andras Papp, Raphael S. Steiner


# Create an empty list for the executables
set( executable_list )

# Function to create an executable
macro( _add_executable name)
  # Add the executable to the list
  list( APPEND executable_list ${name} )
  # Create the executable
  add_executable( ${name} ${name}.cpp )
  target_link_libraries( ${name} PRIVATE OneStopParallel ProjectExecutableFlags )
endmacro()


_add_executable( graph_analyser )

_add_executable( graph_converter )

if(Boost_FOUND)

_add_executable( osp_turnus )

_add_executable ( osp )

configure_file(config/osp_config.json osp_config.json COPYONLY)

_add_executable ( bsp_test_suite )

_add_executable( coarser_plotter )

if(Eigen3_FOUND)
_add_executable ( sptrsv_test_suite )
endif()

if (COPT_FOUND)
_add_executable( ilp_bsp_scheduler )
_add_executable( ilp_hypergraph_partitioner )
_add_executable( ilp_partitioning_test_suite )
endif()

endif()

# Custom target to compile all the executables
add_custom_target( build_executables DEPENDS ${executable_list} )
