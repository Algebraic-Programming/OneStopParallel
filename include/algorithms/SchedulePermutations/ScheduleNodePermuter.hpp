/*
Copyright 2024 Huawei Technologies Co., Ltd.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

@author Toni Boehnlein, Benjamin Lozes, Pal Andras Papp, Raphael S. Steiner   
*/

#pragma once

#include <vector>
#include <algorithm>
#include <queue>

#include "model/BspSchedule.hpp"

enum SCHEDULE_NODE_PERMUTATION_MODES { LOOP_PROCESSORS, SNAKE_PROCESSORS, PROCESSOR_FIRST };

/**
 * @brief Computes a permutation to improve locality of a schedule, looping through processors
 * 
 * @param sched BSP Schedule
 * @param mode ordering of processors
 * @return std::vector<size_t> vec[prev_node_name] = new_node_name(location)
 */
std::vector<size_t> schedule_node_permuter(const BspSchedule& sched, unsigned cache_line_size, const SCHEDULE_NODE_PERMUTATION_MODES mode = LOOP_PROCESSORS, const bool simplified = false);

/**
 * @brief Computes a permutation to improve locality of a schedule, looping through processors
 * 
 * @param sched BSP Schedule
 * @param mode ordering of processors
 * @return std::vector<size_t> vec[prev_node_name] = new_node_name(location)
 */
std::vector<size_t> schedule_node_permuter_basic(const BspSchedule& sched, const SCHEDULE_NODE_PERMUTATION_MODES mode = LOOP_PROCESSORS);

/**
 * @brief Improves the sorting of nodes for data locality in the interior of a superstep
 * 
 * @param nodes Node subset
 * @param sched BSP Schedule
 */
void topological_sort_for_data_locality_interior(std::vector<size_t>& nodes, const BspSchedule& sched, unsigned cache_line_size);

/**
 * @brief Improves the sorting of nodes for data locality in the interior of a superstep
 * 
 * @param nodes Node subset
 * @param sched BSP Schedule
 */
void topological_sort_for_data_locality_interior_basic(std::vector<size_t>& nodes, const BspSchedule& sched);

/**
 * @brief Improves the sorting of nodes for data locality on the beginning of a superstep
 * 
 * @param nodes Node subset
 * @param sched BSP Schedule
 */
void topological_sort_for_data_locality_begin(std::vector<size_t>& nodes, const BspSchedule& sched, unsigned cache_line_size, const std::vector<std::vector<std::vector<size_t>>>& allocation);

/**
 * @brief Improves the sorting of nodes for data locality on the end of a superstep
 * 
 * @param nodes Node subset
 * @param sched BSP Schedule
 */
void topological_sort_for_data_locality_end(std::vector<size_t>& nodes, const BspSchedule& sched, unsigned cache_line_size, const std::vector<std::vector<std::vector<size_t>>>& allocation);
