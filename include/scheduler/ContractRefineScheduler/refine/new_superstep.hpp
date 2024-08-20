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

#include "auxiliary/auxiliary.hpp"
#include "structures/dag.hpp"
#include "structures/union_find.hpp"
#include <algorithm>
#include <cassert>
#include <queue>
#include <tuple>
#include <unordered_set>
#include <utility>
#include <vector>

enum CutType { Balanced, Shaving };

// binary search for weight-balanced cut decision
// returns cut (cut between cut-1 and cut), new bias
int binary_search_weight_bal_cut_along_chain(const SubDAG &graph, const std::vector<int> chain, const int bias = 0);

// Introduces a weight-balanced superstep into a directed acyclic graph thus cutting it into into multiple
// weakly-connected components Returns node assigments (top, bottom) by node name of super-dag
std::pair<std::unordered_set<int>, std::unordered_set<int>> dag_weight_bal_cut(const SubDAG &graph, int bias = 0);

// Introduces a superstep, by shaving from the top, to either increase source nodes of bottom or to have multiple
// components in top Returns node assigments (top, bottom) by node name of super-dag
std::pair<std::unordered_set<int>, std::unordered_set<int>> top_shave_few_sources(const SubDAG &graph,
                                                                                  const int min_comp_generation = 3,
                                                                                  const float mult_size_cap = 0.5);

// Introduces a superstep, by shaving from the bottom, to either increase source nodes of top or to have multiple
// components in bottom Returns node assigments (top, bottom) by node name of super-dag
std::pair<std::unordered_set<int>, std::unordered_set<int>> bottom_shave_few_sinks(const SubDAG &graph,
                                                                                   const int min_comp_generation = 3,
                                                                                   const float mult_size_cap = 0.5);