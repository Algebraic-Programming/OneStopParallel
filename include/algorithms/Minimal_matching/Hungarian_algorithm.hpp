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

#include <cassert>
#include <iostream>
#include <limits.h>
#include <set>
#include <stack>
#include <stdexcept>
#include <vector>

/// @brief Does DFS for Hungarian algorithm
/// @param
/// @return
std::vector<long unsigned> dfs_for_Hungarian_algorithm(std::set<long unsigned> &Z_S, std::set<long unsigned> &Z_T,
                                                  const std::set<long unsigned> &not_assigned_S,
                                                  const std::set<long unsigned> &not_assigned_T,
                                                  const std::vector<std::set<long unsigned>> &tight_forward,
                                                  const std::vector<long int> &tight_backward, std::vector<long unsigned> &path,
                                                  bool last_in_S);

/// @brief Implements the (slower) Hungarian algorithm for complete n*n bipartite graph
/// @param costs edge cost (i,j) = costs[i][j]
/// @return perfect matching (i,out[i]) for i = 0,1,...,n-1
std::vector<unsigned> min_perfect_matching_for_complete_bipartite(const std::vector<std::vector<long long unsigned>> &costs);

std::vector<unsigned> max_perfect_matching_for_complete_bipartite(const std::vector<std::vector<long long unsigned>> &savings);