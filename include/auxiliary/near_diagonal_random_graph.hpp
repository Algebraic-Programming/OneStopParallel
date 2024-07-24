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
#include <stdexcept>
#include <random>
#include <algorithm>
#include <iterator>

#include "auxiliary/auxiliary.hpp"
#include "structures/dag.hpp"

/**
 * @brief Generates a random graph where an edge (i,j), with i<j, is included with probability prob*exp(-(j-i-1)/bandwidth)
 * 
 * @param num_vertices Number of vertices of the graph
 * @param bandwidth chance/num_vertices is the probability of edge inclusion
 * @param prob probability of an edge immediately off the diagonal to be included
 * @return DAG 
 */
DAG near_diag_random_graph( unsigned num_vertices, double bandwidth , double prob);