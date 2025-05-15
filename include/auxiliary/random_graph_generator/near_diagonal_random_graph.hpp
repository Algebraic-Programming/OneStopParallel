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

#include <random>
#include <vector>

#include "concepts/constructable_computational_dag_concept.hpp"

namespace osp {

/**
 * @brief Generates a random graph where an edge (i,j), with i<j, is included with probability
 * prob*exp(-(j-i-1)/bandwidth)
 *
 * @param num_vertices Number of vertices of the graph
 * @param bandwidth chance/num_vertices is the probability of edge inclusion
 * @param prob probability of an edge immediately off the diagonal to be included
 * @return DAG
 */
template<typename Graph_t>
void near_diag_random_graph(Graph_t &dagout, vertex_idx_t<Graph_t> num_vertices, double bandwidth, double prob) {

    static_assert(is_constructable_cdag_v<Graph_t>, "Graph_t must be a constructable computational DAG type");

    dag_out = Graph_t(num_vertices);

    std::random_device rd;
    std::mt19937 gen(rd());

    std::vector<std::vector<int>> in_(num_vertices);
    std::vector<std::vector<int>> out_(num_vertices);

    for (long int i = 1; i <  (long int) num_vertices; i++) {
        std::binomial_distribution<> bino_dist( num_vertices-i , prob * std::exp( (1-i) / bandwidth ) );
        unsigned off_diag_edges_num = bino_dist(gen);

        // std::cout << "Probability: " << prob << " product: " << prob * std::exp(  (1-i) / bandwidth) << " exponent: " << (1-i) / bandwidth << std::endl;
        // std::cout << "Off diag: " << i << " Number: " << off_diag_edges_num << std::endl;
        
        std::vector<unsigned> range(num_vertices-i,0);
        std::iota(range.begin(), range.end(), 0);
        std::vector<unsigned> sampled;

        std::sample(range.begin(), range.end(), std::back_inserter(sampled), off_diag_edges_num, gen);

        for ( const unsigned& j : sampled ) {
            out_[j].emplace_back(j+i);
            in_[j+i].emplace_back(j);
        }
    }
}

} // namespace osp