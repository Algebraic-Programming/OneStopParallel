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

#include <numeric>
#include <random>
#include <unordered_set>
#include <vector>

#include "osp/concepts/constructable_computational_dag_concept.hpp"

namespace osp {

/**
 * @brief Generates a Erdos Renyi random directed graph
 *
 * @param num_vertices Number of vertices of the graph
 * @param chance chance/num_vertices is the probability of edge inclusion
 * @return DAG
 */
template <typename GraphT>
void ErdosRenyiGraphGen(GraphT &dagOut, vertex_idx_t<Graph_t> numVertices, double chance) {
    static_assert(IsConstructableCdagV<Graph_t>, "Graph_t must be a constructable computational DAG type");

    dagOut = GraphT(num_vertices);

    std::random_device rd;
    std::mt19937 gen(rd());

    for (const auto &v : dagOut.vertices()) {
        const auto one = static_cast<vertex_idx_t<Graph_t>>(1);
        std::binomial_distribution<vertex_idx_t<Graph_t>> binoDist(numVertices - one - v, chance / double(num_vertices));
        auto outEdgesNum = bino_dist(gen);

        std::unordered_set<vertex_idx_t<Graph_t>> outEdges;
        while (outEdges.size() < static_cast<size_t>(out_edges_num)) {
            std::uniform_int_distribution<vertex_idx_t<Graph_t>> dist(0, num_vertices - one - v);
            vertex_idx_t<Graph_t> edge = v + one + dist(gen);

            if (outEdges.find(edge) != out_edges.cend()) {
                continue;
            }

            outEdges.emplace(edge);
        }

        for (auto &j : out_edges) {
            dag_out.add_edge(v, j);
        }
    }
}

}    // namespace osp
