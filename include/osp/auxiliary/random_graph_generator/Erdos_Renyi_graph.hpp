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
template <typename Graph_t>
void erdos_renyi_graph_gen(Graph_t &dag_out, vertex_idx_t<Graph_t> num_vertices, double chance) {
    static_assert(is_constructable_cdag_v<Graph_t>, "Graph_t must be a constructable computational DAG type");

    dag_out = Graph_t(num_vertices);

    std::random_device rd;
    std::mt19937 gen(rd());

    for (const auto &v : dag_out.vertices()) {
        const auto one = static_cast<vertex_idx_t<Graph_t>>(1);
        std::binomial_distribution<vertex_idx_t<Graph_t>> bino_dist(num_vertices - one - v, chance / double(num_vertices));
        auto out_edges_num = bino_dist(gen);

        std::unordered_set<vertex_idx_t<Graph_t>> out_edges;
        while (out_edges.size() < static_cast<size_t>(out_edges_num)) {
            std::uniform_int_distribution<vertex_idx_t<Graph_t>> dist(0, num_vertices - one - v);
            vertex_idx_t<Graph_t> edge = v + one + dist(gen);

            if (out_edges.find(edge) != out_edges.cend()) { continue; }

            out_edges.emplace(edge);
        }

        for (auto &j : out_edges) { dag_out.add_edge(v, j); }
    }
}

}    // namespace osp
