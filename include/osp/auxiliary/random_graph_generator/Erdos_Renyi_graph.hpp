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
 * @param numVertices Number of vertices of the graph
 * @param chance chance/numVertices is the probability of edge inclusion
 * @return DAG
 */
template <typename GraphT>
void ErdosRenyiGraphGen(GraphT &dagOut, VertexIdxT<GraphT> numVertices, double chance) {
    static_assert(IsConstructableCdagV<GraphT>, "Graph_t must be a constructable computational DAG type");

    dagOut = GraphT(numVertices);

    std::random_device rd;
    std::mt19937 gen(rd());

    for (const auto &v : dagOut.Vertices()) {
        const auto one = static_cast<VertexIdxT<GraphT>>(1);
        std::binomial_distribution<VertexIdxT<GraphT>> binoDist(numVertices - one - v, chance / double(numVertices));
        auto outEdgesNum = binoDist(gen);

        std::unordered_set<VertexIdxT<GraphT>> outEdges;
        while (outEdges.size() < static_cast<size_t>(outEdgesNum)) {
            std::uniform_int_distribution<VertexIdxT<GraphT>> dist(0, numVertices - one - v);
            VertexIdxT<GraphT> edge = v + one + dist(gen);

            if (outEdges.find(edge) != outEdges.cend()) {
                continue;
            }

            outEdges.emplace(edge);
        }

        for (auto &j : outEdges) {
            dagOut.AddEdge(v, j);
        }
    }
}

}    // namespace osp
