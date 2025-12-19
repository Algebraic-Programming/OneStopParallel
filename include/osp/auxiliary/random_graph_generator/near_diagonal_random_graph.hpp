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

#include "osp/concepts/constructable_computational_dag_concept.hpp"

namespace osp {

/**
 * @brief Generates a random graph where an edge (i,j), with i<j, is included with probability
 * prob*exp(-(j-i-1)/bandwidth)
 *
 * @param numVertices Number of vertices of the graph
 * @param bandwidth chance/numVertices is the probability of edge inclusion
 * @param prob probability of an edge immediately off the diagonal to be included
 * @return DAG
 */
template <typename GraphT>
void NearDiagRandomGraph(GraphT &dagOut, VertexIdxT<GraphT> numVertices, double bandwidth, double prob) {
    static_assert(isConstructableCdagV<GraphT>, "GraphT must be a constructable computational DAG type");

    dagOut = GraphT(numVertices);

    std::random_device rd;
    std::mt19937 gen(rd());

    for (VertexIdxT<GraphT> v = 1; v < numVertices; ++v) {
        std::binomial_distribution<VertexIdxT<GraphT>> binoDist(VertexIdxT<GraphT>(numVertices - v),
                                                                prob * std::exp(1.0 - static_cast<double>(v) / bandwidth));
        VertexIdxT<GraphT> offDiagEdgesNum = binoDist(gen);

        std::vector<VertexIdxT<GraphT>> range(numVertices - v, 0);
        std::iota(range.begin(), range.end(), 0);
        std::vector<VertexIdxT<GraphT>> sampled;

        std::sample(range.begin(), range.end(), std::back_inserter(sampled), offDiagEdgesNum, gen);

        for (const auto &j : sampled) {
            dagOut.AddEdge(j, j + 1);
        }
    }
}

}    // namespace osp
