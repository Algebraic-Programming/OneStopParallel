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

#include <set>
#include <stdexcept>
#include <type_traits>
#include <vector>

#include "osp/auxiliary/datastructures/union_find.hpp"
#include "osp/coarser/Coarser.hpp"
#include "osp/graph_algorithms/directed_graph_coarsen_util.hpp"
#include "osp/graph_algorithms/directed_graph_path_util.hpp"

namespace osp {

namespace SquashAParams {

enum class Mode { EDGE_WEIGHT, TRIANGLES };

struct Parameters {
    double geomDecayNumNodes{17.0 / 16.0};
    double poissonPar{0.0};
    unsigned noise{0U};
    std::pair<unsigned, unsigned> edgeSortRatio{3, 2};
    unsigned numRepWithoutNodeDecrease{4};
    double temperatureMultiplier{1.125};
    unsigned numberOfTemperatureIncreases{14};
    Mode mode{Mode::EDGE_WEIGHT};
    bool useStructuredPoset{false};
    bool useTopPoset{true};
};

}    // end namespace SquashAParams

template <typename GraphTIn, typename GraphTOut>
class SquashA : public CoarserGenExpansionMap<GraphTIn, GraphTOut> {
  private:
    SquashAParams::Parameters params_;

    std::vector<int> GeneratePosetInMap(const GraphTIn &dag_in);

    template <typename T, typename CMP>
    std::vector<std::vector<VertexIdxT<GraphTIn>>> GenExpMapFromContractableEdges(
        const std::multiset<std::pair<EdgeDescT<GraphTIn>, T>, CMP> &edgeWeights,
        const std::vector<int> &posetIntMapping,
        const GraphTIn &dagIn) {
        static_assert(std::is_arithmetic_v<T>, "T must be of arithmetic type!");

        auto lowerThirdIt = edgeWeights.begin();
        std::advance(lowerThirdIt, edgeWeights.size() / 3);
        T lowerThirdWt = std::max(lowerThirdIt->second, static_cast<T>(1));    // Could be 0

        UnionFindUniverse<VertexIdxT<GraphTIn>, VertexIdxT<GraphTIn>, VWorkwT<GraphTIn>, VMemwT<GraphTIn>> connectedComponents;
        for (const auto &vert : dagIn.Vertices()) {
            connectedComponents.AddObject(vert, dagIn.VertexWorkWeight(vert), dagIn.VertexMemWeight(vert));
        }

        std::vector<bool> mergedNodes(dagIn.NumVertices(), false);

        VertexIdxT<GraphTIn> numNodesDecrease = 0;
        VertexIdxT<GraphTIn> numNodesAim
            = dagIn.NumVertices()
              - static_cast<VertexIdxT<GraphTIn>>(static_cast<double>(dagIn.NumVertices()) / params_.geomDecayNumNodes);

        double temperature = 1;
        unsigned temperatureIncreaseIteration = 0;
        while (numNodesDecrease < numNodesAim && temperatureIncreaseIteration <= params_.numberOfTemperatureIncreases) {
            for (const auto &wtEdge : edgeWeights) {
                const auto &edgeD = wtEdge.first;
                const VertexIdxT<GraphTIn> edgeSource = Source(edgeD, dagIn);
                const VertexIdxT<GraphTIn> edgeTarget = Target(edgeD, dagIn);

                // Previously merged
                if (mergedNodes[edgeSource]) {
                    continue;
                }
                if (mergedNodes[edgeTarget]) {
                    continue;
                }

                // weight check
                if (connectedComponents.GetWeightOfComponentByName(edgeSource)
                        + connectedComponents.GetWeightOfComponentByName(edgeTarget)
                    > static_cast<double>(lowerThirdWt) * temperature) {
                    continue;
                }

                // no loops criteria check
                bool checkFailed = false;
                // safety check - this should already be the case
                assert(abs(posetIntMapping[edgeSource] - posetIntMapping[edgeTarget]) <= 1);
                // Checks over all affected edges
                // In edges first
                for (const auto &node : dagIn.Parents(edgeSource)) {
                    if (node == edgeTarget) {
                        continue;
                    }
                    if (!mergedNodes[node]) {
                        continue;
                    }
                    if (posetIntMapping[edgeSource] >= posetIntMapping[node] + 2) {
                        continue;
                    }
                    checkFailed = true;
                    break;
                }
                if (checkFailed) {
                    continue;
                }
                // Out edges first
                for (const auto &node : dagIn.Children(edgeSource)) {
                    if (node == edgeTarget) {
                        continue;
                    }
                    if (!mergedNodes[node]) {
                        continue;
                    }
                    if (posetIntMapping[node] >= posetIntMapping[edgeSource] + 2) {
                        continue;
                    }
                    checkFailed = true;
                    break;
                }
                if (checkFailed) {
                    continue;
                }
                // In edges second
                for (const auto &node : dagIn.Parents(edgeTarget)) {
                    if (node == edgeSource) {
                        continue;
                    }
                    if (!mergedNodes[node]) {
                        continue;
                    }
                    if (posetIntMapping[edgeTarget] >= posetIntMapping[node] + 2) {
                        continue;
                    }
                    checkFailed = true;
                    break;
                }
                if (checkFailed) {
                    continue;
                }
                // Out edges second
                for (const auto &node : dagIn.Children(edgeTarget)) {
                    if (node == edgeSource) {
                        continue;
                    }
                    if (!mergedNodes[node]) {
                        continue;
                    }
                    if (posetIntMapping[node] >= posetIntMapping[edgeTarget] + 2) {
                        continue;
                    }
                    checkFailed = true;
                    break;
                }
                if (checkFailed) {
                    continue;
                }

                // merging
                connectedComponents.JoinByName(edgeSource, edgeTarget);
                mergedNodes[edgeSource] = true;
                mergedNodes[edgeTarget] = true;
                numNodesDecrease++;
            }

            temperature *= params_.temperatureMultiplier;
            temperatureIncreaseIteration++;
        }

        // Getting components to contract and adding graph contraction
        std::vector<std::vector<VertexIdxT<GraphTIn>>> partitionVec;

        VertexIdxT<GraphTIn> minNodeDecrease = dagIn.NumVertices()
                                               - static_cast<VertexIdxT<GraphTIn>>(static_cast<double>(dagIn.NumVertices())
                                                                                   / std::pow(params_.geomDecayNumNodes, 0.25));
        if (numNodesDecrease > 0 && numNodesDecrease >= minNodeDecrease) {
            partitionVec = connectedComponents.GetConnectedComponents();

        } else {
            partitionVec.reserve(dagIn.NumVertices());
            for (const auto &vert : dagIn.Vertices()) {
                std::vector<VertexIdxT<GraphTIn>> vect;
                vect.push_back(vert);
                partitionVec.emplace_back(vect);
            }
        }

        return partitionVec;
    }

  public:
    virtual std::vector<std::vector<VertexIdxT<GraphTIn>>> generate_vertex_expansion_map(const GraphTIn &dag_in) override;

    SquashA(SquashAParams::Parameters params = SquashAParams::Parameters()) : params_(params) {};

    SquashA(const SquashA &) = default;
    SquashA(SquashA &&) = default;
    SquashA &operator=(const SquashA &) = default;
    SquashA &operator=(SquashA &&) = default;
    virtual ~SquashA() override = default;

    inline SquashAParams::Parameters &GetParams() { return params_; }

    inline void SetParams(SquashAParams::Parameters params) { params_ = params; }

    std::string GetCoarserName() const override { return "SquashA"; }
};

template <typename GraphTIn, typename GraphTOut>
std::vector<int> SquashA<GraphTIn, GraphTOut>::GeneratePosetInMap(const GraphTIn &dag_in) {
    std::vector<int> posetIntMapping;
    if (!params_.useStructuredPoset) {
        posetIntMapping = GetStrictPosetIntegerMap<GraphTIn>(params_.noise, params_.poissonPar, dag_in);
    } else {
        if (params_.useTopPoset) {
            posetIntMapping = GetTopNodeDistance<GraphTIn, int>(dag_in);
        } else {
            std::vector<int> botDist = GetBottomNodeDistance<GraphTIn, int>(dag_in);
            posetIntMapping.resize(botDist.size());
            for (std::size_t i = 0; i < botDist.size(); i++) {
                posetIntMapping[i] = -botDist[i];
            }
        }
    }
    return posetIntMapping;
}

template <typename GraphTIn, typename GraphTOut>
std::vector<std::vector<VertexIdxT<GraphTIn>>> SquashA<GraphTIn, GraphTOut>::GenerateVertexExpansionMap(const GraphTIn &dagIn) {
    static_assert(isDirectedGraphEdgeDescV<GraphTIn>, "Graph_t_in must satisfy the directed_graph_edge_desc concept");
    static_assert(isComputationalDagEdgeDescV<GraphTIn>, "Graph_t_in must satisfy the is_computational_dag_edge_desc concept");
    // static_assert(has_hashable_edge_desc_v<Graph_t_in>, "Graph_t_in must have hashable edge descriptors");

    std::vector<int> posetIntMapping = GeneratePosetInMap(dagIn);

    if constexpr (hasEdgeWeightsV<GraphTIn>) {
        if (params_.mode == SquashAParams::Mode::EDGE_WEIGHT) {
            auto edgeWCmp = [](const std::pair<EdgeDescT<GraphTIn>, ECommwT<GraphTIn>> &lhs,
                               const std::pair<EdgeDescT<GraphTIn>, ECommwT<GraphTIn>> &rhs) { return lhs.second < rhs.second; };
            std::multiset<std::pair<EdgeDescT<GraphTIn>, ECommwT<GraphTIn>>, decltype(edgeWCmp)> edgeWeights(edgeWCmp);
            {
                std::vector<EdgeDescT<GraphTIn>> contractableEdges
                    = GetContractableEdgesFromPosetIntMap<GraphTIn>(posetIntMapping, dagIn);
                for (const auto &edge : contractableEdges) {
                    if constexpr (hasEdgeWeightsV<GraphTIn>) {
                        edgeWeights.emplace(edge, dagIn.EdgeCommWeight(edge));
                    } else {
                        edgeWeights.emplace(edge, dagIn.vertex_comm_weight(source(edge, dagIn)));
                    }
                }
            }

            return GenExpMapFromContractableEdges<ECommwT<GraphTIn>, decltype(edgeWCmp)>(edgeWeights, posetIntMapping, dagIn);
        }
    }
    if (params_.mode == SquashAParams::Mode::TRIANGLES) {
        auto edgeWCmp = [](const std::pair<EdgeDescT<GraphTIn>, std::size_t> &lhs,
                           const std::pair<EdgeDescT<GraphTIn>, std::size_t> &rhs) { return lhs.second < rhs.second; };
        std::multiset<std::pair<EdgeDescT<GraphTIn>, std::size_t>, decltype(edgeWCmp)> edgeWeights(edgeWCmp);
        {
            std::vector<EdgeDescT<GraphTIn>> contractableEdges
                = GetContractableEdgesFromPosetIntMap<GraphTIn>(posetIntMapping, dagIn);
            for (const auto &edge : contractableEdges) {
                std::size_t numCommonTriangles = NumCommonParents(dagIn, Source(edge, dagIn), Target(edge, dagIn));
                numCommonTriangles += NumCommonChildren(dagIn, Source(edge, dagIn), Target(edge, dagIn));
                edgeWeights.emplace(edge, numCommonTriangles);
            }
        }

        return GenExpMapFromContractableEdges<std::size_t, decltype(edgeWCmp)>(edgeWeights, posetIntMapping, dagIn);

    } else {
        throw std::runtime_error("Edge sorting mode not recognised.");
    }

    return {};
}

}    // end namespace osp
