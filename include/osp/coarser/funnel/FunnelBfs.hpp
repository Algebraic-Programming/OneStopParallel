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
#include <limits>

#include "osp/coarser/Coarser.hpp"
#include "osp/graph_algorithms/directed_graph_edge_desc_util_parallel.hpp"

namespace osp {

/**
 * @brief Acyclic graph contractor that contracts groups of nodes with only one vertex with incoming/outgoing edges
 * (from outside the group)
 *
 */
template <typename GraphTIn, typename GraphTOut, bool useArchitectureMemoryContraints = false>
class FunnelBfs : public CoarserGenExpansionMap<GraphTIn, GraphTOut> {
  public:
    /**
     * @brief Parameters for Funnel coarsener
     *
     */
    struct FunnelBfsParameters {
        bool funnelIncoming_;

        bool useApproxTransitiveReduction_;

        VWorkwT<GraphTIn> maxWorkWeight_;
        VMemwT<GraphTIn> maxMemoryWeight_;

        unsigned maxDepth_;

        FunnelBfsParameters(VWorkwT<GraphTIn> maxWorkWeight = std::numeric_limits<VWorkwT<GraphTIn>>::max(),
                            VMemwT<GraphTIn> maxMemoryWeight = std::numeric_limits<VMemwT<GraphTIn>>::max(),
                            unsigned maxDepth = std::numeric_limits<unsigned>::max(),
                            bool funnelIncoming = true,
                            bool useApproxTransitiveReduction = true)
            : funnelIncoming_(funnelIncoming),
              useApproxTransitiveReduction_(useApproxTransitiveReduction),
              maxWorkWeight_(maxWorkWeight),
              maxMemoryWeight_(maxMemoryWeight),
              maxDepth_(maxDepth) {};

        ~FunnelBfsParameters() = default;
    };

    FunnelBfs(FunnelBfsParameters parameters = FunnelBfsParameters()) : parameters_(parameters) {}

    virtual ~FunnelBfs() = default;

    virtual std::vector<std::vector<VertexIdxT<GraphTIn>>> GenerateVertexExpansionMap(const GraphTIn &graph) override {
        if constexpr (useArchitectureMemoryContraints) {
            if (maxMemoryPerVertexType_.size() < graph.NumVertexTypes()) {
                throw std::runtime_error("FunnelBfs: max_memory_per_vertex_type has insufficient size.");
            }
        }

        std::vector<std::vector<VertexIdxT<GraphTIn>>> partition;

        if (parameters_.funnelIncoming_) {
            RunInContraction(graph, partition);
        } else {
            RunOutContraction(graph, partition);
        }

        return partition;
    }

    std::string GetCoarserName() const override { return "FunnelBfs"; }

    std::vector<VMemwT<GraphTIn>> &GetMaxMemoryPerVertexType() { return maxMemoryPerVertexType_; }

  private:
    FunnelBfsParameters parameters_;

    std::vector<VMemwT<GraphTIn>> maxMemoryPerVertexType_;

    void RunInContraction(const GraphTIn &graph, std::vector<std::vector<VertexIdxT<GraphTIn>>> &partition) {
        using VertexIdxT = VertexIdxT<GraphTIn>;

        const std::unordered_set<EdgeDescT<GraphTIn>> edgeMask = parameters_.useApproxTransitiveReduction_
                                                                     ? LongEdgesInTrianglesParallel(graph)
                                                                     : std::unordered_set<EdgeDescT<GraphTIn>>();

        std::vector<bool> visited(graph.NumVertices(), false);

        const std::vector<VertexIdxT> topOrder = GetTopOrder(graph);

        for (auto revTopIt = topOrder.rbegin(); revTopIt != topOrder.crend(); revTopIt++) {
            const VertexIdxT &bottomNode = *revTopIt;

            if (visited[bottomNode]) {
                continue;
            }

            VWorkwT<GraphTIn> workWeightOfGroup = 0;
            VMemwT<GraphTIn> memoryWeightOfGroup = 0;

            std::unordered_map<VertexIdxT, VertexIdxT> childrenNotInGroup;
            std::vector<VertexIdxT> group;

            std::deque<VertexIdxT> vertexProcessingFifo({bottomNode});
            std::deque<VertexIdxT> nextVertexProcessingFifo;

            unsigned depthCounter = 0;

            while ((not vertexProcessingFifo.empty()) || (not nextVertexProcessingFifo.empty())) {
                if (vertexProcessingFifo.empty()) {
                    vertexProcessingFifo = nextVertexProcessingFifo;
                    nextVertexProcessingFifo.clear();
                    depthCounter++;
                    if (depthCounter > parameters_.maxDepth_) {
                        break;
                    }
                }

                VertexIdxT activeNode = vertexProcessingFifo.front();
                vertexProcessingFifo.pop_front();

                if (graph.VertexType(activeNode) != graph.VertexType(bottomNode)) {
                    continue;
                }

                if (workWeightOfGroup + graph.VertexWorkWeight(activeNode) > parameters_.maxWorkWeight_) {
                    continue;
                }

                if (memoryWeightOfGroup + graph.VertexMemWeight(activeNode) > parameters_.maxMemoryWeight_) {
                    continue;
                }

                if constexpr (useArchitectureMemoryContraints) {
                    if (memoryWeightOfGroup + graph.VertexMemWeight(activeNode)
                        > maxMemoryPerVertexType_[graph.VertexType(bottomNode)]) {
                        continue;
                    }
                }

                group.emplace_back(activeNode);
                workWeightOfGroup += graph.VertexWorkWeight(activeNode);
                memoryWeightOfGroup += graph.VertexMemWeight(activeNode);

                for (const auto &inEdge : InEdges(activeNode, graph)) {
                    if (parameters_.useApproxTransitiveReduction_ && (edgeMask.find(inEdge) != edgeMask.cend())) {
                        continue;
                    }

                    const VertexIdxT &par = Source(inEdge, graph);

                    if (childrenNotInGroup.find(par) != childrenNotInGroup.cend()) {
                        childrenNotInGroup[par] -= 1;

                    } else {
                        if (parameters_.useApproxTransitiveReduction_) {
                            childrenNotInGroup[par] = 0;

                            for (const auto outEdge : OutEdges(par, graph)) {
                                if (edgeMask.find(outEdge) != edgeMask.cend()) {
                                    continue;
                                }
                                childrenNotInGroup[par] += 1;
                            }

                        } else {
                            childrenNotInGroup[par] = graph.OutDegree(par);
                        }
                        childrenNotInGroup[par] -= 1;
                    }
                }
                for (const auto &inEdge : InEdges(activeNode, graph)) {
                    if (parameters_.useApproxTransitiveReduction_ && (edgeMask.find(inEdge) != edgeMask.cend())) {
                        continue;
                    }

                    const VertexIdxT &par = Source(inEdge, graph);
                    if (childrenNotInGroup[par] == 0) {
                        nextVertexProcessingFifo.emplace_back(par);
                    }
                }
            }

            partition.push_back(group);

            for (const auto &node : group) {
                visited[node] = true;
            }
        }
    }

    void RunOutContraction(const GraphTIn &graph, std::vector<std::vector<VertexIdxT<GraphTIn>>> &partition) {
        using VertexIdxT = VertexIdxT<GraphTIn>;

        const std::unordered_set<EdgeDescT<GraphTIn>> edgeMask = parameters_.useApproxTransitiveReduction_
                                                                     ? LongEdgesInTrianglesParallel(graph)
                                                                     : std::unordered_set<EdgeDescT<GraphTIn>>();

        std::vector<bool> visited(graph.NumVertices(), false);

        for (const auto &topNode : TopSortView(graph)) {
            if (visited[topNode]) {
                continue;
            }

            VWorkwT<GraphTIn> workWeightOfGroup = 0;
            VMemwT<GraphTIn> memoryWeightOfGroup = 0;

            std::unordered_map<VertexIdxT, VertexIdxT> parentsNotInGroup;
            std::vector<VertexIdxT> group;

            std::deque<VertexIdxT> vertexProcessingFifo({topNode});
            std::deque<VertexIdxT> nextVertexProcessingFifo;

            unsigned depthCounter = 0;

            while ((not vertexProcessingFifo.empty()) || (not nextVertexProcessingFifo.empty())) {
                if (vertexProcessingFifo.empty()) {
                    vertexProcessingFifo = nextVertexProcessingFifo;
                    nextVertexProcessingFifo.clear();
                    depthCounter++;
                    if (depthCounter > parameters_.maxDepth_) {
                        break;
                    }
                }

                VertexIdxT activeNode = vertexProcessingFifo.front();
                vertexProcessingFifo.pop_front();

                if (graph.VertexType(activeNode) != graph.VertexType(topNode)) {
                    continue;
                }

                if (workWeightOfGroup + graph.VertexWorkWeight(activeNode) > parameters_.maxWorkWeight_) {
                    continue;
                }

                if (memoryWeightOfGroup + graph.VertexMemWeight(activeNode) > parameters_.maxMemoryWeight_) {
                    continue;
                }

                if constexpr (useArchitectureMemoryContraints) {
                    if (memoryWeightOfGroup + graph.VertexMemWeight(activeNode)
                        > maxMemoryPerVertexType_[graph.VertexType(topNode)]) {
                        continue;
                    }
                }

                group.emplace_back(activeNode);
                workWeightOfGroup += graph.VertexWorkWeight(activeNode);
                memoryWeightOfGroup += graph.VertexMemWeight(activeNode);

                for (const auto &outEdge : OutEdges(activeNode, graph)) {
                    if (parameters_.useApproxTransitiveReduction_ && (edgeMask.find(outEdge) != edgeMask.cend())) {
                        continue;
                    }

                    const VertexIdxT &child = Target(outEdge, graph);

                    if (parentsNotInGroup.find(child) != parentsNotInGroup.cend()) {
                        parentsNotInGroup[child] -= 1;

                    } else {
                        if (parameters_.useApproxTransitiveReduction_) {
                            parentsNotInGroup[child] = 0;

                            for (const auto inEdge : InEdges(child, graph)) {
                                if (edgeMask.find(inEdge) != edgeMask.cend()) {
                                    continue;
                                }
                                parentsNotInGroup[child] += 1;
                            }

                        } else {
                            parentsNotInGroup[child] = graph.InDegree(child);
                        }
                        parentsNotInGroup[child] -= 1;
                    }
                }
                for (const auto &outEdge : OutEdges(activeNode, graph)) {
                    if (parameters_.useApproxTransitiveReduction_ && (edgeMask.find(outEdge) != edgeMask.cend())) {
                        continue;
                    }

                    const VertexIdxT &child = Target(outEdge, graph);
                    if (parentsNotInGroup[child] == 0) {
                        nextVertexProcessingFifo.emplace_back(child);
                    }
                }
            }

            partition.push_back(group);

            for (const auto &node : group) {
                visited[node] = true;
            }
        }
    }
};

}    // namespace osp
