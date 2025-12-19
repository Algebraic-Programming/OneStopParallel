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
#include <algorithm>    // for std::reverse
#include <vector>

#include "osp/auxiliary/datastructures/union_find.hpp"

namespace osp {

/**
 * @struct WavefrontStatistics
 * @brief Holds statistical data for a single wavefront.
 */
template <typename GraphT>
struct WavefrontStatistics {
    using VertexType = VertexIdxT<GraphT>;

    std::vector<VWorkwT<GraphT>> connectedComponentsWeights_;
    std::vector<VMemwT<GraphT>> connectedComponentsMemories_;
    std::vector<std::vector<VertexType>> connectedComponentsVertices_;
};

/**
 * @class WavefrontStatisticsCollector
 * @brief Computes forward and backward wavefront statistics for a given DAG.
 */
template <typename GraphT>
class WavefrontStatisticsCollector {
    using VertexType = VertexIdxT<GraphT>;
    using UnionFind = UnionFindUniverseT<GraphT>;

  public:
    WavefrontStatisticsCollector(const GraphT &dag, const std::vector<std::vector<VertexType>> &levelSets)
        : dag_(dag), levelSets_(levelSets) {}

    /**
     * @brief Computes wavefront statistics by processing levels from start to end.
     * @return A vector of statistics, one for each level.
     */
    std::vector<WavefrontStatistics<GraphT>> ComputeForward() const {
        std::vector<WavefrontStatistics<GraphT>> stats(levelSets_.size());
        UnionFind uf;

        for (size_t i = 0; i < levelSets_.size(); ++i) {
            UpdateUnionFind(uf, i);
            CollectStatsForLevel(stats[i], uf);
        }
        return stats;
    }

    /**
     * @brief Computes wavefront statistics by processing levels from end to start.
     * @return A vector of statistics, one for each level (in original level order).
     */
    std::vector<WavefrontStatistics<GraphT>> ComputeBackward() const {
        std::vector<WavefrontStatistics<GraphT>> stats(levelSets_.size());
        UnionFind uf;

        for (size_t i = levelSets_.size(); i > 0; --i) {
            size_t levelIdx = i - 1;
            UpdateUnionFind(uf, levelIdx);
            CollectStatsForLevel(stats[levelIdx], uf);
        }
        return stats;
    }

  private:
    void UpdateUnionFind(UnionFind &uf, size_t levelIdx) const {
        // Add all vertices from the current level to the universe
        for (const auto vertex : levelSets_[levelIdx]) {
            uf.AddObject(vertex, dag_.VertexWorkWeight(vertex), dag_.VertexMemWeight(vertex));
        }
        // Join components based on edges connecting to vertices already in the universe
        for (const auto &node : levelSets_[levelIdx]) {
            for (const auto &child : dag_.Children(node)) {
                if (uf.IsInUniverse(child)) {
                    uf.JoinByName(node, child);
                }
            }
            for (const auto &parent : dag_.Parents(node)) {
                if (uf.IsInUniverse(parent)) {
                    uf.JoinByName(parent, node);
                }
            }
        }
    }

    void CollectStatsForLevel(WavefrontStatistics<GraphT> &stats, UnionFind &uf) const {
        const auto components = uf.GetConnectedComponentsWeightsAndMemories();
        stats.connectedComponentsVertices_.reserve(components.size());
        stats.connectedComponentsWeights_.reserve(components.size());
        stats.connectedComponentsMemories_.reserve(components.size());

        for (const auto &comp : components) {
            auto &[vertices, weight, memory] = comp;
            stats.connectedComponentsVertices_.emplace_back(vertices);
            stats.connectedComponentsWeights_.emplace_back(weight);
            stats.connectedComponentsMemories_.emplace_back(memory);
        }
    }

    const GraphT &dag_;
    const std::vector<std::vector<VertexType>> &levelSets_;
};

}    // end namespace osp
