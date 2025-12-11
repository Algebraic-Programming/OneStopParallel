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
template <typename Graph_t>
struct WavefrontStatistics {
    using VertexType = vertex_idx_t<Graph_t>;

    std::vector<v_workw_t<Graph_t>> connected_components_weights;
    std::vector<v_memw_t<Graph_t>> connected_components_memories;
    std::vector<std::vector<VertexType>> connected_components_vertices;
};

/**
 * @class WavefrontStatisticsCollector
 * @brief Computes forward and backward wavefront statistics for a given DAG.
 */
template <typename Graph_t>
class WavefrontStatisticsCollector {
    using VertexType = vertex_idx_t<Graph_t>;
    using UnionFind = union_find_universe_t<Graph_t>;

  public:
    WavefrontStatisticsCollector(const Graph_t &dag, const std::vector<std::vector<VertexType>> &level_sets)
        : dag_(dag), level_sets_(level_sets) {}

    /**
     * @brief Computes wavefront statistics by processing levels from start to end.
     * @return A vector of statistics, one for each level.
     */
    std::vector<WavefrontStatistics<Graph_t>> compute_forward() const {
        std::vector<WavefrontStatistics<Graph_t>> stats(level_sets_.size());
        UnionFind uf;

        for (size_t i = 0; i < level_sets_.size(); ++i) {
            update_union_find(uf, i);
            collect_stats_for_level(stats[i], uf);
        }
        return stats;
    }

    /**
     * @brief Computes wavefront statistics by processing levels from end to start.
     * @return A vector of statistics, one for each level (in original level order).
     */
    std::vector<WavefrontStatistics<Graph_t>> compute_backward() const {
        std::vector<WavefrontStatistics<Graph_t>> stats(level_sets_.size());
        UnionFind uf;

        for (size_t i = level_sets_.size(); i > 0; --i) {
            size_t level_idx = i - 1;
            update_union_find(uf, level_idx);
            collect_stats_for_level(stats[level_idx], uf);
        }
        return stats;
    }

  private:
    void update_union_find(UnionFind &uf, size_t level_idx) const {
        // Add all vertices from the current level to the universe
        for (const auto vertex : level_sets_[level_idx]) {
            uf.add_object(vertex, dag_.vertex_work_weight(vertex), dag_.vertex_mem_weight(vertex));
        }
        // Join components based on edges connecting to vertices already in the universe
        for (const auto &node : level_sets_[level_idx]) {
            for (const auto &child : dag_.children(node)) {
                if (uf.is_in_universe(child)) { uf.join_by_name(node, child); }
            }
            for (const auto &parent : dag_.parents(node)) {
                if (uf.is_in_universe(parent)) { uf.join_by_name(parent, node); }
            }
        }
    }

    void collect_stats_for_level(WavefrontStatistics<Graph_t> &stats, UnionFind &uf) const {
        const auto components = uf.get_connected_components_weights_and_memories();
        stats.connected_components_vertices.reserve(components.size());
        stats.connected_components_weights.reserve(components.size());
        stats.connected_components_memories.reserve(components.size());

        for (const auto &comp : components) {
            auto &[vertices, weight, memory] = comp;
            stats.connected_components_vertices.emplace_back(vertices);
            stats.connected_components_weights.emplace_back(weight);
            stats.connected_components_memories.emplace_back(memory);
        }
    }

    const Graph_t &dag_;
    const std::vector<std::vector<VertexType>> &level_sets_;
};

}    // end namespace osp
