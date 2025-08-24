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


namespace osp {

/**
 * @struct WavefrontStatistics
 * @brief Holds statistical data for a single wavefront.
 */
template<typename Graph_t>
struct WavefrontStatistics {
    using VertexType = vertex_idx_t<Graph_t>;

    std::size_t number_of_connected_components = 0;
    std::vector<v_workw_t<Graph_t>> connected_components_weights;
    std::vector<v_memw_t<Graph_t>> connected_components_memories;
    std::vector<std::vector<VertexType>> connected_components_vertices;
};

/**
 * @class WavefrontStatisticsCollector
 * @brief Computes forward and backward wavefront statistics for a given DAG.
 */
template<typename Graph_t>
class WavefrontStatisticsCollector {
    using VertexType = vertex_idx_t<Graph_t>;

public:
    WavefrontStatisticsCollector(const Graph_t &dag, const std::vector<std::vector<VertexType>> &level_sets)
        : dag_(dag), level_sets_(level_sets) {}

    std::vector<WavefrontStatistics<Graph_t>> compute_forward() {
        std::vector<WavefrontStatistics<Graph_t>> stats(level_sets_.size());
        union_find_universe_t<Graph_t> uf;

        for (size_t i = 0; i < level_sets_.size(); ++i) {
            update_union_find(uf, i);
            collect_stats_for_level(stats[i], uf);
        }
        return stats;
    }

private:
    void update_union_find(union_find_universe_t<Graph_t>& uf, size_t level_idx) {
        for (const auto vertex : level_sets_[level_idx]) {
            uf.add_object(vertex, dag_.vertex_work_weight(vertex), dag_.vertex_mem_weight(vertex));
        }
        for (const auto &node : level_sets_[level_idx]) {
            for (const auto &child : dag_.children(node)) {
                if (uf.is_in_universe(child)) uf.join_by_name(node, child);
            }
            for (const auto &parent : dag_.parents(node)) {
                if (uf.is_in_universe(parent)) uf.join_by_name(parent, node);
            }
        }
    }

    void collect_stats_for_level(WavefrontStatistics<Graph_t>& stats, const union_find_universe_t<Graph_t>& uf) {
        const auto components = uf.get_connected_components_weights_and_memories();
        stats.number_of_connected_components = components.size();
        stats.connected_components_vertices.reserve(components.size());
        stats.connected_components_weights.reserve(components.size());
        stats.connected_components_memories.reserve(components.size());
        for (const auto& comp : components) {
            stats.connected_components_vertices.emplace_back(std::get<0>(comp));
            stats.connected_components_weights.emplace_back(std::get<1>(comp));
            stats.connected_components_memories.emplace_back(std::get<2>(comp));
        }
    }

    const Graph_t &dag_;
    const std::vector<std::vector<VertexType>> &level_sets_;
};

};