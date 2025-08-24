/*
Copyright 2024 Huawei Technologies Co., Ltd.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,.
See the License for the specific language governing permissions and
limitations under the License.

@author Toni Boehnlein, Benjamin Lozes, Pal Andras Papp, Raphael S. Steiner
*/
#pragma once

#include <vector>
#include <numeric>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include "osp/auxiliary/datastructures/union_find.hpp"
#include "SequenceSplitter.hpp"
#include "SequenceGenerator.hpp"
#include "osp/dag_divider/DagDivider.hpp"

namespace osp {

/**
 * @class AbstractWavefrontDivider
 * @brief Base class for wavefront-based DAG dividers.
 */
template<typename Graph_t>
class AbstractWavefrontDivider : public IDagDivider<Graph_t> {
    static_assert(is_computational_dag_v<Graph_t>,
                  "AbstractWavefrontDivider can only be used with computational DAGs.");

protected:
    using VertexType = vertex_idx_t<Graph_t>;

    const Graph_t* dag_ptr_ = nullptr;

    /**
     * @brief Helper to get connected components for a specific range of levels.
     * This method is now const-correct.
     */
    std::vector<std::vector<VertexType>> get_components_for_range(
        size_t start_level, size_t end_level,
        const std::vector<std::vector<VertexType>>& level_sets) const {
        
        union_find_universe_t<Graph_t> uf;
        for (size_t i = start_level; i < end_level; ++i) {
            for (const auto vertex : level_sets[i]) {
                uf.add_object(vertex, dag_ptr_->vertex_work_weight(vertex), dag_ptr_->vertex_mem_weight(vertex));
            }
            for (const auto& node : level_sets[i]) {
                for (const auto& child : dag_ptr_->children(node)) {
                    if (uf.is_in_universe(child)) uf.join_by_name(node, child);
                }
                for (const auto& parent : dag_ptr_->parents(node)) {
                    if (uf.is_in_universe(parent)) uf.join_by_name(parent, node);
                }
            }
        }
        return uf.get_connected_components();
    }

    /**
     * @brief Computes wavefronts for the entire DAG.
     * This method is now const.
     */
    std::vector<std::vector<VertexType>> compute_wavefronts(const Graph_t& dag) const {
        std::vector<VertexType> all_vertices(dag.num_vertices());
        std::iota(all_vertices.begin(), all_vertices.end(), 0);
        return compute_wavefronts_for_subgraph(all_vertices);
    }

    /**
     * @brief Computes wavefronts for a specific subset of vertices.
     * This method is now const.
     */
    std::vector<std::vector<VertexType>> compute_wavefronts_for_subgraph(
        const std::vector<VertexType>& vertices) const {
        
        if (vertices.empty()) return {};

        std::vector<std::vector<VertexType>> level_sets;
        std::unordered_set<VertexType> vertex_set(vertices.begin(), vertices.end());
        std::unordered_map<VertexType, int> in_degree;
        std::queue<VertexType> q;

        for (const auto& v : vertices) {
            in_degree[v] = 0;
            for (const auto& p : dag_ptr_->parents(v)) {
                if (vertex_set.count(p)) {
                    in_degree[v]++;
                }
            }
            if (in_degree[v] == 0) {
                q.push(v);
            }
        }

        while (!q.empty()) {
            size_t level_size = q.size();
            std::vector<VertexType> current_level;
            for (size_t i = 0; i < level_size; ++i) {
                VertexType u = q.front();
                q.pop();
                current_level.push_back(u);
                for (const auto& v : dag_ptr_->children(u)) {
                    if (vertex_set.count(v)) {
                        in_degree[v]--;
                        if (in_degree[v] == 0) {
                            q.push(v);
                        }
                    }
                }
            }
            level_sets.push_back(current_level);
        }
        return level_sets;
    }
};

};