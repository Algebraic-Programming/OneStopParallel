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

#include <numeric>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "SequenceGenerator.hpp"
#include "SequenceSplitter.hpp"
#include "osp/auxiliary/datastructures/union_find.hpp"
#include "osp/concepts/computational_dag_concept.hpp"
#include "osp/dag_divider/DagDivider.hpp"

namespace osp {

/**
 * @class AbstractWavefrontDivider
 * @brief Base class for wavefront-based DAG dividers.
 */
template <typename GraphT>
class AbstractWavefrontDivider : public IDagDivider<GraphT> {
    static_assert(IsComputationalDagV<GraphT>, "AbstractWavefrontDivider can only be used with computational DAGs.");

  protected:
    using VertexType = VertexIdxT<GraphT>;

    const GraphT *dagPtr_ = nullptr;

    /**
     * @brief Helper to get connected components for a specific range of levels.
     * This method is now const-correct.
     */
    std::vector<std::vector<VertexType>> GetComponentsForRange(size_t startLevel,
                                                               size_t endLevel,
                                                               const std::vector<std::vector<VertexType>> &levelSets) const {
        union_find_universe_t<GraphT> uf;
        for (size_t i = startLevel; i < endLevel; ++i) {
            for (const auto vertex : level_sets[i]) {
                uf.add_object(vertex, dag_ptr_->VertexWorkWeight(vertex), dag_ptr_->VertexMemWeight(vertex));
            }
            for (const auto &node : level_sets[i]) {
                for (const auto &child : dag_ptr_->Children(node)) {
                    if (uf.is_in_universe(child)) {
                        uf.join_by_name(node, child);
                    }
                }
                for (const auto &parent : dag_ptr_->Parents(node)) {
                    if (uf.is_in_universe(parent)) {
                        uf.join_by_name(parent, node);
                    }
                }
            }
        }
        return uf.get_connected_components();
    }

    /**
     * @brief Computes wavefronts for the entire DAG.
     * This method is now const.
     */
    std::vector<std::vector<VertexType>> ComputeWavefronts() const {
        std::vector<VertexType> allVertices(dagPtr_->NumVertices());
        std::iota(all_vertices.begin(), all_vertices.end(), 0);
        return compute_wavefronts_for_subgraph(all_vertices);
    }

    /**
     * @brief Computes wavefronts for a specific subset of vertices.
     * This method is now const.
     */
    std::vector<std::vector<VertexType>> ComputeWavefrontsForSubgraph(const std::vector<VertexType> &vertices) const {
        if (vertices.empty()) {
            return {};
        }

        std::vector<std::vector<VertexType>> levelSets;
        std::unordered_set<VertexType> vertexSet(vertices.begin(), vertices.end());
        std::unordered_map<VertexType, int> inDegree;
        std::queue<VertexType> q;

        for (const auto &v : vertices) {
            in_degree[v] = 0;
            for (const auto &p : dag_ptr_->Parents(v)) {
                if (vertex_set.count(p)) {
                    in_degree[v]++;
                }
            }
            if (in_degree[v] == 0) {
                q.push(v);
            }
        }

        while (!q.empty()) {
            size_t levelSize = q.size();
            std::vector<VertexType> currentLevel;
            for (size_t i = 0; i < levelSize; ++i) {
                VertexType u = q.front();
                q.pop();
                currentLevel.push_back(u);
                for (const auto &v : dag_ptr_->Children(u)) {
                    if (vertex_set.count(v)) {
                        in_degree[v]--;
                        if (in_degree[v] == 0) {
                            q.push(v);
                        }
                    }
                }
            }
            levelSets.push_back(current_level);
        }
        return level_sets;
    }
};

}    // end namespace osp
