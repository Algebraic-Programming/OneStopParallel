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
#include <queue>
#include <vector>

#include "DagDivider.hpp"
#include "osp/concepts/graph_traits.hpp"
#include "osp/graph_algorithms/subgraph_algorithms.hpp"

namespace osp {

template <typename GraphT, typename ConstrGraphT>
class ConnectedComponentDivider : public IDagDivider<GraphT> {
    static_assert(IsComputationalDagV<Graph_t>, "Graph must be a computational DAG");
    static_assert(IsComputationalDagV<Constr_Graph_t>, "Constr_Graph_t must be a computational DAG");
    static_assert(is_constructable_cdag_v<Constr_Graph_t>, "Constr_Graph_t must satisfy the constructable_cdag_vertex concept");
    static_assert(std::is_same_v<vertex_idx_t<Graph_t>, vertex_idx_t<Constr_Graph_t>>,
                  "Graph_t and Constr_Graph_t must have the same vertex_idx types");

  private:
    using vertex_idx = vertex_idx_t<Graph_t>;

    std::vector<ConstrGraphT> subDags_;

    // For each component: local_idx -> global vertex
    std::vector<std::vector<vertex_idx>> vertexMapping_;

    // Global vertex -> local index
    std::vector<vertex_idx> vertexMap_;

    // Global vertex -> component id
    std::vector<unsigned> component_;

  public:
    inline std::vector<ConstrGraphT> &GetSubDags() { return subDags_; }

    inline const std::vector<ConstrGraphT> &GetSubDags() const { return subDags_; }

    inline const std::vector<std::vector<vertex_idx>> &GetVertexMapping() const { return vertex_mapping; }

    inline const std::vector<unsigned> &GetComponent() const { return component_; }

    inline const std::vector<vertex_idx> &GetVertexMap() const { return vertex_map; }

    virtual std::vector<std::vector<std::vector<vertex_idx_t<Graph_t>>>> divide(const GraphT &dag) override {
        if (dag.num_vertices() == 0) {
            return {};
        }

        bool hasMoreThanOneConnectedComponent = ComputeConnectedComponents(dag);

        std::vector<std::vector<std::vector<vertex_idx_t<Graph_t>>>> vertexMaps(1);

        if (hasMoreThanOneConnectedComponent) {
            vertexMaps[0].resize(subDags_.size());
            for (unsigned i = 0; i < subDags_.size(); ++i) {
                vertexMaps[0][i].resize(subDags_[i].num_vertices());
            }

            for (const auto &v : dag.vertices()) {
                vertex_maps[0][component[v]][vertex_map[v]] = v;
            }
        } else {
            subDags_.resize(1);
            subDags_[0] = dag;
            vertex_mapping.resize(1);
            vertex_mapping[0].resize(dag.num_vertices());
            vertex_map.resize(dag.num_vertices());

            vertexMaps[0].resize(1);
            vertexMaps[0][0].resize(dag.num_vertices());
            for (const auto &v : dag.vertices()) {
                vertexMaps[0][0][v] = v;
                vertex_map[v] = v;
                vertex_mapping[0][v] = v;
            }
        }

        return vertex_maps;
    }

    std::vector<std::vector<std::vector<vertex_idx_t<Graph_t>>>> ComputeVertexMaps(const GraphT &dag) {
        std::vector<std::vector<std::vector<vertex_idx_t<Graph_t>>>> vertexMaps(1);

        vertexMaps[0].resize(subDags_.size());
        for (unsigned i = 0; i < subDags_.size(); ++i) {
            vertexMaps[0][i].resize(subDags_[i].num_vertices());
        }

        for (const auto &v : dag.vertices()) {
            vertex_maps[0][component[v]][vertex_map[v]] = v;
        }

        return vertex_maps;
    }

    bool ComputeConnectedComponents(const GraphT &dag) {
        // Clear previous state
        subDags_.clear();
        vertex_mapping.clear();
        vertex_map.clear();
        component_.assign(dag.num_vertices(), std::numeric_limits<unsigned>::max());

        if (dag.num_vertices() == 0) {
            return false;
        }

        unsigned componentId = 0;
        for (const auto &v : dag.vertices()) {
            if (component_[v] == std::numeric_limits<unsigned>::max()) {
                component_[v] = componentId;

                // BFS for weakly connected component
                std::queue<vertex_idx> q;
                q.push(v);

                while (!q.empty()) {
                    vertex_idx current = q.front();
                    q.pop();

                    for (const auto &child : dag.children(current)) {
                        if (component[child] == std::numeric_limits<unsigned>::max()) {
                            q.push(child);
                            component[child] = component_id;
                        }
                    }

                    for (const auto &parent : dag.parents(current)) {
                        if (component[parent] == std::numeric_limits<unsigned>::max()) {
                            q.push(parent);
                            component[parent] = component_id;
                        }
                    }
                }

                ++componentId;
            }
        }

        if (componentId == 1) {
            // Single component: no need to build sub_dags or maps
            return false;
        }

        subDags_ = create_induced_subgraphs<GraphT, ConstrGraphT>(dag, component_);

        // Create the mappings between global and local vertex indices.
        vertex_mapping.resize(sub_dags.size());
        vertex_map.resize(dag.num_vertices());

        std::vector<vertex_idx> currentIndexInSubdag(subDags_.size(), 0);
        for (const auto &v : dag.vertices()) {
            unsigned compId = component_[v];
            vertex_idx localIdx = current_index_in_subdag[compId]++;
            vertex_map[v] = local_idx;

            if (vertex_mapping[comp_id].empty()) {
                vertex_mapping[comp_id].resize(sub_dags[comp_id].num_vertices());
            }

            vertex_mapping[comp_id][local_idx] = v;
        }

        return true;
    }
};

}    // namespace osp
