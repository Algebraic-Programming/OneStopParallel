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
#include <unordered_map>
#include <vector>

#include "DagDivider.hpp"
#include "osp/concepts/graph_traits.hpp"
#include "osp/graph_algorithms/subgraph_algorithms.hpp"

namespace osp {

template<typename Graph_t, typename Constr_Graph_t>
class ConnectedComponentDivider : public IDagDivider<Graph_t> {

    static_assert(is_computational_dag_v<Graph_t>, "Graph must be a computational DAG");

    static_assert(is_computational_dag_v<Constr_Graph_t>, "Constr_Graph_t must be a computational DAG");

    static_assert(is_constructable_cdag_v<Constr_Graph_t>,
                  "Constr_Graph_t must satisfy the constructable_cdag_vertex concept");

    static_assert(std::is_same_v<vertex_idx_t<Graph_t>, vertex_idx_t<Constr_Graph_t>>,
                  "Graph_t and Constr_Graph_t must have the same vertex_idx types");

  private:
    using vertex_idx = vertex_idx_t<Graph_t>;

    std::vector<Constr_Graph_t> sub_dags;
    std::vector<std::unordered_map<vertex_idx, vertex_idx>> vertex_mapping;

    std::vector<unsigned> component; // vertex id -> component id

    std::vector<vertex_idx> vertex_map;

  public:
    inline const std::vector<Constr_Graph_t> &get_sub_dags() const { return sub_dags; }

    inline const std::vector<std::unordered_map<vertex_idx, vertex_idx>> &get_vertex_mapping() const {
        return vertex_mapping;
    }

    inline const std::vector<unsigned> &get_component() const { return component; }

    inline const std::vector<vertex_idx> &get_vertex_map() const { return vertex_map; }

    virtual std::vector<std::vector<std::vector<vertex_idx_t<Graph_t>>>> divide(const Graph_t &dag) override {

        compute_connected_components(dag);

        std::vector<std::vector<std::vector<vertex_idx_t<Graph_t>>>> vertex_maps(1);
        vertex_maps[0].resize(sub_dags.size());

        for (const auto &v : dag.vertices()) {

            vertex_maps[0][component[v]].push_back(v);
        }
        return vertex_maps;
    }

    void compute_connected_components(const Graph_t &dag) {

        vertex_mapping.clear();
        component = std::vector<unsigned>(dag.num_vertices(), std::numeric_limits<unsigned>::max());
        vertex_map = std::vector<vertex_idx>(dag.num_vertices(), 0);

        unsigned component_id = 0;
        for (const auto &v : dag.vertices()) {
            if (component[v] == std::numeric_limits<unsigned>::max()) {

                component[v] = component_id;
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

                ++component_id;
            }
        }

        sub_dags = create_induced_subgraphs<Graph_t, Constr_Graph_t>(dag, component);
        vertex_mapping.resize(sub_dags.size());
        std::vector<vertex_idx> current_index_in_subdag(sub_dags.size(), 0);
        for (const auto &v : dag.vertices()) {
            vertex_map[v] = current_index_in_subdag[component[v]];
            std::unordered_map<vertex_idx, vertex_idx> &current_vertex_mapping = vertex_mapping[component[v]];
            current_vertex_mapping[current_index_in_subdag[component[v]]] = v;
            ++current_index_in_subdag[component[v]];
        }
    }
};

} // namespace osp