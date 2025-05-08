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

#include <iostream>
#include <limits>
#include <queue>
#include <unordered_map>
#include <vector>

#include "concepts/graph_traits.hpp"
#include "graph_algorithms/subgraph_algorithms.hpp"

namespace osp {

template<typename Graph_t>
class ConnectedComponentDivider {

    static_assert(is_computational_dag_v<Graph_t>, "Graph must be a computational DAG");

  private:
    using vertex_idx = typename directed_graph_traits<Graph_t>::vertex_idx;

    std::vector<Graph_t> sub_dags;
    std::vector<std::unordered_map<vertex_idx, vertex_idx>> vertex_mapping;

    std::vector<unsigned> component; // vertex id -> component id

    std::vector<unsigned> vertex_map;

  public:
    inline const std::vector<Graph_t> &get_sub_dags() const { return sub_dags; }

    inline const std::vector<std::unordered_map<vertex_idx, vertex_idx>> &get_vertex_mapping() const {
        return vertex_mapping;
    }

    inline const std::vector<unsigned> &get_component() const { return component; }

    inline const std::vector<unsigned> &get_vertex_map() const { return vertex_map; }

    void compute_connected_components(const Graph_t &dag) {

        vertex_mapping.clear();
        component = std::vector<unsigned>(dag.numberOfVertices(), std::numeric_limits<unsigned>::max());
        vertex_map = std::vector<unsigned>(dag.numberOfVertices(), 0);

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

        sub_dags = create_induced_subgraphs(dag, component);
        vertex_mapping.resize(sub_dags.size());
        std::vector<unsigned> current_index_in_subdag(sub_dags.size(), 0);
        for (const auto &v : dag.vertices()) {
            vertex_map[v] = current_index_in_subdag[component[v]];
            std::unordered_map<vertex_idx, vertex_idx> &current_vertex_mapping = vertex_mapping[component[v]];
            current_vertex_mapping[current_index_in_subdag[component[v]]] = v;
            ++current_index_in_subdag[component[v]];
        }

        std::cout << "size 0: " << sub_dags[0].num_vertices() << std::endl;
    }
};

} // namespace osp