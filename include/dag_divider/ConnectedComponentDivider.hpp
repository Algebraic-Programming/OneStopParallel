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
#include "model/ComputationalDag.hpp"

class ConnectedComponentDivider {

  private:
    std::vector<ComputationalDag> sub_dags;
    std::vector<std::unordered_map<unsigned, unsigned>> vertex_mapping;
    std::vector<unsigned> component;
    std::vector<unsigned> vertex_map;

  public:
    inline const std::vector<ComputationalDag> &get_sub_dags() const { return sub_dags; }

    inline const std::vector<std::unordered_map<unsigned, unsigned>> &get_vertex_mapping() const {
        return vertex_mapping;
    }

    inline const std::vector<unsigned> &get_component() const { return component; }

    inline const std::vector<unsigned> &get_vertex_map() const { return vertex_map; }

    void compute_connected_components(const ComputationalDag &dag) {

        vertex_mapping.clear();
        component = std::vector<unsigned>(dag.numberOfVertices(), std::numeric_limits<unsigned>::max());
        vertex_map = std::vector<unsigned>(dag.numberOfVertices(), 0);

        unsigned component_id = 0;
        for (unsigned v = 0; v < dag.numberOfVertices(); v++) {
            if (component[v] == std::numeric_limits<unsigned>::max()) {

                component[v] = component_id;
                std::queue<unsigned> q;
                q.push(v);

                while (!q.empty()) {

                    unsigned current = q.front();
                    q.pop();

                    for (const auto &out_edge : dag.out_edges(current)) {
                        const auto &child = out_edge.m_target;

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

        sub_dags = dag.createInducedSubgraphs(component);
        vertex_mapping.resize(sub_dags.size());
        std::vector<unsigned> current_index_in_subdag(sub_dags.size(), 0);
        for (unsigned v = 0; v < dag.numberOfVertices(); v++) {
            vertex_map[v] = current_index_in_subdag[component[v]];
            std::unordered_map<unsigned, unsigned> &current_vertex_mapping = vertex_mapping[component[v]];
            current_vertex_mapping[current_index_in_subdag[component[v]]] = v;
            ++current_index_in_subdag[component[v]];
        }

        std::cout << "size 0: " << sub_dags[0].numberOfVertices() << std::endl;
    }
};