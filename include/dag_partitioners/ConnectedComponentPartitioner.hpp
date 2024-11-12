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

class ConnectedComponentPartitioner {

  private:

    std::vector<ComputationalDag> sub_dags;
    std::vector<std::unordered_map<unsigned, unsigned>> vertex_mapping;
    std::vector<unsigned> component;
    std::vector<unsigned> vertex_map;


  public:

    inline const std::vector<ComputationalDag>& get_sub_dags() const {
        return sub_dags;
    }

    inline const std::vector<std::unordered_map<unsigned, unsigned>>& get_vertex_mapping() const {
        return vertex_mapping;
    }

    inline const std::vector<unsigned>& get_component() const {
        return component;
    }

    inline const std::vector<unsigned>& get_vertex_map() const {
        return vertex_map;
    }

    void compute_connected_components(const ComputationalDag &dag) {

        sub_dags.clear();
        vertex_mapping.clear();
        component = std::vector<unsigned>(dag.numberOfVertices(), std::numeric_limits<unsigned>::max());
        vertex_map = std::vector<unsigned>(dag.numberOfVertices(), 0);

        for (unsigned v = 0; v < dag.numberOfVertices(); v++) {
            if (component[v] == std::numeric_limits<unsigned>::max()) {

                sub_dags.push_back(ComputationalDag());
                vertex_mapping.push_back(std::unordered_map<unsigned, unsigned>());

                ComputationalDag &current_sub_dag = sub_dags.back();
                std::unordered_map<unsigned, unsigned>& current_vertex_mapping = vertex_mapping.back();

                component[v] = sub_dags.size() - 1;
                current_sub_dag.addVertex(dag.nodeWorkWeight(v), dag.nodeCommunicationWeight(v), dag.nodeMemoryWeight(v), dag.nodeType(v));
                current_vertex_mapping[0] = v;
                vertex_map[v] = 0;

                std::queue<unsigned> q;
                q.push(v);

                while (!q.empty()) {

                    unsigned current = q.front();
                    q.pop();

                    for (const auto &out_edge : dag.out_edges(current)) {
                        const auto child = out_edge.m_target;

                        if (component[child] == std::numeric_limits<unsigned>::max()) {    
                            q.push(child);
                            component[child] = sub_dags.size() - 1;
                            current_sub_dag.addVertex(dag.nodeWorkWeight(child), dag.nodeCommunicationWeight(child), dag.nodeMemoryWeight(child), dag.nodeType(child));
                            
                            current_vertex_mapping[current_sub_dag.numberOfVertices() - 1] = child;
                            vertex_map[child] = current_sub_dag.numberOfVertices() - 1;

                        }

                        current_sub_dag.addEdge(vertex_map[current], vertex_map[child], dag.edgeCommunicationWeight(out_edge));
                    }

                    for (const auto &parent : dag.parents(current)) {
                       
                        if (component[parent] == std::numeric_limits<unsigned>::max()) {
                            q.push(parent);
                            component[parent] = sub_dags.size() - 1;
                            current_sub_dag.addVertex(dag.nodeWorkWeight(parent), dag.nodeCommunicationWeight(parent), dag.nodeMemoryWeight(parent), dag.nodeType(parent));
                            
                            current_vertex_mapping[current_sub_dag.numberOfVertices() - 1] = parent;
                            vertex_map[parent] = current_sub_dag.numberOfVertices() - 1;
                        }
                    }
                }
            }
        }
    }
};