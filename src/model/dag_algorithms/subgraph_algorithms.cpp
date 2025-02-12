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
#include "model/dag_algorithms/subgraph_algorithms.hpp"


ComputationalDag dag_algorithms::create_induced_subgraph(const ComputationalDag &dag, const std::vector<unsigned> &selected_nodes) {

    ComputationalDag subdag;
    std::map<unsigned, unsigned> local_idx;

    for (unsigned node : selected_nodes) {
        local_idx[node] = subdag.numberOfVertices();
        subdag.addVertex(dag.nodeWorkWeight(node), dag.nodeCommunicationWeight(node), dag.nodeMemoryWeight(node),
                         dag.nodeType(node));
    }

    std::set<unsigned> selected_nodes_set(selected_nodes.begin(), selected_nodes.end());

    for (unsigned node : selected_nodes)
        for (const auto &in_edge : dag.in_edges(node)) {
            const unsigned pred = in_edge.m_source;
            if (selected_nodes_set.find(pred) != selected_nodes_set.end())
                subdag.addEdge(local_idx[pred], local_idx[node], dag.edgeCommunicationWeight(in_edge));
        }

    return subdag;
}

ComputationalDag dag_algorithms::create_induced_subgraph_sorted(const ComputationalDag &dag, std::vector<unsigned> &selected_nodes) {

    std::sort(selected_nodes.begin(), selected_nodes.end());

    ComputationalDag subdag;
    std::map<unsigned, unsigned> local_idx;

    for (unsigned node : selected_nodes) {
        local_idx[node] = subdag.numberOfVertices();
        subdag.addVertex(dag.nodeWorkWeight(node), dag.nodeCommunicationWeight(node), dag.nodeMemoryWeight(node),
                         dag.nodeType(node));
    }

    for (unsigned node : selected_nodes) {
        for (const auto &in_edge : dag.in_edges(node)) {
            const unsigned pred = in_edge.m_source;
            if (std::binary_search(selected_nodes.begin(), selected_nodes.end(), pred)) {
                subdag.addEdge(local_idx[pred], local_idx[node], dag.edgeCommunicationWeight(in_edge));
            }
        }
    }

    return subdag;
}

